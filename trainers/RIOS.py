import torch
import numpy as np
import random
import math
import logging
import time
import os
from datasets import dataset_entrypoints
from models import model_entrypoints

from util.misc import NestedTensor, reduce_dict
import gc
import wandb
import torch.cuda.amp as amp
from util.misc import all_gather, SmoothedValue, MetricLogger, reduce_scalar ,setup_for_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import torch.distributed as dist

import shutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import decode
from models.swin_transformer import compute_mask
from util.debug import visualize_test_batch
from tqdm import tqdm
from PIL import Image

# region helpers
def get_optimizer(param_dicts, configs):
    if configs.optimizer == 'AdamW':
        return torch.optim.AdamW(param_dicts,
                                 lr=configs.lr,
                                 weight_decay=configs.wd
                                 )
    else:
        raise NotImplementedError

def get_scheduler(optimizer, configs):
    if configs.scheduler == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    **vars(configs.scheduler_fn_kwargs))
    elif configs.scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               **vars(configs.scheduler_fn_kwargs))
    elif configs.scheduler == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **vars(configs.scheduler_fn_kwargs))
    else:
        raise NotImplementedError

def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample

def save_beta_schedule(betas, alphas_cumprod, timesteps, directory):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    ax[0].plot(torch.arange(timesteps),betas)
    ax[1].plot(torch.arange(timesteps),alphas_cumprod)
    plt.savefig(f'{directory}/beta_and_alpha_cum.png')
    plt.close()
    
def init_process_group_and_set_device(world_size, process_id, device_id, config):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    config.world_size = world_size
    config.rank = process_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device

# endregion
# resume_eval_first = True
class RIOS_Trainer:
    def __init__(self, configs, process_id, device_id, num_processes):
        self.configs = configs
        
        # distributed
        self.world_size = num_processes
        self.distributed = num_processes > 1
        self.process_id = process_id
        self.is_main_process = (process_id == 0)
        self.device =init_process_group_and_set_device(world_size=num_processes,
                                                       process_id=process_id,
                                                       device_id=device_id,
                                                       config = configs)
        
        # seed
        seed = configs.seed + configs.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
        # data
        self.dataset_name = configs.dataset_name
        create_dataset = dataset_entrypoints(configs.data.dataset_entrypoint)
        train_dataset, test_dataset = create_dataset(**vars(configs.data), is_distributed=self.distributed)
        if self.distributed:
            self.sampler_train = DistributedSampler(train_dataset,
                                                    num_replicas=configs.world_size,
                                                    rank = configs.rank,
                                                    shuffle=True,
                                                    seed=configs.seed,
                                                    drop_last=False)
        else:
            self.sampler_train = None
        self.trn_loader = DataLoader(train_dataset,
                                     batch_size=configs.optimization.batch_size,
                                     sampler=self.sampler_train,
                                     collate_fn=train_dataset.collator, 
                                     num_workers=configs.runtime.num_workers,
                                     pin_memory=True,
                                     shuffle=self.sampler_train is None,
                                     persistent_workers=True)
        sampler_val = DistributedSampler(test_dataset, 
                                        num_replicas=configs.world_size, 
                                        rank=configs.rank, 
                                        shuffle=False) 
        self.test_loader = DataLoader(test_dataset, 
                                      batch_size=configs.runtime.eval_batch_size, 
                                    sampler=sampler_val, drop_last=False,
                                    collate_fn=test_dataset.collator,
                                    num_workers=configs.runtime.num_workers,
                                    pin_memory=True,
                                    persistent_workers=True)
    
        # model
        create_model = model_entrypoints(configs.model.model_entrypoint)
        model, postprocessor = create_model(device = configs.device,
                            model_configs = configs.model,
                            fig_directory=configs.output_dir,
                            dataset_name=configs.dataset_name)
        model.to(self.device)
        
        model_without_ddp = model
        if self.distributed:
            model = DDP(model, device_ids=[device_id],find_unused_parameters=True)
            model_without_ddp = model.module
        self.model = model
        self.backbone_name = configs.model.backbone_name
        self.checkpoint_dir_path = configs.output_dir
        if self.is_main_process:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f'number of params:{n_parameters}')
            
        self.postprocessor = postprocessor
        # evaluate
        if self.dataset_name == 'a2ds' or self.dataset_name == 'jhmdb':
            self.evaluate = self.evaluate_a2d_sentences
        elif self.dataset_name == 'youtube':
            self.evaluate = self.evaluate_refer_youtube_vos
        else:
            raise NotImplementedError(f'Dataset {self.dataset_name} not found!')
        # optimizer, scheduler, amp 
        # param_dicts = [
        #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and "text_encoder" not in n and p.requires_grad]},
        #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        #     "lr": configs.optimization.lr_backbone},
        #     {"params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
        #         "lr": configs.optimization.text_encoder_lr}, 
        # ]
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "vid_backbone" in n and p.requires_grad],
            "lr": configs.optimization.lr_backbone},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "text_backbone" in n and p.requires_grad],
                "lr": configs.optimization.text_encoder_lr}, 
        ]
        self.optimizer = get_optimizer(param_dicts, configs.optimization)

        self.scheduler = get_scheduler(self.optimizer, configs.optimization)
        self.grad_scaler = amp.GradScaler(enabled=configs.optimization.enable_amp)
        self.max_norm = configs.optimization.clip_max_norm
        if configs.dataset_name == 'a2ds':
            assert configs.wandb_project_name == 'referdiff'
        if self.is_main_process:
            wandb.init(project=configs.wandb_project_name,
               group=configs.wandb_group_name, 
               # job_type='',  
               id=configs.wandb_id, 
               resume=configs.wandb_resume, 
               name=configs.wandb_name, # config_postfix
               config=configs,
               mode=configs.wandb_mode,
            )
            logging.info(configs)
        
        # count record
        self.epoch = 0
        self.iteration = 0
        self.total_epochs = configs.optimization.epochs
        self.total_iterations = self.total_epochs * len(self.trn_loader) 
        self.best_mAP = 0
        self.best_loss = math.inf
        if configs.model.checkpoint_path != '':
            self.load_checkpoint(configs.model.checkpoint_path)
        self.wandb_step = configs.optimization.batch_size / 4.
        
    def train(self):
        assert self.configs.mode == 'train'
        configs = self.configs
        for self.epoch in range(self.epoch, self.total_epochs):
            self.model.train()
            if self.distributed:
                self.sampler_train.set_epoch(self.epoch)
                
            if self.is_main_process:
                metric_logger = MetricLogger(delimiter='\t')
                metric_logger.add_meter('iteration_time',SmoothedValue(window_size=1,fmt='{value:2f}',handler='value') )
                metric_logger.add_meter('loss',SmoothedValue(window_size=5, fmt='{value:.6f}',handler='value') )
                metric_logger.add_meter('lr', SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value'))
                metric_logger.add_meter('lr_backbone', SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value'))
                
            epoch_header = f'Epoch[{self.epoch:{int(math.log10(self.total_epochs))+1}}/{self.total_epochs}]'
            debug_exit_iteration = False
            for idx, bactch_dict in enumerate(self.trn_loader):
                if debug_exit_iteration == True:
                    break
                # NT(t b 3 H W)
                samples = bactch_dict['samples'].to(self.device)
                # list[ [None..None], [Dict,dict] ], t
                targets = to_device(bactch_dict['targets'], self.device)
                # list[str], b
                text_queries = bactch_dict['text_queries']
                # tensor(int) for a2ds
                valid_indices = torch.tensor([i for i, v in enumerate(targets) if None not in v]).to(self.device)
                
                # list[list[dict], b], valid_time
                targets = [targets[i] for i in valid_indices]
                
                start = time.time()
                # TODO amp fp16 for fast training
                with amp.autocast(enabled=configs.optimization.enable_amp):
                    diffusion_loss = self.model(samples, valid_indices, text_queries, targets)

                loss_reduced = reduce_scalar(diffusion_loss)    

                if not math.isfinite(loss_reduced):
                    raise RuntimeError('Loss is {}, stop training'.format(loss_reduced))
                    
                self.optimizer.zero_grad()
                self.grad_scaler.scale(diffusion_loss).backward()
                if self.max_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.max_norm, 
                                                   error_if_nonfinite=False)
                self.grad_scaler.step(self.optimizer,)
                self.grad_scaler.update()
                iteration_time = time.time() - start   
                if self.iteration % 2000 == 0:
                    self.clear_memory()
                    
                if self.is_main_process:
                    metric_logger.update(loss=loss_reduced)
                    metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                    metric_logger.update(iteration_time=iteration_time)
                    metric_logger.update(lr_backbone=self.optimizer.param_groups[1]["lr"])
                    self.iteration +=1
                    header = f'{epoch_header} Itera[{(self.iteration):{int(math.log10(self.total_iterations))+1}d}/{self.total_iterations}]'
                    logging.info(f'{header} {str(metric_logger)}')
                    
                    is_not_last_iteration = (idx != len(self.trn_loader)-1)
                    # wandb.log(metric_logger.to_dict(), commit=True, step=int(self.iteration*self.wandb_step))
                    wandb.log(metric_logger.to_dict(), commit=is_not_last_iteration, step=self.iteration)
             
            self.clear_memory()           
            # eval_metrics = {'P@0.9':0,}
            # wandb.log(eval_metrics, commit=True, step=self.iteration)
            # self.save_checkpoint(0.)
            # criterion = 0.
            # self.scheduler.step(criterion)
            # try:
            #     eval_metrics = self.evaluate()
            # except:
            #     if self.is_main_process:
            #         self.save_checkpoint(0.)
            try:
                eval_metrics = self.evaluate()
            except:
                if self.is_main_process:
                    self.save_checkpoint(0.)
                raise ValueError()
            # step_criterion = 'P@0.9'
            if self.is_main_process:
                if self.dataset_name == 'a2ds':
                    mAP_score = eval_metrics.get('mAP 0.5:0.95')
                    self.save_checkpoint(mAP_score)
                else:  # refer-youtube-vos:
                    self.save_checkpoint(metric_logger.loss.total)
                wandb.log(eval_metrics, commit=True, step=self.iteration)
            # criterion = eval_metrics.get(step_criterion)
            self.scheduler.step()
            self.clear_memory() 
            if self.distributed:
                dist.barrier() 

                
    @torch.no_grad()
    def evaluate_a2d_sentences(self):
        self.model.eval()
        # list[{'image_id', 'segmentation: (Original H W)', 'score: scalar'}], num_batches* t_target * b * num_queries
        predictions = []
        for batch_dict in tqdm(self.test_loader):
            samples = batch_dict['samples'].to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            text_queries = batch_dict['text_queries']

            # keep only the valid targets (targets of frames which are annotated):
            valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t], device=self.device)
            targets = [targets[i] for i in valid_indices.tolist()]
            
            # 'pred_masks': T(t b n h w), [-, +] float
            # 'pred_is_referred': T(t b n 2)
            if isinstance(self.model, DDP):
                outputs = self.model.module.sample(samples, valid_indices, text_queries)
            else:
                outputs = self.model.sample(samples, valid_indices, text_queries)
            outputs.pop('aux_outputs', None)
            outputs, targets = flatten_temporal_batch_dims(outputs, targets)
            # visualize_test_batch(path, outputs=outputs, targets=targets, samples=samples.tensors, batch_size=4, window_size=7, num_valid=1)
            # list[{'scores':(num_queries, ), 'masks':(num_quries, 1, Original_H W), 
            # 'rle_masks':list[{'size','counts'}], num_queries}], t_target*b
            processed_outputs = self.postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                   resized_sample_sizes=[t['size'] for t in targets],
                                                   orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]
            for p, image_id in zip(processed_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})

        if self.distributed:
            # gather and merge predictions from all processes:
            gathered_pred_lists = all_gather(predictions)
            predictions = [p for p_list in gathered_pred_lists for p in p_list]
        eval_metrics = {}
        if self.is_main_process:
            coco_gt = COCO(self.configs.data.dataset_coco_gt_format_path)
            coco_pred = coco_gt.loadRes(predictions)
            coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
            coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
            ap_metrics = coco_eval.stats[:6]
            eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
            if self.configs.runtime.calculate_precision_and_iou_metrics:
                precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
                eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
                eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
            logging.info(eval_metrics)
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        return eval_metrics

    @torch.no_grad()
    def evaluate_refer_youtube_vos(self):
        self.model.eval()
        predictions = []
        for batch_dict in tqdm(self.test_loader, disable=not self.is_main_process):
            samples = batch_dict['samples'].to(self.device)
            text_queries = batch_dict['text_queries']
            valid_indices = torch.arange(len(samples.tensors)).to(self.device)
            if isinstance(self.model, DDP):
                outputs = self.model.module.sample(samples, valid_indices, text_queries)
            else:
                outputs = self.model.sample(samples, valid_indices, text_queries)
            outputs.pop('aux_outputs', None)
            videos_metadata = batch_dict['videos_metadata']
            sample_shape_with_padding = samples.tensors.shape[-2:]
            preds_by_video = self.postprocessor(outputs, videos_metadata, sample_shape_with_padding)
            predictions.extend(preds_by_video)
        # save the predictions as zip
        validation_output_dir = os.path.join(self.checkpoint_dir_path, 'validation_outputs')
        epoch_validation_output_dir = os.path.join(validation_output_dir, f'epoch_{self.epoch}')
        annotations_dir = os.path.join(epoch_validation_output_dir, 'Annotations')
        print('saving predictions...')
        for p in tqdm(predictions, disable=not self.is_main_process):
            pred_dir_path = os.path.join(annotations_dir, p['video_id'], p['exp_id'])
            os.makedirs(pred_dir_path, exist_ok=True)
            for f_mask, f_idx in zip(p['pred_masks'], p['frame_indices']):
                pred_mask_path = os.path.join(pred_dir_path, f'{f_idx}.png')
                pred_mask = Image.fromarray((255 * f_mask.squeeze()).numpy())
                pred_mask.save(pred_mask_path)
        if self.distributed:
            dist.barrier()  # make sure all processes finished saving their predictions before creating the zip file
        if self.is_main_process:
            print('creating a zip file with the predictions...')
            # create zip file to be submitted to refer-youtube-vos validation server:
            zip_file_path = os.path.join(validation_output_dir, f'submission_epoch_{self.epoch}')
            shutil.make_archive(zip_file_path, 'zip', root_dir=epoch_validation_output_dir, base_dir='Annotations')
            print('a zip file was successfully created.')
            shutil.rmtree(epoch_validation_output_dir)  # remove the uncompressed annotations for memory efficiency
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        return {}  # return an empty metrics dict as all validation metrics will be computed on the server later
  

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self.epoch = checkpoint['epoch'] # the epoch after the one saved is about to begin
        self.epoch = checkpoint['epoch'] + 1
        self.iteration = checkpoint['iteration']
        self.total_epochs = checkpoint['total_epochs']
        self.total_iterations = checkpoint['total_iterations']
        if self.dataset_name == 'a2ds':
            self.best_mAP = checkpoint['best_mAP']
        else:
            # refer-youtube-vos
            self.best_loss = checkpoint['best_loss']
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    def save_checkpoint(self, epoch_score):
        if not self.is_main_process:
            return
        is_best = False
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'total_iterations': self.total_iterations,
            'total_epochs': self.total_epochs,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        if self.dataset_name == 'a2ds':
            is_best_mAP = epoch_score > self.best_mAP
            if is_best_mAP:
                self.best_mAP = epoch_score
                is_best = True
            checkpoint_dict['best_mAP'] = self.best_mAP
        else:  # refer-youtube-vos
            is_best_loss = epoch_score < self.best_loss
            if is_best_loss:
                self.best_loss = epoch_score
                is_best = True
            checkpoint_dict['best_loss'] = self.best_loss
        filename = self.get_checkpoint_filename()
        torch.save(checkpoint_dict, filename)
        logging.info(f'saved checkpoint: {filename}')
        if is_best:
            best_filename = self.get_checkpoint_filename(is_best=True)
            shutil.copyfile(filename, best_filename)
        # self.remove_extra_checkpoints()
        
    def get_checkpoint_filename(self, is_best=False):
        basename = 'best' if is_best else f'{self.epoch:02d}'
        return os.path.join(self.checkpoint_dir_path, f'{basename}.pth.tar')

    def remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self.checkpoint_dir_path))
        max_num_checkpoints = 15
        num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self.checkpoint_dir_path, filename))
    
    def clear_memory(self):
        if self.backbone_name == 'swin-t':
            compute_mask.cache_clear()  # empty cache of SwinT
        gc.collect()
        torch.cuda.empty_cache() 




def flatten_temporal_batch_dims(outputs, targets):
    # pred_mask: t_target b num_queries H/4 W/4
    # pred_is_referred: t_target b num_queries 2
    # targets: list[list[dict], b], t_valid
    for k in outputs.keys():
        if isinstance(outputs[k], torch.Tensor):
            outputs[k] = outputs[k].flatten(0, 1)
        else:  # list
            outputs[k] = [i for step_t in outputs[k] for i in step_t]
    # list[dict], t_target*b
    targets = [frame_t_target for step_t in targets for frame_t_target in step_t]
    # pred_mask: t_target*b num_queries, H/4, W/4
    return outputs, targets  

def compute_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    iou = (intersection + EPS) / (union + EPS)  # EPS is used to avoid division by zero
    return iou, intersection, union


def calculate_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in coco_gt.imgs.keys():  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_mask = decode(gt_annot['segmentation'])
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_mask = decode(pred_annot['segmentation'])
        iou, intersection, union = compute_iou(torch.tensor(pred_mask).unsqueeze(0),
                                               torch.tensor(gt_mask).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou