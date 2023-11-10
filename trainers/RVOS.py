import torch
import numpy as np
import random
import math
import logging
import time
import os
import sys
from data_schedule import data_schedule_entrypoints
from models import model_entrypoint
from util.misc import NestedTensor, reduce_dict, to_device
import gc
import wandb
from util.misc import all_gather, SmoothedValue, MetricLogger, reduce_scalar ,setup_for_distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import datetime

from models.video_swin import compute_mask
from tqdm import tqdm
from PIL import Image
from .registry import register_task
__all__ = ['rvos']
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def init_process_group_and_set_device(world_size, process_id, device_id):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    if world_size > 1:
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        setup_for_distributed(process_id == 0)
    return device

class Trainer:
    def __init__(self, process_id, device_id, num_processes,
                 output_directory,
                 data_configs,
                 model_configs,
                 trainer_ckpt,
                 wandb_configs,
                 visualize_configs,
                 seed,
                 resume):
            
        self.distributed = num_processes > 1
        self.is_main_process = (process_id == 0)
        self.device = init_process_group_and_set_device(world_size=num_processes, process_id=process_id, device_id=device_id,)

        seed = seed + process_id
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        # data
        create_data_schedule = data_schedule_entrypoints(data_configs['name'])
        self.train_loader, self.train_sampler,\
        self.validate_loader, self.validate_function, \
        self.test_loader, self.test_function \
            = create_data_schedule(data_configs, is_distributed=self.distributed, num_processes=num_processes, process_id=process_id)
        # 定义: validate set是用来调整model和参数的
        # [train, None, test]: 比如yrvos 要进行真实测试
        # [train, vali, None]: 比如不想生成zip的youtube-vos / a2ds
        # [train, vali, test]: 比如yrvos 分出来一点调整参数

        # model
        create_model = model_entrypoint(model_configs['name'])

        model, optimizer, scheduler, scheduler_unit = create_model(device=self.device, configs=model_configs)  # optimization属于model的部分
        if self.distributed:
            model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
        self.model = model 
        self.optimizer = optimizer
        self.clip_gradient_norm = model_configs['optimization']['clip_max_norm'] # 0.1
        self.scheduler = scheduler
        self.schedule_unit = scheduler_unit
         
        # trainer
        self.out_dir = output_directory
        self.total_epochs = model_configs['optimization']['epochs']
        self.total_iterations = self.total_epochs * len(self.train_loader) 
        self.epoch = -1
        self.iteration = -1
        if trainer_ckpt != "":
            self.load_ckpt(trainer_ckpt, resume=resume)
                        
        # visualize
        # 可视化model output 和 中间结果, to be added
        self.trainset_vis_idxs = visualize_configs['trainset_idxs']
        if self.validate_loader is not None:
            self.validateset_vis_idxs = visualize_configs['validateset_idxs']
        if self.test_loader is not None:
            self.testset_vis_idxs = visualize_configs['testset_idxs']
            
        if self.is_main_process:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f'number of params:{n_parameters}')
            if wandb_configs is not None:
                wandb.init(   
                    project=wandb_configs['project'],
                    group=wandb_configs['group'],   
                    id=wandb_configs['id'], 
                    resume=wandb_configs['resume'], 
                    name=wandb_configs['name'],
                    config=wandb_configs['configs'],
                    mode=wandb_configs['mode'],
                )
        
    def train(self):   
        for self.epoch in range(self.epoch + 1, self.total_epochs):
            self.model.train()
            # 每个进程获得相同的一个permutation后, 然后间隔抽取
            self.train_sampler.set_epoch(self.epoch)
            if self.is_main_process:
                metric_logger = MetricLogger(delimiter='\t')
                metric_logger.add_meter('iteration_time',SmoothedValue(window_size=1,fmt='{value:2f}',handler='value') )
                metric_logger.add_meter('loss_value',SmoothedValue(window_size=1, fmt='{value:.6f}',handler='value') )
                for gn in range(len(self.optimizer.param_groups)):
                    if len(self.optimizer.param_groups[gn]['params']) != 0:
                        metric_logger.add_meter(f'lr_group{gn}', SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value'))
                metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.6f}',handler='value') )
            
            epoch_header = f'Epoch[{self.epoch:{int(math.log10(self.total_epochs))+1}}/{self.total_epochs-1}]'
            debug_data_loding = False
            debug_step_iteration = False
            for idx, batch_dict in enumerate(self.train_loader):
                if debug_data_loding:
                    continue
                if debug_step_iteration:
                    break
                self.optimizer.zero_grad() #对于gradient sampling来说, 内部会先进行梯度下降

                samples = to_device(batch_dict['samples'], self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_query']
                auxiliary = to_device(batch_dict['auxiliary'], self.device)
                if idx < 10:
                    logging.info(auxiliary['sample_idx'])
                
                start = time.time()
                loss_dict_unscaled, loss_dict_scaled, gradient_norm = self.model(samples, text_queries, auxiliary, targets) 
                
                if self.clip_gradient_norm > 0:
                    its = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient_norm)
                    assert torch.isclose(its, gradient_norm).item()
                self.optimizer.step()                
                iteration_time = time.time() - start 

                # logging
                loss_dict_unscaled_reduced = reduce_dict(loss_dict_unscaled) 
                if self.distributed: 
                    dist.barrier()
                    dist.all_reduce(gradient_norm)
                loss_value = sum(reduce_dict(loss_dict_scaled).values()).item()  
                self.iteration +=1
                if self.is_main_process:
                    metric_logger.update(loss_value=loss_value, 
                                         iteration_time=iteration_time,
                                         grad_norm=gradient_norm,
                                         **loss_dict_unscaled_reduced)
                    
                    for gn in range(len(self.optimizer.param_groups)):
                        if len(self.optimizer.param_groups[gn]['params']) != 0:
                            metric_logger.update(**{f'lr_group{gn}':self.optimizer.param_groups[gn]["lr"]})

                    eta = datetime.timedelta(seconds=(self.total_iterations - self.iteration) * iteration_time)

                    header = f'Eta: {str(eta)}{epoch_header} Itera[{(self.iteration):{int(math.log10(self.total_iterations))+1}d}/{self.total_iterations}]'
                    logging.info(f'{header} {str(metric_logger)}')
                    wandb.log(metric_logger.to_dict(), step=self.iteration+self.epoch)

                if ((idx % 2000 == 0) and (self.iteration!=0)) or (idx == (len(self.train_loader)-1)):
                    compute_mask.cache_clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                
                if (self.schedule_unit == 'step') and (self.scheduler is not None):
                    self.scheduler.step()
            # self.evaluate_ckpt()
            try:
                self.evaluate_ckpt()
            except:
                print('error happens in model sampling')
                self.save_ckpt(None) # 先存下来
            if self.distributed:
                dist.barrier() 

            if (self.schedule_unit == 'epoch') and (self.scheduler is not None):
                self.scheduler.step()  

    @torch.no_grad()
    def evaluate_ckpt(self): # validate, test, visualize
        self.model.eval()
        if isinstance(self.model, DDP):
            eval_model = self.model.module
        else:
            eval_model = self.model
        epoch_dir = os.path.join(self.out_dir, f'epoch_{self.epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        validate_metrics, test_metrics = {}, {}
        if self.validate_loader is not None:
            validate_metrics = self.validate_function(self.validate_loader, eval_model, 
                                                        self.device, 
                                                        self.distributed, 
                                                        self.is_main_process,
                                                        output_dir=epoch_dir)
        if self.test_loader is not None:
            test_metrics = self.test_function(self.test_loader, eval_model, 
                                                self.device, 
                                                self.distributed, 
                                                self.is_main_process,
                                                output_dir=epoch_dir)   
        if self.is_main_process:
            validate_metrics.update(test_metrics)
            wandb.log(validate_metrics, step=self.iteration+1+self.epoch)
            logging.info(validate_metrics)
            self.save_ckpt(validate_metrics) 
        if self.distributed:
            dist.barrier()
                              
    def visualize_ckpt(self):        
        # 在训练集上visualize
        visualize_dir = f'{self.out_dir}/epoch_{self.epoch}/trainset_vis'
        self.train_sampler.set_epoch(-1)
        for batch_idx, batch_dict in enumerate(self.train_loader):
            if batch_idx in self.trainset_vis_idxs:
                saved_path = f'{visualize_dir}/batch{batch_idx}'
                samples = batch_dict['samples'].to(self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_query']
                auxiliary = batch_dict['auxiliary']
                
                self.optimizer.zero_grad() #对于gradient sampling来说, 内部会先进行梯度下降
                self.model(samples, text_queries, targets, auxiliary, 
                                     saved_path=saved_path, visualize=True)
                if batch_idx == self.visualize_train_set_end_idx:
                    break
        
        if self.validate_loader is not None:
            visualize_dir = f'{self.out_dir}/epoch_{self.epoch}/validateset_vis'
            for batch_idx, batch_dict in enumerate(self.validate_loader):
                if batch_idx in self.validateset_vis_idxs:
                    saved_path = f'{visualize_dir}/batch{batch_idx}'
                    samples = to_device(batch_dict['samples'], self.device)
                    targets = to_device(batch_dict['targets'], self.device)
                    text_queries = batch_dict['text_query']
                    auxiliary = batch_dict['auxiliary']
                    
                    self.model.sample(samples, text_queries, targets, auxiliary, 
                                        saved_path=saved_path, visualize=True)
                    
                    if batch_idx == self.visualize_test_set_end_idx:
                        break

        if self.test_loader is not None:
            visualize_dir = f'{self.out_dir}/epoch_{self.epoch}/testset_vis'
            for batch_idx, batch_dict in enumerate(self.test_loader):
                if batch_idx in self.testset_vis_idxs:
                    saved_path = f'{visualize_dir}/batch{batch_idx}'
                    samples = to_device(batch_dict['samples'], self.device)
                    targets = to_device(batch_dict['targets'], self.device)
                    text_queries = batch_dict['text_query']
                    auxiliary = batch_dict['auxiliary']
                    
                    self.model.sample(samples, text_queries, targets, auxiliary, 
                                        saved_path=saved_path, visualize=True)
                    
                    if batch_idx == self.visualize_test_set_end_idx:
                        break

    def save_ckpt(self, metrics):
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        # rng_state_dict = {
        #     'cpu_rng_state': torch.get_rng_state(),
        #     'gpu_rng_state': torch.cuda.get_rng_state(),
        #     'numpy_rng_state': np.random.get_state(),
        #     'py_rng_state': random.getstate()
        # }
        checkpoint_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            # 'rng_state': rng_state_dict
        }
        epoch_dir = os.path.join(self.out_dir, f'epoch_{self.epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        filename = os.path.join(epoch_dir, f'{self.epoch:02d}.pth.tar')
        if metrics is not None:
            checkpoint_dict['metrics'] = metrics
        
        torch.save(checkpoint_dict, filename)
        logging.info(f'保存了模型 {filename} {"但是还没有测试" if metrics is None else ""}')

    def load_ckpt(self, ckpt_path, resume=False):
        assert os.path.exists(ckpt_path)
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if resume:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # 不resume, 只load 模型, optimizer是config给的
            pass
    
@register_task
def rvos(configs, process_id, device_id, num_processes,):
    return Trainer(
        process_id, device_id, num_processes,
        output_directory=configs['out_dir'],
        data_configs=configs['data'],
        model_configs=configs['model'],
        trainer_ckpt=configs['trainer_ckpt'],
        wandb_configs=configs['wandb'],
        visualize_configs=configs['visualize'],
        seed=configs['seed'],
        resume=configs['mode'] == 'train_resume'
    )
