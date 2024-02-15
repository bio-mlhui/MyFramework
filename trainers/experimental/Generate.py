import torch
import numpy as np
import random
import math
import logging
import time
import os
import sys
from utils.misc import reduce_dict, to_device
import gc
import wandb
from utils.misc import  SmoothedValue, MetricLogger
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import datetime
from fvcore.nn.precise_bn import get_bn_modules

from detectron2.utils import comm
from detectron2.engine import hooks
from models.backbone.swin import compute_mask
from .registry import register_task
from data_schedule import data_schedule_entrypoints
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
import weakref

from .registry import TRAINER_REGISTRY

@TRAINER_REGISTRY.register()
class Generate(TrainerBase):
    def __init__(self, 
                  configs):
        super().__init__()
        train_loader = self.build_train_loader(configs)

        create_data_schedule = data_schedule_entrypoints(data_configs['name'])
        self.train_loader, self.train_sampler,\
        self.validate_loader, self.validate_function, \
        self.test_loader, self.test_function \
            = create_data_schedule(configs)
        # 定义: validate set是用来调整model和参数的
        # [train, None, test]: 比如yrvos 要进行真实测试
        # [train, vali, None]: 比如不想生成zip的youtube-vos / a2ds
        # [train, vali, test]: 比如yrvos 分出来一点调整参数

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        scheduler = self.build_lr_scheduler(cfg, model, optimizer)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, train_loader, optimizer
        )       
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model=model,
            save_dir=configs['OUTPUT_DIR'],
            trainer=weakref.proxy(self),
        )      

        self.start_iter = 0
        self.max_iter = configs['SOLVER']['MAX_ITER']
        self.cfg = configs

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


    @classmethod
    def build_train_loader(csl, cfg,):
        pass

    @classmethod
    def build_lr_scheduler(cls, cfg, model, optimizer):
        return model.get_lr_sheduler(cfg=cfg, optimizer=optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return model.get_optimizer(cfg=cfg)

    @classmethod
    def build_model(cls, cfg):
        from detectron2.modeling import build_model
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model


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
                if idx < 10:
                    logging.info(batch_dict['auxiliary']['sample_idx'])
                start = time.time()
                loss_dict_unscaled, loss_dict_scaled, gradient_norm = self.model(batch_dict, visualize=False) 
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
        
        # torch.save(checkpoint_dict, filename)
        logging.info(f'保存了模型 {filename} {"但是还没有测试" if metrics is None else ""}')
        print(f'保存了模型 {filename} {"但是还没有测试" if metrics is None else ""}')

    def load_ckpt(self, 
                  ckpt_path, 
                  strict_load=True,
                  resume=False,):
        assert os.path.exists(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'], 
                                          strict=strict_load)
        if resume:
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # 不resume, 只load 模型, optimizer是config给的
            pass
