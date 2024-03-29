import torch
import numpy as np
import random
import math
import logging
import time
import os
import sys
from utils.misc import reduce_dict, to_device, reduce_scalar, is_dist_avail_and_initialized
# from models.optimization.utils import get_total_grad_norm
import gc
import wandb
from utils.misc import  SmoothedValue, MetricLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import detectron2.utils.comm as comm
import datetime
# import traceback
# import torch.distributed.rpc as dist_rpc
import torch.distributed as dist
from models import model_entrypoint
from utils.misc import to_device

# 每个loader是完全不同的数据集, train调用不同loader, 调用不同eval_func
class Trainer_Render:
    def __init__(self, configs):
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        seed = configs['model_schedule_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(seed)        

        # model and data
        create_model_schedule = model_entrypoint(configs['model']['name'])
        self.model, self.optimizer, self.scheduler, self.log_lr_group_name_to_idx, \
            self.train_loaders, self.eval_functions = create_model_schedule(configs, device=self.device,)

        # trainer
        self.out_dir = configs['out_dir']
        self.ckpted_iters = configs['optim']['ckpted_iters'] # list[int]
        self.num_iterations = 0 
        self.num_samples = 0 # wandb的横坐标, 模型见过(forward-backward成功)的samples的数量
        assert self.train_samplers[0].start_idx == self.num_samples

        # init_ckpt 和 train_resume 不同;  更改之前的起始点和不同的线
        if configs['initckpt']['path'] != '':
            self.load_ckpt(configs['initckpt']['path'], 
                           load_random=configs['initckpt']['load_random'], # 随机状态
                           load_model=configs['initckpt']['load_model'], 
                           load_schedule=configs['initckpt']['load_schedule'], # 数据流idx
                           load_optimize=configs['initckpt']['load_optimizer'])

        if comm.is_main_process():
            # default smooth value: window_size=1, fmt='{value:.6f}', handler='value'
            metric_logger = MetricLogger(delimiter='\t')
            metric_logger.add_meter('iteration_time', SmoothedValue(window_size=1,fmt='{value:2f}',handler='value') )
            logging.debug(f'模型的总参数数量:{sum(p.numel() for p in self.model.parameters())}')
            logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            for log_lr_group_name in self.log_lr_group_name_to_idx.keys():
                metric_logger.add_meter(f'lr_group_{log_lr_group_name}', SmoothedValue(window_size=1,fmt='{value:.8f}', handler='value'))
            self.metric_logger = metric_logger

        self.save_ckpt() # 存储初始点
        if configs['initckpt']['eval_init_ckpt']:
            self.evaluate() # 测试的随机状态独立于训练的状态
            self.load_ckpt(os.path.join(self.iteration_dir, 'ckpt.pth.tar'),  # 训练的随机状态
                       load_random=True, load_schedule=False, load_model=False, load_optimize=False,)


    def train(self):   
        manual_stop_train = False
        for loader in self.train_loaders:
            for idx, batch_dict in enumerate(loader):
                if manual_stop_train:
                    self.save_ckpt()
                self.model.train()
                meta_idxs = batch_dict.pop('meta_idxs')
                visualize = batch_dict.pop('visualize')
                batch_dict = to_device(batch_dict, self.device)
                batch_dict['visualize_paths'] = self.visualize_path(meta_idxs=meta_idxs, 
                                                                    visualize=visualize) # visualize model训练过程
                iteration_time = time.time()
                loss_dict_unscaled, loss_weight = self.model(batch_dict)
                loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
                assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
                loss.backward()       
                self.optimizer.step()
                iteration_time = time.time() - iteration_time
                self.optimizer.zero_grad(set_to_none=True) # delete gradient 
                self.scheduler.step() 
                sample_idxs = comm.all_gather(meta_idxs) 
                sample_idxs = [taylor for cardib in sample_idxs for taylor in cardib]
                self.num_samples += len(sample_idxs)
                self.num_iterations += 1
                loss_dict_unscaled_item = {key: torch.tensor(value.detach().item(), device=self.device) for key, value in loss_dict_unscaled.items()}
                del loss, loss_dict_unscaled # delete graph
                self._log(loss_dict_unscaled=loss_dict_unscaled_item,
                          loss_weight=loss_weight,
                          sample_idxs=sample_idxs,
                          iteration_time=iteration_time)


                