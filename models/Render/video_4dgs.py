
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
import os
import json
from torch.cuda.amp import autocast
import logging
import torchvision.transforms.functional as Trans_F

from models.registry import register_model
from models.optimization.scheduler import build_scheduler 
from data_schedule import build_schedule
from detectron2.modeling import META_ARCH_REGISTRY
from models.backbone.utils import VideoMultiscale_Shape
from models.Render.model.GS3D import GS_3D
from detectron2.data import MetadataCatalog
from models.Render.render_utils.loss_utils import l1_loss, ssim
from detectron2.utils import comm

class GS3D_OptimizeBased_Optimizer:
    def __init__(self, 
                 GS3D=None,
                 configs=None,
                 scene_features=None,) -> None:
        """
        高斯点已经初始化了
        高四点放到cuda了
        """
        GS3D.training_step()
        self.renderer = GS3D
        self.log_lr_group_name_to_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation': 5}    

    def step(self, 
             closure=None,
             num_iterations=None,
             optimizer_step_dict=None,
             **kwargs):
        
        self.renderer.optimize(closure=closure,
                               num_iterations=num_iterations,
                               optimizer_step_dict=optimizer_step_dict,
                               kwargs=kwargs)


    def state_dict(self, **kwargs):
        return {
            'optimizer_dict': self.optimizer.state_dict(),
            'densify_from_iter': self.densify_from_iter,
            'densification_interval': self.densification_interval,
            'opacity_reset_interval': self.opacity_reset_interval,
            'densify_grad_threshold': self.densify_grad_threshold,
            'densify_until_iter': self.densify_until_iter,
            'scene_white_background': self.scene_white_background,
            'scene_cameras_extent': self.scene_cameras_extent     
        }
   
    def load_state_dict(self, state_dict=None, **kwargs):
        self.optimizer.load_state_dict(state_dict['optimizer_dict'])
        self.densify_from_iter = state_dict['densify_from_iter']
        self.densification_interval = state_dict['densification_interval']
        self.opacity_reset_interval = state_dict['opacity_reset_interval']
        self.densify_grad_threshold = state_dict['densify_grad_threshold']
        self.densify_until_iter = state_dict['densify_until_iter']
        self.scene_white_background = state_dict['scene_white_background']
        self.scene_cameras_extent = state_dict['scene_white_background']


    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(set_to_none=self.set_to_none) # # delete gradient  


    def get_log_lr_dicts(self,):
        llg = {}
        for log_lr_group_name, log_lr_group_idx in self.log_lr_group_name_to_idx.items():
            if log_lr_group_idx is None:
                llg[f'lr_group_{log_lr_group_name}'] = 0
            else:
                llg[f'lr_group_{log_lr_group_name}'] = self.renderer.optimizer.param_groups[log_lr_group_idx]["lr"]
        return llg 


class GS3D_OptimizeBased_Scheduler:
    def __init__(self, 
                gaussians=None,
                optimizer=None, 
                configs=None,):
        self.gaussians = gaussians # gaussians的
        self.optimizer = optimizer
        spatial_lr_scale = configs['spatial_lr_scale']
        from models.optimization.gs_optimizer import get_expon_lr_func
        self.xyz_sheduler_args = get_expon_lr_func(lr_init=configs['position_lr_init']*spatial_lr_scale,
                                                lr_final=configs['position_lr_final']*spatial_lr_scale,
                                                lr_delay_mult=configs['position_lr_delay_mult'],
                                                max_steps=configs['position_lr_max_steps'])
    def step(self,
             num_iterations=None,
             **kwargs):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_sheduler_args(num_iterations)
                param_group['lr'] = lr
                return lr
    
    def state_dict(self, 
                   **kwargs):
        return {}
    
    def load_state_dict(self, 
                        state_dict=None,
                        **kwargs,):
        assert state_dict == {}

class Image_3DGS_OptimizeBased(nn.Module):

    def __init__(self, configs,):
        super().__init__()
        self.gaussians =  META_ARCH_REGISTRY.get(configs['model']['render']['name'])(configs['model']['render'])

        self.loss_weight = configs['model']['loss_weight']

        loss_value_sum = sum(list(self.loss_weight.values()))

        if (loss_value_sum != 1.0) and comm.is_main_process():
            logging.warning('loss权重之和不是0')
            
    @property
    def device(self):
        return self.gaussians._xyz.device
    
    def load_ckpt(self, 
                  ckpt_dict=None, 
                  iteration=None,
                  configs=None):
        
        self.gaussians.restore(ckpt_dict, configs)
    
    def forward(self, batch_dict):
        viewpoint_cam = batch_dict['viewpoint_cam']
        pipe = batch_dict['pipe']
        bg = batch_dict['bg']
        
        assert self.training
        render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg)
        
        image, viewspace_point_tensor, visibility_filter, radii \
            = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.to(self.device)
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)

        return {'loss_l1': Ll1,
                'loss_ssim': ssim_loss}, self.loss_weight   

    @torch.no_grad()
    def sample(self, batch_dict): 
        # list of test viewpoints -> list of renderings
        assert not self.training
        return {
            'renderings': None
        }
   

@register_model
def image_3dgs_optim_based(configs, device):
    from .render_aux_mapper import Image_3DGS_Optimize_AuxMapper
    model_input_mapper = Image_3DGS_Optimize_AuxMapper(configs['model']['input_aux'])

    renderer = Image_3DGS_OptimizeBased(configs)
    train_samplers, train_loaders, eval_function, scene_features = build_schedule(configs, 
                                                                    model_input_mapper.mapper, 
                                                                    partial(model_input_mapper.collate,))
    
    renderer.create_from_pcd()
    renderer.to(device)

    optimizer = GS3D_OptimizeBased_Optimizer(gaussians=renderer.gaussians,
                                             configs=configs['optim'],
                                             scene_features=scene_features)
    
    log_lr_group_idx = optimizer.log_lr_group_idx
    scheduler = GS3D_OptimizeBased_Scheduler(gaussians=renderer.gaussians,
                                            optimizer=optimizer,
                                            configs=configs['optim'],
                                            scene_features=scene_features)

    return model, optimizer, scheduler, train_samplers, train_loaders, log_lr_group_idx, eval_function
