
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange, reduce
from functools import partial

from models.registry import register_model
from data_schedule import build_schedule
from detectron2.modeling import META_ARCH_REGISTRY
from models.Render.representation.GS3D import GaussianModel
from models.Render.representation.loss import l1_loss, ssim
from models.Render.representation.gaussian_renderer import render

from models.optimization.gs_optimizer import get_expon_lr_func
import copy
from detectron2.utils import comm
import logging

from detectron2.data import MetadataCatalog
from argparse import Namespace
import math

class Image_3DGS_OptimizeBased(GaussianModel):
    def __init__(self, configs,):
        super().__init__(configs=configs)
        self.loss_weight = configs['model']['loss_weight']
        assert sum(list(self.loss_weight.values())) == 1.0, 'loss权重不是1'
        self.random_background = configs['optim']['random_background']
        # sample_setup
        self.pipe = Namespace(convert_SHs_python=configs['model']['sample']['convert_SHs_python'],
                         compute_cov3D_python=configs['model']['sample']['compute_cov3D_python'],
                         debug= configs['model']['sample']['debug'])
    @property
    def device(self):
        return self._xyz.device
    
    def load_ckpt(self, 
                  ckpt_dict=None, 
                  iteration=None,
                  configs=None):
        self.restore(ckpt_dict, configs)
    
    def __call__(self, batch_dict):
        from data_schedule.render.apis import Multiview3D_Optimize_Mapper
        scene_meta = MetadataCatalog.get(batch_dict['scene_dict']['metalog_name'])
        wbcg = scene_meta.get('white_background')
        cameras_extent = scene_meta.get('cameras_extent')
        bg_color = [1,1,1] if wbcg else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 
        bg = torch.rand((3), device="cuda") if self.random_background else background

        view_camera = batch_dict['view_dict']['view_camera']
        render_pkg = render(view_camera, self, self.pipe, bg)
        
        image, viewspace_point_tensor, visibility_filter, radii \
            = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = view_camera.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)

        return {'loss_l1': Ll1,
                'loss_ssim': ssim_loss}, self.loss_weight, {'viewspace_points':viewspace_point_tensor, 
                                                            'visibility_filter': visibility_filter,
                                                             'radii': radii,
                                                              'white_background':wbcg,
                                                            'cameras_extent': cameras_extent }  

    @torch.no_grad()
    def sample(self, batch_dict): 
        # list of test viewpoints -> list of renderings
        wbcg = MetadataCatalog.get(batch_dict['scene_dict']['metalog_name']).get('white_background')
        bg_color = [1,1,1] if wbcg else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  
        rendering = render(batch_dict['view_dict']['view_camera'], self, self.pipe, background)["render"] # 0-1, float
        return {
            'rendering': rendering.cpu()
        }

    def optimize_setup(self, optimize_configs):
        """
        模型已经根据场景初始化了, 并且放到了gpu里
        """
        self.percent_dense = optimize_configs['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        
        l = [
            {'params': [self._xyz], 'lr': optimize_configs['position_lr_init'] * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': optimize_configs['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': optimize_configs['feature_lr'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': optimize_configs['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': optimize_configs['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': optimize_configs['rotation_lr'], "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init= optimize_configs['position_lr_init']* self.spatial_lr_scale,
                                    lr_final= optimize_configs['position_lr_final']* self.spatial_lr_scale,
                                    lr_delay_mult=optimize_configs['position_lr_delay_mult'],
                                    max_steps=optimize_configs['position_lr_max_steps'])   

        self.log_lr_group_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation':5}
        # self.scene_white_background = MetadataCatalog.get(scene_features['scene_name']).get('white_background')
        # self.scene_cameras_extent = MetadataCatalog.get(scene_features['scene_name']).get('cameras_extent')
        
        
        self.position_lr_init = optimize_configs['position_lr_init']
        self.position_lr_final = optimize_configs['position_lr_final']
        self.position_lr_delay_mult = optimize_configs['position_lr_delay_mult']
        self.position_lr_max_steps = optimize_configs['position_lr_max_steps']
        
        self.densify_from_iter = optimize_configs['densify_from_iter']
        self.densification_interval = optimize_configs['densification_interval']
        self.opacity_reset_interval = optimize_configs['opacity_reset_interval']
        self.densify_grad_threshold = optimize_configs['densify_grad_threshold']
        self.densify_until_iter = optimize_configs['densify_until_iter']
        
    def optimize(self, 
                 loss_weight=None,
                 loss_dict_unscaled=None,
                 closure=None,
                 num_iterations=None,
                 optimize_dict=None,
                 **kwargs):
        num_iterations += 1
        self.update_learning_rate(num_iterations)  
        if num_iterations % 1000 == 0:
            self.oneupSHdegree()

        # loss backward了嘛?
        loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  

        viewspace_point_tensor = optimize_dict['viewspace_points']
        visibility_filter = optimize_dict['visibility_filter']
        radii = optimize_dict['radii']
        cameras_extent = optimize_dict['cameras_extent']
        white_background = optimize_dict['white_background']

        if num_iterations < self.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if num_iterations > self.densify_from_iter and num_iterations % self.densification_interval == 0:
                size_threshold = 20 if num_iterations > self.opacity_reset_interval else None
                self.densify_and_prune(self.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
            
            if num_iterations % self.opacity_reset_interval == 0 or (white_background and num_iterations == self.densify_from_iter):
                self.reset_opacity()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    def get_lr_group_dicts(self,):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'spatial_lr_scale': self.spatial_lr_scale,

            'defaults':{
                'percent_dense': self.percent_dense,
                'position_lr_init': self.position_lr_init,
                'feature_lr': self.optimizer.param_groups[1]['lr'],
                'opacity_lr': self.optimizer.param_groups[3]['lr'],
                'scaling_lr': self.optimizer.param_groups[4]['lr'],
                'rotation_lr': self.optimizer.param_groups[5]['lr'],

                'position_lr_final': self.position_lr_final,
                'position_lr_delay_mult': self.position_lr_delay_mult,
                'position_lr_max_steps': self.position_lr_max_steps, 

                
                'densify_from_iter': self.densify_from_iter,
                'densification_interval': self.densification_interval,
                'opacity_reset_interval': self.opacity_reset_interval,
                'densify_grad_threshold': self.densify_grad_threshold,
                'densify_until_iter': self.densify_until_iter,                

                }
        },
        
    def load_optimize_state_dict(self, state_dict):
        # 先init当前状态对应的默认optimize状态
        self.optimize_setup(state_dict['defaults']) # 应该是config的形式

        self.xyz_gradient_accum = state_dict['xyz_gradient_accum']  # tensor
        self.denom = state_dict['denom']  # tensor
        self.max_radii2D = state_dict['max_radii2D'] # tensor
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.spatial_lr_scale = state_dict['spatial_lr_scale']

    def load_state_dict(self, ckpt, strict=True):
        # 只load model
        from models.trainer_model_api import Trainer_Model_API
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,) = ckpt       

    def state_dict(self,):
        from models.trainer_model_api import Trainer_Model_API
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,          
        )


@register_model
def image_3dgs_optim_based(configs, device):
    # 假设只有一张卡, 并且全部用.cuda() 操作
    from .aux_mapper import Image_3DGS_Optimize_AuxMapper
    model_input_mapper = Image_3DGS_Optimize_AuxMapper(configs['model']['input_aux'])

    renderer = Image_3DGS_OptimizeBased(configs)
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))
    from detectron2.data import MetadataCatalog
    scene_list = MetadataCatalog.get('global_dataset').get('subset_list')
    assert len(scene_list) == 1, '基于optimize的必须是一个scene'
    renderer.create_from_pcd(pcd=MetadataCatalog.get(scene_list[0]).get('point_cloud'), 
                             spatial_lr_scale=MetadataCatalog.get(scene_list[0]).get('cameras_extent'))
    renderer.optimize_setup(optimize_configs=configs['optim'])
    assert comm.is_main_process(), '3d重建只需要一张卡'
    # logging.debug(f'初始化的总参数数量:{sum(p.numel() for p in renderer.parameters())}')
    # logging.debug(f'初始化的可训练参数数量:{sum(p.numel() for p in renderer.parameters() if p.requires_grad)}')

    return renderer, train_samplers, train_loaders, eval_function
