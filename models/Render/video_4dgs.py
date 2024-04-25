
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
from models.Render.model.GS3D import GaussianModel
from models.Render.model.loss import l1_loss, ssim
from models.optimization.gs_optimizer import get_expon_lr_func



class Image_3DGS_OptimizeBased(GaussianModel):
    def __init__(self, configs,):
        super().__init__(configs=configs)
        self.loss_weight = configs['model']['loss_weight']
        assert sum(list(self.loss_weight.values())) == 1.0, 'loss权重不是1'

    @property
    def device(self):
        return self._xyz.device
    
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

    def optimize_setup(self, configs):
        """
        模型已经根据场景初始化了, 并且放到了gpu里
        """
        optimize_configs = configs['optim']
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
        # loss backward了嘛?
        viewspace_point_tensor = optimize_dict['viewspace_points']
        visibility_filter = optimize_dict['visibility_filter']
        radii = optimize_dict['radii']
        scene_cameras_extent = optimize_dict['scene_cameras_extent']
        scene_white_background = optimize_dict['scene_white_background']

        if num_iterations < self.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if num_iterations > self.densify_from_iter and num_iterations % self.densification_interval == 0:
                size_threshold = 20 if num_iterations > self.opacity_reset_interval else None
                self.densify_and_prune(self.densify_grad_threshold, 0.005, scene_cameras_extent, size_threshold)
            
            if num_iterations % self.opacity_reset_interval == 0 or (scene_white_background and num_iterations == self.optimizer.densify_from_iter):
                self.reset_opacity()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(num_iterations)
                param_group['lr'] = lr            
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if num_iterations % 1000 == 0:
            self.oneupSHdegree()
        
    def get_lr_group_dicts(self,):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'defaults': {
                'densify_from_iter': self.densify_from_iter,
                'densification_interval': self.densification_interval,
                'opacity_reset_interval': self.opacity_reset_interval,
                'densify_grad_threshold': self.densify_grad_threshold,
                'densify_until_iter': self.densify_until_iter,
                'percent_dense': self.percent_dense,
                'xyz_gradient_accum': self.xyz_gradient_accum,
                'denom': self.denom,
            },
            'scheduler': {
                'position_lr_init': self.position_lr_init,
                'position_lr_final': self.position_lr_final,
                'position_lr_delay_mult': self.position_lr_delay_mult,
                'position_lr_max_steps': self.position_lr_max_steps, 
            },
        }
        
    def load_optimize_state_dict(self, state_dict):
        gs_defaults = state_dict['defaults']
        self.percent_dense = gs_defaults['percent_dense']
        self.xyz_gradient_accum = gs_defaults['xyz_gradient_accum']  # tensor
        self.denom = gs_defaults['denom']  # tensor
        self.densify_from_iter = gs_defaults['']
        self.densification_interval = gs_defaults['densification_interval']
        self.opacity_reset_interval = gs_defaults['opacity_reset_interval']
        self.densify_grad_threshold = gs_defaults['densify_grad_threshold']
        self.densify_until_iter = gs_defaults['densify_until_iter']
        self.optimizer.load_state_dict(state_dict['optimizer'])

        scheduler = state_dict['scheduler']
        self.xyz_scheduler_args = get_expon_lr_func(lr_init= scheduler['position_lr_init']* self.spatial_lr_scale,
                                    lr_final= scheduler['position_lr_final']* self.spatial_lr_scale,
                                    lr_delay_mult=scheduler['position_lr_delay_mult'],
                                    max_steps=scheduler['position_lr_max_steps']) 
        assert self.position_lr_init == scheduler['position_lr_init']
        assert self.position_lr_final == scheduler['position_lr_final']
        assert self.position_lr_delay_mult == scheduler['position_lr_delay_mult']
        assert self.position_lr_max_steps == scheduler['position_lr_max_steps']


@register_model
def image_3dgs_optim_based(configs, device):
    from .render_aux_mapper import Image_3DGS_Optimize_AuxMapper
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

    renderer.optimize_setup(configs=configs)
    
    renderer.to(device)
    
    return renderer, train_samplers, train_loaders, eval_function
