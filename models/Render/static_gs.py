
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
from einops.layers.torch import Rearrange
from torch import einsum
import math
from typing import List, Optional
from utils.misc import NestedTensor
import numpy as np
from models.layers.mamba.ss2d import SS2D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
import logging
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
import torchvision.transforms.functional as Trans_F
import copy
from models.layers.mamba.vss_layer_3d import VSSLayer_3d
from models.layers.matching import dice_loss, ce_mask_loss
__all__ = ['video_u_mamba']

from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from data_schedule import build_schedule
from models.layers.utils import _get_clones
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from .gaussian_splatting import GaussianModel
from models.backbone.utils import VideoMultiscale_Shape

# train: 训练视角, sample: 测试视角
class Static_GS(nn.Module):
    def __init__(
        self,
        configs,):
        super().__init__()
        sh_degree = configs['sh_degree']
        self.gaussian = GaussianModel(sh_degree=sh_degree)

        
        self.loss_weight = configs['model']['loss_weight']
        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        self.fusion_encoder = META_ARCH_REGISTRY.get(configs['model']['fusion']['name'])(configs['model']['fusion'],
                                                                                   multiscale_shapes=self.video_backbone.multiscale_shapes)
          
        same_dim_multiscale_shapes = VideoMultiscale_Shape.set_multiscale_same_dim(shape_by_dim=self.video_backbone.multiscale_shapes,
                                                                                   same_dim=configs['model']['fusion']['d_model'])  
         
        self.decoder = META_ARCH_REGISTRY.get(configs['model']['decoder']['name'])(configs['model']['decoder'],
                                                                                   multiscale_shapes=same_dim_multiscale_shapes)
        if configs['model']['fusion']['name'] == 'Video_Deform2D_DividedTemporal_MultiscaleEncoder_v2':
            self.fusion_encoder.hack_ref(query_norm=self.decoder.temporal_query_norm, mask_mlp=self.decoder.query_mask)

    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict,):
        # b 3 t h w -> b 3 t h w
        multiscales = self.video_backbone(x=videos) # b c t h w
        multiscales = self.fusion_encoder(multiscales, video_aux_dict=video_aux_dict) 
        return self.decoder(multiscales, video_aux_dict=video_aux_dict)

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        videos = batch_dict['video_dict']['videos'] # b T 3 h w, 0-1
        targets = batch_dict['targets']
        batch_size, nf = videos.shape[:2]
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        # plt.imsave('./mask.png', mask[0][0].cpu().numpy())
        pred1          = self.model_preds(videos.permute(0, 2, 1, 3, 4), 
                                          video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        pred1_loss = self.decoder.compute_loss(pred1, targets=targets, frame_targets=batch_dict['frame_targets'],
                                               video_aux_dict=batch_dict['video_dict'])

        loss_value_dict = {key: pred1_loss[key] for key in list(self.loss_weight.keys())}
        # gradient_norm = get_total_grad_norm(self.model.parameters(), norm_type=2)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        # 如果是List的话, 那么取最后一层
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear') > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    # @staticmethod
    # def get_optim_params_group(model: Static_GS, configs):
        # self.percent_dense = configs['percent_dense']
        # self.xyz_gradient_accum = torch.zeros((self.gaussian.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.gaussian.get_xyz.shape[0], 1), device="cuda")
        
        # l = [
        #     {'params': [self.gaussian._xyz], 'lr': configs['position_lr_init'] * self.gaussian.spatial_lr_scale, "name": "xyz"},
        #     {'params': [self.gaussian._features_dc], 'lr': configs['feature_lr'], "name": "f_dc"},
        #     {'params': [self.gaussian._features_rest], 'lr': configs['feature_lr'] / 20.0, "name": "f_rest"},
        #     {'params': [self.gaussian._opacity], 'lr': configs['opacity_lr'], "name": "opacity"},
        #     {'params': [self.gaussian._scaling], 'lr': configs['scaling_lr'], "name": "scaling"},
        #     {'params': [self.gaussian._rotation], 'lr': configs['rotation_lr'], "name": "rotation"}
        # ]

   
        # return l, {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation': 5}


@register_model
def static_gs(configs, device):
    from .aux_mapper import AUXMapper_v1
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    model = Static_GS(configs)
    model.to(device)
    log_lr_group_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation': 5}
    
    optimizer = model.gaussian.optimizer

    assert configs['optim']['scheduler']['name'] == 'static_gs_xyz'
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    
    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx
