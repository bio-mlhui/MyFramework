
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

# train: 训练视角, sample: 测试视角
class Image_3DGS_Optimize(nn.Module):
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
        pass
        # return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict): # 渲染/整个3D参数
        assert not self.training
        return {
            'renderings': None
        }

@register_model
def image_3dgs_optimize(configs, device):
    from .render_aux_mapper import Image_3DGS_Optimize_AuxMapper
    model_input_mapper = Image_3DGS_Optimize_AuxMapper(configs['model']['input_aux'])

    model = Image_3DGS_Optimize(configs)
    model.to(device)
    log_lr_group_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation': 5}
    optimizer = model.gaussian.optimizer
    assert configs['optim']['scheduler']['name'] == 'static_gs_xyz'
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)

    train_samplers, train_loaders, eval_function, dataset_features = build_schedule(configs, 
                                                                                    model_input_mapper.mapper, 
                                                                                    partial(model_input_mapper.collate, max_stride=model.max_stride))

    # dataset_specific initialization

    return model, optimizer, scheduler, train_samplers, train_loaders, log_lr_group_idx, eval_function
