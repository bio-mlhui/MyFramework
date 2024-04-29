



from typing import Any, Optional, List, Dict, Set
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as Trans_F
from einops import repeat, reduce, rearrange
from utils.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from functools import partial
from utils.misc import to_device
from models.utils.visualize_amr import save_model_output
from models.registry import register_model
from data_schedule.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from models.optimization.scheduler import build_scheduler 
from detectron2.config import configurable
from models.registry import register_model
import detectron2.utils.comm as comm
import copy
from models.optimization.utils import get_total_grad_norm
from models.optimization.optimizer import get_optimizer

from models.DYNAMIC_RENDER.scene import Scene, GaussianModel

class Gaussian4D(nn.Module): # frame decoder做object segmentation, temporal decoder做referent segmentation
    # model知道整个module-submodule结构
    def __init__(self,
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],

                 sh_degree=None,
                 hyper=None,
                 dataset=None
                ) -> None:
        super().__init__()
        
        gaussians = GaussianModel(sh_degree, hyper)
        scene = Scene(dataset, gaussians, load_coarse=None)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        video_cams = scene.getVideoCameras()
        test_cams = scene.getTestCameras()
        train_cams = scene.getTrainCameras()


    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['weight_decay']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'backbone':[], 'base':[]}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_text_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if len(log_lr_group_idx['backbone']) == 0:
                        log_lr_group_idx['backbone'].append(len(params))

                else:
                    if len(log_lr_group_idx['base']) == 0:
                        log_lr_group_idx['base'].append(len(params))  
                                     
                # pos_embed, norm, embedding的weight decay特殊对待
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    logging.debug(f'setting weight decay of {module_name}.{module_param_name} to zero')
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
   
        return params, log_lr_group_idx

    def forward(self, batch_dict):
        pass
        # return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample(self, batch_dict):
        pass
        # return {
        #     'video': orig_video,
        #     'pred_masks': pred_masks,
        #     'pred_boxes': pred_boxes,
        # }



@register_model
def decode_frame_query(configs, device):
    model =  Decode_FrameQuery(configs)
    from .aux_mapper import AUXMapper_v1
    model.to(device)
    params_group, log_lr_group_idx = Decode_FrameQuery.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx
