
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
from util.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from functools import partial
from util.misc import to_device
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
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import matplotlib.pyplot as plt
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann

__all__ = ["unet_polyp_seg_video_diff" ] 

class UnetPolyPSeg_VideoDiff(nn.Module):

    def __init__(self,                 
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.loss_weight = configs['model']['loss_weights']
        video_backbone_configs = configs['model']['video_backbone'] 
        multiscale_encoder_configs = configs['model']['multiscale_encoder']
        unet_configs = configs['model']['unet']
        
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        multiscale_encoder_cls = META_ARCH_REGISTRY.get(multiscale_encoder_configs['name'])
        self.multiscale_encoder = multiscale_encoder_cls(multiscale_encoder_configs,
                                                         multiscale_shapes=self.video_backbone.multiscale_shapes,)
        
        # unet_cls = META_ARCH_REGISTRY.get(unet_configs['name'])
        # self.unet = unet_cls(unet_configs,)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict):
        VIS_TrainAPI_clipped_video
        # 看成instance个数是1的instance segmentation
        videos = batch_dict.pop('video_dict')['videos'] # b t 3 h w, 0-1
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = batch_dict.pop('targets')
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch
        batch_size, T, _, H, W = videos.shape
        
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4).contiguous(),)
        # self.video_backbone.compute_loss(multiscales, targets)
        multiscales = self.multiscale_encoder(multiscales=multiscales)   
        unet_losses = self.unet(multiscales=multiscales, targets=targets)

        unet_losses = {f'unet_{key}': value for key, value in unet_losses.items()}
        loss_value_dict = {}
        loss_value_dict.update(unet_losses)
        assert set(list(self.loss_weight.keys())).issubset(set(list(loss_value_dict.keys())))
        assert set(list(loss_value_dict.keys())).issubset(set(list(self.loss_weight.keys())))
        loss = sum([loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()])

        if not math.isfinite(loss.item()):
            logging.debug("Loss is {}, stopping training".format(loss.item()))
            raise RuntimeError()
        
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
    
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
            
    @torch.no_grad()
    def sample(self, batch_dict):
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict.pop('video_dict')['videos'] # b t 3 h w, 0-1
        _, pad_T, _, pad_H, pad_W = videos.shape
        orig_t, orig_h, orig_w = batch_dict.pop('orig_size')[0]
        assert len(videos) == 1
        videos = (videos - self.pixel_mean) / self.pixel_std
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch
        batch_size, T, _, H, W = videos.shape
        
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4),)
        # 对比学习
        # self.video_backbone.compute_loss(multiscales, targets)
        multiscales = self.multiscale_encoder(multiscales=multiscales)   

        # 做成只有一个物体
        # {'pred_masks': b T H W, 'pred_boxes': b T 4, 'pred_class': b c} # 每个类别的概率, 类别数量可以是10, 可以是2
        unet_output = self.unet.sample(multiscales=multiscales)
        pred_masks = unet_output['pred_masks'][0].sigmoid() > 0.5 # T H W
        pred_boxes = unet_output['pred_boxes'][0].sigmoid() # T 4, cxcywh, 0-1
        pred_class = unet_output['pred_class'][0].softmax(-1) # c
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes) # x1y1x2y2
        pred_boxes = pred_boxes * torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=torch.float) # 绝对坐标

        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        pred_masks = pred_masks[:orig_t, :orig_h, :orig_w] # t h w
        pred_boxes = pred_boxes[:orig_t] # t 4
        pred_boxes[:, 0::2].clamp_(min=0, max=orig_w) # 对box进行clip
        pred_boxes[:, 1::2].clamp_(min=0, max=orig_h) # x1y1x2y2
        pred_masks = [pred_mk.unsqueeze(0) for pred_mk in pred_masks] # list[1 h w], t
        pred_boxes = [pred_bx.unsqueeze(0) for pred_bx in pred_boxes] # list[1 4], t
        # polyp的类别有6个, evaluator是vis的接口, 但是eval fn是polyp自己的
        pred_class = pred_class[None, None, :].repeat(orig_t, 1, 1).unbind(0) # list[1 c], t

        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
        VIS_AugAPI
        return {
            'video': [orig_video], # t 3 h w
            'pred_masks': [pred_masks], # list[1 h w], t, bool
            'pred_class': [pred_class], # list[1 c], t, probability
            'pred_boxes': [pred_boxes] # list[1 4], t, x1y1x2y1
        }

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
        log_lr_group_idx = {'backbone':None, 'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if log_lr_group_idx['backbone'] is None:
                        log_lr_group_idx['backbone'] = len(params)

                else:
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                                     
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

@register_model
def unet_polyp_seg_video_diff(configs, device):
    from .decode_frame_query import AUXMapper_v1
    model = UnetPolyPSeg_VideoDiff(configs)
    model.to(device)
    params_group, log_lr_group_idx = UnetPolyPSeg_VideoDiff.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx
