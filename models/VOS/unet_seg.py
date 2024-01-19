
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

from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.registry import VOS_TrainAPI_clipped_video
from data_schedule.registry import VOS_EvalAPI_clipped_video_request_ann

__all__ = ["conditional_unetseg" ] 


class Conditional_UnetSeg(nn.Module):
    def __init__(self,                 
                 configs=None,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
       
        # encoder的对比学习损失, decoder的mask, box损失
        self.loss_weight = configs['model']['loss_weights']
        # video_dict -> multiscale
        video_backbone_configs = configs['model']['video_backbone'] 
        multiscale_encoder_configs = configs['model']['multiscale_encoder']
        unet_configs = configs['model']['unet']
        
        # video_backbone.video_backbone/text_backbone
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        multiscale_encoder_cls = META_ARCH_REGISTRY.get(multiscale_encoder_configs['name'])
        self.multiscale_encoder = multiscale_encoder_cls(multiscale_encoder_configs,
                                                         multiscale_shapes=self.video_backbone.multiscale_shapes,)
        
        unet_cls = META_ARCH_REGISTRY.get(unet_configs['name'])
        self.unet = unet_cls(unet_configs,)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict):
        VOS_TrainAPI_clipped_video
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # 0-1, float, b t 3 h w
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = to_device(batch_dict.pop('targets'), self.device)
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch

        batch_size, T, _, H, W = videos.shape

        # 抽特征
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4),)  # b c t h w
        # 对比学习
        # self.video_backbone.compute_loss(multiscales, text_inputs)

        # b c t h w
        fencoded_multiscales = self.multiscale_encoder(multiscales={scale_name:scale_feat.clone() \
                                                       for scale_name, scale_feat in multiscales.items()})   
        # {'pred_masks', 'pred_boxes'}
        unet_output = self.unet(multiscales=fencoded_multiscales)
        
        unet_losses = self.unet.compute_loss(unet_output, targets)
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
        VOS_TrainAPI_clipped_video
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # B T 3 H W
        assert len(videos) == 1
        visualize_paths = batch_dict.pop('visualize_paths') 
        orig_t, orig_h, orig_w = batch_dict.pop('orig_size')[0]

        batch_size, T, _, H, W = videos.shape

        # 抽特征
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4),)  # b c t h w
        # 对比学习
        # self.video_backbone.compute_loss(multiscales, text_inputs)

        # b c t h w
        fencoded_multiscales = self.multiscale_encoder(multiscales={scale_name:scale_feat.clone() \
                                                       for scale_name, scale_feat in multiscales.items()})   
        # {'pred_masks', 'pred_boxes'}
        unet_output = self.unet(multiscales=fencoded_multiscales)

        pred_masks = unet_output['pred_masks'].unsqueeze(1) # b 1 t h w
        pred_obj = torch.ones(pred_masks.shape[:2]).float().to(self.device) # b 1
        # unpad
        padded_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        pred_masks = pred_masks[0][:, :orig_t, :orig_h, :orig_w] # nq t h w
        pred_obj = pred_obj[0] # nq
        # unnormalize
        orig_video = Trans_F.normalize(padded_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
        
        # 输出应该进入aug callback的接口
        return {
            'video': [orig_video], # t 3 h w
            'pred_masks': [pred_masks], # nq t h w, float
            'pred_obj': [pred_obj] # nq, 0-1
        }

        # prob = temporal_decoder_output[-1]['pred_class'].softmax(-1)[0][:, 0] # b nq c -> nq
        # pred_masks = temporal_decoder_output[-1]['pred_masks'][0].sigmoid() > 0.5 # nq t h w
        # max_query_idx = prob.argmax(-1) # b nq
        # pred_masks = pred_masks[max_query_idx] # t h w
        # import matplotlib.pyplot as plt
        # plt.imsave('./test.png', pred_masks[0].cpu())

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
def conditional_unetseg(configs, device):
    from .decode_frame_query import AUXMapper_v1
    model =  Conditional_UnetSeg(configs)
    model.to(device)
    params_group, log_lr_group_idx = Conditional_UnetSeg.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx
