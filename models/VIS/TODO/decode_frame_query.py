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
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import matplotlib.pyplot as plt
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
__all__ = ['decode_frame_query']
import time


class Decode_FrameQuery_VIS(nn.Module): # frame decoder做object segmentation, temporal decoder做referent segmentation
    # model知道整个module-submodule结构
    def __init__(self,
                 configs,
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
        # (multiscale, text_dict) 在帧间parsing 
        frame_encoder_configs = configs['model']['frame_encoder']
        # frame query聚合每帧的信息
        frame_decoder_configs = configs['model']['frame_decoder']

        # temporal encoder使用的frame decoder的层数
        self.num_frame_decoder_layers_used = configs['model']['num_frame_decoder_layers_used']

        # frame query每帧的信息进行交流
        temporal_encoder_configs = configs['model']['temporal_encoder']
        # temporal query聚合整个video的信息
        temporal_decoder_configs = configs['model']['temporal_decoder']

        # video_backbone.video_backbone/text_backbone
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        frame_encoder_cls = META_ARCH_REGISTRY.get(frame_encoder_configs['name'])
        self.frame_encoder = frame_encoder_cls(frame_encoder_configs,
                                                multiscale_shapes=self.video_backbone.multiscale_shapes,)
        
        frame_decoder_cls = META_ARCH_REGISTRY.get(frame_decoder_configs['name'])
        self.frame_decoder = frame_decoder_cls(frame_decoder_configs,)
        
        temporal_encoder_cls = META_ARCH_REGISTRY.get(temporal_encoder_configs['name'])
        self.temporal_encoder = temporal_encoder_cls(temporal_encoder_configs)
        
        temporal_decoder_cls = META_ARCH_REGISTRY.get(temporal_decoder_configs['name'])
        self.temporal_decoder = temporal_decoder_cls(temporal_decoder_configs)

        self.decoder_mask_stride = self.frame_decoder.mask_stride

 
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

    def forward(self, batch_dict):
        VIS_TrainAPI_clipped_video
        videos = batch_dict.pop('video_dict')['videos'] # 0-1, float, b t 3 h w
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = batch_dict.pop('targets')
        frame_targets = batch_dict.pop('frame_targets')
        # biparte matching
        tgt_masks = targets['masks'] # list[t' H W], batch
        batch_size = len(tgt_masks)
        for btc_idx in range(batch_size):
            start = int(self.decoder_mask_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, start::self.decoder_mask_stride, start::self.decoder_mask_stride] 
            assert tgt_masks[btc_idx].size(1) * self.decoder_mask_stride == im_h
            assert tgt_masks[btc_idx].size(2) * self.decoder_mask_stride == im_w
        targets['masks'] = tgt_masks
        
        frame_tgt_masks = frame_targets['masks'] # list[ni H W], bt'
        batch_size = len(frame_tgt_masks)
        for btc_idx in range(batch_size):
            start = int(self.decoder_mask_stride // 2)
            im_h, im_w = frame_tgt_masks[btc_idx].shape[-2:]
            frame_tgt_masks[btc_idx] = frame_tgt_masks[btc_idx][:, start::self.decoder_mask_stride, start::self.decoder_mask_stride] 
            assert frame_tgt_masks[btc_idx].size(1) * self.decoder_mask_stride == im_h
            assert frame_tgt_masks[btc_idx].size(2) * self.decoder_mask_stride == im_w
        frame_targets['masks'] = frame_tgt_masks
        
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch

        batch_size, T, _, H, W = videos.shape
        # 抽特征
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4).contiguous())  # b c t h w
        # 对比学习
        # self.video_backbone.compute_loss(multiscales, text_inputs)
        fencoded_multiscales = self.frame_encoder(multiscales={scale_name:scale_feat.clone() \
                                                               for scale_name, scale_feat in multiscales.items()})
        # bt c h w -> {'queries': list[b t nqf c], 'pred_masks': b t n h w, 'pred_boxes': b t n 4, 'pred_class': b t n class+1}
        frame_decoder_output = \
            self.frame_decoder(video_multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in fencoded_multiscales.items()},)
        frame_queries_by_layer = [layer_output['frame_queries'] for layer_output in frame_decoder_output]
        frame_queries_by_layer = frame_queries_by_layer[(-1 * self.num_frame_decoder_layers_used):] # list[ b t nq c ]
        
        # b t nqf c -> L b t nqf c -> Lb t nqf c
        frame_queries = torch.stack(frame_queries_by_layer, dim=0).flatten(0, 1).contiguous()

        # Lb t nqf c, 
        tencoded_frame_queries = self.temporal_encoder(frame_queries)
        # Lb t nqf c -> {'queries': list[Lb nq c], 'predictions': }
        temporal_decoder_output = self.temporal_decoder(frame_query=tencoded_frame_queries, 
                                                        multiscales=fencoded_multiscales)
        frame_decoder_losses = self.frame_decoder.compute_loss(frame_decoder_output, frame_targets)
        frame_decoder_losses = {f'frame_decoder_{key}': value for key, value in frame_decoder_losses.items()}
        temporal_decoder_losses = self.temporal_decoder.compute_loss(temporal_decoder_output, targets)
        temporal_decoder_losses = {f'temporal_decoder_{key}': value for key, value in temporal_decoder_losses.items()}
        assert len(set(list(frame_decoder_losses.keys())) & set(list(temporal_decoder_losses.keys()))) == 0
        loss_value_dict = {}
        loss_value_dict.update(frame_decoder_losses)
        loss_value_dict.update(temporal_decoder_losses)
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
        # from models.backbone.swin import compute_mask
        # compute_mask.cache_clear()        
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample(self, batch_dict):
        VIS_EvalAPI_clipped_video_request_ann
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # B T 3 H W
        assert len(videos) == 1
        visualize_paths = batch_dict.pop('visualize_paths') 
        orig_t, orig_h, orig_w = batch_dict.pop('orig_size')[0]

        batch_size, T, _, H, W = videos.shape
        
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4),)  # b c t h w

        fencoded_multiscales = self.frame_encoder(multiscales={scale_name:scale_feat.clone() \
                                                               for scale_name, scale_feat in multiscales.items()})

        frame_decoder_output = \
            self.frame_decoder(video_multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in fencoded_multiscales.items()},)

        frame_queries = frame_decoder_output[-1]['frame_queries'] # b t nqf c
        tencoded_frame_queries = self.temporal_encoder(frame_queries)
        temporal_decoder_output = self.temporal_decoder(frame_query=tencoded_frame_queries, 
                                                        multiscales=fencoded_multiscales)

        pred_masks = temporal_decoder_output[-1]['pred_masks'] # b n t h w
        pred_obj = temporal_decoder_output[-1]['pred_class'].softmax(-1)[:, :, 0] # b n 
        assert len(pred_masks) == 1
        assert len(pred_obj) == 1
        # unpad
        padded_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        pred_masks = pred_masks[0][:, :orig_t, :orig_h, :orig_w] # nq t h w
        pred_obj = pred_obj[0] # nq

        # unnormalize
        orig_video = Trans_F.normalize(padded_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
        VIS_AugAPI
        return {
            'video': [orig_video], # t 3 h w
            'pred_masks': [pred_masks.unbind(1)], # list[nq h w], t
            'pred_obj': [pred_obj.unsqueeze(0).repeat(orig_t, 1).unbind(0)] # list[nq], t
        }

        # prob = temporal_decoder_output[-1]['pred_class'].softmax(-1)[0][:, 0] # b nq c -> nq
        # pred_masks = temporal_decoder_output[-1]['pred_masks'][0].sigmoid() > 0.5 # nq t h w
        # max_query_idx = prob.argmax(-1) # b nq
        # pred_masks = pred_masks[max_query_idx] # t h w
        # import matplotlib.pyplot as plt
        # plt.imsave('./test.png', pred_masks[0].cpu())

    @torch.no_grad()
    def tta_sample(self, asembles, visualize_dir=None):
        # test time agumentation 
        preds_by_aug = []
        for sample in asembles:
            preds_by_aug.append(self.sample(asembles, visualize_dir=visualize_dir))
        return preds_by_aug


@register_model
def decode_frame_query_vis(configs, device):
    from .aux_mapper import AUXMapper_v1
    model =  Decode_FrameQuery_VIS(configs)
    model.to(device)
    params_group, log_lr_group_idx = Decode_FrameQuery_VIS.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx

