
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
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann

from models.backbone.video_swin import PatchEmbed3D, PatchMerging
from models.layers.mamba.vss_layer_3d import VSSLayer_3d


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
    from .aux_mapper import AUXMapper_v1
    model = UnetPolyPSeg_VideoDiff(configs)
    model.to(device)
    params_group, log_lr_group_idx = UnetPolyPSeg_VideoDiff.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx




# 单类语义分割, 没有分类曾
class Video_Umamba(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.loss_weight = configs['model']['loss_weight'] # loss_mask, loss_dice
        self.patch_size=configs['model']['patch_size'] # (1, 4, 4)
        self.depths = configs['model']['depths'] # [2, 2, 2, 2]
        self.dims=configs['model']['dims'] # [96, 192, 384, 768],
        self.d_state=configs['model']['d_state'] # 16
        self.attn_drop_rate=configs['model']['attn_drop_rate'] # 0
        self.drop_rate = configs['model']['drop_rate'] # 0
        self.max_stride = 32
        # drop_path_rate: 每一个Path的dropout的概率
        dpr = [x.item() for x in torch.linspace(0, configs['model']['drop_path_rate'], sum(self.depths))]
                
        self.norm_layer= nn.LayerNorm
        self.num_features = self.dims[-1]
    
        self.decoder_is_mamba = configs['model']['decoder_is_mamba']
        # 如果decoder也是Mamba话, 那么每一层是 mamba 
        # 如果decoder不是Mamba话, 那么每层就是 卷积
    
        # ssm block的版本: 0: thw -> thw. 1: thw -> (thw, htw, hwt)   
        self.ssm_version = configs['model']['ssm_version'] 

        self.num_layers = len(self.depths) # 
        self.embed_dim = self.dims[0]
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.patch_embed = PatchEmbed3D(patch_size=self.patch_size, embed_dim=self.embed_dim, norm_layer=nn.LayerNorm)

        self.down_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_3d(
                dim=self.dims[i_layer],
                depth=self.depths[i_layer],
                d_state=math.ceil(self.dims[0] / 6) if self.d_state is None else self.d_state, # 20240109
                drop=self.drop_rate, 
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=False,
                ssm_version=self.ssm_version
            )
            self.down_layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)

        # 96， 192， 384， 768
        up_dims = self.dims[::-1][1:] # 384， 192， 96
        # 2， 2， 2， 2
        up_depths = self.depths[::-1][1:]
        dpr = dpr[::-1][self.depths[-1]:]
        self.up_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList() # upsample, concate, conv3d, norm
        for i_layer in range(self.num_layers - 1):
            self.upsample_layers.append(
                nn.ModuleList(
                    [
                        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=False), # 768
                        # 768 + 384 -> 384
                        nn.Conv3d(up_dims[i_layer] * 3, up_dims[i_layer], kernel_size=3, padding=1),
                        nn.GroupNorm(32, num_channels=up_dims[i_layer]), # norm
                    ]
                )
            )
            if self.decoder_is_mamba:
                layer = VSSLayer_3d(
                    dim=up_dims[i_layer],
                    depth=up_depths[i_layer],
                    d_state=math.ceil(up_dims[0] / 6) if self.d_state is None else self.d_state, # 20240109
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[sum(up_depths[:i_layer]):sum(up_depths[:i_layer + 1])],
                    norm_layer=nn.LayerNorm,
                    downsample=None,
                    use_checkpoint=False,
                    ssm_version=self.ssm_version
                )
            else:
                layer = nn.Sequential(
                    nn.Conv3d(up_dims[i_layer], up_dims[i_layer], kernel_size=3, padding=1),
                    nn.GroupNorm(32, num_channels=up_dims[i_layer]),
                    nn.Conv3d(up_dims[i_layer], up_dims[i_layer], kernel_size=3, padding=1),
                    nn.GroupNorm(32, num_channels=up_dims[i_layer]),
                )

            self.up_layers.append(layer)

        # encoder inflate vmamba的预训练网络
        self.downsample_layers = nn.ModuleList()
        for downlayer in self.down_layers:
            self.downsample_layers.append(downlayer.downsample)
            downlayer.downsample = None
        assert self.downsample_layers[-1] == None
        self.mask_head = nn.Conv3d(self.embed_dim, 1, kernel_size=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def unet_forward(self, video, ):
        # video: b c t h w
        x = self.patch_embed(video) # b c t h/4 w/4
        x = self.pos_drop(x) # b patch_embed t h/4 w/4

        h = [] # 4, 8, 16, 32, b c t h w

        for layer, downsample_layer in zip(self.down_layers, self.downsample_layers):
            x = layer(x.contiguous())
            if downsample_layer is not None:
                h.append(x)
                x = x.permute(0, 2, 3, 4, 1).contiguous() # b t h w c
                x = downsample_layer(x)
                x = x.permute(0, 4, 1,2,3).contiguous()

        x = self.norm(x.permute(0, 2, 3, 4, 1).contiguous()) # middle_norm
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        for layer, upsample_layer, enc_feats in zip(self.up_layers, self.upsample_layers, h[::-1]):
            batch_size, dim, nf, h, w = x.shape
            x = upsample_layer[0](x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous())  # bt c h w
            x = rearrange(x, '(b t) c h w -> b c t h w',b=batch_size, t=nf)
            x = torch.cat([x, enc_feats], dim=1).contiguous()  # b 3c t h w
            x = upsample_layer[1](x) # b c t h w
            x = upsample_layer[2](x)
            x = layer(x.contiguous())

        return {'pred_masks': self.mask_head(x)} # b 1 t h w
    
    @property
    def device(self):
        return self.pixel_mean.device

    def unet_loss(self, unet_output, targets):
        has_ann = targets['has_ann'].flatten(0,1) # bT
        gt_masks = targets['masks'] # list[n T' h w], b
        gt_masks = torch.stack([mk.squeeze(0) for mk in gt_masks], dim=0).float().flatten(0,1) # bT' h w
        H, W = gt_masks.shape[-2:]
        # b 1 t h w
        pred_masks = unet_output['pred_masks'].squeeze(1).flatten(0, 1)[has_ann].unsqueeze(1) # bT' 1 h/4 w/4
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False)[:, 0] # bT' H W
        loss_ce        = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)   

        pred_masks = pred_masks.sigmoid()
        inter          = (pred_masks*gt_masks).sum(dim=(1,2))
        union          = (pred_masks+gt_masks).sum(dim=(1,2))
        loss_dice      = 1-(2*inter/(union+1)).mean()
        # plt.imsave('./mask.png', gt_masks[0].cpu().numpy())
              
        return {
            'loss_dice': loss_dice,
            'loss_mask': loss_ce
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
        log_lr_group_idx = {'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)

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
        assert self.training
        orig_videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (orig_videos - self.pixel_mean) / self.pixel_std
        
        # b 3 t h w -> b 1 t h w
        unet_output = self.unet_forward(videos.permute(0, 2, 1, 3, 4).contiguous())
        loss_value_dict = self.unet_loss(unet_output, targets=batch_dict['targets']) # b 1 t h w

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
        assert not self.training
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        batch_size, T, _, H, W = videos.shape
        assert batch_size == 1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        
        # b t 3 h w -> b 3 t h w -> b 1 t h w
        unet_output = self.unet_forward(videos.permute(0, 2, 1, 3, 4).contiguous())
        pred_masks = unet_output['pred_masks'][0] # 1 t H/4 W/4
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False) > 0 # 1 t h w
        # 2
        pred_class = torch.tensor([1, 0]).to(self.device) # 0类是前景, 1是后景

        VIS_Aug_CallbackAPI
        pred_masks = pred_masks[:, :, :orig_h, :orig_w] # 1 t h w
        pred_class = pred_class.unsqueeze(0).repeat(orig_t, 1).unsqueeze(1) # t 1 c

        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
        # plt.imsave('./frame.png', orig_video[0].permute(1,2,0).cpu().numpy())
        # 每帧多个预测
        pred_masks = pred_masks.transpose(0, 1).cpu().unbind(0) # list[1 h w], t
        pred_class = pred_class.cpu().unbind(0) # list[1 c], t
        return {
            'video': [orig_video.cpu()], # [t 3 h w], 1
            'pred_masks': [pred_masks], # list[[1 h w], t], 1
            'pred_class': [pred_class], # list[[1 c], t], 1
        }
    
    @torch.no_grad()
    def tta_sample(self, asembles, visualize_dir=None):
        # test time agumentation 
        preds_by_aug = []
        for sample in asembles:
            preds_by_aug.append(self.sample(asembles, visualize_dir=visualize_dir))
        return preds_by_aug


@register_model
def vid_mamba(configs, device):
    from .aux_mapper import AUXMapper_v1
    model =  Video_Umamba(configs)
    model.to(device)
    params_group, log_lr_group_idx = Video_Umamba.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx

