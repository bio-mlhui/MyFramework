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
import time

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4           = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        out          = x2*x3*x4
        return out

class WeakPolyP(nn.Module):
    def __init__(self,                 
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride
        multiscale_shapes = self.video_backbone.multiscale_shapes
        feature_dimensions = [shape.dim for key, shape in multiscale_shapes.items()]

        self.fusion = Fusion(feature_dimensions)
        self.linear = nn.Conv2d(64, 1, kernel_size=1)

    @property
    def device(self):
        return self.pixel_mean.device

    def model_preds(self, videos):
        # b 3 h w -> b 3 1 h w
        bbout = self.video_backbone(x=videos.unsqueeze(2).contiguous(),)
        res2, res3, res4, res5 = bbout['res2'], bbout['res3'], bbout['res4'], bbout['res5']
        
        # b c t h w
        res2 = res2.squeeze(2).contiguous() # b c h w
        res3 = res3.squeeze(2).contiguous() # b c h w
        res4 = res4.squeeze(2).contiguous() # b c h w
        res5 = res5.squeeze(2).contiguous() # b c h w   
        pred = self.fusion(res2, res3, res4, res5) 
        pred = self.linear(pred)    

        return pred

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        # 看成instance个数是1的instance segmentation
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[1] == 1, 'weak_poly只输入单帧训练'
        image = videos[:, 0] # b 3 h w
        masks = batch_dict['targets']['masks'] # list[n t' h w], b
        mask = torch.stack([mk.squeeze() for mk in masks], dim=0).unsqueeze(1).float() # b 1 h w
        # plt.imsave('./mask.png', mask[0][0].cpu().numpy())
        size1          = np.random.choice([256, 288, 320, 352, 384, 416, 448])
        image1         = F.interpolate(image, size=size1, mode='bilinear')
        pred1          = self.model_preds(image1) # b 1 h w
        pred1          = F.interpolate(pred1, size=352, mode='bilinear') # b 1 h w
        size2          = np.random.choice([256, 288, 320, 352, 384, 416, 448])
        image2         = F.interpolate(image, size=size2, mode='bilinear')
        pred2          = self.model_preds(image2)
        pred2          = F.interpolate(pred2, size=352, mode='bilinear') # b 1 h w
        
        ## loss_sc
        loss_sc        = (torch.sigmoid(pred1)-torch.sigmoid(pred2)).abs()
        loss_sc        = loss_sc[mask[:,0:1]==1].mean()
        ## M2B transformation
        pred           = torch.cat([pred1, pred2], dim=0) # 2b 1 h w
        mask           = torch.cat([mask, mask], dim=0) # 2b 1 h w
        predW, predH   = pred.max(dim=2, keepdim=True)[0], pred.max(dim=3, keepdim=True)[0]
        pred           = torch.minimum(predW, predH)
        pred, mask     = pred[:,0], mask[:,0]
        ## loss_ce + loss_dice 
        loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
        pred           = torch.sigmoid(pred)
        inter          = (pred*mask).sum(dim=(1,2))
        union          = (pred+mask).sum(dim=(1,2))
        loss_dice      = 1-(2*inter/(union+1)).mean()
        loss           = loss_ce + loss_dice + loss_sc

        loss.backward()       
        if not math.isfinite(loss.item()):
            logging.debug("Loss is {}, stopping training".format(loss.item()))
            raise RuntimeError()
        
        loss_value_dict = {
            'loss_ce': loss_ce,
            'loss_dice': loss_dice,
            'loss_sc': loss_sc
        }
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {k: v for k, v in loss_value_dict.items()} 
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
            
    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[1] == 1, 'weak_poly只输入单帧训练和测试' 
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        # videos = 
        pred_mask = self.model_preds(videos[:, 0].contiguous()) # b 1 h w
        pred_mask = F.interpolate(pred_mask, size=(H, W), mode='bilinear') > 0 # b 1 h w
        # h w
        pred_mask = pred_mask[0][0]
        # 2
        pred_class = torch.tensor([1, 0]).to(self.device)

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        pred_mask = pred_mask[:orig_h, :orig_w] # h w
        pred_masks = pred_mask.unsqueeze(0) # 1 h w
        pred_class = pred_class.unsqueeze(0) # 1 c

        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])

        return {
            'video': [orig_video.cpu()], # [t 3 h w], 1
            'pred_masks': [[pred_masks.cpu()]], # [list[1 h w], t, bool], 1
            'pred_class': [[pred_class.cpu()]], # [list[1 c], t, probability], 1
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
def weak_polyp(configs, device):
    from ..aux_mapper import AUXMapper_v1
    model = WeakPolyP(configs)
    model.to(device)
    params_group, log_lr_group_idx = WeakPolyP.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx

