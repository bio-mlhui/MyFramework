"""
Modified from DETR https://github.com/facebookresearch/detr
"""
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
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from models.layers.utils import _get_clones
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

from models.backbone.video_swin import PatchEmbed3D, PatchMerging

@META_ARCH_REGISTRY.register()
class ImageConv_MaskDecoder(nn.Module):
    """
    将最大的scale进行conv
    """
    def __init__(self, 
                 configs,) -> None:
        super().__init__()
        self.classified_scale = configs['classified_scale']
        scale_dim = configs['scale_dim']
        self.meask_head = nn.Conv2d(scale_dim, 1, kernel_size=1)
    
    def forward(self, multiscale):
        # b c h w
        batch_size = multiscale[self.classified_scale].shape[0]
        return {
            'pred_masks': self.meask_head(multiscale[self.classified_scale]), # b 1 h w
            # b 1 2
            'pred_classes': torch.tensor([1, 0])[None, None, :].repeat(batch_size, 1, 1)
        }
        
    def compute_loss(self, dec_output):
        pass



@META_ARCH_REGISTRY.register()
class Video2D_ImageConv_MaskDecoder(nn.Module):
    def __init__(self, configs,) -> None:
        super().__init__()
        self.image_homo = ImageConv_MaskDecoder(configs)

    def forward(self, multiscales):
        batch_sisze, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}
        output = self.image_homo(multiscales)

        pred_masks = output['pred_masks'] # bt 1 h w
        pred_masks = rearrange(pred_masks, '(b t) nq h w -> b t nq h w',b=batch_sisze, t=nf)
        
        pred_classes = output['pred_classes'] # bt nq c
        pred_classes = rearrange(pred_classes, '(b t) nq c -> b t nq c',b=batch_sisze, t=nf)
        return {'pred_masks': pred_masks, 'pred_classes': pred_classes}

    def compute_loss(self, dec_output, targets, frame_targets):
        has_ann = targets['has_ann'].flatten() # bT
        masks = targets['masks'] # list[n T' h w], b
        gt_masks = torch.cat([mk.squeeze(0) for mk in masks], dim=0).float() # bT' h w
        mask_shape = gt_masks.shape[-2:]

        pred_masks = dec_output['pred_masks'] # b T 1 h w
        pred_masks = F.interpolate(pred_masks.flatten(0,1), size=mask_shape, mode='bilinear') # bT 1 h w
        pred_masks = pred_masks.squeeze(1)[has_ann] # bT' h w

        loss_ce    = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        pred_masks = torch.sigmoid(pred_masks)
        inter      = (pred_masks*gt_masks).sum(dim=(1,2))
        union      = (pred_masks+gt_masks).sum(dim=(1,2))
        loss_dice  = 1-(2*inter/(union+1)).mean()

        return {
            'loss_dice': loss_dice,
            'loss_ce': loss_ce
        }

    def compute_loss_two_outputs(self, dec_output1, dec_output2, targets):
        has_ann = targets['has_ann'].flatten() # bT
        masks = targets['masks'] # list[n T' h w], b
        gt_masks = torch.cat([mk.squeeze(0) for mk in masks], dim=0).float() # bT' h w
        mask_shape = gt_masks.shape[-2:]

        pred1 = dec_output1['pred_masks'] # b T 1 h w
        pred1 = F.interpolate(pred1.flatten(0,1), size=mask_shape, mode='bilinear') # bT 1 h w
        pred1 = pred1.squeeze(1)[has_ann] # bT' h w

        pred2 = dec_output2['pred_masks'] # b T 1 h w
        pred2 = F.interpolate(pred2.flatten(0,1), size=mask_shape, mode='bilinear') # bT 1 h w
        pred2 = pred2.squeeze(1)[has_ann] # bT' h w

        ## loss_sc
        loss_sc        = (torch.sigmoid(pred1)-torch.sigmoid(pred2)).abs()
        loss_sc        = loss_sc[gt_masks==1].mean()
        return {
            'loss_sc': loss_sc
        }