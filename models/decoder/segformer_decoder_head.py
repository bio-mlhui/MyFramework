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
from transformers import SegformerForSemanticSegmentation
import os

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



# 单个物体
# target: list[1 t' h w], batch
# model output: b t 1 h w
class Video_SingleObjMaskLoss(nn.Module):
    def __init__(self, 
                 loss_config,) -> None:
        super().__init__()
        self.losses = loss_config['losses'] 
        self.register_buffer('device_util', torch.tensor([2]))
    @property
    def device(self,):
        return self.device_util.device
    
    def compute_loss(self, 
                     model_outs, 
                     targets,
                     video_aux_dict,
                     **kwargs):
        loss_values = {
            'mask_dice':0., 'mask_ce':0.,
            'mask_dice_smobj':0., 'mask_ce_smobj':0.,
        }
        assert len(model_outs) == 1
        model_outs = model_outs[0]
        for loss in self.losses:
            loss_extra_param = self.losses[loss]
            if loss == 'mask_dice_ce' :
                loss_dict = self.loss_mask_dice_ce(model_outs, targets,  loss_extra_param=loss_extra_param)
            elif loss == 'point_mask_dice_ce':
                loss_dict = self.loss_point_mask_dice_ce(model_outs, targets, loss_extra_param=loss_extra_param)
            else:
                raise ValueError()
            for key, value in loss_dict.items():
                loss_values[key] = loss_values[key] + value
        return loss_values      


    def loss_mask_dice_ce(self, outputs, targets, loss_extra_param):
        has_ann = targets['has_ann'] # b t
        batch_size = has_ann.shape[0]

        # b t 1 h w -> b 1 t h w
        src_masks = outputs['pred_masks'].permute(0, 2, 1, 3, 4).contiguous() # b nq t h w
        src_masks = [haosen[:, hann]  for haosen, hann in zip(src_masks, has_ann)] # list[nq t' h w]
        src_masks = torch.stack(src_masks, dim=0) # b 1 t' h w
        src_masks = src_masks.flatten(0, 1) # b t' h w

        tgt_masks = targets['masks'] # list[n t' h w]
        tgt_masks = torch.stack(tgt_masks, dim=0).to(src_masks) # b 1 t' h w
        tgt_masks = tgt_masks.flatten(0, 1) # b t' h w

        losses = {
            "mask_ce": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=tgt_masks.shape[0]), # n thw
            "mask_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=tgt_masks.shape[0]),  # n thw
        }
        return losses   

    def loss_point_mask_dice_ce(self, outputs, targets, loss_extra_param):
        has_ann = targets['has_ann'] # b t
        batch_size = has_ann.shape[0]
        nf = has_ann.shape[1]

        # b t 1 h w -> b 1 t h w
        src_masks = outputs['pred_masks'].permute(0, 2, 1, 3, 4).contiguous() # b nq t h w
        src_masks = [haosen[:, hann]  for haosen, hann in zip(src_masks, has_ann)] # list[nq t' h w]
        src_masks = torch.stack(src_masks, dim=0) # b 1 t' h w
        src_masks = src_masks.flatten(0, 1) # b t' h w

        tgt_masks = targets['masks'] # list[n t' h w]
        tgt_masks = torch.stack(tgt_masks, dim=0).to(src_masks) # b 1 t' h w
        tgt_masks = tgt_masks.flatten(0, 1) # b t' h w

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1).unsqueeze(1).contiguous() # nt' 1 h w
        target_masks = tgt_masks.flatten(0, 1).unsqueeze(1).contiguous() 

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                loss_extra_param['num_points'],
                loss_extra_param['oversample_ratio'],
                loss_extra_param['importance_sample_ratio'],
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1) # nt' s

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1) # nt' s

        point_logits = rearrange(point_logits, '(n t) s -> n (t s)',t=nf)
        point_labels = rearrange(point_labels, '(n t) s -> n (t s)',t=nf)

        losses = {
            "mask_dice": ce_mask_loss(point_logits, point_labels, num_boxes=point_labels.shape[0]),
            "mask_ce": dice_loss(point_logits, point_labels, num_boxes=point_labels.shape[0]), 
        }

        del src_masks
        del target_masks
        return losses        


# 多个语义
# target:list[n t' h w]
# model output: 
@META_ARCH_REGISTRY.register()
class ImageSegformer_SingleObjMaskDecoder(nn.Module):
    """
    将最大的scale进行conv
    """
    def __init__(self, 
                 configs,
                 multiscale_shapes) -> None:
        super().__init__()
        from transformers.models.segformer.modeling_segformer import SegformerMLP
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(os.path.join(os.getenv('PT_PATH'), "segformer"),
                                                                            num_labels=1,
                                                                            decoder_hidden_size=configs['decoder_hidden_size'],
                                                                            ignore_mismatched_sizes=True)
        self.segformer_head = pretrained_model.decode_head
        
        # change input mlp
        mlps = []
        for i in ['res2', 'res3', 'res4', 'res5']:
            mlp = SegformerMLP(pretrained_model.config, input_dim=multiscale_shapes[i].dim)
            mlps.append(mlp)
        del self.segformer_head.linear_c
        del self.segformer_head.classifier
        self.segformer_head.linear_c = nn.ModuleList(mlps)
        # self.segformer_head.classifier = nn.Identity()
        self.segformer_head.classifier = nn.Conv2d(pretrained_model.config.decoder_hidden_size, 1, kernel_size=1)
        self.mask_spatial_stride = 4

    def forward(self, multiscale):
        decoder_head_inputs = [multiscale[haosen] for haosen in ['res2', 'res3', 'res4', 'res5']]
        pred_masks = self.segformer_head(decoder_head_inputs) # bt 1 h w
        batch_size = pred_masks.shape[0]
        # b 
        pred_classes = torch.tensor([1, 0])[None, None, :].repeat(batch_size, 1, 1).float()  # b 1 2

        if self.training:
            return {'pred_masks': pred_masks, 'pred_class': pred_classes}
        else:
            pred_masks = F.interpolate(pred_masks, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)
            return {'pred_masks': pred_masks, 'pred_class': pred_classes.softmax(-1)}


@META_ARCH_REGISTRY.register()
class Video_Image2D_SingleObjSegformer_MaskDecoder(nn.Module):
    def __init__(self, configs, 
                 multiscale_shapes=None,
                 ) -> None:
        super().__init__()
        self.image_homo = ImageSegformer_SingleObjMaskDecoder(configs,
                                                              multiscale_shapes=multiscale_shapes)
        self.loss_module = Video_SingleObjMaskLoss(loss_config=configs['loss'], )
    
    def forward(self, 
                multiscales,
                video_aux_dict):
        batch_sisze, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}
        image2d_output = self.image_homo(multiscales) # logits
        pred_masks = image2d_output['pred_masks'] # bt 1 h w
        pred_classes = image2d_output['pred_class'] # bt 1 2

        pred_masks = rearrange(pred_masks, '(b t) nq h w -> b t nq h w',b=batch_sisze, t=nf)
        pred_classes = rearrange(pred_classes, '(b t) nq c -> b t nq c',b=batch_sisze, t=nf)

        return [{'pred_masks': pred_masks, 'pred_class': pred_classes}]


    def compute_loss(self, dec_output, targets, **kwargs):
        return self.loss_module.compute_loss(
            model_outs=dec_output,
            targets=targets,
            video_aux_dict=None
        )


@META_ARCH_REGISTRY.register()
class ImageSegformer_SingleObjMaskDecoder_PT(nn.Module):
    """
    将最大的scale进行conv
    """
    def __init__(self, 
                 configs,
                 multiscale_shapes) -> None:
        super().__init__()
        from transformers.models.segformer.modeling_segformer import SegformerMLP
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(os.path.join(os.getenv('PT_PATH'), "segformer"),
                                                                            num_labels=1,
                                                                            ignore_mismatched_sizes=True)
        self.segformer_head = pretrained_model.decode_head
        self.mask_spatial_stride = 4

    def forward(self, multiscale):
        decoder_head_inputs = [multiscale[haosen] for haosen in ['res2', 'res3', 'res4', 'res5']]
        pred_masks = self.segformer_head(decoder_head_inputs) # bt 1 h w
        batch_size = pred_masks.shape[0]
        # b 
        pred_classes = torch.tensor([1, 0])[None, None, :].repeat(batch_size, 1, 1).float()  # b 1 2

        if self.training:
            return {'pred_masks': pred_masks, 'pred_class': pred_classes}
        else:
            pred_masks = F.interpolate(pred_masks, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)
            return {'pred_masks': pred_masks, 'pred_class': pred_classes.softmax(-1)}


@META_ARCH_REGISTRY.register()
class Video_Image2D_SingleObjSegformer_MaskDecoder_PT(nn.Module):
    def __init__(self, configs, 
                 multiscale_shapes=None,
                 ) -> None:
        super().__init__()
        self.image_homo = ImageSegformer_SingleObjMaskDecoder_PT(configs,
                                                              multiscale_shapes=multiscale_shapes)
        self.loss_module = Video_SingleObjMaskLoss(loss_config=configs['loss'], )
    
    def forward(self, 
                multiscales,
                video_aux_dict):
        batch_sisze, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}
        image2d_output = self.image_homo(multiscales) # logits
        pred_masks = image2d_output['pred_masks'] # bt 1 h w
        pred_classes = image2d_output['pred_class'] # bt 1 2

        pred_masks = rearrange(pred_masks, '(b t) nq h w -> b t nq h w',b=batch_sisze, t=nf)
        pred_classes = rearrange(pred_classes, '(b t) nq c -> b t nq c',b=batch_sisze, t=nf)

        return [{'pred_masks': pred_masks, 'pred_class': pred_classes}]


    def compute_loss(self, dec_output, targets, **kwargs):
        return self.loss_module.compute_loss(
            model_outs=dec_output,
            targets=targets,
            video_aux_dict=None
        )