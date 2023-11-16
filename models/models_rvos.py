from typing import Any, Optional
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from einops import repeat, reduce, rearrange
from util.misc import NestedTensor
import matplotlib.pyplot as plt
import copy
import torch_geometric.nn as geo_nn
from torch_geometric.data import Batch
from scipy.optimize import linear_sum_assignment

###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
from .position_encoding import build_position_encoding
from .model_utils import find_scale_from_multiscales, find_scales_from_multiscales, pad_1d_feats, \
    register_model, get_optimizer, get_total_grad_norm,\
        visualization_for_AMR_V0, zero_module, _get_clones
from .layers_unimodal_attention import FeatureResizer, CrossAttentionLayer, MLP, SelfAttentionLayer, FFNLayer 
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from .transformer import TransformerEncoder, TransformerEncoderLayer
import pycocotools.mask as mask_util
import util.box_ops as box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride, nested_tensor_from_tensor_list_with_stride
from functools import partial
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def ce_mask_loss(inputs, targets, num_boxes):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: 
            b=n_sigma thw
        targets: b=n_sigma thw
            (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # n_sigma=b thw
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # mean(b mean(thw)), 对于a2d来说，num_boxes=
    return ce_loss.mean(1).sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1) # [N, M, THW]
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1) # [N, M, THW]

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2) # [N, M]

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

class Fpn2D(nn.Module):
    def __init__(self, dim, cascaded_scales) -> None:
        """
        cascaded_scales: ['1','4'],  ['1','16'], ['1','32']
        """
        super().__init__()
        # from small to big
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert len(cascaded_scales) > 1
        cascaded_scales = cascaded_scales[::-1] # ['1','32'], ['1','16'], ['1','4'],
        for (temporal_stride, spatial_stride), (next_temporal_stride, next_spatial_stride) \
            in zip(cascaded_scales[:-1], cascaded_scales[1:]):
            assert temporal_stride == next_temporal_stride, 'the temporal stride must be the same for the FPN 2D'
            self.adapters.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.upsamples.append(nn.Upsample(scale_factor=spatial_stride//next_spatial_stride, mode='bilinear'))
            self.convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(32, dim))
        
        self.cascaded_scales = cascaded_scales
    
    def forward(self, multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales):
        """ bt c h w"""
        idxs = find_scales_from_multiscales(video_feat_scales, self.cascaded_scales) 
        fused_feats = [multiscales[idx] for idx in idxs]  # 从小到大

        for idx, (small_feat, large_feat) in enumerate(zip(fused_feats[:-1], fused_feats[1:])): # from small map to large map 
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feat) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            fused_feats[idx+1] = large_feat
        
        for idx, scale_idx in enumerate(idxs):
            multiscales[scale_idx] = fused_feats[idx]

        return multiscales

class Fpn2D_multiple(nn.Module):
    def __init__(self, dim, cascaded_scales) -> None:
        """
        cascaded_scales: ['1','4'],  ['1','16'], ['1','32']
        """
        super().__init__()
        # from small to big
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert len(cascaded_scales) > 1
        cascaded_scales = cascaded_scales[::-1] # ['1','32'], ['1','16'], ['1','4'],
        for (temporal_stride, spatial_stride), (next_temporal_stride, next_spatial_stride) \
            in zip(cascaded_scales[:-1], cascaded_scales[1:]):
            assert temporal_stride == next_temporal_stride, 'the temporal stride must be the same for the FPN 2D'
            self.adapters.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.upsamples.append(nn.Upsample(scale_factor=spatial_stride//next_spatial_stride, mode='bilinear'))
            self.convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(32, dim))
        
        self.cascaded_scales = cascaded_scales
    
    def forward(self, multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales):
        """ bt c h w"""
        idxs = find_scales_from_multiscales(video_feat_scales, self.cascaded_scales) 
        fused_feats = [multiscales[idx] for idx in idxs]  # 从小到大
        new_fused_feats = []
        new_fused_feats.append(fused_feats[0])
        for idx, large_feat in enumerate(fused_feats[1:]): # from small map to large map 
            small_feats = new_fused_feats[-1]
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feats) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            new_fused_feats.append(large_feat)
        
        for idx, scale_idx in enumerate(idxs):
            multiscales[scale_idx] = new_fused_feats[idx]

        return multiscales

class DeformVideo2D_with_FPN(nn.Module):
    def __init__(self, 
                 d_model,
                d_ffn=2048,
                dropout=0.,
                activation='relu',
                nheads=8,
                # important
                fused_scales=None, 
                fpn_strides=None,

                npoints=4, 
                nlayers=6,
                 ) -> None:
        super().__init__()
        n_levels = len(fused_scales)
        self.fused_scales = fused_scales
        encoder = DeformableTransformerEncoder(
                DeformableTransformerEncoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    n_levels=n_levels,
                    n_heads=nheads,
                    n_points=npoints,
                ),
                nlayers
        )
        self.deform_encoder = encoder
        self.level_embed = nn.Embedding(n_levels, d_model)
        self.num_feature_levels = n_levels

        if fpn_strides is not None:
            self.fpn = Fpn2D(dim=d_model, cascaded_scales=fpn_strides)
        else:
            self.fpn = None
        
    def get_valid_ratio(self, mask):
        """
        Input:
            - mask:
                bt h w
        Output:
            - int
        """
        _, H, W = mask.shape
        # T(bt, )
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # T(bt, )
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # T(bt, 2)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, multiscales, multiscales_pad_masks, multiscales_poses, video_feat_scales):
        """ bt c h w"""
        fused_scale_idxs = find_scales_from_multiscales(video_feat_scales, self.fused_scales)
        srcs = [multiscales[idx] for idx in fused_scale_idxs]
        masks = [multiscales_pad_masks[idx] for idx in fused_scale_idxs]
        pos_embeds = [multiscales_poses[idx] for idx in fused_scale_idxs]

        src_flatten = []
        mask_flattn = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bt, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = rearrange(src, 'bt c h w -> bt (h w) c')
            mask = rearrange(mask, 'bt h w -> bt (h w)')
            pos_embed = rearrange(pos_embed, 'bt c h w -> bt (h w) c')
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl][None, None, :]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
            src_flatten.append(src)
            mask_flattn.append(mask)
            
        # bt \sigma(hi wi) c
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flattn, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        
        # #levels, 2
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # (0, h0*wo, h1*w1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # bt num_levels 2
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # bt (h_sigma, w_sigma) c  # bt hw_sigma heads num_scales npoints 2
        memory, sampling_locations_by_layer, attention_weights_by_layer = \
            self.deform_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                lvl_pos_embed_flatten, mask_flatten)
        
        memory_features = []
        spatial_index = 0
        for lvl in range(self.num_feature_levels):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index: (spatial_index + h*w), :].contiguous()
            memory_lvl = rearrange(memory_lvl, 'bt (h w) c -> bt c h w',h=h, w=w)
            memory_features.append(memory_lvl)
            spatial_index += h*w
        
        for idx, scale_idx in enumerate(fused_scale_idxs):
            multiscales[scale_idx] = memory_features[idx]

        multiscales = self.fpn(multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales)
        return multiscales, sampling_locations_by_layer, attention_weights_by_layer

class Scale32CatText_Encoder_FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

# from natten import NeighborhoodAttention3D
# class Neighborhood3D_with_FPN(nn.Module):
#     def __init__(self, 
#                  d_model,
#                 d_ffn=2048,
#                 dropout=0.,
#                 activation='relu',
#                 nheads=8,
#                 # important
#                 fused_scales=None, 
#                 fpn_strides=None,

#                 npoints=4, 
#                 nlayers=6,
#                  ) -> None:
#         super().__init__()
#         n_levels = len(fused_scales)
#         self.fused_scales = fused_scales
        
#         n3d_layer = NeighborhoodAttention3D(
#             dim=d_model,
            
#         )
#         encoder = DeformableTransformerEncoder(
#                 DeformableTransformerEncoderLayer(
#                     d_model=d_model,
#                     d_ffn=d_ffn,
#                     dropout=dropout,
#                     activation=activation,
#                     n_levels=n_levels,
#                     n_heads=nheads,
#                     n_points=npoints,
#                 ),
#                 nlayers
#         )
#         self.deform_encoder = encoder
#         self.level_embed = nn.Embedding(n_levels, d_model)
#         self.num_feature_levels = n_levels

#         if fpn_strides is not None:
#             self.fpn = Fpn2D(dim=d_model, cascaded_scales=fpn_strides)
#         else:
#             self.fpn = None
        
#     def get_valid_ratio(self, mask):
#         """
#         Input:
#             - mask:
#                 bt h w
#         Output:
#             - int
#         """
#         _, H, W = mask.shape
#         # T(bt, )
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         # T(bt, )
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         # T(bt, 2)
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
    
#     def forward(self, multiscales, multiscales_pad_masks, multiscales_poses, video_feat_scales):
#         """ bt c h w"""
#         fused_scale_idxs = find_scales_from_multiscales(video_feat_scales, self.fused_scales)
#         srcs = [multiscales[idx] for idx in fused_scale_idxs]
#         masks = [multiscales_pad_masks[idx] for idx in fused_scale_idxs]
#         pos_embeds = [multiscales_poses[idx] for idx in fused_scale_idxs]

#         src_flatten = []
#         mask_flattn = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
#             bt, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
            
#             src = rearrange(src, 'bt c h w -> bt (h w) c')
#             mask = rearrange(mask, 'bt h w -> bt (h w)')
#             pos_embed = rearrange(pos_embed, 'bt c h w -> bt (h w) c')
#             lvl_pos_embed = pos_embed + self.level_embed.weight[lvl][None, None, :]
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
            
#             src_flatten.append(src)
#             mask_flattn.append(mask)
            
#         # bt \sigma(hi wi) c
#         src_flatten = torch.cat(src_flatten, dim=1)
#         mask_flatten = torch.cat(mask_flattn, dim=1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        
#         # #levels, 2
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         # (0, h0*wo, h1*w1)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         # bt num_levels 2
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
#         # bt (h_sigma, w_sigma) c  # bt hw_sigma heads num_scales npoints 2
#         memory, sampling_locations_by_layer, attention_weights_by_layer = \
#             self.deform_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
#                                 lvl_pos_embed_flatten, mask_flatten,)
        
#         memory_features = []
#         spatial_index = 0
#         for lvl in range(self.num_feature_levels):
#             h, w = spatial_shapes[lvl]
#             memory_lvl = memory[:, spatial_index: (spatial_index + h*w), :].contiguous()
#             memory_lvl = rearrange(memory_lvl, 'bt (h w) c -> bt c h w',h=h, w=w)
#             memory_features.append(memory_lvl)
#             spatial_index += h*w
        
#         for idx, scale_idx in enumerate(fused_scale_idxs):
#             multiscales[scale_idx] = memory_features[idx]

#         multiscales = self.fpn(multiscales, multiscales_pad_masks,  multiscales_poses, video_feat_scales)
#         return multiscales, sampling_locations_by_layer, attention_weights_by_layer
def get_parsing_encoder(name, configs):
    if name == 'deform_video_2d_fpn':
        return DeformVideo2D_with_FPN(**configs)
    
    elif name == 'split_obj_ref_deform_video_2d_fpn':
        obj_seg_nlayers = configs.pop('obj_seg_nlayers')
        ref_seg_nlayers = configs.pop('ref_seg_nlayers')
        assert obj_seg_nlayers > 0
        obj_parsing_encoder = DeformVideo2D_with_FPN(**configs, nlayers=obj_seg_nlayers)
        if ref_seg_nlayers == 0:
            ref_parsing_encoder = None
        else:
            ref_parsing_encoder = DeformVideo2D_with_FPN(**configs, nlayers=ref_seg_nlayers)
        return obj_parsing_encoder, ref_parsing_encoder
    elif name == 'fpn2d':
        return Fpn2D_multiple(dim=configs['d_model'],
                     cascaded_scales=configs['cascaded_scales'])
    else:
        raise ValueError()

def get_fusion(name, configs):
    if name == 'VisionLanguageFusionModule':
        return VisionLanguageFusionModule(**configs)
    elif name == 'self_encoder':
        encoder_nlayers = configs.pop('nlayers')
        return TransformerEncoder(
            TransformerEncoderLayer(
                **configs
            ),encoder_nlayers
        )
    elif name == 'none':
        return None

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None):
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask) # b tgt src, float, 0,1
        tgt = tgt * tgt2
        return tgt, attn_weights

###########################################################################
# amr without variable
###########################################################################
class AMR_v0(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                    
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        # video encoder
        from .video_swin import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=os.path.join(pt_dir, swint_pretrained_path),
                                                running_mode=swint_runnning_mode)
        if swint_freeze:
            for p in self.video_swint.parameters():
                p.requires_grad_(False) 
                 
        assert len(video_projs) == len(video_feat_scales)
        self.video_feat_scales = video_feat_scales
        backbone_channels, backbone_scales = self.video_swint.get_desc()
        assert len(backbone_channels) == len(backbone_scales)
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        # amr encoder
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        
        self.fusion_amr_who_cross = fusion.pop('amr_cross', None)
        fusion_name = fusion.pop('name')
        self.cross_product = get_fusion(fusion_name, fusion)

        self.deform_multiscale_2dencoder = get_parsing_encoder(parsing_encoder.pop('name'),
                                                               parsing_encoder)

        self.decoder_used_scales = refdecoder['used_scales']
        self.decoder_conved_scale = refdecoder['conved_scale']
        self.decoder_nlayers = refdecoder['nlayers']
        self.decoder_level_embed = nn.Embedding(len(self.decoder_used_scales), d_model)
        # amr_cross_video layer
        amr_cross_video_layer = refdecoder['amr_cross_video_layer']
        assert amr_cross_video_layer.pop('name') == 'cross_attention'
        self.decoder_amr_who_cross = amr_cross_video_layer.pop('amr_cross')
        self.decoder_amr_cross_video_layers = _get_clones(CrossAttentionLayer(**amr_cross_video_layer),
                                                                   self.decoder_nlayers)
        self.decoder_nheads = amr_cross_video_layer['nhead']
        # amr self layer
        amr_self_layer = refdecoder['amr_self_layer']
        amr_self_layer_name = amr_self_layer.pop('name')
        from .layer_graph import graphLayer_entrypoint
        create_graph_layer = graphLayer_entrypoint(amr_self_layer_name)
        graph_layer = create_graph_layer(amr_self_layer)
        self.decoder_amr_self_layers = _get_clones(graph_layer, self.decoder_nlayers)
        ffn_layer = refdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                        self.decoder_nlayers)
        # norm, mask out, box, mask
        self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3)
        self.decoder_mask_out_stride = refdecoder['mask_out_stride'] 
        self.decoder_mask_threshold = refdecoder['mask_threshold']
        self.decoder_choose_who = refdecoder['choose_who']

    
    def init_parameters(self,): 
        for proj in self.video_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
    
    def encode_video(self, samples):
        bb_out = self.video_swint(samples)  
        nf, batch_size, *_ = bb_out[0].tensors.shape
        orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src.clone())
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(orig_pad_mask.float(),
                                             size=src_proj_l.shape[-2:],mode='nearest') > 0.5
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses
    
    def encode_text(self, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        return amrs, amr_token_feats, amr_token_seg_ids
    
    def get_fusion_amr_cross(self, amr_token_feats, amr_token_seg_ids):
        # b (V+E)max c, b (V+E)max
        if self.fusion_amr_who_cross == '只有2/3':
            who_fuse_with_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
        elif self.fusion_amr_who_cross == '所有':
            who_fuse_with_video =  (amr_token_seg_ids!=0)
        else:
            raise ValueError()
        amr_fusion_tokens = [bt_feat[who_cross] for bt_feat, who_cross in zip(amr_token_feats, who_fuse_with_video)]
        amr_fusion_tokens, amr_fusion_pad_masks = pad_1d_feats(amr_fusion_tokens)
        return amr_fusion_tokens.permute(1, 0, 2), amr_fusion_pad_masks  

    def forward_refdecoder_heads(self, output, mask_features, attn_mask_target_size=None):
        decoder_output = self.decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_box = self.decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        if attn_mask_target_size is None:
            return outputs_mask, outputs_box, None
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_mask, outputs_box, attn_mask

    def get_refdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features
    
    def get_refdecoder_amr_cross(self, seg_ids, layer_idx):
        if self.decoder_amr_who_cross[layer_idx] == '只有2/3':
            return torch.logical_or(seg_ids==2, seg_ids==3)
        elif self.decoder_amr_who_cross[layer_idx] == '所有':
            return seg_ids!=0
        pass   
    
    def build_multimodal_features_along_edge(self, memory, memory_pos, memory_pad_mask, num_edges_by_bt):
        """
        memory: hw bt c
        num_edges_by_bt: list[int]
        """
        memory_by_edge = []
        memory_pos_by_edge = []
        memory_pad_mask_by_edge = []
        for bt_memory,  bt_memory_pos, bt_memory_pad_mask, num_edges in zip(memory.permute(1,0,2), memory_pos.permute(1, 0, 2), memory_pad_mask, num_edges_by_bt):
            memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            memory_pos_by_edge.append(repeat(bt_memory_pos, 'hw c -> E hw c', E=num_edges))
            memory_pad_mask_by_edge.append(repeat(bt_memory_pad_mask, 'hw -> E hw', E=num_edges))
        return {'video_mem': torch.cat(memory_by_edge, dim=0),
                'video_mem_pos': torch.cat(memory_pos_by_edge, dim=0),
                'video_mem_pad_mask': torch.cat(memory_pad_mask_by_edge, dim=0)} # btE hw
    
    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer
        
        # 准备decoder的东西  thw b c
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.decoder_nlayers):
            scale_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories[scale_index], # hw bt c
                memory_mask=attn_mask,  # bt*head (V+E)max hw
                memory_key_padding_mask=memories_pad_masks[scale_index], 
                pos=memories_poses[scale_index],  # thw b c
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            # output = output2 * (1 - amr_who_cross_video.permute(1,0).float().unsqueeze(-1)) +\
            #     output * (amr_who_cross_video.permute(1,0).float().unsqueeze(-1))
            graph_self_memory = self.build_multimodal_features_along_edge(memories[scale_index].clone(), 
                                                                          memories_poses[scale_index].clone(),
                                                                          memories_pad_masks[scale_index].clone(),
                                                                          num_edges_by_batch)
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, 
                                                                         attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        # choose who
        
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize } 

    def get_decoder_preds(self, model_outs):
        refseg_src = model_outs['refdecoder_refseg']
        if self.decoder_choose_who == '第一个':
            for i in range(-1, self.decoder_nlayers):
                layer_pred = refseg_src[f'layer{i}_preds']
                refseg_src[f'layer{i}_preds']['pred_mask_logits'] = layer_pred['pred_mask_logits'][:, 0]
                refseg_src[f'layer{i}_preds']['pred_box_logits'] = layer_pred['pred_box_logits'][:, 0]
                if 'queries' in layer_pred:
                    refseg_src[f'layer{i}_preds']['queries'] = layer_pred['queries'][0] # bt c
        return refseg_src        
    
    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        decoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
        last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        out_masks_logits =  last_layer_preds['pred_mask_logits'] 
        # bt 1 h w
        query_pred_masks = F.interpolate(out_masks_logits.unsqueeze(1), 
                                         scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
    
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        refseg_src = self.get_decoder_preds(model_outs)
        # bT n H/4 W/4 -> bT H/4 W/4 
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
            
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    # task loss
    def refdecoder_refseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        loss_weight = self.loss_weight
        if 'mask_loss_type' in self.tasks['refdecoder_refseg']:
            mask_loss_type = self.tasks['refdecoder_refseg']['mask_loss_type']
        else:
            mask_loss_type = 'ce'
        
        # list[t] -> bt
        target_valid = torch.cat([t["valid"][t['referent_idx']] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        

        loss_value = {'refdecoder_mask': torch.tensor(0, device=target_valid.device).float(), 
                      'refdecoder_bbox': torch.tensor(0, device=target_valid.device).float(), 
                      'refdecoder_giou': torch.tensor(0, device=target_valid.device).float(),
                      'refdecoder_dice': torch.tensor(0, device=target_valid.device).float(), }

        for i in range(-1, self.decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT H W
                if loss_weight['refdecoder_mask'] != 0 or loss_weight['refdecoder_dice'] !=0:
                    masks_losses = AMR_v0.refdecoder_masks_loss(layer_pred, targets, num_boxes, self.decoder_mask_out_stride, mask_loss_type)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['refdecoder_bbox'] != 0 or loss_weight['refdecoder_giou'] !=0:
                    boxes_losses = AMR_v0.refdecoder_boxes_loss(layer_pred, targets, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
        return loss_value         
 
    @staticmethod
    def refdecoder_boxes_loss(outputs, targets,num_boxes): 
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        src_boxes = outputs['pred_box_logits'].sigmoid()  # bt 4
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0).to(src_boxes)  
        
        src_boxes = src_boxes[is_consistent]  # bt 4
        target_boxes = target_boxes[is_consistent] # bt 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['refdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['refdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    @staticmethod
    def refdecoder_masks_loss(outputs, targets, num_boxes, decoder_mask_out_stride, mask_loss_type):
        # list[n t] -> list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        
        src_masks = outputs["pred_mask_logits"]  # bT h w  )
        
        # list[n t h w] -> list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0).to(src_masks) # list[t h w] -> bt h w

        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        
        src_masks = src_masks[is_consistent].flatten(1) # bt hw
        target_masks = target_masks[is_consistent].flatten(1) # bt hw
        
        if mask_loss_type == 'ce':
            mask_loss = ce_mask_loss(src_masks, target_masks, num_boxes)
        elif mask_loss_type == 'focal':
            mask_loss = sigmoid_focal_loss(src_masks, target_masks, num_boxes)
        losses = {
            "refdecoder_mask": mask_loss,
            "refdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses 
    
class AMR_v0_detectObj(AMR_v0):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                # object_classes=[4194, 1928, 1011, 5103, 512, 4758, 2335],  # Ġadult, Ġbaby, Ġball, Ġbird, Ġcar, Ġcat, Ġdog
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'split_obj_ref_deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'obj_seg_nlayers':3,
                    'ref_seg_nlayers':3
                    },
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                    
                objdecoder={ 
                    'num_classes': 7,
                    'nqueries': 100,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },) -> None:
        parsing_encoder_name = parsing_encoder['name']
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)
        # self.obj_decoder_class_tokens = object_classes
        if parsing_encoder_name == 'deform_video_2d_fpn':
            self.obj_parsing_encoder = self.deform_multiscale_2dencoder
            self.ref_parsing_encoder = None
        elif parsing_encoder_name == 'split_obj_ref_deform_video_2d_fpn':
            self.obj_parsing_encoder = self.deform_multiscale_2dencoder[0]
            self.ref_parsing_encoder = self.deform_multiscale_2dencoder[1]
        elif parsing_encoder_name == 'fpn2d':
            self.obj_parsing_encoder = self.deform_multiscale_2dencoder
            self.ref_parsing_encoder = None
        else:
            raise ValueError()
        # obj decoder
        self.obj_decoder_query_embed = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_query_feats = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_used_scales = objdecoder['used_scales']
        self.obj_decoder_conved_scale = objdecoder['conved_scale']
        self.obj_decoder_nlayers = objdecoder['nlayers']
        self.obj_decoder_nqueries = objdecoder['nqueries']
        self.obj_decoder_level_embed = nn.Embedding(len(self.obj_decoder_used_scales), d_model)
        cross_layer = objdecoder['cross_layer']
        assert cross_layer.pop('name') == 'cross_attention'
        self.obj_decoder_cross_video_layers = _get_clones(CrossAttentionLayer(**cross_layer),
                                                                   self.obj_decoder_nlayers)
        self.obj_decoder_nheads = cross_layer['nhead']
        self_layer = objdecoder['self_layer']
        assert self_layer.pop('name') == 'self_attention'
        self.obj_decoder_self_layers = _get_clones(SelfAttentionLayer(**self_layer),
                                                            self.obj_decoder_nlayers)  
        ffn_layer = objdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.obj_decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                            self.obj_decoder_nlayers) 
        # norm, mask out, box, cls, mask
        self.obj_decoder_class_embed = nn.Linear(d_model, objdecoder['num_classes']+1)
        self.obj_decoder_nclasses = objdecoder['num_classes'] + 1
        self.obj_decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.obj_decoder_mask_out_stride = objdecoder['mask_out_stride']
        self.obj_decoder_mask_threshold = objdecoder['mask_threshold'] 

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_classes, outputs_mask, outputs_box, attn_mask
    
    def get_objdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.obj_decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.obj_decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features

    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries':output.clone() }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':output.clone() }
        # bt n
        output_mask = torch.zeros_like(output.detach())[..., 0].permute(1,0).bool()
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), output_mask, decoder_layer_preds # n bt c

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        if self.ref_parsing_encoder is not None:
            multiscales, _, _\
                = self.ref_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)          
        
        # 准备decoder的东西  thw b c
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries': amr_token_feats.clone() }
        for i in range(self.decoder_nlayers):
            scale_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=torch.cat([memories[scale_index], obj_queries], dim=0), # hw bt c
                memory_mask=F.pad(attn_mask.float(), pad=(0, len(obj_queries))).bool(),  # bt*head (V+E)max hw
                memory_key_padding_mask=torch.cat([memories_pad_masks[scale_index], obj_queries_mask], dim=1), 
                pos=torch.cat([memories_poses[scale_index], torch.zeros_like(obj_queries)],dim=0),  # thw b c
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            # output = output2 * (1 - amr_who_cross_video.permute(1,0).float().unsqueeze(-1)) +\
            #     output * (amr_who_cross_video.permute(1,0).float().unsqueeze(-1))
            graph_self_memory = self.build_multimodal_features_along_edge(torch.cat([memories[scale_index], obj_queries], dim=0), 
                                                                          torch.cat([memories_poses[scale_index], torch.zeros_like(obj_queries)],dim=0),
                                                                          torch.cat([memories_pad_masks[scale_index], obj_queries_mask], dim=1),
                                                                          num_edges_by_batch)
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, 
                                                                         attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries': amr_token_feats.clone() }
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        # choose who
        
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

    def obj_decoder_targets_handler(self, targets):
        # list[n h w], bt
        # list[n t h w] -> list[n h w], bt
        batch_size = len(targets)
        target_masks = []
        for bth_idx in range(batch_size):
            # n t h w
            t_m = targets[bth_idx]["masks"].split(1, dim=1) # list[n 1 h w], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n h w]
            target_masks.extend(t_m)
            
        for idx in range(len(target_masks)):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = target_masks[idx].shape[-2:]
            target_masks[idx] = target_masks[idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert target_masks[idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert target_masks[idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        
        # list[n 4], bt
        target_boxes = []
        for bth_idx in range(batch_size):
            # n t 4
            t_m = targets[bth_idx]["boxes"].split(1, dim=1) # list[n 1 4], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n 4], t
            target_boxes.extend(t_m)

        # list[n], bt
        target_classes = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_labels = targets[bth_idx]['class_labels'].unsqueeze(-1).repeat(1, bth_valids.shape[1]) # n t
            bth_labels = torch.where(bth_valids, bth_labels, self.obj_decoder_nclasses-1)
            t_m = bth_labels.split(1, dim=1) # list[n 1], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n], t
            target_classes.extend(t_m)    

        # list[n], bt
        is_valid = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].split(1, dim=1) # n t -> list[n 1], t
            bth_valids = [bv.squeeze(1) for bv in bth_valids] # list[n], t
            is_valid.extend(bth_valids)
        referent_idxs = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_refidx = [targets[bth_idx]['referent_idx']] * bth_valids.shape[1] # list[int], t
            referent_idxs.extend(bth_refidx)        
        return {
            'masks': target_masks,
            'boxes': target_boxes,
            'class_labels': target_classes,
            'is_valid': is_valid,
            'referent_idx': referent_idxs
        }
                        
        
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        refseg_src = self.get_decoder_preds(model_outs)
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict.update(self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets)[0])  
                  
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 


    # task loss
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n h w], bt
        tgt_masks = targets['masks']
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_class': torch.tensor(0, device=tgt_masks[0].device).float(),}
        matching_indices_by_layer = []
        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = AMR_v0_detectObj.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.decoder_mask_out_stride)
                matching_indices_by_layer.append(layer_matching_indices)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detectObj.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detectObj.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detectObj.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]
        return loss_value,matching_indices_by_layer       

    @staticmethod
    def obj_decoder_class_loss(outputs, targets, indices, class_weight):
        """
        indices: [[], []], bt
        """

        src_logits = outputs["pred_class_logits"] # bt nq c

        # list[n], bt
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['class_labels'], indices)]) # btn_sigma
    
        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], len(class_weight)-1, dtype=torch.int64, device=src_logits.device
        ) # bt n
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.tensor(class_weight).to(src_logits))
        losses = {"objdecoder_class": loss_ce}
        return losses

    
    @staticmethod
    def obj_decoder_boxes_loss(outputs, targets, indices, num_boxes): 
        src_idx = get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_box_logits'].sigmoid()[src_idx]  # bt nq 4 -> btn_sigma 4
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n 4], bt -> btn_sigma 4
        target_boxes = torch.cat([t[J] for t, (_, J) in zip(targets['boxes'], indices)]).to(src_boxes)
            

        src_boxes = src_boxes[is_consistent]  # btn_sigma 4
        target_boxes = target_boxes[is_consistent] # btn_sigma 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['objdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['objdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    

    @staticmethod
    def obj_decoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        src_idx = get_src_permutation_idx(indices)
        src_masks = outputs["pred_mask_logits"][src_idx]  # bt nq h w -> btn_sigma h w
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n h w], bt -> btn_sigma h w
        target_masks = torch.cat([t[J] for t, (_, J) in zip(targets['masks'], indices)]).to(src_masks)
        
        
        src_masks = src_masks[is_consistent].flatten(1) # btn_sigma hw
        target_masks = target_masks[is_consistent].flatten(1) # btn_sigma hw
        
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "objdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    @staticmethod
    @torch.no_grad()
    def obj_decoder_matching(outputs, targets, matching_costs, class_weight, decoder_mask_out_stride):
        src_class_prob = outputs["pred_class_logits"].softmax(dim=-1) # bt n c
        src_boxes = outputs["pred_box_logits"].sigmoid()   # bt n 4
        src_masks_logits = outputs["pred_mask_logits"]  # bt n h w
        bt, nq, h, w = src_masks_logits.shape 
        
        target_boxes = targets['boxes'] # [n 4], bt
        target_masks = targets['masks'] # n h w, bt
        target_classes = targets['class_labels'] # n, bt

        indices = [] 
        for i in range(bt):
            out_prob = src_class_prob[i] # nq c
            out_bbox = src_boxes[i]  # nq 4
            out_mask = src_masks_logits[i]  # nq h w

            tgt_bbox = target_boxes[i].to(out_bbox)# n 4
            tgt_mask = target_masks[i].to(out_mask)# n h w
            tgt_cls = target_classes[i] # n

            cost_class = -out_prob[:, tgt_cls] # nq n

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 4
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = matching_costs['class'] * cost_class +\
                matching_costs['bbox'] * cost_bbox + \
                matching_costs['giou'] * cost_giou + \
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

class AMR_v0_detectObj_onlyObj(AMR_v0_detectObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        if self.ref_parsing_encoder is not None:
            multiscales, _, _\
                = self.ref_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)          
        
        memories = obj_queries # nq bt c  
        memories_pos = query_embed       
        
        # 准备decoder的东西  thw b c
        *_, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=None)
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box}
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=memories_pos,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features,attn_mask_target_size=None)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 
from .layer_graph import batching_graph
import networkx as nx
class AMR_v0_detOnlyObj_Grounding(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                objdecoder={ 
                    'num_classes': 7,
                    'nqueries': 100,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                is_pretraining_seg=False,
                detach_refdecoder_memory=False,
                freeze_obj_decoder=False,
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        # video encoder
        from .video_swin import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=os.path.join(pt_dir, swint_pretrained_path),
                                                running_mode=swint_runnning_mode)
        if swint_freeze:
            for p in self.video_swint.parameters():
                p.requires_grad_(False) 
                 
        assert len(video_projs) == len(video_feat_scales)
        self.video_feat_scales = video_feat_scales
        backbone_channels, backbone_scales = self.video_swint.get_desc()
        assert len(backbone_channels) == len(backbone_scales)
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        # amr encoder
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
        
        self.fusion_amr_who_cross = fusion.pop('amr_cross', None)
        fusion_name = fusion.pop('name')
        self.cross_product = get_fusion(fusion_name, fusion)

        self.obj_parsing_encoder = get_parsing_encoder(parsing_encoder.pop('name'),
                                                               parsing_encoder)
        self.build_obj_decoder(objdecoder, d_model)
        self.build_ref_decoder(refdecoder)
        self.is_pretraining_seg = is_pretraining_seg
        self.detach_refdecoder_memory = detach_refdecoder_memory
        
        if freeze_obj_decoder:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    if ('obj_decoder' in n) or ('video_proj' in n) or ('obj_parsing_encoder' in n) or ('cross_product' in n):
                        p.requires_grad_(False)

    def build_obj_decoder(self, objdecoder, d_model):
        # obj decoder
        self.obj_decoder_query_embed = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_query_feats = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_used_scales = objdecoder['used_scales']
        self.obj_decoder_conved_scale = objdecoder['conved_scale']
        self.obj_decoder_nlayers = objdecoder['nlayers']
        self.obj_decoder_nqueries = objdecoder['nqueries']
        self.obj_decoder_level_embed = nn.Embedding(len(self.obj_decoder_used_scales), d_model)
        cross_layer = objdecoder['cross_layer']
        assert cross_layer.pop('name') == 'cross_attention'
        self.obj_decoder_cross_video_layers = _get_clones(CrossAttentionLayer(**cross_layer),
                                                                   self.obj_decoder_nlayers)
        self.obj_decoder_nheads = cross_layer['nhead']
        self_layer = objdecoder['self_layer']
        assert self_layer.pop('name') == 'self_attention'
        self.obj_decoder_self_layers = _get_clones(SelfAttentionLayer(**self_layer),
                                                            self.obj_decoder_nlayers)  
        ffn_layer = objdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.obj_decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                            self.obj_decoder_nlayers) 
        # norm, mask out, box, cls, mask
        self.obj_decoder_class_embed = nn.Linear(d_model, objdecoder['num_classes']+1)
        self.obj_decoder_nclasses = objdecoder['num_classes'] + 1
        self.obj_decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.obj_decoder_mask_out_stride = objdecoder['mask_out_stride']
        self.obj_decoder_mask_threshold = objdecoder['mask_threshold'] 

    def build_ref_decoder(self, refdecoder,):
        from .layer_graph import graphLayer_entrypoint
        reason_layer = refdecoder['reason_layer']
        reason_layer_name = reason_layer.pop('name')
        self.decoder_reason_layer_nheads = reason_layer['nheads']
        self.decoder_reason_layer_choose_who = reason_layer['choose_who']
        create_reason_layer = graphLayer_entrypoint(reason_layer_name)
        self.decoder_reason_layer = create_reason_layer(reason_layer)

        trans_layer = refdecoder.pop('trans_layer', None)
        if trans_layer is None:
            self.decoder_trans_nlayers = 0
            self.decoder_trans_layers = None
        else:
            trans_layer_name = trans_layer.pop('name')
            if trans_layer_name == 'none':
                self.decoder_trans_nlayers = 0
                self.decoder_trans_layers = None
            else:
                create_layer = graphLayer_entrypoint(trans_layer_name)
                graph_layer = create_layer(trans_layer)
                self.decoder_trans_nlayers = trans_layer['nlayers']
                self.decoder_trans_layers = _get_clones(graph_layer, self.decoder_trans_nlayers)

    def encode_video(self, samples):
        bb_out = self.video_swint(samples)  
        nf, batch_size, *_ = bb_out[0].tensors.shape
        orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src.clone())
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(orig_pad_mask.float(),
                                             size=src_proj_l.shape[-2:],mode='nearest') > 0.5
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses
    
    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def get_fusion_amr_cross(self, amr_token_feats, amr_token_seg_ids):
        # b (V+E)max c, b (V+E)max
        if self.fusion_amr_who_cross == '只有2/3':
            who_fuse_with_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
        elif self.fusion_amr_who_cross == '所有':
            who_fuse_with_video =  (amr_token_seg_ids!=0)
        else:
            raise ValueError()
        amr_fusion_tokens = [bt_feat[who_cross] for bt_feat, who_cross in zip(amr_token_feats, who_fuse_with_video)]
        amr_fusion_tokens, amr_fusion_pad_masks = pad_1d_feats(amr_fusion_tokens)
        return amr_fusion_tokens.permute(1, 0, 2), amr_fusion_pad_masks  

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_classes, outputs_mask, outputs_box, attn_mask
 
    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries':output.clone() }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=None, # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':output.clone() }
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), query_embed, decoder_layer_preds # n bt c

    def get_objdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.obj_decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.obj_decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b smax c  
        if self.cross_product is not None:
            fusion_mem = torch.cat([text_feats, amr_token_feats], dim=1) 
            fusion_mem_pad_mask = torch.cat([text_pad_masks, amr_token_seg_ids==0], dim=-1)
            fusion_mem_pos = torch.cat([text_pos, torch.zeros_like(amr_token_feats)], dim=1)   
            for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
                bs, nf, _, h, w = feat.shape
                feat = rearrange(feat, 'b t c h w -> (t h w) b c')
                poses = rearrange(poses, 'b t c h w -> (t h w) b c')
                feat, attn_weight = self.cross_product(tgt=feat,
                                                        memory=fusion_mem.permute(1,0,2), 
                                                        memory_key_padding_mask=fusion_mem_pad_mask,
                                                        pos=fusion_mem_pos.permute(1,0,2), 
                                                        query_pos=poses)
                check_visualize[f'scale{lvl} attention weights'] = attn_weight
                multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann]
        text_feats = repeat(text_feats, 'b s c -> (b t) s c',t=nf)[perFrame_has_ann]
        # list[list[int], vi], batch
        # batch -> bt
        repeated_node_alignments = [] 
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_node_alignments.append(copy.deepcopy(node_alignments[idx]))
        filtered_rnas = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_rnas.append(repeated_node_alignments[idx])
        assert len(filtered_rnas) != 0
        node_alignments = filtered_rnas
        
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _ = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, query_embed, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                        [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                        [scale_pos.clone() for scale_pos in multiscales_poses])
        if self.is_pretraining_seg:
            return {'objdecoder_objseg': objdecoder_layer_preds}
        if self.detach_refdecoder_memory:
            memories = obj_queries.detach() # nq bt c
            memories_pos = query_embed.detach() # nq bt c
        else:
            memories = obj_queries
            memories_pos = query_embed
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index,  node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), memories_pos.permute(1,0,2).clone(),
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) # V nq
        g_score_by_batch = []
        for bch_idx in range(bt):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        decoder_layer_preds[f'layer{-1}_preds'] = {'grounding_score': g_score_by_batch}

        if self.decoder_trans_layers is not None:
            for layer_idx in range(self.decoder_trans_nlayers):
                node_feats, edge_feats = self.decoder_trans_layers[layer_idx](node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                                            node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                                            node_feats=node_feats, edge_feats=edge_feats,
                                                                            node_memories=node_memories, edge_memories=edge_memories,
                                                                            edge_index=edge_index,
                                                                            node_subseqs=node_subseqs) # V nq
                grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                            node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                            node_feats=node_feats, edge_feats=edge_feats,
                                            node_memories=node_memories, edge_memories=edge_memories,
                                            edge_index=edge_index,
                                            node_subseqs=node_subseqs,
                                            node_dsends=node_dsends) # V nq
                g_score_by_batch = []
                for bch_idx in range(bt):
                    bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
                    g_score_by_batch.append(bch_node_score) # vi nq
                decoder_layer_preds[f'layer{layer_idx}_preds'] = {'grounding_score': g_score_by_batch}

        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 


    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        objseg_preds = decoder_layer_preds['objdecoder_objseg']
        obj_last_layer_preds = objseg_preds[f'layer{self.obj_decoder_nlayers-1}_preds']

        if self.is_pretraining_seg:
            out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            obj_decoder_targets = self.obj_decoder_targets_handler(targets)
            _, matching_result = self.obj_decoder_objseg_loss(objseg_preds, obj_decoder_targets)
            matching_result = matching_result[-1] # list(tgt, src), bt
            gt_referent_idx = obj_decoder_targets['referent_idx'] # list[int], bt
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]] 
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx) in zip(out_mask_logits, gt_referent_idx, matching_result)], dim=0)
        else:
            out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w

            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # bt nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }


    def get_decoder_preds(self, model_outs):
        refseg_src = model_outs['refdecoder_refseg']
        if self.decoder_reason_layer_choose_who == '第一个':
            for i in range(-1, self.decoder_trans_nlayers):
                # list[vi nq], b -> b nq
                layer_gscore = refseg_src[f'layer{i}_preds']['grounding_score']
                layer_gscore = torch.stack([lg[0] for lg in layer_gscore], dim=0)
                refseg_src[f'layer{i}_preds']['grounding_score'] = layer_gscore
        return refseg_src  

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict, matching_result = self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets)

        if not self.is_pretraining_seg:
            refseg_src = self.get_decoder_preds(model_outs)
            loss_value_dict.update(self.ref_choose_loss(refseg_src, obj_decoder_targets, matching_result[-1]))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
   
    def ref_choose_loss(self, refseg_src, targets, decoder_last_layer_matching_results):
        """
        Args:
            refseg_src: dict{layer-1pred: {queries: bt c}}
            targets (_type_): batch[dict{'masks', 'referent_idx'}]
            matching_result_by_layer: list[(tgt_idx, src_idx), bt]
        """
        tgt_masks = targets['masks']
        referent_idx = targets['referent_idx'] # list[int], bt
        device=tgt_masks[0].device 
        num_boxes = sum([t[refidx].flatten().any().int() for t, refidx in zip(tgt_masks,referent_idx)])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        is_valid = targets['is_valid'] # list[ni], bt

        ref_is_valid = torch.tensor([isva[ridx] for isva, ridx in zip(is_valid, referent_idx)]).bool() # bt

        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  decoder_last_layer_matching_results): # bt
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(device) # bt
        match_as_gt_indices = match_as_gt_indices[ref_is_valid]

        refdecoder_choose_loss = 0.
        for layer_idx in range(-1, self.decoder_trans_nlayers):
            layer_weight = layer_weights[layer_idx] 
            if layer_weight != 0: # bt c
                refdecoder_gscore = refseg_src[f'layer{layer_idx}_preds']['grounding_score'] # bt nq
                refdecoder_gscore = refdecoder_gscore[ref_is_valid]
                choose_loss = F.cross_entropy(refdecoder_gscore, match_as_gt_indices, reduction='none') # bt
                choose_loss = choose_loss.sum() / num_boxes
                refdecoder_choose_loss += (choose_loss * layer_weight)
        return {'refdecoder_choose': refdecoder_choose_loss}

    def obj_decoder_targets_handler(self, targets):
        # list[n h w], bt
        # list[n t h w] -> list[n h w], bt
        batch_size = len(targets)
        target_masks = []
        for bth_idx in range(batch_size):
            # n t h w
            t_m = targets[bth_idx]["masks"].split(1, dim=1) # list[n 1 h w], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n h w]
            target_masks.extend(t_m)
            
        for idx in range(len(target_masks)):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = target_masks[idx].shape[-2:]
            target_masks[idx] = target_masks[idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert target_masks[idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert target_masks[idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        
        # list[n 4], bt
        target_boxes = []
        for bth_idx in range(batch_size):
            # n t 4
            t_m = targets[bth_idx]["boxes"].split(1, dim=1) # list[n 1 4], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n 4], t
            target_boxes.extend(t_m)

        # list[n], bt
        target_classes = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_labels = targets[bth_idx]['class_labels'].unsqueeze(-1).repeat(1, bth_valids.shape[1]) # n t
            bth_labels = torch.where(bth_valids, bth_labels, self.obj_decoder_nclasses-1)
            t_m = bth_labels.split(1, dim=1) # list[n 1], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n], t
            target_classes.extend(t_m)    

        # list[n], bt
        is_valid = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].split(1, dim=1) # n t -> list[n 1], t
            bth_valids = [bv.squeeze(1) for bv in bth_valids] # list[n], t
            is_valid.extend(bth_valids)
        referent_idxs = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_refidx = [targets[bth_idx]['referent_idx']] * bth_valids.shape[1] # list[int], t
            referent_idxs.extend(bth_refidx)        
        return {
            'masks': target_masks,
            'boxes': target_boxes,
            'class_labels': target_classes,
            'is_valid': is_valid,
            'referent_idx': referent_idxs
        }

    # task loss
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n h w], bt
        tgt_masks = targets['masks']
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_class': torch.tensor(0, device=tgt_masks[0].device).float(),}
        matching_indices_by_layer = []
        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = AMR_v0_detOnlyObj_Grounding.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.obj_decoder_mask_out_stride)
                matching_indices_by_layer.append(layer_matching_indices)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.obj_decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]
        return loss_value,matching_indices_by_layer       

    @staticmethod
    def obj_decoder_class_loss(outputs, targets, indices, class_weight):
        """
        indices: [[], []], bt
        """

        src_logits = outputs["pred_class_logits"] # bt nq c

        # list[n], bt
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['class_labels'], indices)]) # btn_sigma
    
        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], len(class_weight)-1, dtype=torch.int64, device=src_logits.device
        ) # bt n
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.tensor(class_weight).to(src_logits))
        losses = {"objdecoder_class": loss_ce}
        return losses

    
    @staticmethod
    def obj_decoder_boxes_loss(outputs, targets, indices, num_boxes): 
        src_idx = get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_box_logits'].sigmoid()[src_idx]  # bt nq 4 -> btn_sigma 4
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n 4], bt -> btn_sigma 4
        target_boxes = torch.cat([t[J] for t, (_, J) in zip(targets['boxes'], indices)]).to(src_boxes)
            

        src_boxes = src_boxes[is_consistent]  # btn_sigma 4
        target_boxes = target_boxes[is_consistent] # btn_sigma 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['objdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['objdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    

    @staticmethod
    def obj_decoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        src_idx = get_src_permutation_idx(indices)
        src_masks = outputs["pred_mask_logits"][src_idx]  # bt nq h w -> btn_sigma h w
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n h w], bt -> btn_sigma h w
        target_masks = torch.cat([t[J] for t, (_, J) in zip(targets['masks'], indices)]).to(src_masks)
        
        
        src_masks = src_masks[is_consistent].flatten(1) # btn_sigma hw
        target_masks = target_masks[is_consistent].flatten(1) # btn_sigma hw
        
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "objdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    @staticmethod
    @torch.no_grad()
    def obj_decoder_matching(outputs, targets, matching_costs, class_weight, decoder_mask_out_stride):
        src_class_prob = outputs["pred_class_logits"].softmax(dim=-1) # bt n c
        src_boxes = outputs["pred_box_logits"].sigmoid()   # bt n 4
        src_masks_logits = outputs["pred_mask_logits"]  # bt n h w
        bt, nq, h, w = src_masks_logits.shape 
        
        target_boxes = targets['boxes'] # [n 4], bt
        target_masks = targets['masks'] # n h w, bt
        target_classes = targets['class_labels'] # n, bt

        indices = [] 
        for i in range(bt):
            out_prob = src_class_prob[i] # nq c
            out_bbox = src_boxes[i]  # nq 4
            out_mask = src_masks_logits[i]  # nq h w

            tgt_bbox = target_boxes[i].to(out_bbox)# n 4
            tgt_mask = target_masks[i].to(out_mask)# n h w
            tgt_cls = target_classes[i] # n

            cost_class = -out_prob[:, tgt_cls] # nq n

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 4
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = matching_costs['class'] * cost_class +\
                matching_costs['bbox'] * cost_bbox + \
                matching_costs['giou'] * cost_giou + \
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

from torch_geometric.nn.inits import glorot
from .layer_fusion import fusion_entrypoint
class AMR_v0_detOnlyObj_Grounding_ptObjDet(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # obj decoder
                 obj_decoder = {
                     'name':None,
                     'path': None,
                     'freeze': True,
                 },
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                is_pretraining_seg=False,
                multiclass_choose=False,
                choose_threshold=None
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.is_pretraining_seg = is_pretraining_seg

        create_obj_decoder = pt_obj_decoder_entrypoint(obj_decoder['name'])
        self.obj_decoder = create_obj_decoder(obj_decoder, pt_dir)
        self.obj_query_proj = nn.Linear(self.obj_decoder.out_dim, d_model)
        
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        
        create_fusion_module = fusion_entrypoint(fusion.pop('name'))
        self.fusion_module = create_fusion_module(fusion)

        self.build_ref_decoder(refdecoder)
        torch.nn.init.uniform_(self.obj_query_proj.weight)
        torch.nn.init.zeros_(self.obj_query_proj.bias)
        torch.nn.init.uniform_(self.amrtext_wordEmbedding_proj.fc.weight)
        torch.nn.init.zeros_(self.amrtext_wordEmbedding_proj.fc.bias)
        self.obj_decoder_mask_out_stride = 4
        self.obj_decoder_mask_threshold = 0.5
        self.multiclass_choose = multiclass_choose
        self.choose_threshold=choose_threshold

    def build_ref_decoder(self, refdecoder,):
        from .layer_graph import graphLayer_entrypoint
        reason_layer = refdecoder['reason_layer']
        reason_layer_name = reason_layer.pop('name')
        self.decoder_reason_layer_nheads = reason_layer['nheads']
        self.decoder_reason_layer_choose_who = reason_layer['choose_who']
        create_reason_layer = graphLayer_entrypoint(reason_layer_name)
        self.decoder_reason_layer = create_reason_layer(reason_layer)

        trans_layer = refdecoder.pop('trans_layer', None)
        if trans_layer is None:
            self.decoder_trans_nlayers = 0
            self.decoder_trans_layers = None
        else:
            trans_layer_name = trans_layer.pop('name')
            if trans_layer_name == 'none':
                self.decoder_trans_nlayers = 0
                self.decoder_trans_layers = None
            else:
                create_layer = graphLayer_entrypoint(trans_layer_name)
                graph_layer = create_layer(trans_layer)
                self.decoder_trans_nlayers = trans_layer['nlayers']
                self.decoder_trans_layers = _get_clones(graph_layer, self.decoder_trans_nlayers)
 
    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def get_decoder_preds(self, model_outs):
        refseg_src = model_outs['refdecoder_refseg']
        if self.decoder_reason_layer_choose_who == '第一个':
            for i in range(-1, self.decoder_trans_nlayers):
                # list[vi nq], b -> b nq
                layer_gscore = refseg_src[f'layer{i}_preds']['grounding_score']
                layer_gscore = torch.stack([lg[0] for lg in layer_gscore], dim=0)
                refseg_src[f'layer{i}_preds']['grounding_score'] = layer_gscore
        return refseg_src  

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device) 

        obj_decoder_output = self.obj_decoder(samples, 
                                            text_feats=text_feats, 
                                            text_pad_masks=text_pad_masks,
                                            amr_feats=amr_token_feats,
                                            amr_pad_masks=amr_token_seg_ids==0)
        # b nq c, b t nq h w
        obj_queries, pred_masks = obj_decoder_output['obj_queries'], obj_decoder_output['pred_masks']
        obj_queries = self.obj_query_proj(obj_queries)
        if self.is_pretraining_seg:
            return {'objdecoder_objseg': pred_masks}
        # list[Graph], b (V+E)max c, b (V+E)max 

        obj_queries, amr_token_feats, text_feats = self.fusion_module(query_feat=obj_queries, 
                                                                    text_feats=text_feats,
                                                                    amr_feats=amr_token_feats,
                                                                    amr_pad_masks = amr_token_seg_ids==0,
                                                                    text_pad_masks=text_pad_masks)
        memories = obj_queries.permute(1,0,2)
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index,  node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), None,
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) # V nq
        g_score_by_batch = []
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        decoder_layer_preds[f'layer{-1}_preds'] = {'grounding_score': g_score_by_batch}

        if self.decoder_trans_layers is not None:
            for layer_idx in range(self.decoder_trans_nlayers):
                node_feats, edge_feats = self.decoder_trans_layers[layer_idx](node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                                            node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                                            node_feats=node_feats, edge_feats=edge_feats,
                                                                            node_memories=node_memories, edge_memories=edge_memories,
                                                                            edge_index=edge_index,
                                                                            node_subseqs=node_subseqs) # V nq
                grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                            node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                            node_feats=node_feats, edge_feats=edge_feats,
                                            node_memories=node_memories, edge_memories=edge_memories,
                                            edge_index=edge_index,
                                            node_subseqs=node_subseqs,
                                            node_dsends=node_dsends) # V nq
                g_score_by_batch = []
                for bch_idx in range(bt):
                    bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
                    g_score_by_batch.append(bch_node_score) # vi nq
                decoder_layer_preds[f'layer{layer_idx}_preds'] = {'grounding_score': g_score_by_batch}

        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': pred_masks} 


    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
        # b nq T h w
        out_mask_logits = decoder_layer_preds['objdecoder_objseg'].permute(0,2,1,3,4)
        if self.is_pretraining_seg:
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            # list[n t h w]
            tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
            for btc_idx in range(batch_size):
                start = int(self.obj_decoder_mask_out_stride // 2)
                im_h, im_w = tgt_masks[btc_idx].shape[-2:]
                tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
                assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
                assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        # b nq T h w
        out_mask_logits = model_outs['objdecoder_objseg'].permute(0,2,1,3,4)
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        # list[n t h w]
        tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
        for btc_idx in range(batch_size):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        isvalid = [targets[idx]['valid'] for idx in range(batch_size)] # list[n t], b
        loss_value_dict, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)

        if not self.is_pretraining_seg:
            refseg_src = self.get_decoder_preds(model_outs)
            loss_value_dict.update(self.ref_choose_loss(refseg_src, tgt_masks, gt_referent_idx, isvalid, matching_result, out_mask_logits, perFrame_has_ann))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
   
    def get_gt_class_prob(self, best_query_idx, obj_decoder_out, perFrame_has_ann):
        # b
        # b nq T h w, b T
        # list[t h w], batch
        batch_size, nq, *_ = obj_decoder_out.shape
        device = obj_decoder_out.device
        if not self.multiclass_choose:
            gt_probs = torch.zeros([batch_size, nq]).float().to(device)
            for btc_idx in range(batch_size):
                best_idx = best_query_idx[btc_idx]
                gt_probs[btc_idx][best_idx] = 1.
            return gt_probs
        else:
            dice_with_best = [] # b nq
            for btc_idx in range(batch_size):
                best_idx = best_query_idx[btc_idx]
                has_ann = perFrame_has_ann[btc_idx]
                best_mask = (obj_decoder_out[btc_idx][best_idx][has_ann].sigmoid() > 0.5).flatten() # thw
                other_mask = (obj_decoder_out[btc_idx][:, has_ann].sigmoid() > 0.5).flatten(1) # nq thw
                pairwise_dice = self.get_dice(other_mask, best_mask) # nq
                dice_with_best.append(pairwise_dice)
            dice_with_best = torch.stack(dice_with_best, dim=0) # b nq
            gt_probs = (dice_with_best >= self.choose_threshold).float()
            gt_probs = gt_probs / gt_probs.sum(dim=-1, keepdim=True)
            return gt_probs
    def get_dice(self, all_mask, best_mask):
        # nq c, c, Bool
        all_mask = all_mask.float()
        best_mask = best_mask.float().unsqueeze(0)
        numerator = 2 * torch.einsum("nc,mc->nm", all_mask, best_mask) # nq m
        # nq 1, 1 m -> nq m
        denominator = all_mask.sum(-1)[:, None] + best_mask.sum(-1)[None, :] 
        dice = (numerator + 1) / (denominator + 1) # nq m
        return dice.squeeze(-1) # nq
    def ref_choose_loss(self, refseg_src, tgt_masks, referent_idx, is_valid,  decoder_last_layer_matching_results, obj_decoder_out, perFrame_has_ann):
        """
        Args:
            refseg_src: dict{layer-1pred: {queries: bt c}}
            list[n t h w], batch
            list[src, tgt], batch
            list[n t], batch
        """
        device=tgt_masks[0].device 
        # thw.any()
        num_boxes = sum([t[refidx].flatten().any().int() for t, refidx in zip(tgt_masks,referent_idx)])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b

        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  decoder_last_layer_matching_results): # b
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(device) # b
        gt_probs = self.get_gt_class_prob(match_as_gt_indices, obj_decoder_out, perFrame_has_ann) # b nq
        gt_probs = gt_probs[ref_is_valid]
        refdecoder_choose_loss = 0.
        for layer_idx in range(-1, self.decoder_trans_nlayers):
            layer_weight = layer_weights[layer_idx] 
            if layer_weight != 0: # bt c
                refdecoder_gscore = refseg_src[f'layer{layer_idx}_preds']['grounding_score'] # b nq
                refdecoder_gscore = refdecoder_gscore[ref_is_valid]
                choose_loss = F.cross_entropy(refdecoder_gscore, gt_probs, reduction='none') # b
                choose_loss = choose_loss.sum() / num_boxes
                refdecoder_choose_loss += (choose_loss * layer_weight)

        return {'refdecoder_choose': refdecoder_choose_loss}

    # task loss
    def obj_decoder_objseg_loss(self, out_mask_logits, perFrame_has_ann, tgt_masks):
        # b nq T h w
        # b T
        # list[n t h w]
        loss_weight = self.loss_weight
        # n thw -> n
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),}
        
        matching_indices = self.obj_decoder_matching(out_mask_logits, perFrame_has_ann, tgt_masks)
        if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
            masks_losses = self.obj_decoder_masks_loss(out_mask_logits, perFrame_has_ann, tgt_masks, matching_indices, num_boxes)
            for k in masks_losses.keys():
                loss_value[k] += masks_losses[k]
        return loss_value, matching_indices       

    @torch.no_grad()
    def obj_decoder_matching(self, out_mask_logits, perFrame_has_ann, tgt_masks):
        # b nq T h w
        # b T
        # list[n t h w]
        src_masks_logits = out_mask_logits  # b nq T h w
        batch_size, nq, T, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t h w
            tgt_mask = tgt_masks[i].to(out_mask) # n t H W
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder_objseg']['matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder_objseg']['matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def binary_cross_entropy_mask_loss(self, src_masks, has_ann, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks[:, has_ann].flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, has_ann, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks[:, has_ann].flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def obj_decoder_masks_loss(self, out_mask_logits, perFrame_has_ann, tgt_masks, matching_indices, num_boxes):
        # b nq T h w
        # b T
        # list[n t h w], b
        batch_size = len(out_mask_logits)

        # list[n T h w], b
        src_masks = [t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)]

        # list[n t h w], b 
        tgt_masks = [t[J] for t, (_, J) in zip(tgt_masks, matching_indices)]
        

        mask_losses = [] 
        mask_dice_losses = [] 
        for btc_idx in range(batch_size):
            mask_ce_loss = self.binary_cross_entropy_mask_loss(src_masks[btc_idx], perFrame_has_ann[btc_idx], tgt_masks[btc_idx])
            mask_dice_loss = self.dice_mask_loss(src_masks[btc_idx], perFrame_has_ann[btc_idx], tgt_masks[btc_idx])
            mask_losses.append(mask_ce_loss)
            mask_dice_losses.append(mask_dice_loss)

        losses = {
            "objdecoder_mask": torch.cat(mask_losses).sum() / num_boxes,
            "objdecoder_dice": torch.cat(mask_dice_losses).sum() / num_boxes,
        }
        return losses    

class AMR_v0_detOnlyObj_Grounding_ptObjDet_v2(AMR_v0_detOnlyObj_Grounding_ptObjDet):
    def __init__(self, d_model=256, max_stride=64, pt_dir='/home/xhh/pt', obj_decoder={ 'name': None,'path': None,'freeze': True }, amrbart_wordEmbedding_freeze=True, amrtext_wordEmbedding_proj={ 'name': 'FeatureResizer','input_feat_size': 1024,'output_feat_size': 256,'dropout': 0,'do_ln': True }, fusion={ 'name': 'VisionLanguageFusionModule','d_model': 256,'nheads': 8,'dropout': 0 }, loss_weight={ 'refdecoder_mask': 5,'refdecoder_dice': 5,'refdecoder_giou': 0,'refdecoder_bbox': 0 }, tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 } } }, refdecoder={ 'nlayers': 9,'amr_cross_video_layer': { 'name': 'cross_attention','amr_cross': ['只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3'],'d_model': 256,'nhead': 8,'dropout': 0 },'amr_self_layer': { 'name': 'graph_layer_v1','d_model': 256,'flow': 'source_to_target','aggr': 'min' },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'choose_who': '第一个' }, is_pretraining_seg=False, multiclass_choose=False, choose_threshold=None) -> None:
        super().__init__(d_model, max_stride, pt_dir, obj_decoder, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, loss_weight, tasks, refdecoder, is_pretraining_seg, multiclass_choose, choose_threshold)

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device) 

        # b nq c, list[b t c h w, 32, 16, 8], b t c h w, (b nq t h w)
        obj_decoder_output = self.obj_decoder(samples)
        obj_queries, multiscale_feats, mask_feats, pred_masks_ori = obj_decoder_output['obj_queries'], obj_decoder_output['multiscale_feats'], obj_decoder_output['mask_features'], obj_decoder_output['pred_masks']
        fusion_output = self.fusion_module(video_queries=obj_queries,
                                            multiscale_feats=multiscale_feats,
                                            mask_feats=mask_feats, 
                                            text_feats=text_feats,
                                            amr_feats=amr_token_feats,
                                            amr_pad_masks = amr_token_seg_ids==0,
                                            text_pad_masks=text_pad_masks)
        # b nq c, b t nq h w
        obj_queries, pred_masks = fusion_output['obj_queries'], fusion_output['pred_masks']
        memories = obj_queries.permute(1,0,2)
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index,  node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), None,
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) # V nq
        g_score_by_batch = []
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        decoder_layer_preds[f'layer{-1}_preds'] = {'grounding_score': g_score_by_batch}
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                'objdecoder_objseg_ori': pred_masks_ori,
                 'objdecoder_objseg': pred_masks} 


class AMR_v0_detOnlyObj_Grounding_ptObjDet_v3(AMR_v0_detOnlyObj_Grounding_ptObjDet):
    def __init__(self, d_model=256, max_stride=64, pt_dir='/home/xhh/pt', obj_decoder={ 'name': None,'path': None,'freeze': True }, amrbart_wordEmbedding_freeze=True, amrtext_wordEmbedding_proj={ 'name': 'FeatureResizer','input_feat_size': 1024,'output_feat_size': 256,'dropout': 0,'do_ln': True }, fusion={ 'name': 'VisionLanguageFusionModule','d_model': 256,'nheads': 8,'dropout': 0 }, loss_weight={ 'refdecoder_mask': 5,'refdecoder_dice': 5,'refdecoder_giou': 0,'refdecoder_bbox': 0 }, tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 } } }, refdecoder={ 'nlayers': 9,'amr_cross_video_layer': { 'name': 'cross_attention','amr_cross': ['只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3', '只有2/3'],'d_model': 256,'nhead': 8,'dropout': 0 },'amr_self_layer': { 'name': 'graph_layer_v1','d_model': 256,'flow': 'source_to_target','aggr': 'min' },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'choose_who': '第一个' }, is_pretraining_seg=False, multiclass_choose=False, choose_threshold=None) -> None:
        super().__init__(d_model, max_stride, pt_dir, obj_decoder, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, loss_weight, tasks, refdecoder, is_pretraining_seg, multiclass_choose, choose_threshold)
        assert self.is_pretraining_seg == False

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device) 

        # b nq c, list[b t c h w, 32, 16, 8], b t c h w, (b nq t h w)
        obj_decoder_output = self.obj_decoder(samples)
        obj_queries_before_fusion, multiscale_feats, mask_feats, pred_masks_ori = obj_decoder_output['obj_queries'], obj_decoder_output['multiscale_feats'], obj_decoder_output['mask_features'], obj_decoder_output['pred_masks']

        fusion_output = self.fusion_module(video_queries=obj_queries_before_fusion,
                                            multiscale_feats=multiscale_feats,
                                            mask_feats=mask_feats, 
                                            text_feats=text_feats,
                                            amr_feats=amr_token_feats,
                                            amr_pad_masks = amr_token_seg_ids==0,
                                            text_pad_masks=text_pad_masks)
        # b nq c, b t nq h w
        obj_queries, pred_masks = fusion_output['obj_queries'], fusion_output['pred_masks']
        memories = obj_queries.permute(1,0,2)
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index,  node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), None,
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) # V nq
        g_score_by_batch = []
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        decoder_layer_preds[f'layer{-1}_preds'] = {'grounding_score': g_score_by_batch}

        assert self.decoder_trans_layers is None

        return {'refdecoder_refseg': decoder_layer_preds,
                'mask_logits_before_fusion': pred_masks_ori,
                'queries_before_fusion': obj_queries_before_fusion,
                'mask_logits_after_fusion': pred_masks,
                'queries_after_fusion': obj_queries,} 

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        # b nq T h w
        mask_logits_after_fusion = model_outs['mask_logits_after_fusion'].permute(0,2,1,3,4)
        mask_logits_before_fusion = model_outs['mask_logits_before_fusion'].permute(0,2,1,3,4)
        queries_before_fusion = model_outs['queries_before_fusion']
        queries_after_fusion = model_outs['queries_after_fusion']

        # after-fusion query distiallation loss
        loss_value_dict = self.fusion_distillation_loss(mask_logits_after_fusion=mask_logits_after_fusion, 
                                                        mask_logits_before_fusion=mask_logits_before_fusion,
                                                        query_after_fusion=queries_after_fusion,
                                                        query_before_fusion=queries_before_fusion)
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        # list[n t h w]
        tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
        for btc_idx in range(batch_size):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        isvalid = [targets[idx]['valid'] for idx in range(batch_size)] # list[n t], b
        _, matching_result = self.obj_decoder_objseg_loss(mask_logits_before_fusion, perFrame_has_ann, tgt_masks)

        if not self.is_pretraining_seg:
            refseg_src = self.get_decoder_preds(model_outs)
            loss_value_dict.update(self.ref_choose_loss(refseg_src, tgt_masks, gt_referent_idx, isvalid, matching_result, mask_logits_before_fusion, perFrame_has_ann))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
    
    def fusion_distillation_loss(self, mask_logits_after_fusion, mask_logits_before_fusion, query_after_fusion, query_before_fusion):
        # pixel-wise
        # b nq t h w
        # b nq t h w
        pixel_loss = F.binary_cross_entropy_with_logits(mask_logits_after_fusion,
                                                        mask_logits_before_fusion.sigmoid(), reduction='none') # b nq t h w
        pixel_loss = pixel_loss.flatten(1).mean(-1)
        # query pair-wise
        # b nq c
        # b nq c
        pair_sim_before = F.cosine_similarity(query_before_fusion.unsqueeze(1), query_before_fusion.unsqueeze(2), dim=3) # b nq nq
        pair_sim_after = F.cosine_similarity(query_after_fusion.unsqueeze(1), query_after_fusion.unsqueeze(2), dim=3) # b nq nq
        query_loss =(pair_sim_before - pair_sim_after).square().flatten(1).mean(-1)
        return {
            'pixelwise_fusion_distill': pixel_loss.mean(),
            'query_pairwise_fusion_distill':query_loss.mean(),
        }

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
        # b nq T h w
        out_mask_logits = decoder_layer_preds['mask_logits_before_fusion'].permute(0,2,1,3,4)
        if self.is_pretraining_seg:
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            # list[n t h w]
            tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
            for btc_idx in range(batch_size):
                start = int(self.obj_decoder_mask_out_stride // 2)
                im_h, im_w = tgt_masks[btc_idx].shape[-2:]
                tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
                assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
                assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }

class AMR_Grounding_2DObj(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                 work_dir=None,
                 mode=None,
                loss_weight={},
                tasks = { 'objdecoder':{}},
                pixel_mean = [0.485, 0.456, 0.406],
                pixel_std = [0.229, 0.224, 0.225],
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                # obj decoder
                 obj_decoder = {
                     'name':None,
                     'path': None,
                     'freeze': True,
                 },
                reason_module={},
                temporal_decoder = {},
                fusion={},
                use_we=False,
                loss_type='object',
                word_embedding_random=False,
                ) -> None:
        super().__init__()
        self.use_we = use_we
        self.loss_type = loss_type
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.mode = mode
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if word_embedding_random:
            for p in self.amrbart_wordEmbedding.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        amr_proj_name = amrtext_wordEmbedding_proj.pop('name')
        if amr_proj_name == 'FeatureResizer':
            self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        elif amr_proj_name == 'linear':
            self.amrtext_wordEmbedding_proj = nn.Linear(**amrtext_wordEmbedding_proj)
        # self.amrtext_wordEmbedding_3c_to_c = nn.Linear(1024 * 3, 1024)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        from models.pretrained_image_decoder import pt_obj_2d_decoder_entrypoint
        create_obj_decoder = pt_obj_2d_decoder_entrypoint(obj_decoder['name']) # 里面有load image segmentation
        self.obj_decoder = create_obj_decoder(obj_decoder, pt_dir, work_dir)
        self.obj_decoder_num_layers = self.obj_decoder.num_layers # 10层  # load checkpoing
        self.obj_decoder_mask_out_stride = self.obj_decoder.mask_out_stride
        self.obj_decoder_mask_threshold = self.obj_decoder.mask_threshold
    
        if mode == 'rios测试预训练imseg':
            return
        
        # 和text相关
        logging.info('你在做和text有关的任务')
        print('你在做和text有关的任务')
        from models.layer_fusion import fusion_entrypoint
        
        create_fusion = fusion_entrypoint(fusion['name'])
        fusion_module = create_fusion(fusion)
        self.fusion_module = fusion_module
        # hack obj decoder fusion
        self.obj_decoder.sem_seg_head.pixel_decoder.hack_fusion(fusion_module=self.fusion_module,
                                                                early_fusion=fusion['deform_early'],
                                                                early_fusion_deep_copy=fusion['deform_early_dcopy'],
                                                                early_add_pos=fusion['deform_add_pos'] if 'deform_add_pos' in fusion else True,
                                                                encoder_layer_ref_self=fusion['deform_layer'],
                                                                encoder_layer_deep_copy=fusion['deform_layer_dcopy'],
                                                                encoder_layer_add_pos=fusion['deform_layer_add_pos'] if 'deform_layer_add_pos' in fusion else True)
        from .layer_graph import graphLayer_entrypoint
        create_reason_module = graphLayer_entrypoint(reason_module['graph']['name'])
        self.reason_module = create_reason_module(reason_module['graph'])
        if mode == '只训练rios':
            self.reason_2d_choose = reason_module['2d_choose_who']
            self.reason_2d_layer_if_reason =  self.tasks['2d_layer_if_reason'] # obj_decoder的每层是否reason
            assert len(self.tasks['objdecoder']['loss_layer_weights']) == self.obj_decoder_num_layers
            assert self.reason_2d_layer_if_reason[-1]
            assert len(self.reason_2d_layer_if_reason) == self.obj_decoder_num_layers

        elif mode == 'rios之后rvos' or mode == '只训练rvos' or mode == 'joint':
            self.reason_2d_choose = reason_module['2d_choose_who']
            self.reason_2d_layer_if_reason =  self.tasks['2d_layer_if_reason'] # obj_decoder的每层是否reason
            assert len(self.reason_2d_layer_if_reason) == self.obj_decoder_num_layers
            assert len(self.tasks['objdecoder']['loss_layer_weights']) == self.obj_decoder_num_layers
            from .layer_temporal_decoder import temporal_decoder_entrypoint
            create_temporal_decoder = temporal_decoder_entrypoint(temporal_decoder['name'])
            self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir)
            self.temporal_decoder_num_layers = self.temporal_decoder.num_layers
            self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
            self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
            self.temporal_decoder.hack_fusion(fusion_module,
                                                early_fusion=fusion['swin_early'],
                                                early_fusion_deep_copy=fusion['swin_early_dcopy'], 
                                                early_fusion_add_pos=fusion['swin_early_add_pos'],
                                                encoder_layer_ref_self=fusion['swin_layer'],
                                                encoder_layer_deep_copy=fusion['swin_layer_dcopy'],
                                                encoder_layer_add_pos=fusion['swin_early_add_pos'],)
            self.reason_3d_choose = reason_module['3d_choose_who']
            self.reason_3d_layer_if_reason = self.tasks['3d_layer_if_reason'] # decoder的每层是否reason
            assert self.reason_3d_layer_if_reason[-1]
            assert len(self.tasks['temporal_decoder']['loss_layer_weights']) == self.temporal_decoder.used_layers\
                                                                         * self.temporal_decoder_num_layers # 训练的时候用后三层
            assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers * self.temporal_decoder.used_layers
        else:
            return
    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']

        if self.use_we:
            if not self.training:
                we_size = len(self.amrbart_wordEmbedding.weight)
                global_we = self.amrtext_wordEmbedding_proj(self.amrbart_wordEmbedding.weight)
                global_we = repeat(global_we, 's c -> b s c',b=batch_size)
                global_seg_ids = (amr_token_seg_ids.new_ones([batch_size, we_size]) * 2).int()
            else:
                global_we = text_auxiliary['all_concept_roles'] # b mmax
                global_seg_ids = global_we.new_ones([batch_size, global_we.shape[1]]) * 2 # b mmax
                acc_pad = text_auxiliary['all_concept_roles_pad'] # b max
                global_seg_ids.masked_fill_(acc_pad, 0)
                global_we = self.amrtext_wordEmbedding_proj(self.amrbart_wordEmbedding(global_we))
        else:
            global_we = None
            global_seg_ids = None
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments, global_we, global_seg_ids
        
        # amrs = text_auxiliary['amrs'] # list[Graph]
        # batch_size = len(amrs)
        # text_tokens = text_auxiliary['text_token_ids'] # b smax
        # text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        # text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        # text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        # text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        # for batch_idx in range(batch_size):
        #     text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        # text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        # amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        # amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        # amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        # amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        # # list[list[ti c]] -> list[Vi+Ei c]
        # amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        # for batch_idx in range(batch_size):
        #     amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0) # V + E, c

        # num_nodes_by_batch = [g.num_nodes for g in amrs]
        # num_edges_by_batch = [g.num_edges for g in amrs]
        # batched_amrs = Batch.from_data_list(amrs) # concate
        # edge_index = batched_amrs.edge_index.to(device)  # 2 E
        # batched_node_feats: torch.Tensor = torch.cat([atf[:nnodes] for atf, nnodes in zip(amr_token_feats, num_nodes_by_batch)], dim=0) # V_sum c
        # batched_edge_feats = torch.cat([atf[nnodes:] for atf, nnodes in zip(amr_token_feats, num_nodes_by_batch)], dim=0) # E_sum c
        # src_feats, tgt_feats = batched_node_feats[edge_index[0]],  batched_node_feats[edge_index[1]]
        # batched_edge_feats = self.amrtext_wordEmbedding_3c_to_c(torch.cat([src_feats, tgt_feats, batched_edge_feats], dim=-1))

        # # split
        # batched_node_feats = batched_node_feats.split(num_nodes_by_batch)
        # batched_edge_feats = batched_edge_feats.split(num_edges_by_batch)
        # for batch_idx in range(batch_size):
        #     amr_token_feats[batch_idx] = torch.cat([batched_node_feats[batch_idx],batched_edge_feats[batch_idx]], dim=0) # V + E, c

        # amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        # assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        # assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        # node_alignments = text_auxiliary['node_alignments']

        # amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b (V+E)max c
        # return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, targets=None, visualize_dir=None):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        if len(samples.tensors.shape) == 5:
            # t b c h w
            nf, batch_size, *_ = samples.tensors.shape
            samples.tensors = rearrange(samples.tensors, 't b c h w -> (b t) c h w')
            samples.mask = rearrange(samples.mask, 't b h w -> (b t) h w')
        elif len(samples.tensors.shape) == 4:
            # b 3 h w
            batch_size = samples.tensors.shape[0]
        else:
            raise ValueError()
        device = samples.tensors.device
        if text_queries is not None:
            amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments, global_we, global_we_seg_ids,\
                  = self.encode_text(text_queries, auxiliary, device) 
        else:
            text_feats, text_pad_masks, amr_token_feats, amr_token_seg_ids = None, None, None, None

        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        if self.use_we:
            obj_decoder_output, _, _ = self.obj_decoder(samples,
                                                                    amrs=[None] * len(amrs), 
                                                                    amr_token_feats=global_we,
                                                                    amr_token_seg_ids=global_we_seg_ids, 
                                                                    text_feats=None, 
                                                                    text_pad_masks=None)
        else:
            obj_decoder_output, amr_token_feats, text_feats = self.obj_decoder(samples,
                                                                                amrs= amrs, 
                                                                                amr_token_feats=amr_token_feats,
                                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                                text_feats=text_feats, 
                                                                                text_pad_masks=text_pad_masks)
        obj_queries_by_layer, pred_masks_by_layer, multiscale_feats, \
                            mask_features= obj_decoder_output['obj_queries'], obj_decoder_output['pred_masks'],\
                                                                    obj_decoder_output['multiscale_feats'], obj_decoder_output['mask_features'] # b nq c

        if self.mode == 'rios测试预训练imseg':
            return {'objdecoder': {'pred_masks': pred_masks_by_layer,}}
        elif self.mode == '只训练rios':  
            grounding_score_by_layer = []
            for layer_idx, obj_queries in enumerate(obj_queries_by_layer): 
                if self.reason_2d_layer_if_reason[layer_idx]:
                    if self.use_we:
                        amr_token_feats = contexualized_amr_feats
                    grounding_score = self.reason_module(obj_queries=obj_queries, 
                                                        amrs=amrs,
                                                        amr_token_feats=amr_token_feats,
                                                        amr_token_seg_ids=amr_token_seg_ids,
                                                        node_alignments=node_alignments,
                                                        text_feats=text_feats,
                                                        is_2d=True,
                                                        text_pad_masks=text_pad_masks) # list[vi nq]
                    if self.reason_2d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(grounding_score)
                else:
                    grounding_score_by_layer.append(None)
            return {'objdecoder': {'pred_masks': pred_masks_by_layer, # b nq h w
                                   'reason_2d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint': 
            #可能会有2d的loss计算
            repeated_amrs = []
            for idx in range(batch_size):
                for _ in range(nf):
                    repeated_amrs.append(copy.deepcopy(amrs[idx]))
            grounding_score_2d_by_layer = []
            for layer_idx, obj_queries in enumerate(obj_queries_by_layer): 
                if self.reason_2d_layer_if_reason[layer_idx]:
                    grounding_score_2d = self.reason_module(obj_queries=obj_queries.clone(), 
                                                        amrs=repeated_amrs,
                                                        amr_token_feats=repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf),
                                                        amr_token_seg_ids=repeat(amr_token_seg_ids, 'b s -> (b t) s',t=nf),
                                                        node_alignments=node_alignments,
                                                        text_feats=repeat(text_feats, 'b s c -> (b t) s c', t=nf),
                                                        is_2d=True, is_3d=False,
                                                        text_pad_masks=repeat(text_pad_masks,'b s -> (b t) s', t=nf))
                    if self.reason_2d_choose == '第一个':
                        grounding_score_2d = torch.stack([lg[0] for lg in grounding_score_2d], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_2d_by_layer.append(grounding_score_2d)
                else:
                    grounding_score_2d_by_layer.append(None)

            obj_queries_by_layer = [rearrange(obj_q, '(b t) nq c -> b t nq c',b=batch_size,t=nf) for obj_q in obj_queries_by_layer] 
            # L b s c, L b s c
            if self.use_we:
                temporal_decoder_output, _, _ = self.temporal_decoder(frame_query_by_layer=obj_queries_by_layer, # list[b t nq c]
                                                                                            mask_features=mask_features, # bt c h w
                                                                                            amrs=[None] * len(amrs), 
                                                                                            amr_token_feats=global_we,
                                                                                            amr_token_seg_ids=global_we_seg_ids, 
                                                                                            text_feats=None, 
                                                                                            text_pad_masks=None)
            else:
                temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(frame_query_by_layer=obj_queries_by_layer, # list[b t nq c]
                                                                                            mask_features=mask_features, # bt c h w
                                                                                            amrs= amrs, 
                                                                                            amr_token_feats=amr_token_feats,
                                                                                            amr_token_seg_ids=amr_token_seg_ids, 
                                                                                            text_feats=text_feats, 
                                                                                            text_pad_masks=text_pad_masks)
            # L D b nq c
            # L b t nqf c
            # L D b nq t nqf
            # L D b nq t h w
            # L D b nq class+1
            temporal_queries_by_layer, frame_queries_memory, cross_attn_weights_by_layer, \
              temporal_pred_masks_by_layer, temporal_pred_logits_by_layer,\
                = temporal_decoder_output['temporal_queries'], temporal_decoder_output['frame_queries'], \
                                                                temporal_decoder_output['cross_attn_weights'],\
                                                                temporal_decoder_output['pred_masks'], temporal_decoder_output['pred_logits']
            D = temporal_queries_by_layer.shape[1]
            L = temporal_queries_by_layer.shape[0]
            if self.use_we:
                amr_token_feats = repeat(amr_token_feats, 'b s c -> L D b s c',L=L, D=D)
                text_feats = repeat(text_feats, 'b s c -> L D b s c', L=L, D=D)
            else:
                amr_token_feats = repeat(amr_token_feats, 'L b s c -> L D b s c',D=D)
                text_feats = repeat(text_feats, 'L b s c -> L D b s c', D=D)
            frame_queries_memory = repeat(frame_queries_memory, 'L b t nqf c -> L D b t nqf c',D=D)
            # region
            # repeated_amrs = []
            # for idx in range(batch_size):
            #     for _ in range(nf):
            #         repeated_amrs.append(copy.deepcopy(amrs[idx]))
            # spatial_grounding_score = self.reason_module(obj_queries=frame_queries_memory.flatten(0, 1), # bt nq c 
            #                                             amrs=repeated_amrs,
            #                                             amr_token_feats=repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf),
            #                                             amr_token_seg_ids=repeat(amr_token_seg_ids, 'b s -> (b t) s',t=nf),
            #                                             node_alignments=node_alignments,
            #                                             text_feats=repeat(text_feats, 'b s c -> (b t) s c', t=nf),
            #                                             is_2d=True, is_3d=False,
            #                                             text_pad_masks=repeat(text_pad_masks,'b s -> (b t) s', t=nf),
            #                                         ) # list[vi nqf], batch_size * T
            # # list[vi nqf], b * T -> list[list[vi nqf], T], b
            # spg_by_batch = []
            # for idx in range(batch_size):
            #     spg_by_batch.append(spatial_grounding_score[idx*nf:(idx+1)*nf])
            # spg_scores = [torch.stack(sbb, dim=1) for sbb in spg_by_batch] # list[Vi T nqf]
            # endregion
            grounding_score_by_layer = []
            for layer_idx, (temporal_queries, cross_attn_weights, amr_tok_feat, txt_feat, frame_query_mem) in \
                enumerate(zip(temporal_queries_by_layer.flatten(0,1),
                            cross_attn_weights_by_layer.flatten(0,1), 
                            amr_token_feats.flatten(0,1), text_feats.flatten(0,1), frame_queries_memory.flatten(0,1))):
                if self.reason_3d_layer_if_reason[layer_idx]:
                    grounding_score = self.reason_module(temporal_queries=temporal_queries,  # b nq c
                                                            frame_queries=frame_query_mem, # b t nqf c
                                                            frame_queries_grounding_score=None, 
                                                             cross_attn_weights=cross_attn_weights,  # # b nq t nqf
                                                             is_3d=True, is_2d=False,
                                                             amrs=amrs,
                                                             amr_token_feats=amr_tok_feat,
                                                             amr_token_seg_ids=amr_token_seg_ids,
                                                             node_alignments=node_alignments,
                                                             text_feats=txt_feat,
                                                             text_pad_masks=text_pad_masks) # list[vi nq]
                    # 可视化所有object query的mask
                    if visualize_dir is not None:
                        save_model_output(videos=samples.tensors,
                                        text_query=text_queries[0], 
                                        amr=auxiliary['amrs'][0],
                                        amr_tree_string=auxiliary['amr_tree_strings'][0], 
                                        directory=visualize_dir,
                                        pred_masks=temporal_pred_masks_by_layer.flatten(0,1)[-1][0],
                                        scores=grounding_score[0],)
                    if self.reason_3d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(grounding_score)
                else:
                    grounding_score_by_layer.append(None)
            return {'temporal_decoder': {'pred_masks': temporal_pred_masks_by_layer.flatten(0,1), # list[b nq t h w]
                                         'pred_logits': temporal_pred_logits_by_layer.flatten(0,1), # list[b nq class+1]
                                         'reason_3d': grounding_score_by_layer}, # list[b nq]
                    'objdecoder': {'pred_masks': pred_masks_by_layer, # list[bt nq h w]
                                   'reason_2d': grounding_score_2d_by_layer} ,} # list[None] / list[bt nq]

    def forward_rios(self, samples, text_queries, auxiliary, targets, visualize=False):
        # samples: list[3 h w] -> b 3 H W
        # targets: list[{'masks': ni h w}]
        # loss有 n个物体的mask loss + reason choose loss
        images = [(x - self.pixel_mean) / self.pixel_std for x in samples]
        samples = nested_tensor_from_tensor_list_with_stride(images, max_stride=self.max_stride)
        batch_size, _, H, W = samples.tensors.shape
        new_targets = self.rios_targets_handler(targets, pad_H=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary, targets=new_targets) 
        if self.loss_type == 'referent':
            loss_value_dict = self.objdecoder_loss_referent(model_outs, new_targets)
        else:
            loss_value_dict = self.objdecoder_loss(model_outs, new_targets)

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample_rios(self,samples, text_queries, auxiliary, targets, visualize=False):
        images = [(x - self.pixel_mean) / self.pixel_std for x in samples]
        samples = nested_tensor_from_tensor_list_with_stride(images, max_stride=self.max_stride)
        batch_size, _, H, W = samples.tensors.shape
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)['objdecoder']
        # b nq h w
        out_mask_logits = decoder_layer_preds['pred_masks'][-1]
        if self.mode == 'rios测试预训练imseg': 
            new_targets = self.rios_targets_handler(targets, pad_H=H, pad_W=W) # 缩放gt mask到4x
            matching_result = self.objdecoder_matching(out_mask_logits, new_targets)
            gt_referent_idx = new_targets['gt_referent_idx']
            # list[h w] -> b h w
            out_mask_logits = torch.stack([out_mask[src_idx[tgt_idx.tolist().index(gt_ref_idx)]]
                                            for out_mask, gt_ref_idx, (src_idx, tgt_idx) in zip(out_mask_logits, gt_referent_idx, matching_result, )], dim=0)
        else:
            ref_last_layer_gscore = decoder_layer_preds['reason_2d'][-1] # b nq
            argmax_query_idx = ref_last_layer_gscore.argmax(-1) # b
            # b h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
        # b 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # b 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], b
        size_after_aug = [] #list[h,w], b
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
        return {
            'query_pred_masks': processed_pred_masks, # [n h w], batch
            'query_pred_is_referred_prob': query_pred_is_referred_prob, # [n], batch
        }
    

    def rios_targets_handler(self, targets, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = []
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n h w -> n H W
            tgt_masks.append(F.pad(targets[idx]['masks'].float().unsqueeze(0), pad=(0, pad_W-w, 0, pad_H-h)).bool()[0])
        for btc_idx in range(batch_size):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        isvalid = [targets[idx]['valid'] for idx in range(batch_size)] # list[ni], b
        return {
            'masks': tgt_masks,
            'gt_referent_idx': gt_referent_idx,
            'isvalid': isvalid
        }

    def objdecoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = sum([t.int().sum() for t in isvalid])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss(layer_gscore_output, matching_indices, targets)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, matching_indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss(self, layer_gscore_output, matching_indices,  targets):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        num_refs = (ref_is_valid.int().sum())
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}


    def objdecoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = len(isvalid)
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss_referent(layer_gscore_output, matching_indices, targets, num_objs)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching_referent(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # lis[int], batch
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # list[int], b
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[[J]] for t, J in zip(tgt_masks, referent_idx)], dim=0) # n_simga h w
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss_referent(self, layer_gscore_output, matching_indices,  targets, num_refs):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        assert ref_is_valid.any()

        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}
    
 
    def forward_rvos(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        
        rvos_targets = self.rvos_targets_handler(targets, pad_T=T, pad_H=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        # 可能会有object decoder的loss
        if self.loss_type == 'object':
            loss_value_dict = self.temporal_decoder_loss(model_outs, rvos_targets)
        elif self.loss_type == 'referent':
            loss_value_dict = self.temporal_decoder_loss_referent(model_outs, rvos_targets)

        video_rios_targets = self.video_rios_targets_handler(targets, pad_T=T, pad_H=H, pad_W=W)
        has_ann = video_rios_targets['has_ann'] # bT
        obj_pred_masks = model_outs['objdecoder']['pred_masks'] # list[bt nq h w]
        obj_pred_masks = [opm[has_ann] for opm in obj_pred_masks]
        model_outs['objdecoder']['pred_masks'] = obj_pred_masks

        obj_gscores = model_outs['objdecoder']['reason_2d'] # list[bT nq/None]
        new_obj_gscores = []
        for og in obj_gscores:
            if og is None:
                new_obj_gscores.append(None)
            else:
                new_obj_gscores.append(og[has_ann]) # bt nq -> bt' nq
        model_outs['objdecoder']['reason_2d'] = new_obj_gscores
        if self.loss_type == 'object':
            loss_value_dict.update(self.objdecoder_loss(model_outs, video_rios_targets))
        elif self.loss_type == 'referent':
            loss_value_dict.update(self.objdecoder_loss_referent(model_outs, video_rios_targets))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample_rvos(self, samples, text_queries, auxiliary, targets, visualize_dir=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride) # targets[0]['masks']
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, visualize_dir=visualize_dir,)
        # b nq T h w -> b T nq h w
        out_mask_logits = decoder_layer_preds['temporal_decoder']['pred_masks'][-1].permute(0,2,1,3,4)
        if self.mode == '测试rvos bound':
            raise NotImplementedError()
            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            ref_last_layer_gscore = decoder_layer_preds['temporal_decoder']['reason_3d'][-1]  # b nq
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[:, max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()] # bT -> bt' h w
        # bt' 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.temporal_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.temporal_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # list[(h w)],batch -> list[(h w)], bt'
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        assert len(query_pred_masks) == len(size_original)
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0].bool()
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt' n -> list[n], bt' -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }


    def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
        labels = [t['class_labels'] for t in targets] # list[ni], batch
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
        # list[ni]
        is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> list[T]
        perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

        return {
            'labels': labels, 
            'masks': tgt_masks,
            'is_valid': is_valid,
            'referent_idx': gt_referent_idx,
            'has_ann': perFrame_has_ann
        }

    def video_rios_targets_handler(self, targets, pad_T, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w
            
        # list[n h w], bt'
        rep_tgt_masks = []
        for btc_idx in range(batch_size):
            t_m = tgt_masks[btc_idx].split(1, dim=1)
            t_m = [tm.squeeze(1) for tm in t_m]
            rep_tgt_masks.extend(t_m)
        rep_is_valid = [rtm.flatten(1).any(-1) for rtm in rep_tgt_masks] # list[ni], bt'
        num_anns_by_batch = [tm.shape[1] for tm in tgt_masks] # list[int]
        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        rep_gt_referent_idx = [] # list[int], bt'
        for btc_idx in range(batch_size):
            rep_gt_referent_idx.extend([gt_referent_idx[btc_idx]] * num_anns_by_batch[btc_idx])
                

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> bT
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann])
        # 有annotation并且是valid的

        return {
            'masks': rep_tgt_masks,
            'gt_referent_idx': rep_gt_referent_idx,
            'isvalid': rep_is_valid,
            'has_ann': perFrame_has_ann
        }

    def temporal_decoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idxs = targets['referent_idx'] # list[int]

        num_objs = sum([tm.flatten(0,1).flatten(1).any(-1).int().sum() for tm in tgt_masks])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        num_refs = len(tgt_masks) # 
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_class': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers   
        out_logits = model_outs['temporal_decoder']['pred_logits'] # list[b nq class+1]
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_out_logits, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                    out_logits,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching(layer_mask_output, layer_out_logits, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if self.loss_weight['tempdecoder_class'] != 0:
                    class_losses = self.temporal_decoder_classes_loss(layer_out_logits, targets, matching_indices, num_objs)
                    for k in class_losses.keys():
                        loss_value[k] += layer_loss_weight * class_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, out_class_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        tgt_classes = targets['labels'] # list[ni]
        src_masks_logits = out_mask_logits  # b nq T h w
        src_class_logits = out_class_logits # b nq class+1

        batch_size, nq, T, h, w = src_masks_logits.shape 
        src_class_probs = src_class_logits.softmax(-1) # b nq class+1
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_class_prob = src_class_probs[i] # nq class+1

            tgt_cls = tgt_classes[i] # ni
            cost_class = - out_class_prob[:, tgt_cls] # nq ni

            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W

            scores = []
            for ann_t in range(out_mask.shape[1]):
                out_t_mask = out_mask[:, ann_t] # nq h w
                tgt_t_mask = tgt_mask[:, ann_t] # ni h w
                c_mask = batch_sigmoid_ce_loss(out_t_mask.flatten(1), tgt_t_mask.flatten(1)) # nq ni
                c_dice = batch_dice_loss(out_t_mask.flatten(1), tgt_t_mask.flatten(1)) # nq ni

                t_cost =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * c_mask + \
                    self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * c_dice
                scores.append(t_cost)
            scores = torch.stack(scores, dim=0).mean(0) # n nq ni -> nq ni
            C = scores + self.tasks['temporal_decoder']['objseg_matching_costs']['class'] * cost_class
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_classes_loss(self, layer_out_logits, targets, matching_indices, num_objs):
        # b nq class+1, 
        target_labels = targets['labels'] #list[ni], batch
        # t_sigma
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, matching_indices)]) # n_sigma
    
        idx = get_src_permutation_idx(matching_indices)
        # b nq 充满背景类别
        target_classes = torch.full(
            layer_out_logits.shape[:2], layer_out_logits.shape[2] -1, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o
        class_weights = torch.ones(layer_out_logits.shape[2]).float() # class+1
        class_weights[-1] = self.tasks['temporal_decoder']['eos_weight']

        loss_ce = F.cross_entropy(layer_out_logits.transpose(1,2), target_classes, weight=class_weights.to(layer_out_logits))
        losses = {"tempdecoder_class": loss_ce}
        return losses

    def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
        has_anntation = targets['has_ann'] # list[T]
        # is_valid = targets['is_valid'] # list[ni]

        # b nq T H W -> list[ni t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[ni t' H W], b 
        tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# list[ni_t' h w]
        tgt_masks = torch.cat([tm.flatten(0,1) for tm in tgt_masks],dim=0) # list[ni_t' h w]
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_objs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_objs),
        }
        return losses    

    def temporal_decoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int]
        # list[ni t' h w] -> list[t' hw]
        num_refs = sum([tm[ref_idx].flatten(1).any(-1).int().sum() for tm, ref_idx in zip(tgt_masks, referent_idx)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()

        num_refs_video = len(tgt_masks)
        num_refs_video = torch.as_tensor([num_refs_video], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs_video)
        num_refs_video = torch.clamp(num_refs_video / get_world_size(), min=1).item()

        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers 
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_refs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss_referent(layer_gscore_output, targets, matching_indices, num_refs, num_refs_video)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching_referent(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int], batch
        src_masks_logits = out_mask_logits  # b nq T h w

        batch_size, nq, T, h, w = src_masks_logits.shape
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 t' H W
            tgt_mask = tgt_mask.to(out_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1

            C =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice
            C = C.cpu()
            indices.append(linear_sum_assignment(C))            
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_refs):
        has_anntation = targets['has_ann'] # list[T]
        ref_idx = targets['referent_idx'] # list[int], batch
        # b nq T H W -> list[1 t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[1 t' H W], b 
        tgt_masks = [t[[J]] for t, J in zip(targets['masks'], ref_idx)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# 1_t'_sigma h w
        tgt_masks = torch.cat([tm.flatten(0, 1) for tm in tgt_masks],dim=0) # 1_t'_sigma h w
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
        }
        return losses    

    def temporal_reason_loss_referent(self, layer_gscore_output, targets, matching_indices, num_refs, num_refs_video):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs_video}


    def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, global_num_refs):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        num_refs = (ref_is_valid.int().sum())
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = tgt_idx.tolist().index(ref_idx)
            match_as_gt_idx = src_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs}

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        # 把pixel mean, pixel std加上去
        # samples -> NestedTensor
        if self.mode == 'rios测试预训练imseg':
            raise ValueError() # 只能以evaluate形式运行
        elif self.mode == '只训练rios':
            return self.forward_rios(samples, text_queries, auxiliary, targets)
        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint':
            return self.forward_rvos(samples, text_queries, auxiliary, targets) 
        else:
            raise ValueError()

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize_dir=False):
        if self.mode == 'rios测试预训练imseg':
            return self.sample_rios(samples, text_queries=None, auxiliary=None, targets=targets)
        elif self.mode == '只训练rios':
            return self.sample_rios(samples, text_queries, auxiliary, targets)
        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint':
            return self.sample_rvos(samples, text_queries, auxiliary, targets,visualize_dir=visualize_dir) 
        else:
            raise ValueError()

@register_model
def amr_grounding_2dobj(device, configs):
    model = AMR_Grounding_2DObj(
        d_model=configs['d_model'],
        max_stride=configs['max_stride'],
        use_we=configs['use_we'] if 'use_we' in configs else False,
        pt_dir=configs['pt_dir'],
        work_dir=configs['work_dir'],
        mode=configs['mode'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        pixel_mean=configs['pixel_mean'],
        pixel_std=configs['pixel_std'],
        word_embedding_random=configs['word_embedding_random'] if 'word_embedding_random' in configs else False,
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],

        obj_decoder=configs['obj_decoder'],
        reason_module=configs['reason_module'], 
        temporal_decoder=configs['temporal_decoder'],
        fusion=configs['fusion'],
        loss_type=configs['loss_type'] if 'loss_type' in configs else 'object'
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder.backbone" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder.backbone" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] 
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.info('你没用任何scheduler')
        print('你没用任何scheduler')
        return model, optimizer, None, None
    
    if sch_conf['name'] == 'polynomial_split':
        from models.model_utils import polynomial_decay_lambda
        group_names = ['model', 'vbb', 'text']
        poly_lambdas = []
        for gname in group_names:
            g_poly_conf = sch_conf[gname]
            poly_lambdas.append(partial(polynomial_decay_lambda, initial_learning_rate=g_poly_conf['initial_learning_rate'],
                                                                        end_learning_rate=g_poly_conf['end_learning_rate'],
                                                                        decay_steps=g_poly_conf['decay_steps'],
                                                                        power=g_poly_conf['power']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=poly_lambdas,
                                                      last_epoch=-1,)
        return model, optimizer, scheduler, sch_conf['unit']
    elif sch_conf['name'] == 'polynomial_freezebb':
        from models.model_utils import polynomial_decay_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(polynomial_decay_lambda, 
                                                                        initial_learning_rate=sch_conf['initial_learning_rate'],
                                                                        end_learning_rate=sch_conf['end_learning_rate'],
                                                                        decay_steps=sch_conf['decay_steps'],
                                                                        power=sch_conf['power'],
                                                                        ),
                                                      last_epoch=-1,
                                                      verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'polynomial':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                                        total_iters=sch_conf['total_iters'],
                                                        power=sch_conf['power'],
                                                        last_epoch=-1,
                                                        verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=sch_conf['milestones'],
                                                        gamma=sch_conf['gamma'],
                                                        verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']
    
    elif sch_conf['name'] == 'invert_sqrt':
        from models.model_utils import inverse_sqrt_warmup_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=partial(inverse_sqrt_warmup_lambda,
                                                                         num_warmup_steps=sch_conf['num_warmup_steps'],
                                                                         num_training_steps=sch_conf['num_training_steps']), last_epoch=-1)
        return model, optimizer, scheduler, sch_conf['unit']
    else:
        raise ValueError()


class AMR_Grounding_2DObj_MeVis(AMR_Grounding_2DObj):
    def __init__(self, d_model=256, max_stride=32, pt_dir='/home/xhh/pt', work_dir=None, mode=None, loss_weight={}, tasks={ 'objdecoder': {} }, pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225], amrbart_wordEmbedding_freeze=True, amrtext_wordEmbedding_proj={ 'name': 'FeatureResizer','input_feat_size': 1024,'output_feat_size': 256,'dropout': 0,'do_ln': True }, obj_decoder={ 'name': None,'path': None,'freeze': True }, reason_module={}, temporal_decoder={}, fusion={}, use_we=False, loss_type='object') -> None:
        super().__init__(d_model, max_stride, pt_dir, work_dir, mode, loss_weight, tasks, pixel_mean, pixel_std, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, obj_decoder, reason_module, temporal_decoder, fusion, use_we, loss_type)

        assert 'sigmoid' in reason_module['name']


    def objdecoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = len(isvalid)
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss_referent(layer_gscore_output, matching_indices, targets, num_objs)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching_referent(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # lis[int], batch
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # list[int], b
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[[J]] for t, J in zip(tgt_masks, referent_idx)], dim=0) # n_simga h w
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss_referent(self, layer_gscore_output, matching_indices,  targets, num_refs):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        assert ref_is_valid.any()

        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}
    

    def temporal_decoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int]
        # list[ni t' h w] -> list[t' hw]
        num_refs = sum([tm[ref_idx].flatten(1).any(-1).int().sum() for tm, ref_idx in zip(tgt_masks, referent_idx)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()

        num_refs_video = len(tgt_masks)
        num_refs_video = torch.as_tensor([num_refs_video], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs_video)
        num_refs_video = torch.clamp(num_refs_video / get_world_size(), min=1).item()

        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers 
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_refs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss_referent(layer_gscore_output, targets, matching_indices, num_refs, num_refs_video)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching_referent(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int], batch
        src_masks_logits = out_mask_logits  # b nq T h w

        batch_size, nq, T, h, w = src_masks_logits.shape
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 t' H W
            tgt_mask = tgt_mask.to(out_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1

            C =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice
            C = C.cpu()
            indices.append(linear_sum_assignment(C))            
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_refs):
        has_anntation = targets['has_ann'] # list[T]
        ref_idx = targets['referent_idx'] # list[int], batch
        # b nq T H W -> list[1 t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[1 t' H W], b 
        tgt_masks = [t[[J]] for t, J in zip(targets['masks'], ref_idx)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# 1_t'_sigma h w
        tgt_masks = torch.cat([tm.flatten(0, 1) for tm in tgt_masks],dim=0) # 1_t'_sigma h w
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
        }
        return losses    

    def temporal_reason_loss_referent(self, layer_gscore_output, targets, matching_indices, num_refs, num_refs_video):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs_video}


class AMR_Grounding_3DObj(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                 work_dir=None,
                 mode=None,
                loss_weight={},
                tasks = { 'tempdecoder':{}},
                pixel_mean = [0.485, 0.456, 0.406],
                pixel_std = [0.229, 0.224, 0.225],
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {},
                 temporal_decoder = {},
                reason_module_3d={},
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.mode = mode
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        from models.pretrained_video_instance_decoder import pt_3d_obj_decoder_entrypoint
        create_temporal_decoder = pt_3d_obj_decoder_entrypoint(temporal_decoder['name'])
        self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir, work_dir)
        self.temporal_decoder_num_layers = self.temporal_decoder.num_layers # 4层
        self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
        self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
        if mode == 'rvos测试预训练vis':
            pass
        elif mode == '只训练rvos':
            self.reason_3d_choose = reason_module_3d['choose_who']
            self.reason_3d_layer_if_reason = reason_module_3d['layer_if_reason'] # obj_decoder的每层是否reason
            assert self.reason_3d_layer_if_reason[-1]
            assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers
            from .layer_graph import graphLayer_entrypoint
            create_reason_module = graphLayer_entrypoint(reason_module_3d['graph']['name'])
            self.reason_module_3d = create_reason_module(reason_module_3d['graph'])
        else:
            raise ValueError()
        
    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_ = samples.tensors.shape
        device = samples.tensors.device

        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        # TODO: 添加text 特征变换
        temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(samples, 
                                                                                     
                                                                                    amrs=amrs, 
                                                                                    amr_token_feats=amr_token_feats,
                                                                                    amr_token_seg_ids=amr_token_seg_ids, 
                                                                                    text_feats=text_feats, 
                                                                                    text_pad_masks=text_pad_masks)
         # l b nq c, l b t nqf c, l b nq T nqf
         # l b t nq h w,
        temporal_queries_by_layer, frame_queries_by_layer, cross_attn_weights_by_layer,\
              pred_masks_by_layer, multiscale_feats = temporal_decoder_output['video_queries'], temporal_decoder_output['frame_queries'], \
                                                            temporal_decoder_output['cross_attn_weights'],\
                                                         temporal_decoder_output['pred_masks'], temporal_decoder_output['multiscale_feats']

        if self.mode == 'rvos测试预训练vis':
            return {'tempdecoder': {'pred_masks': pred_masks_by_layer,}}
        elif self.mode == '只训练rvos':  
            grounding_score_by_layer = []
            for layer_idx, (frame_queries, temporal_queries, cross_attn_weights) in enumerate(zip(frame_queries_by_layer, 
                                                                                                  temporal_queries_by_layer, 
                                                                                                  cross_attn_weights_by_layer)):
                if self.reason_3d_layer_if_reason[layer_idx]:
                    grounding_score = self.reason_module_3d(temporal_queries=temporal_queries, 
                                                            frame_queries=frame_queries,
                                                             cross_attn_weights=cross_attn_weights, 
                                                             amrs=amrs,
                                                             amr_token_feats=amr_token_feats,
                                                             amr_token_seg_ids=amr_token_seg_ids,
                                                             node_alignments=node_alignments,
                                                             text_feats=text_feats,
                                                             text_pad_masks=text_pad_masks) # list[vi nq]
                    
                    if self.reason_3d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(torch.stack(grounding_score, dim=0))
                else:
                    grounding_score_by_layer.append(None)

            return {'temporal_decoder': {'pred_masks': pred_masks_by_layer, # list[b nq h w]
                                   'reason_3d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
        # b nq T h w
        out_mask_logits = decoder_layer_preds['objdecoder_objseg'].permute(0,2,1,3,4)
        if self.is_pretraining_seg:
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            # list[n t h w]
            tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
            for btc_idx in range(batch_size):
                start = int(self.obj_decoder_mask_out_stride // 2)
                im_h, im_w = tgt_masks[btc_idx].shape[-2:]
                tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
                assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
                assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
  

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape

        new_targets = self.rvos_targets_handler(targets, pad_h=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        loss_value_dict = self.temporal_decoder_loss(model_outs, new_targets)

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
        # list[ni]
        is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> list[T]
        perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

        return {
            'masks': tgt_masks,
            'is_valid': is_valid,
            'referent_idx': gt_referent_idx,
            'has_ann': perFrame_has_ann
        }

    def temporal_decoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idxs = targets['referent_idx'] # list[int]
        is_valid = targets['is_valid'] # list[ni]
        num_objs = sum([is_v.int().sum() for is_v in is_valid])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        num_refs = sum([is_v[ref_idx].int() for is_v, ref_idx in zip(is_valid, referent_idxs)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers     

        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        src_masks_logits = out_mask_logits  # b nq T h w
        batch_size, nq, T, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni

            C = self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
        has_anntation = targets['has_ann'].bool() # list[T]
        is_valid = targets['is_valid'].bool() # list[ni]
        # b nq T H W -> list[ni t' H W]
        src_masks = [t[J][:, has_ann] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]
        
        # list[ni t' H W], b 
        tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        masks_losses = torch.cat([self.binary_cross_entropy_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)
        dice_losses = torch.cat([self.dice_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)

        losses = {
            "tempdecoder_mask": masks_losses.sum() / num_objs,
            "tempdecoder_dice": dice_losses.sum() / num_objs,
        }
        return losses    

    def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss


    def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, num_refs):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'].bool() # list[ni]
        ref_is_valid = torch.cat([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]) # b
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = tgt_idx.tolist().index(ref_idx)
            match_as_gt_idx = src_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs}


@register_model
def amr_grounding_3dobj(device, configs):
    model = AMR_Grounding_3DObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        work_dir=configs['work_dir'],
        max_stride=configs['max_stride'],
        pixel_mean=configs['pixel_mean'],
        pixel_std=configs['pixel_std'],
        mode=configs['mode'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        temporal_decoder=configs['temporal_decoder'],
        reason_module_3d=configs['reason_module_3d']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder.backbone" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder.backbone" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] 
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.info('你没用任何scheduler')
        print('你没用任何scheduler')
        return model, optimizer, None, None
    
    if sch_conf['name'] == 'polynomial_split':
        from models.model_utils import polynomial_decay_lambda
        group_names = ['model', 'vbb', 'text']
        poly_lambdas = []
        for gname in group_names:
            g_poly_conf = sch_conf[gname]
            poly_lambdas.append(partial(polynomial_decay_lambda, initial_learning_rate=g_poly_conf['initial_learning_rate'],
                                                                        end_learning_rate=g_poly_conf['end_learning_rate'],
                                                                        decay_steps=g_poly_conf['decay_steps'],
                                                                        power=g_poly_conf['power']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=poly_lambdas,
                                                      last_epoch=-1,)
        return model, optimizer, scheduler, sch_conf['unit']
    elif sch_conf['name'] == 'polynomial_freezebb':
        from models.model_utils import polynomial_decay_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(polynomial_decay_lambda, 
                                                                        initial_learning_rate=sch_conf['initial_learning_rate'],
                                                                        end_learning_rate=sch_conf['end_learning_rate'],
                                                                        decay_steps=sch_conf['decay_steps'],
                                                                        power=sch_conf['power'],
                                                                        ),
                                                      last_epoch=-1,
                                                      verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'polynomial':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                                        total_iters=sch_conf['total_iters'],
                                                        power=sch_conf['power'],
                                                        last_epoch=-1,
                                                        verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=sch_conf['milestones'],
                                                        gamma=sch_conf['gamma'],
                                                        verbose=True), 
        return model, optimizer, scheduler, sch_conf['unit']
    
    elif sch_conf['name'] == 'invert_sqrt':
        from models.model_utils import inverse_sqrt_warmup_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=partial(inverse_sqrt_warmup_lambda,
                                                                         num_warmup_steps=sch_conf['num_warmup_steps'],
                                                                         num_training_steps=sch_conf['num_training_steps']), last_epoch=-1)
        return model, optimizer, scheduler, sch_conf['unit']
    else:
        raise ValueError()



class AMR_Grounding_2DObj_withPad(AMR_Grounding_2DObj):
    def __init__(self, d_model=256, max_stride=32, pt_dir='/home/xhh/pt', work_dir=None, mode=None, loss_weight={}, tasks={ 'objdecoder': {} }, pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225], amrbart_wordEmbedding_freeze=True, amrtext_wordEmbedding_proj={ 'name': 'FeatureResizer','input_feat_size': 1024,'output_feat_size': 256,'dropout': 0,'do_ln': True }, obj_decoder={ 'name': None,'path': None,'freeze': True }, reason_module_2d={}, temporal_decoder={}, reason_module_3d={}) -> None:
        super().__init__(d_model, max_stride, pt_dir, work_dir, mode, loss_weight, tasks, pixel_mean, pixel_std, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, obj_decoder, reason_module_2d, temporal_decoder, reason_module_3d)
        assert self.mode == '只训练rios'

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, targets=None):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 

        batch_size = samples.tensors.shape[0]
        device = samples.tensors.device

        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device) 


        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        obj_decoder_output = self.obj_decoder(samples, 
                                            text_feats=text_feats, 
                                            text_pad_masks=text_pad_masks,
                                            amr_feats=amr_token_feats,
                                            amr_pad_masks=amr_token_seg_ids==0)
        obj_queries_by_layer, pred_masks_by_layer, multiscale_feats = obj_decoder_output['obj_queries'], obj_decoder_output['pred_masks'],\
                                                                    obj_decoder_output['multiscale_feats'] # b nq c

        grounding_score_by_layer = []
        for layer_idx, (obj_queries, layer_pred_mask) in enumerate(zip(obj_queries_by_layer, pred_masks_by_layer)):
            if self.reason_2d_layer_if_reason[layer_idx]:
                obj_queries = self.reason_2d_obj_query_proj(obj_queries) # b nq c
                obj_queries_pad_mask = torch.zeros_like(obj_queries[:, :, 0]).bool() # b nq

                # list[vi nq] # 每个batch的每个amr node觉得的分数
                grounding_score = self.reason_module_2d(obj_queries=obj_queries.clone(),
                                                        obj_queries_pad_mask=obj_queries_pad_mask, # b nq c
                                                        amrs=amrs,
                                                        amr_token_seg_ids=amr_token_seg_ids,
                                                        amr_token_feats=amr_token_feats,
                                                        text_feats=text_feats, 
                                                        text_pad_masks=text_pad_masks,
                                                        node_alignments=node_alignments) 
                
                if self.reason_2d_choose == '第一个':
                    grounding_score = [lg[0] for lg in grounding_score]# list[vi nq] -> list[nq]
                else:
                    raise ValueError()
                assert ((obj_queries_pad_mask.float()) * (torch.stack(grounding_score, dim=0).softmax(-1))).sum() == 0
                grounding_score_by_layer.append(torch.stack(grounding_score, dim=0))
            else:
                grounding_score_by_layer.append(None)

        return {'objdecoder': {'pred_masks': pred_masks_by_layer, # b nq h w
                                'reason_2d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

    def ref_choose_2d_loss(self, layer_gscore_output, matching_indices,  targets):
        version = 'v3'
        if version == 'v2':
            # b nq
            is_valid = targets['isvalid'] # list[ni], batch
            referent_idx = targets['gt_referent_idx'] # list[int], batch
            ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
            assert ref_is_valid.any()
            choose_loss_by_batch = []
            for ref_val, ref_idx, (src_idx, tgt_idx), g_score in zip(ref_is_valid, referent_idx, matching_indices, layer_gscore_output):
                if ref_val:
                    # nq -> ni
                    pred = g_score[src_idx]
                    sel_idx = torch.tensor(tgt_idx.tolist().index(ref_idx)).to(self.device)
                    choose_loss_by_batch.append(F.cross_entropy(pred, sel_idx))
            return {'objdecoder_reason': torch.tensor(choose_loss_by_batch).mean()}
        elif version == 'v3':
            # b nq
            is_valid = targets['isvalid'] # list[ni], batch
            referent_idx = targets['gt_referent_idx'] # list[int], batch
            ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
            num_refs = (ref_is_valid.int().sum())
            match_as_gt_indices = [] # list[int], bt
            for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
                sel_idx = src_idx.tolist().index(ref_idx)
                match_as_gt_idx = tgt_idx[sel_idx]
                match_as_gt_indices.append(match_as_gt_idx.item())
            match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
            choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
            return {'objdecoder_reason': choose_loss.sum() / num_refs}
        elif version == 'v4':
            # b nq
            is_valid = targets['isvalid'] # list[ni], batch
            referent_idx = targets['gt_referent_idx'] # list[int], batch
            ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
            num_refs = (ref_is_valid.int().sum())
            gt_probabilities = torch.zeros_like(layer_gscore_output) # b nq
            for btc_idx, (ref_idx, (tgt_idx, src_idx)) in enumerate(zip(referent_idx,  matching_indices)): # b
                sel_idx = src_idx.tolist().index(ref_idx)
                match_as_gt_idx = tgt_idx[sel_idx]
                gt_probabilities[btc_idx][match_as_gt_idx] = 1.
            choose_loss = F.binary_cross_entropy_with_logits(layer_gscore_output[ref_is_valid], gt_probabilities[ref_is_valid], 
                                                             reduction='none') # b nq
            return {'objdecoder_reason': choose_loss.sum() / num_refs}

        elif version == 'v5':
            weight = 0.1
            # b nq
            is_valid = targets['isvalid'] # list[ni], batch
            referent_idx = targets['gt_referent_idx'] # list[int], batch
            ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
            num_refs = (ref_is_valid.int().sum())
            gt_probabilities = torch.zeros_like(layer_gscore_output) # b nq
            weights = torch.ones_like(layer_gscore_output) * weight
            for btc_idx, (ref_idx, (tgt_idx, src_idx)) in enumerate(zip(referent_idx,  matching_indices)): # b
                sel_idx = src_idx.tolist().index(ref_idx)
                match_as_gt_idx = tgt_idx[sel_idx]
                gt_probabilities[btc_idx][match_as_gt_idx] = 1.
                weights[btc_idx][match_as_gt_idx] = 1.
            # b nq
            choose_loss = F.binary_cross_entropy_with_logits(layer_gscore_output[ref_is_valid], gt_probabilities[ref_is_valid], 
                                                             weight=weights,) # b nq
            return {'objdecoder_reason': choose_loss}

    # 这是之前的
    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, perFrame_has_ann, tgt_masks):
        # b nq T h w
        # b T
        # list[n t h w]
        src_masks_logits = out_mask_logits  # b nq T h w
        batch_size, nq, T, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t h w
            tgt_mask = tgt_masks[i].to(out_mask) # n t H W
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder_objseg']['matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder_objseg']['matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_objseg_loss(self, out_mask_logits, perFrame_has_ann, tgt_masks):
        # b nq T h w
        # b T
        # list[n t h w]
        loss_weight = self.loss_weight
        # n thw -> n
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),}
        
        matching_indices = self.obj_decoder_matching(out_mask_logits, perFrame_has_ann, tgt_masks)
        if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
            masks_losses = self.obj_decoder_masks_loss(out_mask_logits, perFrame_has_ann, tgt_masks, matching_indices, num_boxes)
            for k in masks_losses.keys():
                loss_value[k] += masks_losses[k]
        return loss_value, matching_indices       

    def temporal_decoder_masks_loss(self, out_mask_logits, perFrame_has_ann, tgt_masks, matching_indices, num_objs):
        # b nq T h w
        # b T
        # list[n t h w], b
        batch_size = len(out_mask_logits)

        # list[nq T h w], b -> list[ni T h w] 
        src_masks = [t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)]
        # list[ni ti h w], b 
        tgt_masks = [t[J] for t, (_, J) in zip(tgt_masks, matching_indices)]

        ce_losses = []
        dice_losses = []
        for btc_idx in range(batch_size):
            ce_losses.append(self.binary_cross_entropy_mask_loss(src_masks[btc_idx], perFrame_has_ann, tgt_masks[btc_idx]))
            dice_losses.append(self.dice_mask_loss(src_masks[btc_idx], perFrame_has_ann, tgt_masks[btc_idx]))

        losses = {
            "objdecoder_mask": torch.cat(ce_losses).sum() / num_objs,
            "objdecoder_dice": torch.cat(dice_losses).sum() / num_objs,
        }
        return losses    

    def ref_choose_loss(self, refseg_src, tgt_masks, referent_idx, is_valid,  decoder_last_layer_matching_results, obj_decoder_out, perFrame_has_ann):
        """
        Args:
            refseg_src: dict{layer-1pred: {queries: bt c}}
            list[n t h w], batch
            list[src, tgt], batch
            list[n t], batch
        """
        device=tgt_masks[0].device 
        # thw.any()
        num_boxes = sum([t[refidx].flatten().any().int() for t, refidx in zip(tgt_masks,referent_idx)])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b

        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  decoder_last_layer_matching_results): # b
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(device) # b
        gt_probs = self.get_gt_class_prob(match_as_gt_indices, obj_decoder_out, perFrame_has_ann) # b nq
        gt_probs = gt_probs[ref_is_valid]
        refdecoder_choose_loss = 0.
        for layer_idx in range(-1, self.decoder_trans_nlayers):
            layer_weight = layer_weights[layer_idx] 
            if layer_weight != 0: # bt c
                refdecoder_gscore = refseg_src[f'layer{layer_idx}_preds']['grounding_score'] # b nq
                refdecoder_gscore = refdecoder_gscore[ref_is_valid]
                choose_loss = F.cross_entropy(refdecoder_gscore, gt_probs, reduction='none') # b
                choose_loss = choose_loss.sum() / num_boxes
                refdecoder_choose_loss += (choose_loss * layer_weight)

        return {'refdecoder_choose': refdecoder_choose_loss}

    def binary_cross_entropy_mask_loss(self, src_masks, has_ann, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks[:, has_ann].flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, has_ann, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks[:, has_ann].flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss


@register_model
def amr_grounding_2dobj_pad(device, configs):
    model = AMR_Grounding_2DObj_withPad(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        work_dir=configs['work_dir'],
        max_stride=configs['max_stride'],
        pixel_mean=configs['pixel_mean'],
        pixel_std=configs['pixel_std'],
        mode=configs['mode'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        obj_decoder=configs['obj_decoder'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        reason_module_2d=configs['reason_module_2d'], 
        temporal_decoder=configs['temporal_decoder'],
        reason_module_3d=configs['reason_module_3d']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder.backbone" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder.backbone" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 
















class AMR_v0_detOnlyObj_Groudning_multiReason(AMR_v0_detOnlyObj_Grounding):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'nlayers': 6,},
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },
            is_pretraining_seg=False,
            detach_refdecoder_memory=False
            ) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder, is_pretraining_seg, detach_refdecoder_memory)


    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b smax c
        fusion_mem = torch.cat([text_feats, amr_token_feats], dim=1) 
        fusion_mem_pad_mask = torch.cat([text_pad_masks, amr_token_seg_ids==0], dim=-1)
        fusion_mem_pos = torch.cat([text_pos, torch.zeros_like(amr_token_feats)], dim=1)     
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=fusion_mem.permute(1,0,2), 
                                                    memory_key_padding_mask=fusion_mem_pad_mask,
                                                    pos=fusion_mem_pos.permute(1,0,2), 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann]
        text_feats = repeat(text_feats, 'b s c -> (b t) s c',t=nf)[perFrame_has_ann]
        # list[list[int], vi], batch
        # batch -> bt
        repeated_node_alignments = [] 
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_node_alignments.append(copy.deepcopy(node_alignments[idx]))
        filtered_rnas = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_rnas.append(repeated_node_alignments[idx])
        assert len(filtered_rnas) != 0
        node_alignments = filtered_rnas
        
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _ = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, query_embed, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                        [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                        [scale_pos.clone() for scale_pos in multiscales_poses])
        if self.is_pretraining_seg:
            return {'objdecoder_objseg': objdecoder_layer_preds}
        else:
            memories = obj_queries
            memories_pos = query_embed
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index, node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), memories_pos.permute(1,0,2).clone(),
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        # list[list[nq], number_i], batch
        # 每个sample有多个grouding结果, 根据近邻个数的不同
        grounding_scores = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) 
        decoder_layer_preds[f'layer{-1}_preds'] = {'grounding_scores': grounding_scores}

        assert self.decoder_trans_layers is None
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

class AMR_v0_detOnlyObj_Grounding_weightedQuery(AMR_v0_detOnlyObj_Grounding):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                objdecoder={ 
                    'num_classes': 7,
                    'nqueries': 100,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                is_pretraining_seg=False,
                detach_refdecoder_memory=False,
                freeze_obj_decoder=False,
                adpt=None,
                ) -> None:
        super().__init__(d_model, max_stride, pt_dir,
                          swint_pretrained_path, swint_freeze, swint_runnning_mode,
                            video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, 
                            amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, 
                            tasks, refdecoder, objdecoder, is_pretraining_seg, detach_refdecoder_memory,
                            freeze_obj_decoder=freeze_obj_decoder,)
        self.use_adpt = False
        if adpt is not None:
            self.use_adpt = True
            self.adpt_param = adpt

    def forward_obj_decoder(self, memories, memories_poses, memories_pad_masks, conved_features):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries':output.clone() }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=None, # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':output.clone() }
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return output, query_embed, decoder_layer_preds # n bt c

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size,
                                 return_attn_mask=True):
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        if return_attn_mask:
            attn_mask = outputs_mask
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
            
            return outputs_classes, outputs_mask, outputs_box, attn_mask
        else:
            return outputs_classes, outputs_mask, outputs_box

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b smax c  
        if self.cross_product is not None:
            fusion_mem = torch.cat([text_feats, amr_token_feats], dim=1) 
            fusion_mem_pad_mask = torch.cat([text_pad_masks, amr_token_seg_ids==0], dim=-1)
            fusion_mem_pos = torch.cat([text_pos, torch.zeros_like(amr_token_feats)], dim=1)   
            for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
                bs, nf, _, h, w = feat.shape
                feat = rearrange(feat, 'b t c h w -> (t h w) b c')
                poses = rearrange(poses, 'b t c h w -> (t h w) b c')
                feat, attn_weight = self.cross_product(tgt=feat,
                                                        memory=fusion_mem.permute(1,0,2), 
                                                        memory_key_padding_mask=fusion_mem_pad_mask,
                                                        pos=fusion_mem_pos.permute(1,0,2), 
                                                        query_pos=poses)
                check_visualize[f'scale{lvl} attention weights'] = attn_weight
                multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann]
        text_feats = repeat(text_feats, 'b s c -> (b t) s c',t=nf)[perFrame_has_ann]
        # list[list[int], vi], batch
        # batch -> bt
        repeated_node_alignments = [] 
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_node_alignments.append(copy.deepcopy(node_alignments[idx]))
        filtered_rnas = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_rnas.append(repeated_node_alignments[idx])
        assert len(filtered_rnas) != 0
        node_alignments = filtered_rnas
        
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _ = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # bt c h w
        obj_memories, obj_memories_poses, obj_memories_pad_masks, obj_conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        # n bt c, bt n,
        obj_queries, query_embed, objdecoder_layer_preds = self.forward_obj_decoder(obj_memories, obj_memories_poses, obj_memories_pad_masks, obj_conved_features)
        if self.is_pretraining_seg:
            return {'objdecoder_objseg': objdecoder_layer_preds}
        if self.detach_refdecoder_memory:
            memories = obj_queries.detach() # nq bt c
            memories_pos = query_embed.detach() # nq bt c
        else:
            memories = obj_queries
            memories_pos = query_embed
        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats, \
            node_memories,edge_memories,\
            edge_index,  node_subseqs, node_dsends = \
              batching_graph(amrs, amr_token_feats, amr_token_seg_ids, memories.permute(1,0,2).clone(), memories_pos.permute(1,0,2).clone(),
                            text_feats, node_alignments) # memories是dict

        decoder_layer_preds = {}
        grounding_score = self.decoder_reason_layer(node_batch_ids=nodes_batch_ids, edge_batch_ids=edges_batch_ids, 
                                                node_seg_ids=node_seg_ids, edge_seg_ids=edges_seg_ids,
                                                node_feats=node_feats, edge_feats=edge_feats,
                                                node_memories=node_memories, edge_memories=edge_memories,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends) # V nq
        weighted_query_by_batch = []
        for bch_idx in range(bt):
            # vi nq
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            bch_node_probs = bch_node_score.softmax(-1)
            bch_queries = obj_queries[:, bch_idx] # nq c
            # vi c
            weighted_query = bch_node_probs @ bch_queries
            if self.decoder_reason_layer_choose_who == '第一个':
                weighted_query = weighted_query[0] # c
            weighted_query_by_batch.append(weighted_query) # c
        weighted_query_by_batch = torch.stack(weighted_query_by_batch, dim=0).unsqueeze(0) # 1 bt c
        # bt 1 class, bt 1 h w, bt 1 4
        out_class, out_mask, out_box = self.forward_objdecoder_heads(weighted_query_by_batch, obj_conved_features, attn_mask_target_size=None,
                                                                     return_attn_mask=False)
        
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask.squeeze(1), 'pred_box_logits': out_box.squeeze(1), 'pred_class_logits': out_class}
    
        assert self.decoder_trans_layers is None
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict, matching_result = self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets)

        if not self.is_pretraining_seg:
            refseg_src = model_outs['refdecoder_refseg']
            refseg_losses = self.refdecoder_refseg_loss(refseg_src, targets)
            if self.use_adpt:
                # list[([], [])], batch
                objdecoder_last_matching_res = matching_result[-1]
                target_masks = obj_decoder_targets['masks'] # bt n H W
                objseg_pred_masks = objseg_src[f'layer{self.obj_decoder_nlayers-1}_preds']['pred_mask_logits'].detach()
                objseg_pred_masks = (objseg_pred_masks.sigmoid() > 0.5)
                ious = []
                for btc_idx, (src_indices, tgt_indices) in enumerate(objdecoder_last_matching_res):
                    btc_tgt_masks = target_masks[btc_idx][tgt_indices].flatten(1).float() # n hw
                    btc_src_masks = objseg_pred_masks[btc_idx][src_indices].flatten(1).float() # n hw
                    numerator = (btc_src_masks * btc_tgt_masks).sum(1)
                    denominator = btc_src_masks.sum(-1) + btc_tgt_masks.sum(-1)
                    iou = numerator / (denominator + 1e-5) # n
                    for iiou in iou:
                        assert (iiou <= 1.) and (iiou >=0.)
                    ious.append(iou)
                mean_iou = torch.cat(ious).mean() # 0-1
                weight = self.get_adpt_weight(mean_iou)
            else:
                weight = 1.
            for loss_key, loss_value in refseg_losses.items():
                refseg_losses[loss_key] = loss_value * weight
            loss_value_dict.update(refseg_losses)

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    def get_adpt_weight(self, mean_iou):

        if self.adpt_param['name'] == 'sigmoid':
            gamma = self.adpt_param['gamma']
            center =self.adpt_param['center']
            return 1 / (1 + math.exp(-gamma*(mean_iou - center)))
        else:
            raise ValueError()

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        objseg_preds = decoder_layer_preds['objdecoder_objseg']
        obj_last_layer_preds = objseg_preds[f'layer{self.obj_decoder_nlayers-1}_preds']

        if self.is_pretraining_seg:
            out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            obj_decoder_targets = self.obj_decoder_targets_handler(targets)
            _, matching_result = self.obj_decoder_objseg_loss(objseg_preds, obj_decoder_targets)
            matching_result = matching_result[-1] # list(tgt, src), bt
            gt_referent_idx = obj_decoder_targets['referent_idx'] # list[int], bt
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]] 
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx) in zip(out_mask_logits, gt_referent_idx, matching_result)], dim=0)
        else:
            ref_preds = decoder_layer_preds['refdecoder_refseg'][f'layer{self.decoder_trans_nlayers-1}_preds']
            out_mask_logits = ref_preds['pred_mask_logits'] # bt h w
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }



    # task loss
    def refdecoder_refseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        loss_weight = self.loss_weight
        if 'mask_loss_type' in self.tasks['refdecoder_refseg']:
            mask_loss_type = self.tasks['refdecoder_refseg']['mask_loss_type']
        else:
            mask_loss_type = 'ce'
        
        # list[t] -> bt
        target_valid = torch.cat([t["valid"][t['referent_idx']] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        

        loss_value = {'refdecoder_mask': torch.tensor(0, device=target_valid.device).float(), 
                      'refdecoder_bbox': torch.tensor(0, device=target_valid.device).float(), 
                      'refdecoder_giou': torch.tensor(0, device=target_valid.device).float(),
                      'refdecoder_dice': torch.tensor(0, device=target_valid.device).float(), }
        for i in range(-1, self.decoder_trans_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT H W
                if loss_weight['refdecoder_mask'] != 0 or loss_weight['refdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detOnlyObj_Grounding_weightedQuery.refdecoder_masks_loss(layer_pred, targets, num_boxes, self.obj_decoder_mask_out_stride, mask_loss_type)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['refdecoder_bbox'] != 0 or loss_weight['refdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detOnlyObj_Grounding_weightedQuery.refdecoder_boxes_loss(layer_pred, targets, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
        return loss_value         

    @staticmethod
    def refdecoder_boxes_loss(outputs, targets,num_boxes): 
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        src_boxes = outputs['pred_box_logits'].sigmoid()  # bt 4
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0).to(src_boxes)  
        
        src_boxes = src_boxes[is_consistent]  # bt 4
        target_boxes = target_boxes[is_consistent] # bt 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['refdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['refdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    @staticmethod
    def refdecoder_masks_loss(outputs, targets, num_boxes, decoder_mask_out_stride, mask_loss_type):
        # list[n t] -> list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        
        src_masks = outputs["pred_mask_logits"]  # bT h w  )
        
        # list[n t h w] -> list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0).to(src_masks) # list[t h w] -> bt h w

        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        
        src_masks = src_masks[is_consistent].flatten(1) # bt hw
        target_masks = target_masks[is_consistent].flatten(1) # bt hw
        
        if mask_loss_type == 'ce':
            mask_loss = ce_mask_loss(src_masks, target_masks, num_boxes)
        elif mask_loss_type == 'focal':
            mask_loss = sigmoid_focal_loss(src_masks, target_masks, num_boxes)
        losses = {
            "refdecoder_mask": mask_loss,
            "refdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses 

class AMR_v0_detOnlyObj_Grounding_AsObjLoss(AMR_v0_detOnlyObj_Grounding):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'nlayers': 6,},
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },
            is_pretraining_seg=False,
            detach_refdecoder_memory=False,
            layer_if_choose=None
            ) -> None:
        assert refdecoder['trans_layer']['name'] == 'none'
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder, is_pretraining_seg, detach_refdecoder_memory)
        from .layer_graph import graphLayer_entrypoint
        ysyp_config = refdecoder['get_ysyp']
        ysyp_name = ysyp_config.pop('name')
        create_ysyp = graphLayer_entrypoint(ysyp_name)
        self.decoder_ysyp = create_ysyp(ysyp_config)
        from torch_geometric.nn.inits import glorot
        glorot(self.obj_decoder_query_embed.weight)
        glorot(self.obj_decoder_query_feats.weight)
        if layer_if_choose is None:
            self.layer_if_choose = [True] * (self.obj_decoder_nlayers + 1)
        else:
            self.layer_if_choose = layer_if_choose
        assert self.layer_if_choose[-2]

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
                                 
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        bt = decoder_output.shape[0]
        
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        return outputs_classes, outputs_mask, outputs_box, attn_mask

    def get_batch_gscore(self, node_gscore, nodes_batch_ids, bt):
        gscore_by_batch = []
        for bch_idx in range(bt):
            bch_node_score = torch.stack([node_gscore[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            gscore_by_batch.append(bch_node_score) # vi nq
        return gscore_by_batch

    def batching_graph(self, amrs,
                    amr_token_feats,
                    amr_seg_ids,
                    text_feats, node_alignments
                    ):
        """
        Args:
            amrs: list[Graph]
            amr_token_feats: b (v+e)max c
            amr_seg_ids: b (v+e)max
            memories: b nq c
            memories_pos: b nq c
            text_feats: b smax c
            node_alignments: list[list[int], si] batch
        Returns:
            _type_: _description_
        """
        device = amr_token_feats.device
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        nodes_batch_ids = torch.tensor(nodes_batch_ids, device=device)
        edges_batch_ids = torch.tensor(edges_batch_ids, device=device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        edge_index = batched_amrs.edge_index.to(device)

        node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0)
        edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0)

        node_subseqs = [] # list[s c], V
        for btc_text_feat, btc_node_alis in zip(text_feats, node_alignments):
            # s c, list[int]
            for node_ali in btc_node_alis:
                node_subseqs.append(btc_text_feat[:(node_ali+1)])

        node_dsends = [] # list[si c], V
        icgd = list(zip(edge_index[0, :].tolist(), edge_index[1, :].tolist()))
        nx_graph = nx.DiGraph(icgd)
        for node_id in range(len(nodes_batch_ids)):
            # s c, list[int]
            dsends = list(nx.descendants(nx_graph, node_id))
            dsends = [node_id] + dsends
            node_dsends.append(node_feats[dsends])  

        return nodes_batch_ids, edges_batch_ids, \
            node_seg_ids, edges_seg_ids, \
                node_feats, edge_feats, edge_index, node_subseqs, node_dsends

    def batching_memory(self, nodes_batch_ids,  edges_batch_ids,
                        memories, # 
                        memories_pos=None,
                        ):
        # V nq c
        node_memories_feats = torch.stack([memories[bid] for bid in nodes_batch_ids], dim=0)
        node_memories_poses = torch.stack([memories_pos[bid] for bid in nodes_batch_ids], dim=0) if memories_pos is not None else None

        edge_memories_feats = torch.stack([memories[bid] for bid in edges_batch_ids], dim=0)
        edge_memories_poses = torch.stack([memories_pos[bid] for bid in edges_batch_ids], dim=0) if memories_pos is not None else None

        node_memories = {'feat': node_memories_feats, 'pos': node_memories_poses}
        edge_memories = {'feat': edge_memories_feats, 'pos': edge_memories_poses}


        return node_memories, edge_memories

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b smax c
        fusion_mem = torch.cat([text_feats, amr_token_feats], dim=1) 
        fusion_mem_pad_mask = torch.cat([text_pad_masks, amr_token_seg_ids==0], dim=-1)
        fusion_mem_pos = torch.cat([text_pos, torch.zeros_like(amr_token_feats)], dim=1)     
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=fusion_mem.permute(1,0,2), 
                                                    memory_key_padding_mask=fusion_mem_pad_mask,
                                                    pos=fusion_mem_pos.permute(1,0,2), 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann]
        text_feats = repeat(text_feats, 'b s c -> (b t) s c',t=nf)[perFrame_has_ann]
        # list[list[int], vi], batch
        # batch -> bt
        repeated_node_alignments = [] 
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_node_alignments.append(copy.deepcopy(node_alignments[idx]))
        filtered_rnas = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_rnas.append(repeated_node_alignments[idx])
        assert len(filtered_rnas) != 0
        node_alignments = filtered_rnas
        
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _ = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        if not self.is_pretraining_seg:
            node_batch_ids, edge_batch_ids,\
                node_seg_ids, edge_seg_ids, \
                node_feats, edge_feats, \
                edge_index, node_subseqs, node_dsends = \
                self.batching_graph(amrs, amr_token_feats, amr_token_seg_ids,text_feats, node_alignments) # memories是dict
            ys_node_feats, yp_node_feats = self.decoder_ysyp(node_feats=node_feats, edge_feats=edge_feats,
                                                            edge_index=edge_index,
                                                            node_subseqs=node_subseqs,
                                                            node_dsends=node_dsends)

        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 
                                                   'pred_mask_logits': out_mask, 'pred_box_logits': out_box}
                
        if (not self.is_pretraining_seg) and (self.layer_if_choose[-1]):
            # 没Norm?
            # node_memories, edge_memories = self.batching_memory(nodes_batch_ids=node_batch_ids, edges_batch_ids=edge_batch_ids,
            #                                                 memories=output.permute(1,0,2), memories_pos=query_embed.permute(1,0,2))
            node_memories, edge_memories = self.batching_memory(nodes_batch_ids=node_batch_ids, edges_batch_ids=edge_batch_ids,
                                                            memories=self.obj_decoder_norm(output).permute(1,0,2),
                                                            memories_pos=query_embed.permute(1,0,2))
            node_gscore = self.decoder_reason_layer(node_batch_ids=node_batch_ids.clone(), edge_batch_ids=edge_batch_ids.clone(), 
                                                node_seg_ids=node_seg_ids.clone(), edge_seg_ids=edge_seg_ids.clone(),
                                                node_feats=node_feats.clone(), edge_feats=edge_feats.clone(),
                                                edge_index=edge_index.clone(),
                                                node_subseqs=[nsub.clone() for nsub in node_subseqs],
                                                node_dsends=[ndes.clone() for ndes in node_dsends],
                                                ys_node_feats=ys_node_feats.clone(),
                                                yp_node_feats=yp_node_feats.clone(),
                                                node_memories=node_memories,
                                                edge_memories=edge_memories) # V nq
            gscore_by_batch = self.get_batch_gscore(node_gscore, node_batch_ids, bt=bt)
            decoder_layer_preds[f'layer{-1}_preds']['grounding_score'] = gscore_by_batch

        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)],)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                    'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}             
            if (not self.is_pretraining_seg) and (self.layer_if_choose[i]):
                # 没Norm?
                # node_memories, edge_memories = self.batching_memory(nodes_batch_ids=node_batch_ids, edges_batch_ids=edge_batch_ids,
                #                                                 memories=output.permute(1,0,2), memories_pos=query_embed.permute(1,0,2))
                node_memories, edge_memories = self.batching_memory(nodes_batch_ids=node_batch_ids, edges_batch_ids=edge_batch_ids,
                                                            memories=self.obj_decoder_norm(output).permute(1,0,2),
                                                            memories_pos=query_embed.permute(1,0,2))
                node_gscore = self.decoder_reason_layer(node_batch_ids=node_batch_ids.clone(), edge_batch_ids=edge_batch_ids.clone(), 
                                            node_seg_ids=node_seg_ids.clone(), edge_seg_ids=edge_seg_ids.clone(),
                                            node_feats=node_feats.clone(), edge_feats=edge_feats.clone(),
                                            edge_index=edge_index.clone(),
                                            node_subseqs=[nsub.clone() for nsub in node_subseqs],
                                            node_dsends=[ndes.clone() for ndes in node_dsends],
                                            ys_node_feats=ys_node_feats.clone(),
                                            yp_node_feats=yp_node_feats.clone(),
                                            node_memories=node_memories,
                                            edge_memories=edge_memories) # V nq
                gscore_by_batch = self.get_batch_gscore(node_gscore, node_batch_ids, bt=bt)
                decoder_layer_preds[f'layer{i}_preds']['grounding_score'] = gscore_by_batch
            
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return {'objdecoder_objseg': decoder_layer_preds}


    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        objseg_preds = self.get_obj_decoder_preds(decoder_layer_preds)
        obj_last_layer_preds = objseg_preds[f'layer{self.obj_decoder_nlayers-1}_preds']

        if self.is_pretraining_seg:
            out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            obj_decoder_targets = self.obj_decoder_targets_handler(targets)
            _, matching_result = self.obj_decoder_objseg_loss(objseg_preds, obj_decoder_targets)
            matching_result = matching_result[-1] # list(tgt, src), bt
            gt_referent_idx = obj_decoder_targets['referent_idx'] # list[int], bt
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]] 
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx) in zip(out_mask_logits, gt_referent_idx, matching_result)], dim=0)
        else:
            out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
            out_gscore = obj_last_layer_preds['grounding_score'] # bt nq
            argmax_query_idx = out_gscore.argmax(-1)
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }


    def get_obj_decoder_preds(self, model_outs):
        objseg_src = model_outs['objdecoder_objseg']
        if self.is_pretraining_seg:
            return objseg_src
        if self.decoder_reason_layer_choose_who == '第一个':
            for i in range(-1, self.obj_decoder_nlayers):
                # list[vi nq], b -> b nq
                if 'grounding_score' in objseg_src[f'layer{i}_preds']:
                    layer_gscore = objseg_src[f'layer{i}_preds']['grounding_score']
                    layer_gscore = torch.stack([lg[0] for lg in layer_gscore], dim=0)
                    objseg_src[f'layer{i}_preds']['grounding_score'] = layer_gscore
        return objseg_src  

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        objseg_src = self.get_obj_decoder_preds(model_outs)
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict, matching_result = self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets)

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
   
    # task loss
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n h w], bt
        tgt_masks = targets['masks']
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        referent_idx = targets['referent_idx'] # list[int], bt
        num_ref_boxes = sum([t[refidx].flatten().any().int() for t, refidx in zip(tgt_masks,referent_idx)])
        num_ref_boxes = torch.as_tensor([num_ref_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_ref_boxes)
        num_ref_boxes = torch.clamp(num_ref_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_class': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_choose': torch.tensor(0, device=tgt_masks[0].device).float()}
        matching_indices_by_layer = []
        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = AMR_v0_detOnlyObj_Grounding.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.obj_decoder_mask_out_stride)
                matching_indices_by_layer.append(layer_matching_indices)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.obj_decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detOnlyObj_Grounding.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]
                if (loss_weight['objdecoder_choose'] != 0) and ('grounding_score' in layer_pred):
                    chhose_losses = self.ref_choose_loss(layer_pred, targets, layer_matching_indices, num_ref_boxes)
                    for k in chhose_losses.keys():
                        loss_value[k] += layer_weight * chhose_losses[k]
        return loss_value,matching_indices_by_layer 


    def ref_choose_loss(self, layer_pred, targets, decoder_last_layer_matching_results, num_boxes):
        tgt_masks = targets['masks']
        referent_idx = targets['referent_idx'] # list[int], bt
        device=tgt_masks[0].device 
        is_valid = targets['is_valid'] # list[ni], bt
        ref_is_valid = torch.tensor([isva[ridx] for isva, ridx in zip(is_valid, referent_idx)]).bool() # bt
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  decoder_last_layer_matching_results): # bt
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(device) # bt
        match_as_gt_indices = match_as_gt_indices[ref_is_valid]
        
        reason_gscore = layer_pred['grounding_score'] # bt nq
        reason_gscore = reason_gscore[ref_is_valid]
        choose_loss = F.cross_entropy(reason_gscore, match_as_gt_indices, reduction='none') # bt
        choose_loss = choose_loss.sum() / num_boxes
        return {'objdecoder_choose': choose_loss}


# 只有obj进行fusion
class AMR_v0_detectObj_onlyObj_fusionObj(AMR_v0_detectObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },
            fusion={
                'name': 'self_encoder',
                'd_model': 256,
                'nheads': 8,
                'dropout': 0.1,
                'nlayers': 6},
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            ) -> None:
        assert fusion['name'] == 'self_encoder'
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)

    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries':output.clone() }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':output.clone() }
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), query_embed, decoder_layer_preds # n bt c


    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c
        obj_queries, query_embeds, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])

        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        assert self.ref_parsing_encoder is None

        fusion_encoder_input = torch.cat([obj_queries, amr_fusion_tokens], dim=0)
        fusion_encoder_pos = torch.cat([query_embeds, torch.zeros_like(amr_fusion_tokens)], dim=0)
        queries_pads = torch.zeros_like(obj_queries[..., 0]).permute(1,0).bool()
        fusion_encoder_pad_mask = torch.cat([queries_pads, amr_fusion_pad_masks], dim=1)

        fusion_encoder_input = self.cross_product(src=fusion_encoder_input,
                                mask=None,
                                src_key_padding_mask=fusion_encoder_pad_mask,
                                pos=fusion_encoder_pos)
        
        memories = fusion_encoder_input[:len(obj_queries)] # nq bt c  
        memories_pos = query_embeds       
        
        # 准备decoder的东西  thw b c
        *_, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=None)
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box}
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=memories_pos,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features,attn_mask_target_size=None)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

# obj和vidoe都进行fusion
class AMR_v0_detectObj_onlyObj_fusionBoth(AMR_v0_detectObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },
            obj_fusion={
                'name': 'self_encoder',
                'd_model': 256,
                'nheads': 8,
                'dropout': 0.1,
                'nlayers': 6},) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)
        self.obj_fusion_amr_who_cross = obj_fusion.pop('amr_cross', None)
        fusion_name = obj_fusion.pop('name')
        self.obj_queries_fusion = get_fusion(fusion_name, obj_fusion)

    def get_objfusion_amr_cross(self, amr_token_feats, amr_token_seg_ids):
        # b (V+E)max c, b (V+E)max
        if self.obj_fusion_amr_who_cross == '只有2/3':
            who_fuse_with_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
        elif self.obj_fusion_amr_who_cross == '所有':
            who_fuse_with_video =  (amr_token_seg_ids!=0)
        else:
            raise ValueError()
        amr_fusion_tokens = [bt_feat[who_cross] for bt_feat, who_cross in zip(amr_token_feats, who_fuse_with_video)]
        amr_fusion_tokens, amr_fusion_pad_masks = pad_1d_feats(amr_fusion_tokens)
        return amr_fusion_tokens.permute(1, 0, 2), amr_fusion_pad_masks
    
    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box, 'queries':output.clone() }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':output.clone() }
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), query_embed, decoder_layer_preds # n bt c

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, query_embeds, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_objfusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)

        if self.obj_queries_fusion is not None:
            fusion_encoder_input = torch.cat([obj_queries, amr_fusion_tokens], dim=0)
            fusion_encoder_pos = torch.cat([query_embeds, torch.zeros_like(amr_fusion_tokens)], dim=0)
            queries_pads = torch.zeros_like(obj_queries[..., 0]).permute(1,0).bool()
            fusion_encoder_pad_mask = torch.cat([queries_pads, amr_fusion_pad_masks], dim=1)

            fusion_encoder_input = self.obj_queries_fusion(src=fusion_encoder_input,
                                    mask=None,
                                    src_key_padding_mask=fusion_encoder_pad_mask,
                                    pos=fusion_encoder_pos)
            
            memories = fusion_encoder_input[:len(obj_queries)] # nq bt c  
            memories_pos = query_embeds        
        else:
            memories = obj_queries
            memories_pos = query_embeds
        
        # 准备decoder的东西  thw b c
        *_, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=None)
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box}
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=memories_pos,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features,attn_mask_target_size=None)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

class AMR_v0_detectObj_onlyObj_fusionAsLoss(AMR_v0_detectObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            amrbart_freeze= True,
            fusion={'name': 'none',},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },) -> None:
        assert fusion['name'] == 'none'
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)
        proj_to_amrbart = objdecoder['proj_to_amrbart']
        assert proj_to_amrbart.pop('name') == 'Linear'
        self.obj_decoder_proj_to_amrbart = nn.Linear(**proj_to_amrbart)
        from .amr_utils.utils import BartForConditionalGeneration
        self.amrbart_model = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        assert amrbart_freeze
        for p in self.amrbart_model.parameters():
            p.requires_grad_(False)
        self.amrbart_model_encoder = self.amrbart_model.model.encoder
        from models.amr_utils.tokenization_bart import AMRBartTokenizer
        self.amrbart_tokenizer = AMRBartTokenizer.from_pretrained(os.path.join(self.pt_dir,'amr','AMRBART_pretrain'))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_decoder_refer_pos = nn.Embedding(1, self.amrbart_model.config.hidden_size)

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c

        query_as_amrbarts = self.obj_decoder_proj_to_amrbart(decoder_output) # bt n 1024
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_classes, outputs_mask, outputs_box, attn_mask, query_as_amrbarts


    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask, query_to_amrbarts = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, \
                                                   'pred_mask_logits': out_mask, 'pred_box_logits': out_box,\
                                                      'queries':self.obj_decoder_norm(output),
                                                       'query_to_amrbart':query_to_amrbarts  }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask, query_to_amrbarts = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 
                                                      'pred_mask_logits': out_mask,
                                                        'pred_box_logits': out_box,
                                                         'queries':self.obj_decoder_norm(output),
                                                         'query_to_amrbart':query_to_amrbarts }
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), query_embed, decoder_layer_preds # n bt c


    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _ = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, query_embed, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        assert self.ref_parsing_encoder is None          
        memories = obj_queries # nq bt c  
        memories_pos = query_embed  
        *_, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=None)
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=memories_pos,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, _ = self.forward_refdecoder_heads(amr_token_feats, conved_features,attn_mask_target_size=None)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box,}
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

    def obj_decoder_targets_handler(self, targets, auxiliary):
        # list[n h w], bt
        # list[n t h w] -> list[n h w], bt
        batch_size = len(targets)
        target_masks = []
        for bth_idx in range(batch_size):
            # n t h w
            t_m = targets[bth_idx]["masks"].split(1, dim=1) # list[n 1 h w], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n h w]
            target_masks.extend(t_m)
            
        for idx in range(len(target_masks)):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = target_masks[idx].shape[-2:]
            target_masks[idx] = target_masks[idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert target_masks[idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert target_masks[idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        
        # list[n 4], bt
        target_boxes = []
        for bth_idx in range(batch_size):
            # n t 4
            t_m = targets[bth_idx]["boxes"].split(1, dim=1) # list[n 1 4], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n 4], t
            target_boxes.extend(t_m)

        # list[n], bt
        target_classes = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_labels = targets[bth_idx]['class_labels'].unsqueeze(-1).repeat(1, bth_valids.shape[1]) # n t
            bth_labels = torch.where(bth_valids, bth_labels, self.obj_decoder_nclasses-1)
            t_m = bth_labels.split(1, dim=1) # list[n 1], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n], t
            target_classes.extend(t_m)    

        # list[n], bt
        is_valid = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].split(1, dim=1) # n t -> list[n 1], t
            bth_valids = [bv.squeeze(1) for bv in bth_valids] # list[n], t
            is_valid.extend(bth_valids)
        referent_idxs = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_refidx = [targets[bth_idx]['referent_idx']] * bth_valids.shape[1] # list[int], t
            referent_idxs.extend(bth_refidx)  


        linamrs_aux = auxiliary['linamrs_aux']
        linamr_esrctgts = linamrs_aux['Esrctgt_ids'] # text_sigma nmax
        linamr_labels = linamrs_aux['labels'] # text_sigma nmax
        linamrs_obj_ids = linamrs_aux['obj_ids'] # list[list[int], num_text], batch
        has_ann_number_by_batch = [t['has_ann'].int().sum().item() for t in targets] # list[int], batch
        repeated_linamrs_obj_ids = [] # list[list[int], n_text], bt
        repeated_linamr_esrctgts = [] # list[text_i nmax], bt
        repeated_linamr_labels = []
        text_cnt = 0
        for obj_ids, ann_number in zip(linamrs_obj_ids, has_ann_number_by_batch):
            esrc = linamr_esrctgts[text_cnt:(text_cnt+len(obj_ids))] # text_i, nmax
            lab = linamr_labels[text_cnt:(text_cnt+len(obj_ids))] # text_i, nmax
            for _ in range(ann_number):
                repeated_linamrs_obj_ids.append(obj_ids)
                repeated_linamr_esrctgts.append(esrc)
                repeated_linamr_labels.append(lab)
            text_cnt += len(obj_ids)
        linamrs_aux['labels'] = torch.cat(repeated_linamr_labels, dim=0)  # text_sigma
        linamrs_aux['Esrctgts_ids'] = torch.cat(repeated_linamr_esrctgts, dim=0)
        linamrs_aux['obj_ids'] = repeated_linamrs_obj_ids # list[list[int], n_text] bt
        auxiliary['linamrs_aux'] = linamrs_aux

        return {
            'masks': target_masks,
            'boxes': target_boxes,
            'class_labels': target_classes,
            'is_valid': is_valid,
            'referent_idx': referent_idxs
        }, auxiliary

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        refseg_src = self.get_decoder_preds(model_outs)
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
        
        obj_decoder_targets, auxiliary = self.obj_decoder_targets_handler(targets, auxiliary)

        loss_value_dict.update(self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets, auxiliary)[0])  
                  
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    # task loss
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets, text_auxiliary):
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n h w], bt
        tgt_masks = targets['masks']
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_texts = text_auxiliary['linamrs_aux']['Esrctgts_ids'].shape[0]
        num_texts = torch.as_tensor([num_texts], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_texts)
        num_texts = torch.clamp(num_texts / get_world_size(), min=1).item()        
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_class': torch.tensor(0, device=tgt_masks[0].device).float(),
                    'objdecoder_vtc':torch.tensor(0, device=tgt_masks[0].device).float(),
                    'objdecoder_vtg':torch.tensor(0, device=tgt_masks[0].device).float(),
                    'objdecoder_vtm':torch.tensor(0, device=tgt_masks[0].device).float(),}
        matching_indices_by_layer = []
        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = AMR_v0_detectObj.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.decoder_mask_out_stride)
                matching_indices_by_layer.append(layer_matching_indices)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detectObj.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detectObj.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detectObj.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]

                if loss_weight['objdecoder_vtc'] != 0:
                    vtc_losses = self.obj_decoder_vtc_loss(layer_pred, text_auxiliary, layer_matching_indices, num_texts)
                    for k in vtc_losses.keys():
                        loss_value[k] += layer_weight * vtc_losses[k]
                if loss_weight['objdecoder_vtg'] != 0:
                    vtg_losses = self.obj_decoder_vtg_loss(layer_pred, text_auxiliary, layer_matching_indices, num_texts)
                    for k in vtg_losses.keys():
                        loss_value[k] += layer_weight * vtg_losses[k]
                if loss_weight['objdecoder_vtm'] != 0:
                    vtm_losses = self.obj_decoder_vtm_loss(layer_pred, text_auxiliary, layer_matching_indices, num_texts)
                    for k in vtm_losses.keys():
                        loss_value[k] += layer_weight * vtm_losses[k]
        return loss_value,matching_indices_by_layer   

    def obj_decoder_vtc_loss(self, outputs, text_auxiliary, indices, num_texts):
        bt = len(indices)
        obj_queries = outputs["query_to_amrbart"] # bt nq c
        linamrs_aux = text_auxiliary['linamrs_aux'] # 
        linamr_esrctgts = linamrs_aux['Esrctgts_ids'] # text_sigma nmax
        linamrs_obj_ids = linamrs_aux['obj_ids'] # list[list[int], 0-ni], bt

        # 如果一个物体没有text query
        flatten_obj_ids = []
        num_objs_by_bt = [max(loi)+1 for loi in linamrs_obj_ids]
        for bt_idx, lin_ob_id in enumerate(linamrs_obj_ids):
            if bt_idx == 0:
                exist_num_objs = 0
            else:
                exist_num_objs = sum(num_objs_by_bt[:bt_idx])
            new_lin_ob_id = [loi + exist_num_objs for loi in lin_ob_id]
            flatten_obj_ids.extend(new_lin_ob_id)
        flatten_obj_ids = torch.tensor(flatten_obj_ids, device=obj_queries.device).long()

        # list[ni c], bt
        obj_queries = [obj_q[J] for obj_q, (J, _) in zip(obj_queries, indices)]

        tgt_idx_to_query_by_batch = []
        for bt_idx in range(bt):
            J = indices[bt_idx][1]
            bt_tgt_idx_to_query = {idx.item():query for idx, query in zip(J, obj_queries[bt_idx])}
            tgt_idx_to_query_by_batch.append(bt_tgt_idx_to_query)

        # text_sigma nmax c, <s> MASK </s> <g> AMR </g>
        attention_mask = linamr_esrctgts.ne(self.amrbart_tokenizer.pad_token_id)
        linamr_feats = self.amrbart_model_encoder(input_ids = linamr_esrctgts, 
                                                  attention_mask = attention_mask).last_hidden_state
        linamr_feats = linamr_feats[:, 3] # text_sigma c
        obj_feats = []
        for bt_idx in range(bt):
            tgt_idx_to_query = tgt_idx_to_query_by_batch[bt_idx]
            for obj_id in linamrs_obj_ids[bt_idx]:
                obj_feats.append(tgt_idx_to_query[obj_id])
        obj_feats = torch.stack(obj_feats, dim=0) # text_sigma c
        
        linamr_feats = linamr_feats / linamr_feats.norm(dim=1, keepdim=True)
        obj_feats = obj_feats / obj_feats.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_linamr = logit_scale * (linamr_feats @ (obj_feats.t())) # text_sigma text_sigma
        logits_per_obj = logits_per_linamr.t()
        
        affinity = (flatten_obj_ids.unsqueeze(0) == flatten_obj_ids.unsqueeze(1)).int()
        linamr_probs = affinity / affinity.sum(dim=1, keepdim=True)
        obj_probs = (affinity / affinity.sum(dim=0, keepdim=True)).t()

        loss_i = F.cross_entropy(logits_per_linamr, linamr_probs, reduction='none')
        loss_t = F.cross_entropy(logits_per_obj, obj_probs, reduction='none') # text_sigma
        loss = (loss_i + loss_t)/2
        # <g> embed # list[c], n_sigma

        return {
            'objdecoder_vtc': loss.sum() / num_texts
        }

    def obj_decoder_vtg_loss(self, outputs, text_auxiliary, indices, num_texts):
        bt = len(indices)
        obj_queries = outputs["query_to_amrbart"] # bt nq c
        linamrs_aux = text_auxiliary['linamrs_aux'] # 
        linamr_labels = linamrs_aux['labels'] # text_sigma nmax

        linamrs_obj_ids = linamrs_aux['obj_ids'] # list[list[int], 0-ni], bt

        decoder_input = torch.zeros_like(linamr_labels)
        decoder_input[:, 1:] = linamr_labels[:, :-1].clone()
        decoder_input[:, 0] = self.amrbart_tokenizer.amr_bos_token_id
        decoder_attention_mask = decoder_input.ne(-100)
        decoder_input.masked_fill_(~decoder_attention_mask, self.amrbart_tokenizer.pad_token_id)

        encoder_input_by_text = []
        for bt_idx in range(bt):
            btc_input = obj_queries[bt_idx] # nq 1024
            btc_src_idx, btc_tgt_idx = indices[bt_idx]
            btc_obj_ids = linamrs_obj_ids[bt_idx] # list[int], 0-ni
            for obj_id in btc_obj_ids:
                in_idx = btc_tgt_idx.tolist().index(obj_id)
                src_idx = btc_src_idx[in_idx]
                enc_input = btc_input.clone()
                enc_input[src_idx] += self.obj_decoder_refer_pos.weight[0]
                encoder_input_by_text.append(enc_input)
        encoder_input_by_text = torch.stack(encoder_input_by_text, dim=0) # text_sigma nq 1024

        next_amr_token_loss = self.amrbart_model(inputs_embeds=encoder_input_by_text,
                                                attention_mask=None,
                                                decoder_input_ids=decoder_input,
                                                decoder_attention_mask=decoder_attention_mask,
                                                labels=linamr_labels).loss

        return {
            'objdecoder_vtg': next_amr_token_loss
        }

    def obj_decoder_vtm_loss(self, outputs, text_auxiliary, indices, num_boxes):
        """
        indices: [[], []], bt
        """
        raise NotImplementedError
        src_logits = outputs["pred_class_logits"] # bt nq c

        # list[n], bt
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['class_labels'], indices)]) # btn_sigma
    
        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], len(class_weight)-1, dtype=torch.int64, device=src_logits.device
        ) # bt n
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.tensor(class_weight).to(src_logits))
        losses = {"objdecoder_class": loss_ce}
        return losses

class AMR_v0_detectObj_RefChoose(AMR_v0_detectObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
                
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)
        # v1
        # 冻住proj, fusion parsing encoder, objdecoder,
        # 只训练refdecoder
        # for p in self.video_proj.parameters():
        #     p.requires_grad_(False) 
        # for p in self.amrtext_wordEmbedding_proj.parameters():
        #     p.requires_grad_(False)
        # for p in self.cross_product.parameters():
        #     p.requires_grad_(False)

        # for p in self.obj_parsing_encoder.parameters():
        #     p.requires_grad_(False)
        # assert self.ref_parsing_encoder == None
        # for n, p in self.named_parameters():
        #     if 'obj_decoder' in n:
        #         p.requires_grad_(False)

        # # v2 / v3
        # assert self.ref_parsing_encoder == None
        # for n, p in self.named_parameters():
        #     if 'obj_decoder' in n:
        #         p.requires_grad_(False)        

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        refseg_src = self.get_decoder_preds(model_outs)
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict, matching_result = self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets)

        loss_value_dict.update(self.ref_choose_loss(refseg_src, objseg_src, obj_decoder_targets, matching_result[-1]))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
   
    def ref_choose_loss(self, refseg_src, objseg_src, targets, decoder_last_layer_matching_results):
        """
        Args:
            refseg_src: dict{layer-1pred: {queries: bt c}}
            targets (_type_): batch[dict{'masks', 'referent_idx'}]
            matching_result_by_layer: list[(tgt_idx, src_idx), bt]
        """
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        last_obj_decoder_queries = objseg_src[f'layer{self.obj_decoder_nlayers-1}_preds']['queries'] # nq bt c
        referent_idx = targets['referent_idx'] # list[int], bt
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  decoder_last_layer_matching_results): # bt
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        refdecoder_choose_loss = 0.
        for layer_idx in range(-1, self.decoder_nlayers):
            layer_weight = layer_weights[layer_idx] 
            if layer_weight != 0:# bt c
                # nq bt c, 1 bt c -> nq bt
                refdecoder_layer_query = refseg_src[f'layer{layer_idx}_preds']['queries'] # bt c
                logits = (last_obj_decoder_queries.detach() * (refdecoder_layer_query.unsqueeze(0))).sum(dim=-1)
                choose_loss = F.cross_entropy(logits.permute(1,0), torch.tensor(match_as_gt_indices).long().to(logits.device))
                refdecoder_choose_loss += (choose_loss * layer_weight)
        return {'refdecoder_choose': refdecoder_choose_loss}

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        objseg_preds = decoder_layer_preds['objdecoder_objseg']
        obj_last_layer_preds = objseg_preds[f'layer{self.obj_decoder_nlayers-1}_preds']

        obj_last_layer_query = obj_last_layer_preds['queries'] # n bt c
        out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w

        refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
        ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        ref_last_layer_queries = ref_last_layer_preds['queries']  # bt c
        prob_by_query = (obj_last_layer_query * (ref_last_layer_queries.unsqueeze(0))).sum(-1).argmax(dim=0) # n bt -> bt

        # bt h w
        out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, prob_by_query)], dim=0)

        # out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
        # for idx in range(batch_size):
        #     h, w = targets[idx]['masks'].shape[-2:]
        #     # n t h w -> n t H W
        #     targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        # obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        # _, matching_result = self.obj_decoder_objseg_loss(objseg_preds, obj_decoder_targets)
        # matching_result = matching_result[-1] # list(tgt, src), bt
        # gt_referent_idx = obj_decoder_targets['referent_idx'] # list[int], bt
        # out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]] 
        #                                 for out_mask, gt_ref_idx, (tgt_idx, src_idx) in zip(out_mask_logits, gt_referent_idx, matching_result)], dim=0)
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }

class AMR_v0_detectObj_RefChoose_onlyObj(AMR_v0_detectObj_RefChoose):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)

    def get_decoder_preds(self, model_outs):
        refseg_src = model_outs['refdecoder_refseg']
        if self.decoder_choose_who == '第一个':
            for i in range(-1, self.decoder_nlayers):
                layer_pred = refseg_src[f'layer{i}_preds']
                refseg_src[f'layer{i}_preds']['queries'] = layer_pred['queries'][0] # bt c
        return refseg_src        
    

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        if self.ref_parsing_encoder is not None:
            multiscales, _, _\
                = self.ref_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)          
        
        memories = obj_queries # nq bt c
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        decoder_layer_preds['layer-1_preds'] = {'queries': amr_token_feats.clone() }
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                    batched_edge_index, 
                                                                    batched_edge_feats,
                                                                    memory=graph_self_memory,
                                                                    batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            decoder_layer_preds[f'layer{i}_preds'] = {'queries': amr_token_feats.clone() }

            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1

        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

class AMR_v0_detectObj_RefChoose_onlyObj_objencoder(AMR_v0_detectObj_RefChoose_onlyObj):
    def __init__(self, 
                d_model=256,
                max_stride=64,
                pt_dir='/home/xhh/pt',
                # video encoder
                swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                swint_freeze=True,
                swint_runnning_mode='train',
                video_projs = [
                {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                    'bias':True,}],
            video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

            # amrtext
            amrbart_wordEmbedding_freeze=True,
            amrtext_wordEmbedding_proj = {
                'name': 'FeatureResizer',
                'input_feat_size': 1024,
                'output_feat_size': 256,
                'dropout':0,
                'do_ln':True},
            fusion={
                'name': 'VisionLanguageFusionModule',
                'd_model':256,
                'nheads': 8,
                'dropout':0.},
            parsing_encoder={
                'name':'split_obj_ref_deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.,
                'activation': 'relu',
                'nheads': 8,
                'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'obj_seg_nlayers':3,
                'ref_seg_nlayers':3
                },
            loss_weight={'refdecoder_mask': 5,
                            'refdecoder_dice': 5,
                            'refdecoder_giou': 0,
                            'refdecoder_bbox': 0,
                },
            tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                            },
            },
            refdecoder={
                'nlayers': 9,
                'amr_cross_video_layer':{
                    'name': 'cross_attention',
                    'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'amr_self_layer':{
                    'name': 'graph_layer_v1', # 只更新node
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min'
                },
                # add ffn layer
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'choose_who': '第一个'
                },
            objencoder={
                'nlayers': 6,
                'nheads': 8,
            },
            objdecoder={ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                },) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)
        self.obj_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model,
                                    nheads=objencoder['nheads'],),
                                    objencoder['nlayers']
        )

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        # n bt c, bt n,
        multiscales, _, _\
            = self.obj_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)

        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        obj_queries = self.obj_encoder(src=obj_queries,
                                        mask=None,
                                        src_key_padding_mask=None,
                                    pos=query_embed)
        
        if self.ref_parsing_encoder is not None:
            multiscales, _, _\
                = self.ref_parsing_encoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)          
        
        memories = obj_queries # nq bt c
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        decoder_layer_preds['layer-1_preds'] = {'queries': amr_token_feats.clone() }
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=memories, # nq bt c
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                    batched_edge_index, 
                                                                    batched_edge_feats,
                                                                    memory=graph_self_memory,
                                                                    batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            decoder_layer_preds[f'layer{i}_preds'] = {'queries': amr_token_feats.clone() }

            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1

        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

###########################################################################
# 相比v0, 用object decoder
###########################################################################
# 改成3d proj
# 改成matching
# mrbart的encoding训练
# proj变成多层
# 改成MTTR的那种融合方式, 因为如果只关注32x的feature, 能够使用(thw) parsing, 而不是2d parsing
# 加上LLN
# temporal queries
# parsing encoder, 考虑temporal信息
###########################################################################
# 在fusion之后, parsing encoder, decoder, matching都是单帧
# text_sentence的特征加到每个query上
class Text_V0(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                roberta_freeze = True,
                text_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 768,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
            
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_refer': 2,
                             'refdecoder_giou': 2,
                             'refdecoder_bbox': 5,
                            # 现在的模型只有decoder有loss
                            # 其他的module是否有loss
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                'refer_class_weight': [1, 0.1],
                                                'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5, 'box': 5, 'giou': 2 },
                                                },
                },
                refdecoder={ 
                    'nqueries': 5,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        
        self.d_model = d_model
        self.max_stride = max_stride
        # video encoder
        from .video_swin import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=os.path.join(pt_dir, swint_pretrained_path),
                                                running_mode=swint_runnning_mode)
        if swint_freeze:
            for p in self.video_swint.parameters():
                p.requires_grad_(False) 
                 
        assert len(video_projs) == len(video_feat_scales)
        self.video_feat_scales = video_feat_scales
        backbone_channels, backbone_scales = self.video_swint.get_desc()
        assert len(backbone_channels) == len(backbone_scales)
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        from transformers import RobertaModel, RobertaTokenizerFast
        self.roberta = RobertaModel.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        if roberta_freeze:
            for p in self.roberta.parameters():
                p.requires_grad_(False)
        
        assert text_proj.pop('name') == 'FeatureResizer'
        self.txt_proj = FeatureResizer(**text_proj)
        self.text_pos_embed = build_position_encoding(position_embedding_name='1d')
        
        assert fusion.pop('name') == 'VisionLanguageFusionModule'
        self.cross_product = VisionLanguageFusionModule(**fusion)

        assert parsing_encoder.pop('name') == 'deform_video_2d_fpn'
        self.deform_multiscale_2dencoder = DeformVideo2D_with_FPN(**parsing_encoder)

        self.decoder_query_embed = zero_module(nn.Embedding(refdecoder['nqueries'], d_model))
        self.decoder_used_scales = refdecoder['used_scales']
        self.decoder_conved_scale = refdecoder['conved_scale']
        self.decoder_nlayers = refdecoder['nlayers']
        self.decoder_nqueries = refdecoder['nqueries']
        self.decoder_level_embed = nn.Embedding(len(self.decoder_used_scales), d_model)
        cross_layer = refdecoder['cross_layer']
        assert cross_layer.pop('name') == 'cross_attention'
        self.decoder_cross_video_layers = _get_clones(CrossAttentionLayer(**cross_layer),
                                                                   self.decoder_nlayers)
        self.decoder_nheads = cross_layer['nhead']
        self_layer = refdecoder['self_layer']
        assert self_layer.pop('name') == 'self_attention'
        self.decoder_self_layers = _get_clones(SelfAttentionLayer(**self_layer),
                                                            self.decoder_nlayers)  
        ffn_layer = refdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                            self.decoder_nlayers) 
        # norm, mask out, box, cls, mask
        self.decoder_refer_embed = nn.Linear(d_model, 2)
        self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.decoder_mask_out_stride = refdecoder['mask_out_stride']
        self.decoder_mask_threshold = refdecoder['mask_threshold']
 
    def init_parameters(self,): 
        for proj in self.video_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def encode_video(self, samples):
        bb_out = self.video_swint(samples)  
        nf, batch_size, *_ = bb_out[0].tensors.shape
        orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src.clone())
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(orig_pad_mask.float(),
                                             size=src_proj_l.shape[-2:],mode='nearest') > 0.5
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses

    # # 2d pos
    # def encode_video(self, samples):
    #     bb_out = self.video_swint(samples)  
    #     nf, batch_size, *_ = bb_out[0].tensors.shape
    #     orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
    #     for layer_out in bb_out:
    #         layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
    #         layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
    #     multiscales = []
    #     multiscales_pad_masks = []
    #     multiscales_poses = []
    #     for lvl, feat in enumerate(bb_out): 
    #         src, pad_mask = feat.decompose() 
    #         src_proj_l = self.video_proj[lvl](src.clone())
    #         src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
    #         multiscales.append(src_proj_l)
    #         multiscales_pad_masks.append(pad_mask)
    #         pos_2d = self.video_2d_pos(pad_mask.flatten(0, 1), hidden_dim=src_proj_l.shape[2])
    #         pos_2d = rearrange(pos_2d, '(b t) c h w -> b t c h w', b=batch_size, t=nf)
    #         multiscales_poses.append(pos_2d)
    #         if lvl == (len(bb_out) - 1):
    #             for idx in range(lvl+1, len(self.video_proj)):
    #                 src_proj_l = self.video_proj[idx](src.clone())
    #                 src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
    #                 pad_mask = F.interpolate(orig_pad_mask.float(),
    #                                          size=src_proj_l.shape[-2:],mode='nearest') > 0.5
    #                 multiscales.append(src_proj_l)
    #                 multiscales_pad_masks.append(pad_mask)
    #                 pos_2d = self.video_2d_pos(pad_mask.flatten(0, 1), hidden_dim=src_proj_l.shape[2])
    #                 pos_2d = rearrange(pos_2d, '(b t) c h w -> b t c h w', b=batch_size, t=nf)
    #                 multiscales_poses.append(pos_2d)
    #     return multiscales, multiscales_pad_masks, multiscales_poses
   
    def encode_text(self, text_queries, device):
        tokenized = self.roberta_tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.roberta(**tokenized)
        # encoded_text.last_hidden_state: [batch_size, length, 768]
        # encoded_text.pooler_output: [batch_size, 768]
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # text_attention_mask: [batch_size, length]
        text_features = encoded_text.last_hidden_state 
        text_features = self.txt_proj(text_features)    
        text_masks = text_attention_mask              

        text_sentence_features = encoded_text.pooler_output  
        text_sentence_features = self.txt_proj(text_sentence_features)  
        # max b c, b max, b c
        return text_features.permute(1,0,2), text_masks, text_sentence_features
    
    def get_refdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features
    
    def forward_refdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_refer = self.decoder_refer_embed(decoder_output)  # bt n 2
        outputs_box = self.decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_refer, outputs_mask, outputs_box, attn_mask
 
    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """
        samples: t b c h w, t b h w
        frame_has_ann_by_batch: list[t, True/False], b
        """
        check_visualize = {} 
        device = samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # 抽文本的特征 max b c,  b max, b c 
        token_feats, token_pad_masks, token_sentence_feats = self.encode_text(text_queries, auxiliary, device)
        token_pos = self.text_pos_embed(token_pad_masks, hidden_dim=token_feats.shape[-1]).permute(2, 0, 1)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=token_feats, 
                                                    memory_key_padding_mask=token_pad_masks,
                                                    pos=token_pos, 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} fusion attention weights'] = attn_weight # b thw s, float, 0, 1
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> list[t c h w] -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b c -> bT c
        token_sentence_feats = repeat(token_sentence_feats, 'b c -> (b t) c', t=nf)[perFrame_has_ann]
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer
        
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = repeat(token_sentence_feats, 'bt c -> n bt c', n=self.decoder_nqueries,) # n bt c

        decoder_layer_preds = {}
        out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.decoder_nlayers):
            level_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.decoder_ffn_layers[i](
                output # n bt c
            )
            out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }

        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize } 
    

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w, bt' n
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)['refdecoder_refseg']
        last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        out_masks_logits =  last_layer_preds['pred_mask_logits'] 
        out_prob = last_layer_preds['pred_refer_logits'].softmax(dim=-1)
        # bt' n h w
        query_pred_masks = F.interpolate(out_masks_logits, scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False) 
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' n
        query_pred_is_referred_prob = out_prob[..., 0]
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
        
    
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bT n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        
        refseg_src = model_outs['refdecoder_refseg']
        
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
            
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

                
    # task loss
    def refdecoder_refseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        refer_class_weight = self.tasks['refdecoder_refseg']['refer_class_weight']
        matching_costs = self.tasks['refdecoder_refseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[t] -> bt
        target_valid = torch.cat([t["valid"][t['referent_idx']] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        

        loss_value = {'refdecoder_mask': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_bbox': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_giou': torch.tensor(0, device=target_valid.device).float(),
                      'refdecoder_dice': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_refer': torch.tensor(0, device=target_valid.device).float(),}

        for i in range(-1, self.decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = Text_V0.refdecoder_matching(layer_pred, targets, matching_costs, refer_class_weight, self.decoder_mask_out_stride)
                if loss_weight['refdecoder_mask'] != 0 or loss_weight['refdecoder_dice'] !=0:
                    masks_losses = Text_V0.refdecoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['refdecoder_bbox'] != 0 or loss_weight['refdecoder_giou'] !=0:
                    boxes_losses = Text_V0.refdecoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['refdecoder_refer'] != 0:
                    refer_losses = Text_V0.refdecoder_refer_loss(layer_pred, targets, layer_matching_indices, refer_class_weight)
                    for k in refer_losses.keys():
                        loss_value[k] += layer_weight * refer_losses[k]
        return loss_value         

    @staticmethod
    def refdecoder_refer_loss(outputs, targets, indices, refer_class_weight):
        """
        indices: [[], []], bt
        """
        src_logits = outputs['pred_refer_logits']  # bt n 2
        bt, nq, _ = src_logits.shape # bt n 2
        
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool() 
        target_classes = torch.ones([bt, nq], device=src_logits.device).long() # bt n
        
        for batch_idx in range(bt):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        refer_class_weight = torch.tensor(refer_class_weight).to(src_logits)
        # btn 2, btn
        loss_ce = F.cross_entropy(src_logits.flatten(0,1), target_classes.flatten(), refer_class_weight)
        losses = {'refdecoder_refer': loss_ce}

        return losses
    
    @staticmethod
    def refdecoder_boxes_loss(outputs, targets, indices, num_boxes): 
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        src_boxes = outputs['pred_box_logits'].sigmoid()  # bt n 4
        # list[4] -> bt 4
        src_boxes = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0) 
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0).to(src_boxes)  
        
        src_boxes = src_boxes[is_consistent]  # bt 4
        target_boxes = target_boxes[is_consistent] # bt 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['refdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['refdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    @staticmethod
    def refdecoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        # list[n t] -> list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        
        src_masks = outputs["pred_mask_logits"]  # bT n h w  
        # list[h w] -> bT h w
        src_masks = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)
        
        target_masks = torch.zeros_like(src_masks) # bT h w
        # list[n t h w] -> list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0).to(src_masks) # list[t h w] -> bt h w

        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        
        src_masks = src_masks[is_consistent].flatten(1) # bt hw
        target_masks = target_masks[is_consistent].flatten(1) # bt hw
        
        losses = {
            "refdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "refdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    @staticmethod
    @torch.no_grad()
    def refdecoder_matching(outputs, targets, matching_costs, refer_class_weight, decoder_mask_out_stride):
        src_refer_prob = outputs["pred_refer_logits"].softmax(dim=-1) # bt n 2
        src_boxes = outputs["pred_box_logits"].sigmoid()   # bt n 4
        src_masks_logits = outputs["pred_mask_logits"]  # bt n h w
        bt, nq, h, w = src_masks_logits.shape 

        # list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0)
        target_masks = target_masks.to(src_masks_logits)
        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0) 
        # list[t] -> bt 
        is_valid = torch.cat([t['valid'][t['referent_idx']] for t in targets], dim=0).bool()

        indices = [] 
        for i in range(bt):
            out_prob = src_refer_prob[i] # n 2
            out_bbox = src_boxes[i]  # n 4
            out_mask = src_masks_logits[i]  # n h w

            tgt_bbox = target_boxes[i].unsqueeze(0) # 1 4
            tgt_mask = target_masks[i].unsqueeze(0) # 1 h w
            tgt_valid = is_valid[i]    # True/False
            
            tgt_is_referred = (~tgt_valid).long()  # 1/0

            
            cost_refer = -out_prob[:, [tgt_is_referred]] # n 1

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 1
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # n hw : 1 hw -> n 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = matching_costs['refer'] * cost_refer +\
                matching_costs['bbox'] * cost_bbox + \
                matching_costs['giou'] * cost_giou + \
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        return indices

class Text_v0linamr(Text_V0):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                roberta_freeze = True,
                linamrbart_freeze=True,
                text_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 768,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                how_to_encode_linamr='encoder decoder',
                linamr_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                linamr_text_sentence_level_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1792, # + 768 = 1792
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
            
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_refer': 2,
                             'refdecoder_giou': 2,
                             'refdecoder_bbox': 5,
                            # 现在的模型只有decoder有loss
                            # 其他的module是否有loss
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                'refer_class_weight': [1, 0.1],
                                                'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5, 'box': 5, 'giou': 2 },
                                                },
                },
                refdecoder={ 
                    'nqueries': 5,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                ) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)
        
        # from transformers import RobertaModel, RobertaTokenizerFast
        # self.roberta = RobertaModel.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        # self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        # if roberta_freeze:
        #     for p in self.roberta.parameters():
        #         p.requires_grad_(False)
        
        # 
        # self.txt_proj = FeatureResizer(**text_proj)
        # self.text_pos_embed = build_position_encoding(position_embedding_name='1d')
    
        from .amr_utils.utils import BartForConditionalGeneration
        self.linamr_model = BartForConditionalGeneration.from_pretrained(os.path.join(pt_dir, 'amr', 'AMRBART_pretrain'))
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.linamr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_dir, 'amr', 'AMRBART_pretrain'))
        if linamrbart_freeze:
            for p in self.linamr_model.parameters():
                p.requires_grad_(False)
        
        assert linamr_proj.pop('name') == 'FeatureResizer'
        self.linamr_proj = FeatureResizer(**linamr_proj)
        assert linamr_text_sentence_level_proj.pop('name') == 'FeatureResizer'
        self.linamr_text_sentence_level_proj = FeatureResizer(**linamr_text_sentence_level_proj)
        self.how_to_encode_linamr = how_to_encode_linamr
    
    def linamr_model_forward(self, model_inputs, device):
        # input_ids: <s> text </s>
        # srcEtgt_ids: <s> text </s> <g> <MASK> </g>
        # Esrctgt_ids: <s> <MASK> </s> <g> amr </g>
        # labels: amr </g>
        # joint_ids: <s> text </s> <g> amr </g>
        if self.how_to_encode_linamr == 'encoder':
            # Esrctgt, label
            bart_input = model_inputs["Esrctgt_ids"] # b max
            attention_mask = bart_input.ne(self.linamr_tokenizer.pad_token_id).int() 
            bart_input = bart_input.to(device) # <s> <MASK> </s> <g> amr </g> pad
            attention_mask = attention_mask.to(device) # 0代表padding的位置
            # <s> <MASK> </s> <g> amr </g> pad
            encoder_outputs = self.linamr_model.model.encoder(
                input_ids=bart_input,
                attention_mask=attention_mask,
            ).last_hidden_state
            amr_embeds = encoder_outputs[:, 3:]
            amr_pad_masks = ~(attention_mask[:, 3:].bool())
            amr_sentence_level_embed = amr_embeds[:, 0] # b c
            return amr_embeds, amr_pad_masks, amr_sentence_level_embed 
        
        elif self.how_to_encode_linamr == 'encoder decoder':
            # <s> <MASK> </s> <g> amr </g> pad
            bart_input = model_inputs["Esrctgt_ids"] # b max
            attention_mask = bart_input.ne(self.linamr_tokenizer.pad_token_id).int()      
            # amr </g> pad pad
            labels = model_inputs["labels"] # b max
            
            dec_input = labels.new_zeros(labels.size(0), labels.size(1))
            # <g> amr </g> pad -> amr </g> pad pad
            dec_input[:, 1:] = labels[:, :-1].clone()
            dec_input[:, 0] = self.linamr_tokenizer.amr_bos_token_id 
 
            decoder_input_pad_mask = (dec_input == -100) 
            dec_input.masked_fill_(decoder_input_pad_mask, self.linamr_tokenizer.pad_token_id)
            
            bart_input = bart_input.to(device) # <s> <MASK> </s> <g> amr </g> pad
            attention_mask = attention_mask.to(device) # 0代表padding的位置
            labels = labels.to(device) # amr </g> -100
            dec_input = dec_input.to(device) # <g> amr </g> pad
            # self.tokenizer.decode([self.model.lm_head(decoder_output[0][i]).argmax().item() for i in range(len(decoder_output[0]))])
            # amr </g> pad
            amr_embeds = self.linamr_model(input_ids=bart_input,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=dec_input,
                                    labels=labels)
            amr_embeds_pad_mask = decoder_input_pad_mask[:, 1:]
            amr_embeds_pad_mask = F.pad(amr_embeds_pad_mask.float(), [0, 1], value=1.0).bool()
            return amr_embeds, amr_embeds_pad_mask
        
        elif self.how_to_encode_linamr == 'amr+text_encoder amr_decoder':
            # joint, label
            pass
        elif self.how_to_encode_linamr == 'amr+text_encoder amr+text_decoder':
            bart_input = model_inputs["joint_ids"]
            seg_ids = model_inputs['seg_ids'] # 0: text, 1: graph
            labels = model_inputs["joint_ids"].clone()
            labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)
            labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
            dec_input = model_inputs["joint_ids"].clone()
            dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()          # attention mask
            
            # text </s> <g> amr </g>
            decoder_output = self.linamr_model(input_ids=bart_input,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=dec_input,
                                        labels=labels).decoder_hidden_states
            decoder_output = self.text_proj(decoder_output)
            text_feat = decoder_output
            
            return decoder_output, meta_dict['each_token_length'], text_feat, None
     

    def encode_text(self, text_queries, auxiliary, device):
        tokenized = self.roberta_tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.roberta(**tokenized)
        # encoded_text.last_hidden_state: [batch_size, length, 768]
        # encoded_text.pooler_output: [batch_size, 768]
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # text_attention_mask: [batch_size, length]
        text_features = encoded_text.last_hidden_state 
        text_features = self.txt_proj(text_features)  
        text_masks = text_attention_mask              

        text_sentence_features = encoded_text.pooler_output  
        # max b c, b max, b c
        text_seq2seq_feats = text_features.permute(1,0,2) # max b c
        text_seq2seq_pad_masks = text_masks # b max
        text_seq2seq_sent_feats = text_sentence_features  # b c
        
        
        # dict["input_ids", "labels", "joint_ids"]
        linamr_feats, linamr_pad_masks, linamr_sentence_feats \
            = self.linamr_model_forward(auxiliary['model_inputs'], device=device)
                
        linamr_feats = self.linamr_proj(linamr_feats).permute(1,0,2) # max b c
        
        # s b c, b s, b c
        return torch.cat([text_seq2seq_feats, linamr_feats], dim=0),\
            torch.cat([text_seq2seq_pad_masks, linamr_pad_masks], dim=1),\
                self.linamr_text_sentence_level_proj(torch.cat([text_seq2seq_sent_feats, linamr_sentence_feats], dim=1))
        

###########################################################################
import logging


@register_model
def amr_v0(device, configs):
    model = AMR_v0(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detObj(device, configs):
    model = AMR_v0_detectObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detObj_onlyobj(device, configs):
    model = AMR_v0_detectObj_onlyObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 

@register_model
def amr_v0_detObj_onlyobj_fusionObj(device, configs):
    model = AMR_v0_detectObj_onlyObj_fusionObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detObj_onlyobj_fusionBoth(device, configs):
    model = AMR_v0_detectObj_onlyObj_fusionBoth(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder'],
        obj_fusion=configs['obj_fusion']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 




@register_model
def amr_v0_detObj_onlyobj_fusionasLoss(device, configs):
    model = AMR_v0_detectObj_onlyObj_fusionAsLoss(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        amrbart_freeze=configs['amrbart_freeze'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detObjRefChoose(device, configs):
    model = AMR_v0_detectObj_RefChoose(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 



@register_model
def amr_v0_detObjRefChoose_onlyObj(device, configs):
    model = AMR_v0_detectObj_RefChoose_onlyObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 



@register_model
def amr_v0_detOnlyObj_grounding(device, configs):
    model = AMR_v0_detOnlyObj_Grounding(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        detach_refdecoder_memory=configs['detach_refdecoder_memory'] if 'detach_refdecoder_memory' in configs else False,
        freeze_obj_decoder=configs['freeze_obj_decoder'] if 'freeze_obj_decoder' in configs else False
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detOnlyObj_grounding_ptObjDet(device, configs):
    model = AMR_v0_detOnlyObj_Grounding_ptObjDet(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        obj_decoder=configs['obj_decoder'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        multiclass_choose=configs['multiclass_choose'] if 'multiclass_choose' in configs else False,
        choose_threshold=configs['choose_threshold'] if 'choose_threshold' in configs else None
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detOnlyObj_grounding_ptObjDet_v2(device, configs):
    model = AMR_v0_detOnlyObj_Grounding_ptObjDet_v2(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        obj_decoder=configs['obj_decoder'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        multiclass_choose=configs['multiclass_choose'] if 'multiclass_choose' in configs else False,
        choose_threshold=configs['choose_threshold'] if 'choose_threshold' in configs else None
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detOnlyObj_grounding_ptObjDet_v3(device, configs):
    model = AMR_v0_detOnlyObj_Grounding_ptObjDet_v3(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        obj_decoder=configs['obj_decoder'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        multiclass_choose=configs['multiclass_choose'] if 'multiclass_choose' in configs else False,
        choose_threshold=configs['choose_threshold'] if 'choose_threshold' in configs else None
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def amr_v0_detOnlyObj_grounding_weightedquery(device, configs):
    model = AMR_v0_detOnlyObj_Grounding_weightedQuery(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        detach_refdecoder_memory=configs['detach_refdecoder_memory'] if 'detach_refdecoder_memory' in configs else False,
        freeze_obj_decoder=configs['freeze_obj_decoder'] if 'freeze_obj_decoder' in configs else False,
        adpt=configs['adpt'] if 'adpt' in configs else None
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 



@register_model
def amr_v0_detOnlyObj_grounding_asobjLoss(device, configs):
    model = AMR_v0_detOnlyObj_Grounding_AsObjLoss(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder'],
        is_pretraining_seg=configs['is_pretraining_seg'],
        detach_refdecoder_memory=configs['detach_refdecoder_memory'] if 'detach_refdecoder_memory' in configs else False,
        layer_if_choose=configs['layer_if_choose'] if 'layer_if_choose' in configs else None
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 



@register_model
def amr_v0_detObjRefChoose_onlyObj_objencoder(device, configs):
    model = AMR_v0_detectObj_RefChoose_onlyObj_objencoder(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 

@register_model
def text_v0(device, configs):
    model = Text_V0(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("roberta" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


@register_model
def text_v0linamr(device, configs):
    model = Text_v0linamr(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        linamr_proj=configs['linamr_proj'],
        linamr_text_sentence_level_proj=configs['linamr_text_sentence_level_proj'],
        how_to_encode_linamr=configs['how_to_encode_linamr'],
        linamrbart_freeze=configs['linamrbart_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if (("roberta" in n) or ("linamr_model" in n)) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])
    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.info('你没用任何scheduler')
        print('你没用任何scheduler')
        return model, optimizer, None, None
    
    if sch_conf['name'] == 'polynomial_split':
        from models.model_utils import polynomial_decay_lambda
        group_names = ['model', 'vbb', 'text']
        poly_lambdas = []
        for gname in group_names:
            g_poly_conf = sch_conf[gname]
            poly_lambdas.append(partial(polynomial_decay_lambda, initial_learning_rate=g_poly_conf['initial_learning_rate'],
                                                                        end_learning_rate=g_poly_conf['end_learning_rate'],
                                                                        decay_steps=g_poly_conf['decay_steps'],
                                                                        power=g_poly_conf['power']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=poly_lambdas,
                                                      last_epoch=-1,)
        return model, optimizer, scheduler, sch_conf['unit']
    elif sch_conf['name'] == 'polynomial_freezebb':
        from models.model_utils import polynomial_decay_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(polynomial_decay_lambda, 
                                                                        initial_learning_rate=sch_conf['initial_learning_rate'],
                                                                        end_learning_rate=sch_conf['end_learning_rate'],
                                                                        decay_steps=sch_conf['decay_steps'],
                                                                        power=sch_conf['power'],
                                                                        ),
                                                      last_epoch=-1,
                                                      verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'polynomial':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                                        total_iters=sch_conf['total_iters'],
                                                        power=sch_conf['power'],
                                                        last_epoch=-1,
                                                        verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=sch_conf['milestones'],
                                                        gamma=sch_conf['gamma'],
                                                        verbose=True), 
        return model, optimizer, scheduler, sch_conf['unit']
    
    elif sch_conf['name'] == 'invert_sqrt':
        from models.model_utils import inverse_sqrt_warmup_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=partial(inverse_sqrt_warmup_lambda,
                                                                         num_warmup_steps=sch_conf['num_warmup_steps'],
                                                                         num_training_steps=sch_conf['num_training_steps']), last_epoch=-1)
        return model, optimizer, scheduler, sch_conf['unit']
    else:
        raise ValueError()

    return model, optimizer 


from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.visualizer import ColorMode
import torchvision.transforms as transforms
import networkx as nx

def generate_instance_canvas(vid_frames, metadata, H, W, pred_mask):
    """pred_mask: h w, score:float"""
    istce_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([1]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()


from torch_geometric.utils.convert import to_networkx
# 存储一个batch的debug
def save_model_output(videos, text_query, amr, amr_tree_string,  directory, pred_masks, scores):
    # t 3 h w
    # nq t h w
    # vi nq 
    #
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tgt_dir = '/'.join(directory.split('/')[:-1])
    os.makedirs(tgt_dir, exist_ok=True)
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])
    metadata = MetadataCatalog.get('youtube_rvos')

    final_image = []
    # draw video frames
    vid_frames = videos.detach().cpu()
    vid_frames = invTrans(vid_frames)
    vid_frames = torch.clamp(vid_frames, min=0, max=1).permute(2, 0, 3, 1).flatten(1,2)  # t 3 h w -> h t w 3 -> h (t w) 3
    H, W = vid_frames.shape[:2]
    final_image.append(vid_frames*255)
    # draw refer preditions
    pred_masks = pred_masks.permute(0, 2, 1, 3).flatten(2,3).detach().cpu() # t nq h w -> nq h t w -> nq h (t w) 
    pred_masks = (F.interpolate(pred_masks.float().unsqueeze(0), size=[H, W], mode='bilinear', align_corners=False) > 0)[0]
    scores = scores.detach().cpu()# vi nq   
    _, map_nqs = scores.max(-1)
    num_instances = len(pred_masks)
    from joblib import Parallel, delayed
    import multiprocessing
    params_by_instance = [(vid_frames, metadata, H, W, pred_mask) for pred_mask in pred_masks]
    n_jobs = min(multiprocessing.cpu_count(), num_instances)
    instances_canvas = Parallel(n_jobs)(delayed(generate_instance_canvas)(*p) for p in params_by_instance)
    final_image.extend(instances_canvas) # h (t w) 3

    title = [text_query, amr_tree_string]
    amr_tree_lines = len(amr_tree_string.split('\n'))
    max_sentence_length = max([len(tit) for tit in title])
    num_sentences = 2 + amr_tree_lines

    assert amr.num_nodes == len(map_nqs)
    max_nq_string = ' '.join([f'{str(key)} / ' + str(max_nq_idx) + ';' for key, max_nq_idx in zip(list(range(amr.num_nodes)), map_nqs.tolist())])
    title.append(max_nq_string)
    title = '\n'.join(title)
    font_size = 20
    linespacing = 2
    whole_image = np.vstack(final_image) / 255.0 # (# h) (t w) 3

    fig_with = max(whole_image.shape[1], (font_size*max_sentence_length))
    fig_height = whole_image.shape[0] + (num_sentences+linespacing*(num_sentences-1)) * font_size

    sep = whole_image.shape[0] / float(fig_height)
    fig, axs = plt.subplots(1, 2, figsize=(fig_with/100.0, fig_height/100.0))
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[0].imshow(whole_image)
    axs[0].set_position([(0.5 - whole_image.shape[1]/(float(fig_with)*2)),
                        0, 
                        whole_image.shape[1]/float(fig_with), whole_image.shape[0]/float(fig_height)])

    axs[1].xaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    axs[1].set_position([0.3, sep, 1 - sep, 1 - sep])
    G = to_networkx(amr)
    options = {
        "font_size": 20,
        "node_color": "red",
        "edgecolors": "blue",
        "linewidths": 0,
        "width": 5,
        "ax": axs[1],
        "labels": {key: str(key) for key in range(amr.num_nodes)},
    }
    nx.draw(G,**options)

    fig.text(0, sep, title, fontsize=font_size, linespacing=linespacing,)
    fig.savefig(directory)
    plt.close()     

