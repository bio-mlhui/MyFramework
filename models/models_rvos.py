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
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride
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

from models.pretrained_video_instance_decoder.pt_obj_decoder import pt_obj_decoder_entrypoint
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
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride

        create_obj_decoder = pt_obj_decoder_entrypoint(obj_decoder['name'])
        self.obj_decoder = create_obj_decoder(obj_decoder)
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
        self.is_pretraining_seg = is_pretraining_seg

        glorot(self.obj_query_proj)
        glorot(self.amrtext_wordEmbedding_proj)

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
        # b t nq c
        obj_queries, objdecoder_layer_preds = self.obj_decoder(samples)

        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device) 

        obj_queries, amr_token_feats, text_feats = self.fusion_module(query_feat=obj_queries, 
                                                                text_feats=text_feats,
                                                                amr_feats=amr_token_feats,
                                                                amr_pad_masks = amr_token_seg_ids==0,
                                                                text_pad_masks=text_pad_masks)
            
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
        
        if self.is_pretraining_seg:
            return {'objdecoder_objseg': objdecoder_layer_preds}
        memories = obj_queries
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
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
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

    return model, optimizer 





