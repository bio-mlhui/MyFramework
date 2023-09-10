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
from typing import List
import torchvision.transforms.functional as torchv_F
###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
from .position_encoding import build_position_encoding
from .model_utils import find_scale_from_multiscales, find_scales_from_multiscales, pad_1d_feats, \
    register_model, get_optimizer, get_total_grad_norm,\
        visualization_for_AMR_V0, zero_module, _get_clones
from .layers_unimodal_attention import FeatureResizer, CrossAttentionLayer, MLP, SelfAttentionLayer, FFNLayer 
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
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
                                lvl_pos_embed_flatten, mask_flatten,)
        
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

class Graph_Layer_v1(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        out = self.propagate(edge_index, size=None,  # keywords
                             x=x, edge_attr=edge_attr_2) # arguments
    
        # residual
        return out + x
    
  
    def message(self, x_j, x_i, edge_attr):
        """ Compute message for each neighbord
        Input:
            x_j: E dim
            x_i: E dim
            edge_embeds: E dim
        Output:
            message: E dim
        """     
        # 1. 你是一个man
        # x_j: man
        # x_i: m
        # edge: /    
        
        # 2. 你是(:arg1为car)的look, 
        # x_j: c  (/ car)
        # x_i: l  (/ look)
        # edge: :ARG1
        
        # message如果是describable的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car)
        # message如果是entity的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car的look)
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message



###########################################################################
# = swin-t 
# = amrbart word embedding
# = AMR Graph
###########################################################################
class AMR_v0(nn.Module):
    def __init__(self, 
                 d_model=256,
                 loss_weight_dict={
                     'loss_mask': 5,
                     'loss_dice': 5,
                 },
                 # video encoder
                 swint_pretrained_path='/home/xhh/pt/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':False,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':False,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':False,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':False,},],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32]],

                # amrtext
                amrbart_wordEmbedding_freeze = True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'amr_cross': '只有segid=2/3的node',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.}
                ,
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.1,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,}
                ,
                referent_decoder={
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': '只有segid=2/3的node',
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
                    
                    'nlayers': 9,
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'criterion':{
                        'name': 'no_matching_refer',
                        'mask_out_stride':4,
                        'losses': ['masks'],
                        'which_index': '只有第一个node'
                    },
                    'aux_loss': True},
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight_dict = loss_weight_dict
        
        # video encoder
        from .encoder_video import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=swint_pretrained_path,
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
        AMRBart = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        
        assert fusion.pop('name') == 'VisionLanguageFusionModule'
        assert fusion.pop('amr_cross') == '只有segid=2/3的node'
        self.cross_product = VisionLanguageFusionModule(**fusion)

        assert parsing_encoder.pop('name') == 'deform_video_2d_fpn'
        self.deform_multiscale_encoder = DeformVideo2D_with_FPN(**parsing_encoder)
        self.build_referent_decoder(referent_decoder)
    
    def init_parameters(self,): 
        for proj in self.video_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            
    def build_referent_decoder(self, referent_decoder, hidden_dim):
        from .layers_unimodal_attention import CrossAttentionLayer, MLP
        from .transformer import _get_clones
        self.referent_decoder_used_scales = referent_decoder['used_scales']
        self.referent_decoder_conved_scale = referent_decoder['conved_scale']
        self.referent_decoder_aux_loss = referent_decoder['aux_loss']
        self.referent_decoder_nlayers = referent_decoder['nlayers']
        self.referent_decoder_level_embed = nn.Embedding(len(self.referent_decoder_used_scales), hidden_dim)
        # amr_cross_video layer
        amr_cross_video_layer = referent_decoder['amr_cross_video_layer']
        assert amr_cross_video_layer.pop('name') == 'cross_attention'
        assert amr_cross_video_layer.pop('amr_cross') == '只有segid=2/3的node'
        self.referent_decoder_amr_cross_video_layers = _get_clones(CrossAttentionLayer(**amr_cross_video_layer),
                                                                   self.referent_decoder_nlayers)
        # amr self layer
        amr_self_layer = referent_decoder['amr_self_layer']
        assert amr_self_layer.pop('name') == 'graph_layer_v1'
        self.referent_decoder_amr_self_layers = _get_clones(Graph_Layer_v1(**amr_self_layer),
                                                            self.referent_decoder_nlayers)  
        # norm, mask out
    
        self.referent_decoder_norm = nn.LayerNorm(hidden_dim)
        self.referent_decoder_mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3) 
        # criterion
        criterion = referent_decoder['criterion']
        assert criterion.pop('name') == 'no_matching_refer'
        assert criterion.pop('which_index') == '只有第一个node'
        from .criterion_video import No_Matching_Refer
        self.referent_decoder_criterion = No_Matching_Refer(**criterion)
    
    def encode_video(self, samples):
        bb_out = self.video_swint(samples)
        nf, batch_size, *_ = bb_out[0].tensors.shape
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src)
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
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
        
        return amrs, amr_token_feats, amr_token_seg_ids
    
    def get_amr_fusion(self, amr_token_feats, amr_token_seg_ids):
        # b (V+E)max c, b (V+E)max
        who_fuse_with_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
        amr_fusion_tokens = [bt_feat[who_cross] for bt_feat, who_cross in zip(amr_token_feats, who_fuse_with_video)]
        amr_fusion_tokens, amr_fusion_pad_masks = pad_1d_feats(amr_fusion_tokens)
        return amr_fusion_tokens.permute(1, 0, 2), amr_fusion_pad_masks  

    def forward_prediction_heads(self, 
                                 output, mask_features, attn_mask_target_size=None,):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.referent_decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.referent_decoder_mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        assert attn_mask_target_size is not None
        attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
        attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
        
        return outputs_mask, attn_mask

    def get_decoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.referent_decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.referent_decoder_conved_scale)
        mask_features = multiscales[conved_feat_idx]
        multiscales = [multiscales[used_idx] for used_idx in used_feat_idxs]
        multiscales_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        multiscales_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        
        multiscales = [rearrange(scale_feat, 'b t c h w -> (t h w) b c') for scale_feat in multiscales]
        multiscales_poses = [rearrange(scale_pos, 'b t c h w -> (t h w) b c') for scale_pos in multiscales_poses]
        multiscales_pad_masks = [rearrange(scale_pad, 'b t h w -> b (t h w)') for scale_pad in multiscales_pad_masks] 
        multiscales = [scale_feat + self.referent_decoder_level_embed.weight[i][None, None, :] for i, scale_feat in enumerate(multiscales)]
        return multiscales, multiscales_pad_masks, multiscales_poses, mask_features
        
    def model_outputs(self, samples : NestedTensor, text_queries, text_auxiliary):
        """ text_auxiliary
        'graphs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        checkout = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # 抽文本的特征 list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(text_auxiliary, device, batch_size)
        
        # 文本和视频融合(thw和s)  SegID2/3_max b c, b SegID2/3_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_amr_fusion(amr_token_feats, amr_token_seg_ids,)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                      memory=amr_fusion_tokens, memory_key_padding_mask=amr_fusion_pad_masks,
                                       pos=None, query_pos=poses)
            checkout[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
        # 多模态特征进一步parsing
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_encoder((multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales))
        checkout['sampling_locations_by_layer'] = sampling_locations_by_layer
        checkout['attention_weights_by_layer'] = attention_weights_by_layer
        
        # 准备decoder的东西  thw b c
        size_list = [scale.shape[-2:] for scale in multiscales]
        multiscales, multiscales_pad_masks, multiscales_poses, mask_features = self.get_decoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        num_edges_by_batch = [g.num_edges for g in amrs]
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        amr_token_feats = amr_token_feats.permute(1,0,2)

        # decoder第0层尝试进行预测
        predictions_mask = [] # list[b t h w],
        checkout_mask = []
        # b t (V+E)max h w, b*head (V+E)max thw
        pred_masks, attn_mask = self.forward_prediction_heads(amr_token_feats, mask_features, attn_mask_target_size=size_list[0])
        predictions_mask.append(pred_masks[:, :, 0])
        checkout_mask.append(pred_masks)
        
        who_cross_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
        for i in range(self.referent_decoder_nlayers):
            scale_index = i % len(self.referent_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.referent_decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats,  # (V+E)max b c
                memory=multiscales[scale_index], # thw b c
                memory_mask=attn_mask,  # b*head (V+E)max thw
                memory_key_padding_mask=None, 
                pos=multiscales_poses[scale_index],  # thw b c
                query_pos=None,
            )
            # 每一层只有concept/constant nodes和video 融合
            amr_token_feats = amr_token_feats * (1 - who_cross_video.permute(1,0).float().unsqueeze(-1)) +\
                output * (who_cross_video.permute(1,0).float().unsqueeze(-1))

            # 更新节点特征
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.permute(1,0,2), amr_token_seg_ids)], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.permute(1,0,2), amr_token_seg_ids)], dim=0)
            assert sum(num_nodes_by_batch) == len(batched_nodes_feats) and sum(num_edges_by_batch) == len(batched_edge_feats)
            batched_nodes_feats = self.referent_decoder_amr_self_layers[i](batched_nodes_feats, batched_edge_index, 
                                                                          batched_edge_feats.clone())
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            
            pred_masks, attn_mask = self.forward_prediction_heads(amr_token_feats, mask_features, attn_mask_target_size=size_list[(i + 1) % len(self.referent_decoder_used_scales)])
            predictions_mask.append(pred_masks[:, :, 0])
            checkout_mask.append(pred_masks)
            who_cross_video = torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3) # b max
            
        assert len(predictions_mask) == self.referent_decoder_nlayers + 1
        checkout['predictions_mask_by_layer'] = checkout_mask
        return {
            'pred_masks': predictions_mask[-1], # b t H W
            'aux_outputs': [{"pred_masks": b} for b in predictions_mask[:-1]]
        }, checkout

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary):
        refer_pred = self.model_outputs(samples, text_queries, text_auxiliary)[0]
        refer_pred['pred_masks'] = refer_pred['pred_masks'].index_select(index=valid_indices, dim=1)
        
        output = {}
        output['pred_masks'] = rearrange(refer_pred['pred_masks'], 'b t h w -> t b 1 h w')
        nf, batch_size, *_ = output['pred_masks'].shape
        pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=refer_pred['pred_masks'].device) * 100
        pred_is_referred[..., 1] = -100
        output['pred_is_referred'] = pred_is_referred
        return output
    
    def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary):
        refer_pred = self.model_outputs(samples, text_queries, text_auxiliary)[0]
        refer_pred['pred_masks'] = refer_pred['pred_masks'].index_select(index=valid_indices, dim=1)
        refer_pred['aux_outputs'] = [layer_out['pred_masks'].index_select(index=valid_indices, dim=1) for layer_out in refer_pred['aux_outputs']]
        losses = {}
        targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
        outputs_without_aux = {k: v for k, v in refer_pred.items() if k != "aux_outputs"}
        losses = self.referent_decoder_criterion(outputs_without_aux, targets)
        if self.referent_decoder_aux_loss:
            for i, aux_outputs in enumerate(refer_pred['aux_outputs']):
                l_dict_i = self.referent_decoder_criterion(aux_outputs, targets)
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        
        assert set(losses.keys()).issubset(self.loss_weight_dict.keys())

        loss = sum((losses[k] * self.loss_weight_dict[k] for k in losses.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * self.loss_weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm
    
    def visualize(self, samples : NestedTensor, valid_indices : Tensor, text_queries, text_auxiliary, saved_path,
                  is_train_set=False, targets=None):
        refer_pred, checkout = self.model_outputs(samples, text_queries, text_auxiliary)
        if is_train_set:
            losses = {}
            targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
            outputs_without_aux = {k: v for k, v in refer_pred.items() if k != "aux_outputs"}
            losses = self.referent_decoder_criterion(outputs_without_aux, targets)
            if self.referent_decoder_aux_loss:
                for i, aux_outputs in enumerate(refer_pred['aux_outputs']):
                    l_dict_i = self.referent_decoder_criterion(aux_outputs, targets)
                    for k in l_dict_i.keys():
                        assert k in losses
                        losses[k] += l_dict_i[k]  
            
            assert set(losses.keys()).issubset(self.loss_weight_dict.keys())

            loss = sum((losses[k] * self.loss_weight_dict[k] for k in losses.keys()))
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                print(losses)
                sys.exit(1)
            loss.backward()
            grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
            
        visualization_for_AMR_V0(videos=samples.tensors.index_select(0, valid_indices), text_query=text_queries, directory=saved_path, 
                    targets=targets, refer_pred=refer_pred, losses=losses, draw_all_instances=False)
        
        return None
   

# 改成3d proj
class AMR_V0_v1():
    pass

# 改成matching
class AMR_V0_v2():
    pass

# amrbart的encoding训练
class AMR_V0_v3():
    pass

# 加上64的scale
class AMR_V0_v4():
    def encode_video(self, samples):
        bb_out = self.video_swint(samples)
        nf, batch_size, *_ = bb_out[0].tensors.shape
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src)
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(pad_mask.float(), size=src_proj_l.shape[-2:],mode='nearest-exact').bool()
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses
 

# proj变成多层
class AMR_V0_v4():
    pass

# 改成MTTR的那种融合方式, 因为如果只关注32x的feature, 能够使用(thw) parsing, 而不是2d parsing
class AMR_V0_v4():
    pass

# 改变下融合/cross node的选择
class AMR_V0_v4():
    pass

# 加上LLN
class AMR_vo_v4():
    pass

# temporal queries
class AMR_vo_v4():
    pass

###########################################################################
# 相比v0, 用object decoder
###########################################################################
class AMR_V1(nn.Module):
    pass

###########################################################################
# 相比v0, 用amr_without_variable
###########################################################################
class AMR_V2(nn.Module):
    pass

###########################################################################
# 相比v0, 用graph_layer_message_cross_video
###########################################################################
class AMR_V3(nn.Module):
    pass


class LinAMR_V0(nn.Module):
    pass



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
        token_feats, token_pad_masks, token_sentence_feats = self.encode_text(text_queries, device)
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
    def sample(self, asembles, auxiliary, visualize=False):
        """
        asembles: list[list[vid, text, meta, callback], num_of_asemble], batch_size
        auxiliary: list[dict], batch_size
        """
        by_batch_mask_preds = []
        by_batch_preds_probs = []
        for s_asem, s_auxiliary in zip(asembles, auxiliary):
            # list[[vid, text, {has_ann}, callback]], num_asembles
            samples_without_padding, text_queries, asem_metas, callbacks = list(zip(s_asem))
            num_asembles = len(samples_without_padding)
            sizes_without_pad = [samples_without_padding[i].shape[[0, 2, 3]] for i in range(num_asembles)]
            samples = nested_tensor_from_videos_list_with_stride(samples_without_padding, max_stride=self.max_stride)
            T, _, _, H, W, device = *samples.tensors.shape, samples.tensors.device
            
            perFrame_has_ann = [asm['has_ann'] for asm in asem_metas] # list[t]
            
            # t -> T -> numAs T
            perFrame_has_ann = F.pad(perFrame_has_ann.float(), pad=(0, T-len(perFrame_has_ann))).bool().unsqueeze(0).repeat(num_asembles, 1)             
            
            decoder_layer_preds = self.model_outputs(samples, text_queries, s_auxiliary, perFrame_has_ann)['refdecoder_refseg']
            
            last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
            # bt' n H/4 W/4
            out_masks_logits =  last_layer_preds['pred_mask_logits'] 
            num_queries = out_masks_logits.shape[1]
            query_pred_refer_prob = last_layer_preds['pred_refer_logits'] # bt' n 2
            # bt' n H W
            query_pred_masks = F.interpolate(out_masks_logits, scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False) 
            # query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold)
            
            # b T n H W / b T n 2
            asems_mask_preds = torch.zeros([num_asembles, T, num_queries, H, W ], device=device).float().flatten(0, 1)
            asems_refer_prob_preds = torch.zeros([num_asembles, T, num_queries, query_pred_refer_prob.shape[-1]], device=device).float().flatten(0, 1)
            asems_mask_preds[perFrame_has_ann] = query_pred_masks
            asems_refer_prob_preds[perFrame_has_ann] = query_pred_refer_prob
            
            asems_mask_preds = rearrange(asems_mask_preds, '(b T) n H W -> b T n H W', b=num_asembles, T=T)
            asems_refer_prob_preds = rearrange(asems_refer_prob_preds, '(b T) n -> b T n', b=num_asembles, T=T)
            
            # unpad
            pred_masks_by_asem = [] # n t' h w, bool
            pred_probs_by_asem = [] # n t', float
            for asem_vid, asem_text, asem_pred_masks, asem_pred_probs, asem_has_ann, size_without_pad, asem_callback \
                in zip(samples_without_padding, text_queries, asems_mask_preds,
                       asems_refer_prob_preds, perFrame_has_ann, sizes_without_pad, callbacks):
                s_nf, s_h, s_w = size_without_pad
                # T n H W -> t n h w -> n t' h w(after aug)
                asem_pred_masks = asem_pred_masks[:s_nf, :, :s_h, :s_w][asem_has_ann]
                # T n -> t n -> n t'(after aug)
                asem_pred_probs = asem_pred_probs[:s_nf][asem_has_ann]
                asem_pred = {'masks': asem_pred_masks.permute(1, 0, 2, 3), 'refer_prob': asem_pred_probs.permute(1, 0)}
                for cb in asem_callback:
                    asem_vid, asem_text, asem_pred = cb(asem_vid, asem_text, asem_pred)
                pred_masks_by_asem.append(asem_pred['masks'])
                pred_probs_by_asem.append(asem_pred['refer_prob'])
            by_batch_mask_preds.append(
                torch.stack(pred_masks_by_asem, dim=0).mean(dim=0).sigmoid() > self.decoder_mask_threshold
            )
            # list[n t' 2] -> n t' 2 -> n t'
            by_batch_preds_probs.append(torch.stack(pred_probs_by_asem, dim=0).mean(dim=0).softmax(-1)[..., 0])
        # visualization function
        
        return {
            'query_pred_masks': by_batch_mask_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
        
    
    def forward(self, samples, text_queries, auxiliary, targets, visualize=False):
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





















































































































































































































###########################################################################


@register_model
def amr_v0(device, model_configs):
    model = AMR_v0()
    model.to(device)

    optmization_configs = model_configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 

@register_model
def text_v0(device, configs):
    model = Text_V0(
        d_model=configs['d_model'],
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
def text_v0_v1(device, configs):
    model = Text_V0_V1(
        d_model=configs['d_model'],
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
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
        
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



###########################################################################
# 
###########################################################################
# @register_model
# def clip_v0(device, model_configs):
    
#     configs = model_configs
#     model =  CLIP_v0(
#         d_model=configs.d_model,
#         object_classes=configs.object_classes,
#         weight_dict=vars(configs.weight_dict),
#         object_decoder_configs=configs.object_decoder,
#         referent_decoder_configs=configs.referent_decoder,
#     )
#     model.to(device)

#     optmization_configs = configs.optimization
#     param_dicts = [
#         {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
#         {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
#         "lr": optmization_configs.vid_backbone_lr},
#         {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
#         "lr": optmization_configs.text_backbone_lr}, 
#     ] # CHECK params dict every run
#     optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

#     return model, optimizer 
# class CLIP_v0(nn.Module):
#     def __init__(self, 
#                  d_model,
#                  weight_dict,
#                  object_classes,
                 
#                  object_decoder_configs,
#                  referent_decoder_configs) -> None:
#         super().__init__()
#         self.weight_dict = weight_dict
#         self.d_model = d_model
        
#         self.object_classes = object_classes
        
#         from .encoder_fusion import VisionLanguageFusionModule
#         from .encoder_multiscale import multiscale_encoder_entrypoints
#         self.cross_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
#         from .transformer import TransformerEncoder, TransformerEncoderLayer
#         self.self_parser = TransformerEncoder(TransformerEncoderLayer(d_model=d_model,
#                                                                         nheads=8,
#                                                                         dim_feedforward=2048,
#                                                                         dropout=0.1,
#                                                                         activation='relu',
#                                                                         normalize_before=False), 6)

#         self.object_decoder = object_detector(object_decoder_configs, d_model=d_model) 
        
#         self.referent_decoder = referent_decoder_forSequenceText(referent_decoder_configs, d_model=d_model)
#         from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
#         clip_model = CLIPModel.from_pretrained("/home/xhh/pt/clip_base")
#         for p in clip_model.parameters():
#             p.requires_grad_(False)
#         self.clip_video_encoder = clip_model.vision_model
#         self.clip_visual_projection = clip_model.visual_projection
#         self.clip_text_encoder = clip_model.text_model
#         self.clip_text_projection = clip_model.text_projection
#         self.clip_processor = CLIPProcessor.from_pretrained("/home/xhh/pt/clip_base")
#         vocab = self.clip_processor.tokenizer.get_vocab()
#         self.object_class_token_ids = [vocab[w] for w in object_classes]
#         self.vid_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d')
#         self.text_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='1d')
    
#     def proj_video(self, samples):
#         nf, batch_size, *_ = samples.tensors.shape
#         vid_frames = rearrange(samples.tensors, 't b c h w -> (b t) c h w')
        
#         # .last_hidden_state b s c    # 24 50 512
#         # .pooler-output: b c  # 24 1024
#         # .hidden_states: (b s c)
#         vision_outputs = self.clip_video_encoder(
#             pixel_values=vid_frames,
#             output_attentions=True,
#             output_hidden_states=True,
#             return_dict=True,
#         )
        
#         video_feat = vision_outputs.last_hidden_state[:, 1:] # bt seq c

#         video_feat = self.clip_visual_projection(video_feat)
        
#         video_feat = video_feat / video_feat.norm(p=2, dim=-1, keepdim=True)
        
#         video_feat = rearrange(video_feat, '(b t) (h w) c -> b t c h w', b=batch_size,t=nf, h=7, w=7)
#         pos = self.vid_pos_embed(video_feat, None) # b t c h w
#         orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # t b h w -> b t h w
#         pad_mask = F.interpolate(orig_pad_mask.float(), size=video_feat.shape[-2:]).to(torch.bool)
#         return video_feat, pad_mask, pos, [[1,32]]  
    
#     def proj_text(self, text_queries, device):
#         text_encoding = self.clip_processor(text=text_queries,
#                                             return_tensors='pt', padding=True)
#         input_ids = text_encoding['input_ids'].to(device) # b max
#         attention_mask = text_encoding['attention_mask'].to(device) # 0代表没有Padding
#         text_encoder_output = self.clip_text_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#         text_pad_mask = attention_mask.ne(1).bool() # b max 
        
#         text_feats = text_encoder_output.last_hidden_state
#         text_sentence_feats = text_encoder_output.pooler_output
        
#         text_feats = self.clip_text_projection(text_feats)
#         text_sentence_feats = self.clip_text_projection(text_sentence_feats)
        
#         text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
#         text_sentence_feats = text_sentence_feats / text_sentence_feats.norm(p=2, dim=-1, keepdim=True)
        
#         return {
#             'token_feats': text_feats,
#             'token_pad_masks': text_pad_mask,
#             'token_sentence_feats': text_sentence_feats
#         }               
    
#     @torch.no_grad()
#     def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
#         nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device

#         video_feat, video_pad_mask, video_pos, decs = self.proj_video(samples=samples) 
#         text_encoder_output = self.proj_text(text_queries=text_queries, device=device)
#         # b max c, b max
#         crossed_text_feats, crossed_text_pad_mask = text_encoder_output['token_feats'].clone(),\
#             text_encoder_output['token_pad_masks'].clone()
#         object_class_embeds = self.get_word_embeds(self.object_class_token_ids, device=device)  
#         object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
#         crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
#         crossed_text_pos = self.text_pos_embed(crossed_text_pad_mask, hidden_dim=crossed_text_feats.shape[-1]).permute(2, 0, 1) # b c max -> max b c
#         crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
#         crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
#         crossed_text_pos = torch.cat([crossed_text_pos, torch.zeros_like(object_class_embeds)], dim=0)
#         video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
#         video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
#         video_feat = self.cross_module(tgt=video_feat,
#                                 memory=crossed_text_feats,
#                                 memory_key_padding_mask=crossed_text_pad_mask,
#                                 pos=crossed_text_pos,
#                                 query_pos=video_pos) # 6 * 49
#         video_feat = self.self_parser(src=video_feat,
#                                        mask=None, src_key_padding_mask=None,
#                                        pos=video_pos)
#         video_feat = rearrange(video_feat, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         video_pos = rearrange(video_pos, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         decoder_video_input = {
#             'multiscales': [video_feat],
#             'multiscale_pad_masks': [video_pad_mask],
#             'multiscale_poses': [video_pos],
#             'multiscale_des': decs
#         }
#         # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
#         object_decoder_output, _  = self.object_decoder(decoder_video_input, 
#                                                                       return_loss=False,
#                                                                       targets=None,
#                                                                       valid_indices=valid_indices)
        
#         out, _ = self.referent_decoder(decoder_video_input, 
#                                                             object_decoder_output, 
#                                                             text_encoder_output,
#                                                             return_loss=False, 
#                                                             targets=None, 
#                                                             matching_results=None,
#                                                             valid_indices=valid_indices)                                                          
#         # pred_logits: b t n classes, real
#         # pred_boxes: b t n 4, [0, 1]
#         # pred_masks: b t n h w, real
#         # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
#         # final_token_feats: 
#         output = {}
#         if len(out['pred_masks'].shape) == 5:
#             output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
#             output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
#         else:
#             assert len(out['pred_masks'].shape) == 4
#             output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
#             nf, batch_size, *_ = output['pred_masks'].shape
#             pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
#             pred_is_referred[..., 1] = -100
#             output['pred_is_referred'] = pred_is_referred

#         return output
    

#     def get_word_embeds(self, token_ids, device):
#         if type(token_ids[0]) == str:
#             vocab = self.clip_processor.tokenizer.get_vocab()
#             token_ids = [vocab[w] for w in token_ids]
#         token_ids = torch.tensor(token_ids, device=device)
#         token_embeds = self.clip_text_encoder.embeddings.token_embedding(token_ids) # 7 512
#         token_embeds = self.clip_text_projection(token_embeds)
        
#         token_embeds = token_embeds / token_embeds.norm(p=2, dim=-1, keepdim=True)
#         return token_embeds
    
#     # get the loss, and the model has gradients;
#     def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
#                 visualize=False, saved_path=None):
#         """
#         'graphs': list[T(2 E_i)]
#         'seg_ids': b (V+E)max
#         'token_splits': list[list[int]]
#         'tokens_ids': b max
#         """
        
#         losses = {}
        
#         nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device

#         video_feat, video_pad_mask, video_pos, decs = self.proj_video(samples=samples) 
#         text_encoder_output = self.proj_text(text_queries=text_queries, device=device)
#         # b max c, b max
#         crossed_text_feats, crossed_text_pad_mask = text_encoder_output['token_feats'].clone(),\
#             text_encoder_output['token_pad_masks'].clone()
#         object_class_embeds = self.get_word_embeds(self.object_class_token_ids, device=device)  
#         object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
#         crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
#         crossed_text_pos = self.text_pos_embed(crossed_text_pad_mask, hidden_dim=crossed_text_feats.shape[-1]).permute(2, 0, 1) # b c max -> max b c
#         crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
#         crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
#         crossed_text_pos = torch.cat([crossed_text_pos, torch.zeros_like(object_class_embeds)], dim=0)
#         video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
#         video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
#         video_feat = self.cross_module(tgt=video_feat,
#                                 memory=crossed_text_feats,
#                                 memory_key_padding_mask=crossed_text_pad_mask,
#                                 pos=crossed_text_pos,
#                                 query_pos=video_pos) # 6 * 49
#         video_feat = self.self_parser(src=video_feat,
#                                        mask=None, src_key_padding_mask=None,
#                                        pos=video_pos)
#         video_feat = rearrange(video_feat, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         video_pos = rearrange(video_pos, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         decoder_video_input = {
#             'multiscales': [video_feat],
#             'multiscale_pad_masks': [video_pad_mask],
#             'multiscale_poses': [video_pos],
#             'multiscale_des': decs
#         }
#         all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
#                                                             class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_class_token_ids)})
#         # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
#         object_decoder_output, object_loss_dict = self.object_decoder(decoder_video_input, 
#                                                                       return_loss=True,
#                                                                       targets=all_instance_targets,
#                                                                       valid_indices=valid_indices)
#         matching_results = object_loss_dict.pop('matching_results')
#         losses.update(object_loss_dict)
        
#         # 进行referent 推断
#         # referent应该和objects的matching结果一致, 
#         referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
#         refer_pred, refer_loss_dict = self.referent_decoder(decoder_video_input, 
#                                                             object_decoder_output, 
#                                                             text_encoder_output,
#                                                             return_loss=True, 
#                                                             targets=referent_targets, 
#                                                             matching_results=matching_results,
#                                                             valid_indices=valid_indices)
#         losses.update(refer_loss_dict)
        
#         assert set(losses.keys()).issubset(self.weight_dict.keys())

#         loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
#         if not math.isfinite(loss.item()):
#             print("Loss is {}, stopping training".format(loss.item()))
#             print(losses)
#             sys.exit(1)
#         loss.backward()

#         loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
#         loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
#         grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
            
#         return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

# class Referent_Decoder_forSequenceText(nn.Module):
#     def __init__(
#         self, # decoder               
#         in_channels,
#         hidden_dim: int,
        
#         nheads: int,
#         pre_norm: bool,
#         mask_dim: int,
#         enforce_input_project: bool,
#         dim_feedforward,

#         # important
#         nqueries,
#         dec_layers: int,
#         used_scales,
#         conved_scale,
#         matching_configs,
#         aux_loss,
   
#     ):
#         super().__init__()
#         self.query_pos = nn.Embedding(nqueries, hidden_dim)
#         self.nqueries = nqueries
#         self.hidden_dim = hidden_dim
#         self.num_feature_levels = len(used_scales)
#         self.used_scales = used_scales
#         assert dec_layers % self.num_feature_levels == 0
#         self.conved_scale = conved_scale
#         if self.num_feature_levels == 1:
#             self.level_embed = None
#         else:
#             self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
#         self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
#         for _ in range(self.num_feature_levels):
#             if in_channels != hidden_dim or enforce_input_project:
#                 # should be 
#                 raise NotImplementedError()
#             else:
#                 self.input_proj.append(nn.Sequential())  
                     
#         self.num_heads = nheads
#         self.num_layers = dec_layers

#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()

#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )        

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.aux_loss = aux_loss
        
#         self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
#         self.class_embed = nn.Linear(hidden_dim, 2)     
#         create_criterion = matching_entrypoints(matching_configs.name)
#         self.criterion = create_criterion(matching_configs)


#     def forward(self, 
#                 video_features_args,
#                 object_args,
#                 text_args,
#                 return_loss=False,
#                 targets=None,
#                 matching_results=None,
#                 valid_indices=None):
#         # b t c h w
#         multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
#             = video_features_args['multiscales'], video_features_args['multiscale_pad_masks'], \
#                 video_features_args['multiscale_poses'], video_features_args['multiscale_des']
                
#         # n b c
#         objects_queries = object_args['object_embeds'].permute(1,0,2)
#         num_objects, batch_size, _ = objects_queries.shape
        
#         used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
#         used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
#         used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
#         conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
#         mask_features = multiscales[conved_feat_idx]
#         batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
#         cross_memories_by_scale = []
#         cross_memory_poses_by_scale = []
#         size_list = []
#         for i in range(self.num_feature_levels):
#             # 32x 
#             size_list.append(used_video_feats[i].shape[-2:])
#             scale_feats = used_video_feats[i]
#             scale_feats = self.input_proj[i](scale_feats) # b t c h w
#             scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
#             if self.num_feature_levels != 1:
#                 scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
                
#             memory = torch.cat([scale_feats, objects_queries], dim=0) # (thw + n) b c
#             pos = torch.cat([rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'), torch.zeros_like(objects_queries)], dim=0)
#             cross_memories_by_scale.append(memory) # thw+n b c
#             cross_memory_poses_by_scale.append(pos) # thw+n b c
       

#         token_sentence_feats = text_args['token_sentence_feats'] # b c
#         output = repeat(token_sentence_feats, 'b c -> n b c', n=self.nqueries)
#         # output_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=output.shape[0], b=batch_size)
#         output_pos = repeat(self.query_pos.weight, 'n c -> n b c', b=batch_size)
        
        
#         predictions_mask = [] # list[b t n H/4 W/4],
#         predictions_class = [] # b n 2
#         attn_mask_size = size_list[0] 
#         # b t n h w, b*h n thw+num_objects
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size,
#                                                  num_objects=len(objects_queries))
#         predictions_class.append(outputs_class)
#         if valid_indices is not None:
#             outputs_mask = outputs_mask.index_select(dim=1, index=valid_indices)
#         predictions_mask.append(outputs_mask)
        
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels 
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
#             output = self.transformer_cross_attention_layers[i](
#                 tgt=output,  # max b c
#                 memory=cross_memories_by_scale[level_index], # thw b c
#                 memory_mask=attn_mask, # 
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=cross_memory_poses_by_scale[level_index],  # thw b c
#                 query_pos=output_pos, # max b c
#             )

#             output = self.transformer_self_attention_layers[i](
#                 output, # n b c
#                 tgt_mask=None,
#                 tgt_key_padding_mask=None, # b n 
#                 query_pos=output_pos, # n b c
#             )
            
#             output = self.transformer_ffn_layers[i](
#                 output # n b c
#             )
                
#             attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

#             outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size,
#                                                  num_objects=len(objects_queries))
#             predictions_class.append(outputs_class)
#             if valid_indices is not None:
#                 outputs_mask = outputs_mask.index_select(dim=1, index=valid_indices)
#             predictions_mask.append(outputs_mask)
            
#         assert len(predictions_mask) == self.num_layers + 1
#         outputs = {
#             'pred_logits': predictions_class[-1], # b nq 2
#             'pred_masks': predictions_mask[-1], # b t nq H W
#             'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
#         }

#         if return_loss:
#             assert targets is not None and matching_results is not None
#             losses = self.forward_refer_loss(outputs, targets, matching_results)
#             return outputs, losses
#         else:
#             assert targets is None
#             return outputs, None

#     def forward_refer_loss(self, out, targets, matching_results):
#         """
#         Params:
#             targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
#         """
#         losses = {}
        
#         outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
        
#         indices = self.criterion.matching(outputs_without_aux, targets)
        
#         losses = self.criterion(out, targets, indices)
#         if self.aux_loss:
#             for i, aux_outputs in enumerate(out['aux_outputs']):
#                 indices_i = self.criterion.matching(aux_outputs, targets)
#                 l_dict_i = self.criterion(aux_outputs, targets, indices_i)
                
#                 for k in l_dict_i.keys():
#                     assert k in losses
#                     losses[k] += l_dict_i[k]  
#         return losses
    
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         return [
#             {"pred_logits": a, "pred_masks": b}
#             for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#         ]  
    
#     def forward_prediction_heads(self, output, mask_features, 
#                                  attn_mask_target_size=None, 
#                                  return_attn_mask=True,
#                                  num_objects=None):
#         bs, nf, *_= mask_features.shape # b t c h w
#         decoder_output = self.decoder_norm(output)  # n b c
#         decoder_output = decoder_output.transpose(0, 1)  # b n c
        
#         mask_embed = self.mask_embed(decoder_output)  # b n c
#         mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
#         outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
#         attn_mask = None
#         if return_attn_mask:
#             assert attn_mask_target_size is not None and num_objects is not None
#             attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
#             attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
#             attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
#             attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
            
#             pad_objects_cross = F.pad(attn_mask.float(), pad=[0, num_objects, 0, 0], value=0.).bool() # b*h n thw+num_objects
            
#         outputs_class = self.class_embed(decoder_output) 
           
#         return outputs_class, outputs_mask, pad_objects_cross

# def referent_decoder_forSequenceText(decoder_configs, d_model):
#     configs = vars(decoder_configs)
#     return Referent_Decoder_forSequenceText(
#                             in_channels=d_model,
#                             hidden_dim=d_model,
#                             nheads=configs['nheads'],
#                             pre_norm=configs['pre_norm'],
#                             mask_dim=configs['mask_dim'],
#                             enforce_input_project=configs['enforce_proj_input'],
#                             dim_feedforward=configs['dff'],
#                             # important
#                             nqueries=configs['nqueries'],
#                             dec_layers=configs['nlayers'],
#                             used_scales=configs['used_scales'],
#                             conved_scale=configs['conved_scale'],
#                             matching_configs=decoder_configs.matching,
#                             aux_loss=configs['aux_loss'],)

# class ObjectDetector(nn.Module):
#     def __init__(
#         self, # decoder 
#         num_classes,
#         in_channels,
#         hidden_dim: int,
#         nheads: int,
#         dim_feedforward: int,
#         pre_norm: bool,
#         mask_dim: int,
#         enforce_input_project: bool,

#         # important
#         num_queries: int,
#         dec_layers: int,
#         used_scales,
#         conved_scale,
#         matching_configs,
#         aux_loss,
   
#     ):
#         super().__init__()
#         assert num_queries > 10
#         self.query_feat = nn.Embedding(num_queries, hidden_dim)
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.num_queries = num_queries

#         self.hidden_dim = hidden_dim
        
#         self.num_feature_levels = len(used_scales)
#         self.used_scales = used_scales
#         assert dec_layers % self.num_feature_levels == 0
#         self.conved_scale = conved_scale
#         if self.num_feature_levels == 1:
#             self.level_embed = None
#         else:
#             self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
#         self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
#         for _ in range(self.num_feature_levels):
#             if in_channels != hidden_dim or enforce_input_project:
#                 # should be 
#                 raise NotImplementedError()
#             else:
#                 self.input_proj.append(nn.Sequential())  
                     
#         self.num_heads = nheads
#         self.num_layers = dec_layers
#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()

#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.aux_loss = aux_loss

#         self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
#         self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

#         create_criterion = matching_entrypoints(matching_configs.name)
#         self.criterion = create_criterion(matching_configs, num_classes=num_classes)

#     def forward(self, video_args,return_loss=False, targets=None, valid_indices=None):
#         """
#         query_feats: n b c
#         video: b t c h w
#         text: b s c
#         """
#         # make sure that the video features are fused with the text features before
#         multiscales = [scale_feat.clone() for scale_feat in video_args['multiscales']]
#         multiscale_masks = [pad_mask.clone() for pad_mask in video_args['multiscale_pad_masks']]
#         multiscale_poses = [pos.clone() for pos in video_args['multiscale_poses']]
#         multiscale_dec = copy.deepcopy(video_args['multiscale_des'])
        
#         used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
#         used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
#         used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
#         conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
#         mask_features = multiscales[conved_feat_idx]

#         batch_size, nf, *_, device = *mask_features.shape, mask_features.device

#         query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

#         output = query_feats
        
#         srcs = []
#         poses = []
#         size_list = []
#         for i in range(self.num_feature_levels):
#             # 32x -> 16x -> 8x
#             size_list.append(used_video_feats[i].shape[-2:])
#             scale_feats = used_video_feats[i]
#             scale_feats = self.input_proj[i](scale_feats) # b t c h w
#             scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
#             if self.num_feature_levels != 1:
#                 scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
#             srcs.append(scale_feats) # thw b c
#             poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
#         predictions_class = [] # list[b nq k+1], init -> 32x -> 16x -> 8x
#         predictions_mask = [] # list[b nq t H/4 W/4], 
#         attn_mask_size = size_list[0]
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size)
#         if valid_indices is not None: # [3]
#             outputs_mask = outputs_mask.index_select(index=valid_indices, dim=2)
#         predictions_class.append(outputs_class)
#         predictions_mask.append(outputs_mask)
        
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels
#             # b*h n thw
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
#             output = self.transformer_cross_attention_layers[i](
#                 tgt=output,  # n b c
#                 memory=srcs[level_index], # thw b c
#                 memory_mask=attn_mask, # bh n thw
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=poses[level_index],  # thw b c
#                 query_pos=query_pos, # n b c
#             )

#             output = self.transformer_self_attention_layers[i](
#                 output, # n b c
#                 tgt_mask=None,
#                 tgt_key_padding_mask=None, # b n 
#                 query_pos=query_pos, # n b c
#             )
#             output = self.transformer_ffn_layers[i](
#                 output # n b c
#             )
            
#             attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
#             # (b nq 2, real), (b nq t H W, real), bh n thw
#             outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size)
#             predictions_class.append(outputs_class)
#             if valid_indices is not None: # [3]
#                 outputs_mask = outputs_mask.index_select(index=valid_indices, dim=2)
#             predictions_mask.append(outputs_mask)

#         assert len(predictions_class) == self.num_layers + 1
#         outputs = {
#             'object_embeds': output.permute(1, 0, 2), # b n c
#             'pred_logits': predictions_class[-1], # b nq k+1
#             'pred_masks': predictions_mask[-1], # b nq t H W
 
#             'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
#         } # {'object_embeds': b n c, 'object_box_diff':..}

#         if return_loss:
#             assert targets is not None
#             losses, indices = self.forward_object_loss(outputs, targets)
#             losses.update({'matching_results': indices})
#             return outputs, losses
#         else:
#             assert targets is None
#             return outputs, None
    
#     def forward_object_loss(self, out, targets):
#         """
#         Params:
#             targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
#         """
#         losses = {}
        
#         outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
        
#         indices = self.criterion.matching(outputs_without_aux, targets)
        
#         losses = self.criterion(out, targets, indices)
#         if self.aux_loss:
#             for i, aux_outputs in enumerate(out['aux_outputs']):
#                 indices_i = self.criterion.matching(aux_outputs, targets)
#                 l_dict_i = self.criterion(aux_outputs, targets, indices_i)
                
#                 for k in l_dict_i.keys():
#                     assert k in losses
#                     losses[k] += l_dict_i[k]  
#         return losses, indices
    
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         """
#         Input:
#             - output_class:
#                 list[T(tb n classes)]
#             - outputs_seg_masks:
#                 list[T(tb n H W)]
#             - outputs_boxes:
#                 list[T(tb n 4)]
#         """
#         return [
#             {"pred_logits": a, "pred_masks": b}
#             for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#         ]
        
#     def forward_prediction_heads(self, output, mask_features, 
#                                  attn_mask_target_size=None, 
#                                  return_cls=True, return_attn_mask=True, return_box=False):
#         bs, nf, *_= mask_features.shape # b t c h w
#         decoder_output = self.decoder_norm(output)  # n b c
#         decoder_output = decoder_output.transpose(0, 1)  # b n c
        
#         mask_embed = self.mask_embed(decoder_output)  # b n c
#         mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
#         outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features).permute(0, 2, 1, 3, 4)  # b n t h w
        
#         attn_mask = None
#         outputs_class = None
#         if return_attn_mask:
#             assert attn_mask_target_size is not None
#             attn_mask = outputs_mask.detach().flatten(0,1) # bn t h w
#             attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bn t h w, real
#             attn_mask = repeat(attn_mask, '(b n) t h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads,
#                                n=self.num_queries) # b*h n (t h w)
#             attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
#         if return_cls:
#             outputs_class = self.class_embed(decoder_output)  # b n k+1
            
#         return outputs_class, outputs_mask, attn_mask

# def object_detector(decoder_configs, d_model):
#     configs = vars(decoder_configs)
#     return ObjectDetector(
#         num_classes=configs['num_classes'],
#         in_channels=d_model,
#         hidden_dim=d_model,
#         nheads=configs['nheads'],
#         dim_feedforward=configs['dff'],
#         pre_norm=configs['pre_norm'],
#         mask_dim=configs['mask_dim'],
#         enforce_input_project=configs['enforce_proj_input'],
#         # important
#         num_queries=configs['num_queries'],
#         dec_layers=configs['nlayers'],
#         used_scales=configs['used_scales'],
#         conved_scale=configs['conved_scale'],
#         matching_configs=decoder_configs.matching,
#         aux_loss=configs['aux_loss'],)



