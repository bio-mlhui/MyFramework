# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import META_ARCH_REGISTRY
from models.layers.position_encoding import PositionEmbeddingSine
from models.layers.utils import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn_with_GlobalRegisters
from functools import partial
from einops import repeat
import copy
from models.layers.utils import _get_clones
from models.backbone.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from kan import KANLayer

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 num_feature_levels=None,
                 task_to_num_regs=None,
                 res_configs=None, 
                 ffn_configs=None,
                 deform_configs=None,):
        super().__init__()
        dropout, norm_layer = res_configs['dropout'], res_configs['norm_layer']
        if norm_layer == 'layer_norm':
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise ValueError()
        
        # token mixer
        self.norm1 = norm_layer(d_model)
        self.self_attn = MSDeformAttn_with_GlobalRegisters(d_model=d_model,
                                                           num_feature_levels=num_feature_levels,
                                                           task_to_num_regs=task_to_num_regs,
                                                           deform_configs=deform_configs)
        self.dropout1 = nn.Dropout(dropout)
        
        # dim mixer
        self.norm2 = norm_layer(d_model)
        mlp_act, mlp_ratio, mlp_bias, mlp_type =  ffn_configs.pop('mlp_act', 'relu'), ffn_configs.pop('mlp_ratio', 4), ffn_configs.pop('mlp_bias', True), ffn_configs.pop('mlp_type', 'mlp')
        mlp_hidden_dim = int(d_model * mlp_ratio)
        if mlp_act == 'relu':
            mlp_act = nn.ReLU
        else:
            raise ValueError()
        
        if mlp_type == "mlp":
            logging.debug("using MLP layer as FFN")
            ffn_layer = Mlp
        elif mlp_type == "swiglufused" or mlp_type == "swiglu":
            logging.debug("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif mlp_type == "identity":
            logging.debug("using Identity layer as FFN")
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        elif mlp_type == 'kan':
            logging.debug('using Kan layer as FFN')
            ffn_layer = KANLayer
        else:
            raise NotImplementedError  
        self.mlp = ffn_layer(in_features=d_model,
                             hidden_features=mlp_hidden_dim,
                             act_layer=mlp_act,
                             drop=dropout,
                             bias=mlp_bias,)      

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        def attn_residual_func(x):
            x = self.norm1(x)
            return self.dropout1(self.self_attn(self.with_pos_embed(x, pos), reference_points, x, spatial_shapes, level_start_index, padding_mask))

        def ffn_residual_func(x):
            return self.mlp(self.norm2(x))
        
        # b s c + 1 s c
        src = src + attn_residual_func(src)
        src = src + ffn_residual_func(src)

        return src

# 只有8的feature给到4
class FPNLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_fpn_levels,
                 fpn_configs=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_fpn_levels = num_fpn_levels
        
        norm_layer = fpn_configs['norm_layer']
        if norm_layer == 'layer_norm':
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise ValueError()
        self.input_norm = norm_layer(d_model)
        
        lateral_linears = [] 
        output_convs = []
        for idx in range(self.num_fpn_levels):
            lateral_linear = nn.Linear(d_model, d_model,)
            output_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1,)
            weight_init.c2_xavier_fill(lateral_linear)
            weight_init.c2_xavier_fill(output_conv)
            lateral_linears.append(lateral_linear)
            output_convs.append(output_conv)
        self.lateral_linears = nn.ModuleList(lateral_linears)
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, 
                fpn_srcs=None,
                output=None):
        pass

# 8, 16, 32的Feature 都给到4
    
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self,
                 d_model=256,
                 num_encoder_layers=None,
                 num_feature_levels=None,
                 num_fpn_levels=None,
                 task_to_num_regs=None,
                 # res
                 res_configs=None,
                 ffn_configs=None,
                 deform_configs=None,
                 fpn_configs=None,
        ):
        super().__init__()
        self.d_model = d_model
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model=d_model,
                                                            num_feature_levels=num_feature_levels,
                                                            task_to_num_regs=task_to_num_regs,
                                                            res_configs=res_configs,
                                                            deform_configs=deform_configs,
                                                            ffn_configs=ffn_configs,)
        fpn_layer = FPNLayer(d_model=d_model, num_fpn_levels=num_fpn_levels, fpn_configs=fpn_configs,)
        self.encoder = _get_clones(encoder_layer, num_encoder_layers)
        self.fpn_layers = _get_clones(fpn_layer, num_encoder_layers)
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        normal_(self.level_embed)

    # @staticmethod
    def get_reference_points(self, spatial_shapes, valid_ratios, num_registers, device):
        # lsit[h w], L; b L 2
        assert (valid_ratios == 1).all(), '都是1'
        batch_size = valid_ratios.shape[0]
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # b reg 2
            reg_ref_points = torch.zeros([batch_size, self.num_register_tokens, 2], dtype=torch.float32, device=device)
            reference_points_list.append(reg_ref_points)

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)) # h w
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 1 hw / b 1 (h*h_ratio = h_valid_max)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_) # 绝对坐标 / 最大长宽 = 相对坐标
            ref = torch.stack((ref_x, ref_y), -1) # b hw 2
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1) # b reg_hw_sigma 2
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # b reg_hw_sigma 1 2 * b 1 L 2   相对坐标 * 1 (因为mask都是0)

        return reference_points  # b reg_hw_sigma L 2


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) # b
        valid_W = torch.sum(~mask[:, 0, :], 1) # b
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # b 2, w的有效ratio, x的有效ratio
        return valid_ratio

    def forward(self, srcs, shapes, fpn_srcs=None):
        # list[b reg+hw_i c]
        masks = []  # list[b reg_hw_i]
        lvl_pos_embeds = []  # list[1 s c]
        spatial_shapes = shapes
        for lvl in range(len(srcs)):
            length = srcs[lvl].shape[1]
            lvl_pos_embeds.append(repeat(self.level_embed[lvl], 'c -> 1 s c', s=length).contiguous())
            masks.append(torch.zeros_like(srcs[lvl][..., 0]).bool())
        
        srcs = torch.cat(srcs, dim=1) # b reg_hw_i_reg_hw_i c
        masks = torch.cat(masks, dim=1)
        lvl_pos_embeds = torch.cat(lvl_pos_embeds, dim=1)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # b L 2

        
        output = srcs
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=srcs.device,)
        for _, (layer, fpn_layer) in enumerate(zip(self.encoder, self.fpn_layers)):
            output = layer(output, lvl_pos_embeds, reference_points, spatial_shapes, level_start_index, masks)
            fpn_srcs = fpn_layer(output=output, fpn_srcs=fpn_srcs,)
            
        return output, spatial_shapes, level_start_index





@META_ARCH_REGISTRY.register()
class MSRegisters_Encoder(nn.Module):
    # 假设: 每个scale的d_model一样
    def __init__(
        self,
        d_model,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
        task_to_num_regs,
        configs,
    ):  
        super().__init__()
        # 4, 8, 16, 32
        self.multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))
        self.encoded_scales = sorted(configs.pop('encoded_scales'), key=lambda x:self.multiscale_shapes[x].spatial_stride) # res3, res4, res5
        
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            num_encoder_layers=configs['num_encoder_layers'],
            num_feature_levels=len(self.encoded_scales),
            num_fpn_levels=len(self.multiscale_shapes) - len(self.encoded_scales),
            task_to_num_regs=task_to_num_regs,
            ffn_configs= configs['ffn_configs'],
            deform_configs=configs['deform_configs'],
            res_configs=configs['res_configs'],
            fpn_configs=configs['fpn_configs']
        )

    def forward(self, 
                multiscales=None, 
                **kwargs):
        # 'res2': {'feat': b 1+4+task+hw c, 'shape': (h, w), 
        # 'res3': {'feat': b 1+4+task+hw c, 'shape': (h, w),
        # 'res4': {'feat': b 1+4+task+hw c, 'shape': (h, w),
        # 'reg_task_order': list[str], }
        srcs = [] # list[b s_i c]
        shapes = [] # list[h w]
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            srcs.append(self.input_norm(multiscales[scale_name]['feat']))
            shapes.append(multiscales[scale_name]['shape'])
        fpn_srcs = None
        
        memory, fpn_srcs = self.transformer(srcs=srcs, shapes=shapes, fpn_srcs=fpn_srcs,)
        
        return memory, fpn_srcs        


