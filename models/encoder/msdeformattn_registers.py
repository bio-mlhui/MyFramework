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
from models.layers.position_encoding import build_position_encoding
from models.backbone.metaformer_build_tool import  LayerNormGeneral, StarReLU, LayerNormWithoutBias
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 multiscale_shapes=None, fpn_scales=None, deform_scales=None, fpn_configs=None,
                 num_feature_levels=None,
                 task_to_num_regs=None,
                 mlp_configs=None,
                 deform_configs=None,
                 drop_path=None,):
        super().__init__()
        self.multiscale_shapes = multiscale_shapes
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        # token mixer
        self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        self.self_attn = MSDeformAttn_with_GlobalRegisters(d_model=d_model,
                                                           num_feature_levels=num_feature_levels,
                                                           task_to_num_regs=task_to_num_regs,
                                                           deform_configs=deform_configs)

        self.alias_convs = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1) for _ in range(len(self.fpn_scales)-1)])
        
        self.dropout1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # feature mixer
        self.norm2 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        mlp_type = mlp_configs['mlp_type']
        if mlp_type == "mlp":
            self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model *  mlp_configs['mlp_ratio']), act_layer=StarReLU, drop=0, bias=False)
        elif mlp_type == "swiglu" or mlp_type == 'kan':
            raise ValueError()
            # SwiGLUFFNFused
        else:
            raise NotImplementedError  
        self.dropout2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward_fpn(self, fpns, fpn_and_deform_feats):
        # bigger -> small
        assert len(fpns) + len(fpn_and_deform_feats) == len(self.fpn_scales)
        ret = fpns + fpn_and_deform_feats
        ret = ret[::-1]
        output = [ret[0].permute(0, 3, 1, 2).contiguous()] # smallest
        for idx, f in enumerate(ret[1:]): # 
            f = f.permute(0, 3, 1, 2)
            f = f + F.interpolate(output[-1], size=(f.shape[2], f.shape[3]), mode="bilinear", align_corners=False)
            f = self.alias_convs[idx](f)
            output.append(f)
        output = output[-len(fpns):][::-1]
        output = [haosen.permute(0, 2, 3, 1) for haosen in output]
        return output
        

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, fpns, reg_split, scale_to_hw_length, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # b ri_hwi_rj_hwj_rh_hw_h c
        
        src2 = self.norm1(src)
        fpns2 = [self.norm1(haosen) for haosen in fpns]
        
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reg_split, reference_points, src2, spatial_shapes, level_start_index, padding_mask)
        
        scale_hw_length = [haosen[0] * haosen[1] for haosen in scale_to_hw_length]
        assert sum(scale_hw_length) == reg_split[-1][-1]
        hw_feats = src2[:, -(reg_split[-1][-1]):].contiguous().split(scale_hw_length, dim=1)
        fpn_and_deform_feats = [rearrange(hw_feats[haosen], 'b (h w) c -> b h w c',h=scale_to_hw_length[haosen][0], w=scale_to_hw_length[haosen][1]) for haosen in self.fpn_and_deform_idxs]
        fpns2 = self.forward_fpn(fpns2, fpn_and_deform_feats)
        
        src = src + self.dropout1(src2)
        fpns = [fpns[idx]+self.dropout1(fpns2[idx]) for idx in range(len(fpns))]
        
        src = src + self.dropout2(self.mlp(self.norm2(src)))
        
        fpns = [fpns[idx]+self.dropout2(self.mlp(self.norm2(fpns[idx]))) for idx in range(len(fpns))]

        return src, fpns

# 只有8的feature给到4
class FPNLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 num_fpn_levels,
                 fpn_configs=None,
                 drop_path=None,) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_fpn_levels = num_fpn_levels
        
        self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        
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
                 d_model,
                 num_encoder_layers=None, drop_path_rate=None,
                 deform_scales = None, fpn_scales = None, multiscale_shapes=None,
                 task_to_num_regs=None,
                 mlp_configs=None,
                 deform_configs=None,
                 fpn_configs=None,
        ):
        super().__init__()
        self.d_model = d_model
        self.multiscale_shapes = multiscale_shapes
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        self.tasks = deform_configs.pop('tasks')
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        self.deform_layers = nn.ModuleList([MSDeformAttnTransformerEncoderLayer(d_model=d_model,
                                                                                multiscale_shapes=multiscale_shapes, deform_scales=deform_scales, fpn_scales=fpn_scales, fpn_configs=fpn_configs,
                                                                                
                                                                                num_feature_levels=len(self.deform_scales),
                                                                                task_to_num_regs=task_to_num_regs,
                                                                                deform_configs=deform_configs,
                                                                                mlp_configs=mlp_configs,
                                                                                drop_path=dp_rates[j]) for j in range(num_encoder_layers)])
        # self.fpn_layers = nn.ModuleList([FPNLayer(d_model=d_model, 
        #                                           num_fpn_levels=len(self.fpn_scales), 
        #                                           fpn_configs=fpn_configs,
        #                                           drop_path=dp_rates[j]) for j in range(num_encoder_layers)])
        self.pos_2d = build_position_encoding('2d')
        

    # @staticmethod
    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        # lsit[h w], L; b L 2
        assert (valid_ratios == 1).all(), '都是1'
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)) # h w
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 1 hw / b 1 (h*h_ratio = h_valid_max)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_) # 绝对坐标 / 最大长宽 = 相对坐标
            ref = torch.stack((ref_x, ref_y), -1) # b hw 2
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) # b
        valid_W = torch.sum(~mask[:, 0, :], 1) # b
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # b 2, w的有效ratio, x的有效ratio
        return valid_ratio

    @property
    def device(self):
        return self.deform_layers[0].norm1.weight.device
    
    def forward(self, srcs):
        # 相同维度
        # {'res2': 'hw': b h w c, 'task_name': b s c}
        
        fpns = [srcs[haosen]['hw'] for haosen in self.fpn_not_deform]
        deform_scales = [srcs[haosen] for haosen in self.deform_scales] # list[(b r c, b h w c, '')]
    
        # deform
        scale_to_hw_length = [(haosen['hw'].shape[1], haosen['hw'].shape[2]) for haosen in deform_scales]
        hw_feats = [haosen['hw'].flatten(1, 2)  for haosen in deform_scales] # list[b hw c]
        
        hw_feats = torch.cat(hw_feats, dim=1)
        hw_masks = [torch.zeros_like(haosen['hw'][..., 0]).bool() for haosen in deform_scales] # list[b h w]
        hw_pos_embed = torch.cat([self.pos_2d(m, hidden_dim=self.d_model).permute(0, 2, 3, 1).flatten(1, 2) for m in hw_masks], dim=1) # b hw_sigma c
        hw_spatial_shapes = torch.as_tensor([(m.shape[1], m.shape[2]) for m in hw_masks], dtype=torch.long, device=self.device,) # L 2
        hw_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in hw_masks], 1) # b L 2
        hw_masks = torch.cat([m.flatten(1, 2) for m in hw_masks], dim=1) # b hw_sigma 
        hw_level_start_index = torch.cat((hw_spatial_shapes.new_zeros((1, )), hw_spatial_shapes.prod(1).cumsum(0)[:-1]))
        hw_reference_points = self.get_reference_points(hw_spatial_shapes, hw_valid_ratios, device=self.device,)
        
        # reg {'cls': b scale_s c, 'sem_seg': b scale_s c}
        task_to_reg_feats = {}
        for task_name in self.tasks:
            reg_feats = []
            for feat in deform_scales:
                if task_name in feat:
                    reg_feats.append(feat[task_name])
            task_to_reg_feats[task_name] = torch.cat(reg_feats, dim=1)
        reg_feats = torch.cat([task_to_reg_feats[task_name] for task_name in self.tasks], dim=1) # b s c
        reg_split = [(task_name, task_to_reg_feats[task_name].shape[1]) for task_name in self.tasks]
        reg_poses = torch.zeros_like(reg_feats)
        reg_reference_points = reg_feats.new_zeros([reg_feats.shape[0], reg_feats.shape[1], hw_reference_points.shape[-2], hw_reference_points.shape[-1]])
        reg_masks = torch.zeros_like(reg_feats[..., 0]).bool()
        
        # concate
        reg_split.append(('hw', hw_feats.shape[1]))
        
        output = torch.cat([reg_feats, hw_feats], dim=1)  # b reg_sigma+hw_sigma c
        reference_points = torch.cat([reg_reference_points, hw_reference_points], dim=1)
        output_poses = torch.cat([reg_poses, hw_pos_embed], dim=1)
        padding_masks = torch.cat([reg_masks, hw_masks], dim=1)
        for _, deform_layer in enumerate(self.deform_layers):
            output, fpns = deform_layer(src=output, pos=output_poses, fpns=fpns, reg_split=reg_split, scale_to_hw_length=scale_to_hw_length,
                                        reference_points=reference_points, spatial_shapes=hw_spatial_shapes, level_start_index=hw_level_start_index, padding_mask=padding_masks)
            # hw_feats = output[-(reg_split['hw']):].contiguous().split(scale_to_hw_length)
            # fpn_but_not_deform = fpn_layer(scales = fpn_but_not_deform + [hw_feats[haosen] for haosen in self.fpn_and_deform_idxs])
            
        return output, fpns


@META_ARCH_REGISTRY.register()
class MSRegisters_Encoder(nn.Module):
    # 假设: 每个scale的d_model一样
    def __init__(
        self,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
        task_to_num_regs,
        configs,
    ):  
        super().__init__()
        d_model, proj_add_norm, proj_add_star_relu, proj_bias, proj_dropout = configs['d_model'], configs['proj_add_norm'], configs['proj_add_star_relu'], configs['proj_bias'], configs['proj_dropout']
        self.multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))

        input_proj_list = {}
        for huihui in self.multiscale_shapes.keys():
            input_proj_list[huihui]= nn.Sequential(
                partial(LayerNormGeneral, bias=False, eps=1e-6)(self.multiscale_shapes[huihui].dim) if proj_add_norm else nn.Identity(),
                nn.Linear(self.multiscale_shapes[huihui].dim, d_model, bias=proj_bias),
                StarReLU() if proj_add_star_relu else nn.Identity(),
                nn.Dropout(proj_dropout) if proj_dropout != 0 else nn.Identity(),
            )
        self.input_projs = nn.ModuleDict(input_proj_list)
        
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            num_encoder_layers=configs['num_encoder_layers'], drop_path_rate=configs['drop_path_rate'],
            deform_scales=configs['deform_scales'], fpn_scales=configs['fpn_scales'], multiscale_shapes=self.multiscale_shapes,
            task_to_num_regs=task_to_num_regs,
            mlp_configs= configs['mlp_configs'],
            deform_configs=configs['deform_configs'],
            fpn_configs=configs['fpn_configs']
        )
        

    def forward(self, 
                multiscales=None, 
                **kwargs):
        # 'res2': {'hw': b h w c, 'task_name': b s c}
        
        srcs = {}
        for scale_name in self.multiscale_shapes.keys():
            input_proj = self.input_projs[scale_name]
            
            stage_srcs = {}
            for key in multiscales[scale_name].keys():
                stage_srcs[key] = input_proj(multiscales[scale_name][key])
            srcs[scale_name] = stage_srcs
            
        srcs = self.transformer(srcs)
        
        return srcs        


