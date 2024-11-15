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

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, 
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4,
                 temporal_attn=None,
                 scan_configs=None,
        ):
        super().__init__()

        self.d_model = d_model

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels,
                                                            temporal_attn=temporal_attn,
                                                            scan_configs=scan_configs)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters(META_ARCH_REGISTRY.get(scan_configs['ms_deform_version']))

    def _reset_parameters(self, attn_cls):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
        for m in self.modules():
            if isinstance(m, attn_cls):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape # b h w
        valid_H = torch.sum(~mask[:, :, 0], 1) # b 
        valid_W = torch.sum(~mask[:, 0, :], 1) # b
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # b 2
        return valid_ratio

    def forward(self, 
                srcs=None, 
                pos_embeds=None,
                video_aux_dict=None,
                **kwargs):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # b #scale 2

        # encoder
        memory = self.encoder(src=src_flatten, 
                              spatial_shapes=spatial_shapes, 
                              level_start_index=level_start_index, 
                              valid_ratios=valid_ratios, 
                              pos=lvl_pos_embed_flatten, 
                              padding_mask=mask_flatten,
                              video_aux_dict=video_aux_dict)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model, d_ffn,
                 dropout, activation,
                 n_levels,
                 scan_configs=None,
                 temporal_attn=None):
        super().__init__()
        self.add_temporal = temporal_attn['name'] is not None
        if self.add_temporal:
            self.self_attn_temporal = META_ARCH_REGISTRY.get(temporal_attn['name'])(temporal_attn)
            self.dropout_tem = nn.Dropout(dropout)
            self.norm_tem = nn.LayerNorm(d_model)
            self.temporal_fc = nn.Linear(d_model, d_model)


        # self attention
        assert n_levels == scan_configs['n_levels']
        self.self_attn = META_ARCH_REGISTRY.get(scan_configs['ms_deform_version'])(scan_configs)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None,
                video_aux_dict=None):
        
        # temporal attention
        if self.add_temporal:
            # bt hw_sigma c
            src_tem = self.self_attn_temporal(query=src, 
                                              spatial_shapes=spatial_shapes,
                                              level_start_index=level_start_index,
                                              video_aux_dict=video_aux_dict)[0] 
            src_tem = self.temporal_fc(src_tem)
            src = src + self.dropout_tem(src_tem)
            src = self.norm_tem(src)

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # b #scale 2, valid_w(0-1), valid_h(0-1), 整个feature map有多少是非padding的
        # list[h w] #scale
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 1 hw / b 1 -> b hw(0-1), y的绝对坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_) # 1 hw / b 1 -> b hw(0-1), x的绝对坐标
            ref = torch.stack((ref_x, ref_y), -1) # b hw 2
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # b hw_sigma 2, 每个点的相对坐标(0-1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # b hw_sigma 1 2 * b 1 #scale 2
        return reference_points # b hw_sigma #scale 2

    def forward(self, 
                src, 
                spatial_shapes, 
                level_start_index, 
                valid_ratios, 
                pos=None, padding_mask=None,
                video_aux_dict=None):
        output = src # bt hw_sigma c
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(src=output, 
                           pos=pos, 
                           reference_points=reference_points, 
                           spatial_shapes=spatial_shapes, 
                           level_start_index=level_start_index, 
                           padding_mask=padding_mask,
                           video_aux_dict=video_aux_dict)

        return output


import copy
from einops import rearrange
from models.layers.utils import _get_clones
from models.layers.position_encoding import build_position_encoding
# video multiscale, text_dict

@META_ARCH_REGISTRY.register()
class Video_DeformSS2D_TemporalSS1D_MultiscaleEncoder(nn.Module):
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
    ):
        super().__init__()
        d_model = configs['d_model']
        fpn_norm = configs['fpn_norm'] # fpn的norm
        nlayers = configs['nlayers']

        # 4, 8, 16, 32
        self.multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))
        self.encoded_scales = sorted(configs['encoded_scales'], 
                                     key=lambda x:self.multiscale_shapes[x].spatial_stride) # res3, res4, res5
        
        # 4 -> 8 -> 16 -> 32    
        self.scale_dims = [val.dim for val in multiscale_shapes.values()]
        self.video_projs = META_ARCH_REGISTRY.get(configs['video_projs']['name'])(configs=configs['video_projs'],
                                                                            multiscale_shapes=multiscale_shapes, out_dim=d_model)

        self.pos_2d = build_position_encoding(position_embedding_name='2d')

        deform_attn = configs['deform_attn']
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            dropout=deform_attn['dropout'],
            dim_feedforward=deform_attn['dim_feedforward'],
            activation=deform_attn['activation'],
            num_encoder_layers=nlayers,
            num_feature_levels=len(self.encoded_scales),
            temporal_attn=configs['temporal_attn'],
            scan_configs=deform_attn['scan_configs'],
        )

        min_encode_stride = self.multiscale_shapes[self.encoded_scales[0]].spatial_stride # 8
        min_stride = list(self.multiscale_shapes.values())[0].spatial_stride # 4
        self.num_fpn_levels = int(np.log2(min_encode_stride) - np.log2(min_stride))
        lateral_convs = [] 
        output_convs = []
        use_bias = fpn_norm == ""
        for idx, in_channels in enumerate(self.scale_dims[:self.num_fpn_levels]):
            lateral_norm = get_norm(fpn_norm, d_model)
            output_norm = get_norm(fpn_norm, d_model)

            lateral_conv = Conv2d(in_channels, d_model, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)

            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1] # 8 4
        self.output_convs = output_convs[::-1] # 8 4

    def forward(self, 
                multiscales=None, # b c t h w
                video_aux_dict=None, # dict{}
                **kwargs):
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        video_aux_dict['nf'] = nf
        multiscales = self.video_projs(multiscales) 
        assert set(list(multiscales.keys())).issubset(set(list(self.multiscale_shapes.keys())))
        assert set(list(self.multiscale_shapes.keys())).issubset(set(list(multiscales.keys())))

        srcs = []
        poses = [] # 32, 16, 8
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            x = multiscales[scale_name].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous() # bt c h w
            srcs.append(x)
            poses.append(self.pos_2d(torch.zeros_like(x)[:, 0, :, :].bool(), hidden_dim=x.shape[1]))

        memory, spatial_shapes, level_start_index = self.transformer(srcs=srcs, 
                                                                     pos_embeds=poses,
                                                                     video_aux_dict=video_aux_dict)
        bs = memory.shape[0]
        spatial_index = 0
        memory_features = [] # 32 16 8
        for lvl in range(len(self.encoded_scales)):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index : spatial_index + h * w, :].reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()  
            memory_features.append(memory_lvl)
            spatial_index += h * w

        for idx, f in enumerate(list(self.multiscale_shapes.keys())[:self.num_fpn_levels][::-1]):
            x = multiscales[f].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous() # bt c h w
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(memory_features[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[idx](y)
            memory_features.append(y)

        assert len(memory_features) == len(list(self.multiscale_shapes.keys()))

        ret = {}
        for key, out_feat in zip(list(self.multiscale_shapes.keys()), memory_features[::-1]):
            ret[key] = rearrange(out_feat, '(b t) c h w -> b c t h w', b=batch_size, t=nf)
        return ret


@META_ARCH_REGISTRY.register()
class MultiscaleEncoder_None(nn.Module):
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
    ):
        super().__init__()
    
    def forward(self, 
                multiscales=None, # b c t h w
                video_aux_dict=None, # dict{}
                **kwargs):
        return multiscales

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out