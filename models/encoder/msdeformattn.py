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
from .ops.modules import MSDeformAttn

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
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
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
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
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features




class MSDeformAttnTransformerEncoder_fusionText(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        
        self.fusion_rel_self = None # 'before', 'after', None
        self.fusion_modules = None # hack
        self.fusion_add_pos = None
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    
    def forward_before(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                amrs=None, amr_token_feats=None, amr_token_seg_ids=None, text_feats=None, text_pad_masks=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)           
        for _, (layer, fusion_module) in enumerate(zip(self.layers, self.fusion_modules)):
            output, amr_token_feats, text_feats = fusion_module(multiscale_feats=output, 
                                                                multiscale_poses=pos,
                                                                multiscale_is_flattened=True,
                                                                is_image_multiscale=True,
                                                                amrs=amrs, 
                                                                amr_text_add_pos=self.fusion_add_pos,
                                                                amr_token_feats=amr_token_feats,
                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                text_feats=text_feats, 
                                                                text_pad_masks=text_pad_masks)
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output, amr_token_feats, text_feats
    
    def forward_after(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                amrs=None, amr_token_feats=None, amr_token_seg_ids=None, text_feats=None, text_pad_masks=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)           
        for _, (layer, fusion_module) in enumerate(zip(self.layers, self.fusion_modules)):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            output, amr_token_feats, text_feats = fusion_module(multiscale_feats=output, 
                                                                multiscale_poses=pos,
                                                                multiscale_is_flattened=True,
                                                                is_image_multiscale=True,
                                                                amrs=amrs, 
                                                                amr_text_add_pos=self.fusion_add_pos,
                                                                amr_token_feats=amr_token_feats,
                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                text_feats=text_feats, 
                                                                text_pad_masks=text_pad_masks)
        return output, amr_token_feats, text_feats

    def forward_none(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)           
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                amrs=None, amr_token_feats=None, amr_token_seg_ids=None, text_feats=None, text_pad_masks=None):

        if self.fusion_rel_self == None:
            return self.forward_none(src, spatial_shapes, level_start_index, valid_ratios, pos=pos, padding_mask=padding_mask), \
                            amr_token_feats, text_feats
        if self.fusion_rel_self == 'before':
            return self.forward_before(src, spatial_shapes, level_start_index, valid_ratios, pos=pos, padding_mask=padding_mask,
                                        amrs=amrs, 
                                        amr_token_feats=amr_token_feats,
                                        amr_token_seg_ids=amr_token_seg_ids, 
                                        text_feats=text_feats, 
                                        text_pad_masks=text_pad_masks)  
        elif self.fusion_rel_self == 'after':
            return self.forward_after(src, spatial_shapes, level_start_index, valid_ratios, pos=pos, padding_mask=padding_mask,
                                        amrs=amrs, 
                                        amr_token_feats=amr_token_feats,
                                        amr_token_seg_ids=amr_token_seg_ids, 
                                        text_feats=text_feats, 
                                        text_pad_masks=text_pad_masks)  
        else:
            raise ValueError()

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly_fusionText(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder_fusionText(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, 
                amrs=None, 
                amr_token_feats=None, 
                amr_token_seg_ids=None, 
                text_feats=None, 
                text_pad_masks=None):
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
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory, amr_token_feats, text_feats = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten,
                              amrs=amrs, 
                            amr_token_feats=amr_token_feats, 
                            amr_token_seg_ids=amr_token_seg_ids, 
                            text_feats=text_feats, 
                            text_pad_masks=text_pad_masks)

        return memory, spatial_shapes, level_start_index, amr_token_feats, text_feats


import copy
from einops import rearrange
from models.layers.utils import _get_clones

# video multiscale, text_dict

# text和没有转换维度的multiscale进行fusion, multiscale进入encoder
# text作为一个scale进入multiscale encoder

@META_ARCH_REGISTRY.register()
class VideoMultiscale_Text_Deform2d(nn.Module):
    # video, text 先proj, 再fusion, 
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
        text_dim,
        fusion_module,
        norm = "",
    ):

        super().__init__()
        d_model = configs['d_model']
        norm = configs['norm'] # fpn的norm
        nlayers = configs['nlayers']
        deform_attn = configs['deform_attn']
        self.encoded_scales = configs['encoded_scales'] # list[str]
        multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))
        self.multiscale_shapes = multiscale_shapes
        # 4 -> 8 -> 16 -> 32    
        self.scale_dims = [val.dim for val in multiscale_shapes.values()]
        self.video_text_projs = META_ARCH_REGISTRY.get(configs['video_text_projs']['name'])(configs=configs['video_text_projs'],
                                                                                            multiscale_shapes=multiscale_shapes,
                                                                                            text_dim=text_dim,
                                                                                            out_dim=d_model)
        from models.layers.position_encoding import build_position_encoding
        self.pos_2d = build_position_encoding(position_embedding_name='2d')

        self.fusion_module = fusion_module
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            dropout=deform_attn['dropout'],
            nhead=deform_attn['nheads'],
            dim_feedforward=deform_attn['dim_feedforward'],
            activation=deform_attn['activation'],
            num_encoder_layers=nlayers,
            num_feature_levels=len(self.encoded_scales),
            enc_n_points=deform_attn['enc_n_points']
        )
        # 假设encode 的是最高的三层feature, 然后计算最底层的feature和三层feature的最底层之间的距离
        # 8, 16, 32
        # 2, 4
        min_encode_stride = min([v.spatial_stride for k,v in multiscale_shapes.items() if k in self.encoded_scales])
        min_stride = min([v.spatial_stride for k,v in multiscale_shapes.items()])
        self.num_fpn_levels = int(np.log2(min_encode_stride) - np.log2(min_stride))
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.scale_dims[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, d_model)
            output_norm = get_norm(norm, d_model)

            lateral_conv = Conv2d(
                in_channels, d_model, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                d_model,
                d_model,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, 
                multiscales=None, # b t c h w
                text_inputs=None): 
        batch_size, _, nf, *_ = multiscales[list(self.multiscale_shapes.keys())[0]].shape

        assert set(list(multiscales.keys())).issubset(set(list(self.multiscale_shapes.keys())))
        assert set(list(self.multiscale_shapes.keys())).issubset(set(list(multiscales.keys())))
        # transform to 2d
        encoded_scales = {scale: multiscales[scale].clone() for scale in self.encoded_scales}
        encoded_scales, text_inputs = self.video_text_projs(encoded_scales, text_inputs)
        # early fusion
        # encoded_scales, text_dict = self.fusion_module(encoded_scales, text_dict)
        # srcs, text_inputs = self.fusion_module(multiscale_feats=srcs, 
        #                                             multiscale_poses=pos,
        #                                             multiscale_is_flattened=False,
        #                                             is_image_multiscale=True,
        #                                             text_inputs=text_inputs)
        # fusion的时候是 t h w 和 s 进行融合,  至于怎么融合是fusion module的事情
    
        # bt c h w
        encoded_scales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1) for key, value in encoded_scales.items()}

        srcs = []
        poses = []
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            x = encoded_scales[scale_name]  # deformable detr does not support half precision, bt c h w
            srcs.append(x)
            poses.append(self.pos_2d(torch.zeros_like(x)[:, 0, :, :].bool(), hidden_dim=x.shape[1])) # bt h w

        y, spatial_shapes, level_start_index = self.transformer(srcs, poses) # bt s_max c
        bs = y.shape[0]

        split_size_or_sections = [None] * len(self.encoded_scales)
        for i in range(len(self.encoded_scales)):
            if i < len(self.encoded_scales) - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1) # bt hw c

        out = [] # from high to low
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        # bt c h w -> bt
        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(list(self.multiscale_shapes.keys())[:self.num_fpn_levels][::-1]):
            x = multiscales[f].clone().permute(0, 2, 1, 3, 4).flatten(0, 1)
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        assert len(out) == len(list(self.multiscale_shapes.keys()))

        ret = {}
        for key, out_feat in zip(list(self.multiscale_shapes.keys()), out[::-1]):
            ret[key] = rearrange(out_feat, '(b t) c h w -> b c t h w', b=batch_size, t=nf)

        return ret, text_inputs


@META_ARCH_REGISTRY.register()
class VideoMultiscale_Deform2d(nn.Module):
    # video, text 先proj, 再fusion, 
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
        norm = "",
    ):
        super().__init__()
        d_model = configs['d_model']
        fpn_norm = configs['fpn_norm'] # fpn的norm
        nlayers = configs['nlayers']
        deform_attn = configs['deform_attn']
        encoded_scales = configs['encoded_scales'] # list[str]
        multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride)) # 4, 8, 16, 32
        self.multiscale_shapes = multiscale_shapes
        self.encoded_scales = sorted(encoded_scales, key=lambda x:self.multiscale_shapes[x].spatial_stride) # res3, res4, res5
        # 4 -> 8 -> 16 -> 32    
        self.scale_dims = [val.dim for val in multiscale_shapes.values()]
        self.video_projs = META_ARCH_REGISTRY.get(configs['video_projs']['name'])(configs=configs['video_projs'],
                                                                                  multiscale_shapes=multiscale_shapes,
                                                                                  out_dim=d_model)
        from models.layers.position_encoding import build_position_encoding
        self.pos_2d = build_position_encoding(position_embedding_name='2d')

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            dropout=deform_attn['dropout'],
            nhead=deform_attn['nheads'],
            dim_feedforward=deform_attn['dim_feedforward'],
            activation=deform_attn['activation'],
            num_encoder_layers=nlayers,
            num_feature_levels=len(self.encoded_scales),
            enc_n_points=deform_attn['enc_n_points']
        )
        # 假设encode 的是最高的三层feature, 然后计算最底层的feature和三层feature的最底层之间的距离
        # 8, 16, 32
        # 2, 4
        min_encode_stride = min([v.spatial_stride for k,v in multiscale_shapes.items() if k in self.encoded_scales])
        min_stride = min([v.spatial_stride for k,v in multiscale_shapes.items()])
        self.num_fpn_levels = int(np.log2(min_encode_stride) - np.log2(min_stride))
        lateral_convs = []
        output_convs = []

        use_bias = fpn_norm == ""
        for idx, in_channels in enumerate(self.scale_dims[:self.num_fpn_levels]):
            lateral_norm = get_norm(fpn_norm, d_model)
            output_norm = get_norm(fpn_norm, d_model)

            lateral_conv = Conv2d(
                in_channels, d_model, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                d_model,
                d_model,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, 
                multiscales): # b t c h w)
        multiscales = self.video_projs(multiscales) 
        batch_size, _, nf, *_ = multiscales[list(self.multiscale_shapes.keys())[0]].shape

        assert set(list(multiscales.keys())).issubset(set(list(self.multiscale_shapes.keys())))
        assert set(list(self.multiscale_shapes.keys())).issubset(set(list(multiscales.keys())))

        # early fusion
        # encoded_scales, text_dict = self.fusion_module(encoded_scales, text_dict)
        # srcs, text_inputs = self.fusion_module(multiscale_feats=srcs, 
        #                                             multiscale_poses=pos,
        #                                             multiscale_is_flattened=False,
        #                                             is_image_multiscale=True,
        #                                             text_inputs=text_inputs)
        # fusion的时候是 t h w 和 s 进行融合,  至于怎么融合是fusion module的事情
    
        srcs = []
        poses = []
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            x = multiscales[scale_name].permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
            srcs.append(x)
            poses.append(self.pos_2d(torch.zeros_like(x)[:, 0, :, :].bool(), hidden_dim=x.shape[1])) # bt h w

        memory, spatial_shapes, level_start_index = self.transformer(srcs, poses) # bt  [32,16, 8], c
        bs = memory.shape[0]
        spatial_index = 0
        memory_features = [] # 32 16 8
        for lvl in range(len(self.encoded_scales)):
            h, w = spatial_shapes[lvl]
            # [bs*t, c, h, w]
            memory_lvl = memory[:, spatial_index : spatial_index + h * w, :].reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()  
            memory_features.append(memory_lvl)
            spatial_index += h * w

        for idx, f in enumerate(list(self.multiscale_shapes.keys())[:self.num_fpn_levels][::-1]):
            x = multiscales[f].permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(memory_features[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            memory_features.append(y)

        assert len(memory_features) == len(list(self.multiscale_shapes.keys()))

        ret = {}
        for key, out_feat in zip(list(self.multiscale_shapes.keys()), memory_features[::-1]):
            ret[key] = rearrange(out_feat, '(b t) c h w -> b c t h w', b=batch_size, t=nf)

        return ret
    