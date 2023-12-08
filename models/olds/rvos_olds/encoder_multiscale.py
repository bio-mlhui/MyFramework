import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import Any, Optional
from torch import Tensor
from .model_utils import find_scale_from_multiscales, find_scales_from_multiscales
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer


_multiscale_encoder_entrypoints = {}
def register_multiscale_encoder(fn):
    multiscale_encoder_name = fn.__name__
    _multiscale_encoder_entrypoints[multiscale_encoder_name] = fn
    return fn

def multiscale_encoder_entrypoints(multiscale_encoder_name):
    try:
        return _multiscale_encoder_entrypoints[multiscale_encoder_name]
    except KeyError as e:
        print(f'MultiScale Encoder {multiscale_encoder_name} not found')

@register_multiscale_encoder
def no_scale_fusion(configs, d_model):
    return None


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
    
    def forward(self, multiscale_args):
        """
        """
        multiscale_feats, _, _, multiscale_decs = multiscale_args
        idxs = find_scales_from_multiscales(multiscale_decs, self.cascaded_scales) 
    
        fused_feats = [multiscale_feats[idx] for idx in idxs]  # 从小到大
        batch_size, nf = fused_feats[0].shape[:2]

        fused_feats = [rearrange(f, 'b t c h w -> (b t) c h w') for f in fused_feats]

        for idx, (small_feat, large_feat) in enumerate(zip(fused_feats[:-1], fused_feats[1:])): # from small map to large map 
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feat) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            fused_feats[idx+1] = large_feat
        
        fused_feats = [rearrange(f, '(b t) c h w -> b t c h w',t=nf, b=batch_size) for f in fused_feats]

        for idx, scale_idx in enumerate(idxs):
            multiscale_feats[scale_idx] = fused_feats[idx]

        return multiscale_feats

class DeformVideo2D_with_FPN(nn.Module):
    def __init__(self, 
                 d_model,
                d_ffn=2048,
                dropout=0.,
                activation='relu',
                n_heads=8,
                # important
                fused_scales=None, 
                fpn_strides=None,

                n_points=4, 
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
                    n_heads=n_heads,
                    n_points=n_points,
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
    
    def forward(self, multiscale_args):
        """
        假设fused_scales是给到的multiscales的
        """
        video_feats, video_pad_masks, video_poses, multiscale_des = multiscale_args
        fused_scale_idxs = find_scales_from_multiscales(multiscale_des, self.fused_scales)

        batch_size, nf, *_ = video_feats[0].shape

        srcs = [video_feats[idx] for idx in fused_scale_idxs]
        masks = [video_pad_masks[idx] for idx in fused_scale_idxs]
        pos_embeds = [video_poses[idx] for idx in fused_scale_idxs]

        srcs = [rearrange(f, 'b t c h w -> (b t) c h w') for f in srcs]
        masks = [rearrange(m, 'b t h w -> (b t) h w') for m in masks]
        pos_embeds = [rearrange(p, 'b t c h w -> (b t) c h w') for p in pos_embeds]
        
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
        # bt (h_sigma, w_sigma) c
        memory = self.deform_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten,)
        
        memory_features = []
        spatial_index = 0
        for lvl in range(self.num_feature_levels):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index: (spatial_index + h*w), :].contiguous()
            memory_lvl = rearrange(memory_lvl, '(b t) (h w) c -> b t c h w',h=h, w=w, b=batch_size, t=nf)
            memory_features.append(memory_lvl)
            spatial_index += h*w
        
        for idx, scale_idx in enumerate(fused_scale_idxs):
            video_feats[scale_idx] = memory_features[idx]

        video_feats = self.fpn((video_feats, video_pad_masks, video_poses, multiscale_des))
        return video_feats
    

@register_multiscale_encoder
def deform_video_2d_fpn(configs, d_model):
    configs = vars(configs)
    return DeformVideo2D_with_FPN(d_model=d_model,
                        d_ffn=configs['d_ffn'],
                        dropout=configs['dropout'],
                        activation=configs['activation'],
                        n_heads=configs['nheads'],
                        # important
                        fused_scales=configs['fused_scales'],
                        fpn_strides=configs['fpn_strides'],
                        n_points=configs['npoints'],
                        nlayers=configs['nlayers'])
    