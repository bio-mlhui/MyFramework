"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce

from mamba_ssm import Mamba

from detectron2.modeling import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Image_InterpolateTimes_MultiscaleEncoder(nn.Module):
    def __init__(self, 
                 configs,
                 multiscale_shapes,):
        """
        将fusion_scales中所有scale放大到最大的scale, 然后相乘转换最大的scale
        """
        super().__init__()
        d_model = configs['d_model']
        self.image_projs = META_ARCH_REGISTRY.get(configs['image_projs']['name'])(configs=configs['image_projs'],
                                                                                  multiscale_shapes=multiscale_shapes,
                                                                                  out_dim=d_model)
        fusion_scales = configs['fusion_scales']
        self.fusion_scales = sorted(fusion_scales, key=lambda x: multiscale_shapes[x].spatial_stride)
        
    def forward(self, 
                multiscales=None,
                **kwargs): # 传入dict数据不能改变
        ret = self.image_projs(multiscales)
        target_scale = ret[self.fusion_scales[0]] # b c h w
        max_shape = target_scale.shape[-2:]

        for scale_name in self.fusion_scales[1:]:
            scale_feat = ret[scale_name] # b c h w
            itp_scale = F.interpolate(scale_feat, size=max_shape, mode='bilinear')
            target_scale = target_scale * itp_scale
        ret[self.fusion_scales[0]] = target_scale
        return ret

@META_ARCH_REGISTRY.register()
class Video2D_InterpolateTimes_MultiscaleEncoder(nn.Module):
    def __init__(self, 
                 configs,
                 multiscale_shapes,):
        """
        将fusion_scales中所有scale放大到最大的scale, 然后相乘转换最大的scale
        """
        super().__init__()
        self.image_homo = Image_InterpolateTimes_MultiscaleEncoder(configs=configs,
                                                 multiscale_shapes=multiscale_shapes,)
        
    def forward(self, 
                multiscales=None,
                **kwargs):
        batch_sisze, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        # b c t h w -> bt c h w
        multiscales = {key: value.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous() for key,value in multiscales.items()}
        multiscales = self.image_homo(multiscales)
        multiscales = {key: rearrange(value, '(b t) c h w -> b c t h w',b=batch_sisze, t=nf).contiguous()\
                        for key,value in multiscales.items()}
        return multiscales


@META_ARCH_REGISTRY.register()
class Image_MambaMultiscale(nn.Module):
    def __init__(self, 
                 configs,
                 multiscale_shapes,
                 fusion_scales):
        super().__init__()
        d_model = configs['d_model']
        self.image_projs = META_ARCH_REGISTRY.get(configs['image_projs']['name'])(configs=configs['image_projs'],
                                                                                  multiscale_shapes=multiscale_shapes,
                                                                                  out_dim=d_model)
        
        self.fusion_scales = sorted(fusion_scales, key=lambda x: multiscale_shapes[x].spatial_stride)

        self.self_attention = Mamba(d_model=64)
        self.norm = nn.GroupNorm(32, num_channels=64) # N c

    def forward(self, multiscales): 
        multiscales = self.image_projs(multiscales)
        x2 = x2.split(4, dim=2) # list[b c 4 w4], h
        x2 = [taylor.split(4, dim=-1) for taylor in x2] # list[list[b c 4 4] w] h 

        x3 = x3.split(2, dim=2) # list[b c 2 w2], h
        x3 = [taylor.split(2, dim=-1) for taylor in x3] # list[list[b c 2 2] w] h

        x4 = x4.split(1, dim=2) # list[b c 1 w1], h
        x4 = [taylor.split(1, dim=-1) for taylor in x4] # list[list[b c 1 1] w] h

        num_h_splits = len(x4)
        num_w_splits = len(x4[0])

        sequence = [] # b c L
        for h_idx in range(num_h_splits):
            for w_idx in range(num_w_splits):
                hw_seq = torch.cat([
                    x4[h_idx][w_idx].flatten(2).contiguous(),
                    x3[h_idx][w_idx].flatten(2).contiguous(),
                    x2[h_idx][w_idx].flatten(2).contiguous(),
                ], dim=-1) # b c 21
                sequence.append(hw_seq)
        sequence = torch.cat(sequence, dim=-1) # b c hw21
        sequence = sequence.transpose(-1, -2).contiguous() # b L c
        sequence = self.self_attention(sequence)
        sequence = sequence.transpose(-1, -2).contiguous() # b c L

        sequence = sequence.split(21, dim=-1) # list[b c 21], hw
        sequence = [taylor.split([1, 4, 16], dim=-1) for taylor in sequence] # list[[b c 1, b c 4, b c 16]], hw
        
        scale_x4, scale_x3, scale_x2 = list(zip(*sequence)) # list[b c 1], hw

        scale_x4 = torch.stack(scale_x4, dim=2) # b c hw 1
        scale_x4 = rearrange(scale_x4, 'b c (h w) (c1 c2) -> b c (h c1) (w c2)',h=num_h_splits, w=num_w_splits, c1=1,c2=1)

        scale_x3 = torch.stack(scale_x3, dim=2) # b c hw 4
        scale_x3 = rearrange(scale_x3, 'b c (h w) (c1 c2) -> b c (h c1) (w c2)',h=num_h_splits, w=num_w_splits, c1=2,c2=2)

        scale_x2 = torch.stack(scale_x2, dim=2) # b c hw 16
        scale_x2 = rearrange(scale_x2, 'b c (h w) (c1 c2) -> b c (h c1) (w c2)',h=num_h_splits, w=num_w_splits, c1=4,c2=4)
        scale_x2 = self.norm(scale_x2)
        return scale_x2 # b c h/8 w/8
