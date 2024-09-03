import logging
import copy
import os
from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Dict, List, Optional


import torch.nn.functional as F
import torch.nn as nn
import torch
import detectron2.utils.comm as comm
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat

import logging
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

logger = logging.getLogger("dinov2")

from .vision_transformer import DinoVisionTransformer

from models.backbone.dinos.dinov2.layers import MemEffAttention, NestedTensorBlock as Block



@BACKBONE_REGISTRY.register()
class Dinov2_REG(nn.Module):
    @property
    def device(self):
        return self.ssl.register_tokens.device

    def __init__(self, configs,):
        super().__init__()
        dino_name = configs['type']
        dino_configs = DINO_NAME_TO_CONFIGS[dino_name]
        self.ssl = DinoVisionTransformer(**dino_configs)
        if configs['load_pt']:
            name_to_pt_path = {'dinov2_vitb14_reg': 'dinov2/dinov2_vitb14_reg4_pretrain.pth',
                            'dinov2_vits14_reg': 'dinov2/dinov2_vits14_reg4_pretrain.pth',
                            'dinov2_vitg14_reg': 'dinov2/dinov2_vitg14_reg4_pretrain.pth',
                            'dinov2_vitl14': 'dinov2/dinov2_vitl14_pretrain.pth',
                            'dinov2_vitg14': 'dinov2/dinov2_vitg14_pretrain.pth',
                            'dinov2_vitb14': 'dinov2/dinov2_vitb14_pretrain.pth'}
            state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[dino_name]), map_location='cpu')
            self.ssl.load_state_dict(state_dict, strict=True) 
        if configs['freeze_pt']:
            for p in self.ssl.parameters():
                p.requires_grad_(False)  
        self.embed_dim = self.ssl.embed_dim
        self.patch_size = self.ssl.patch_size
        if os.environ.get('GO_TRAIN') == 'true':
            self.ssl = torch.compile(self.ssl)

    # def forward(self, x, masks=None):
    #     # b 3 h w -> b c h w
    #     H, W = x.shape[-2:]
    #     features = self.ssl.forward_features(x)['x_norm_patchtokens'] # b hw c
    #     features = rearrange(features, 'b (h w) c -> b c h w',h=H//self.patch_size, w=W//self.patch_size)
    #     return features
    def forward(self, x, n=1):
        rets = self.ssl(x, n=n)
        return rets
  

DINO_NAME_TO_CONFIGS = {
    'dinov2_vits14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),
    },
    'dinov2_vitb14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),              
    },
    'dinov2_vitl14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),              
    },
    'dinov2_vitg14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'swiglufused',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),             
    },
    'dinov2_vitl14':{
        'img_size': 518, 
        'patch_size': 14, 
        'init_values': 1.0, 
        'ffn_layer': 'mlp', 
        'block_chunks': 0,
        'num_register_tokens': 0, 
        'interpolate_antialias': False, 
        'interpolate_offset': 0.1,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio':4,
        'block_fn': partial(Block, attn_class=MemEffAttention)
        },
    'dinov2_vitg14':{
        'img_size': 518, 
        'init_values': 1.0, 
        'ffn_layer': 'swiglufused', 
        'block_chunks': 0, 
        'interpolate_antialias': False, 
        'interpolate_offset': 0.1,
        'patch_size': 14,
        'num_register_tokens': 0,
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),
        
        },
    'dinov2_vitb14':{
        'img_size': 518, 
        'init_values': 1.0, 
        'ffn_layer': 'mlp', 
        'block_chunks': 0, 
        'interpolate_antialias': False, 
        'interpolate_offset': 0.1,
        'patch_size':14,
        'embed_dim':768,
        'depth':12,
        'num_heads':12,
        'mlp_ratio':4,
        'block_fn':partial(Block, attn_class=MemEffAttention),
        'num_register_tokens': 0,}
}
