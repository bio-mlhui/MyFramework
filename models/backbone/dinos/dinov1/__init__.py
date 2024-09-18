# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torchvision.models.resnet import resnet50
from . import dinov1_adapter
from detectron2.modeling import BACKBONE_REGISTRY
from .vision_transformer import VisionTransformer
dependencies = ["torch", "torchvision"]


import torch.nn as nn
import os

from functools import partial
from einops import rearrange

@BACKBONE_REGISTRY.register()
class Dinov1(nn.Module):

    def __init__(self, configs,):
        super().__init__()
        dino_name = configs['type']
        dino_configs = DINO_NAME_TO_CONFIGS[dino_name]
        self.ssl = VisionTransformer(**dino_configs)

        if configs['load_pt']:
            name_to_pt_path = {'dinov1_vits16': 'dinov1/dino_deitsmall16_pretrain.pth',
                               'dinov1_vitb8': 'dinov1/dino_vitbase8_pretrain.pth',
                               'dinov1_vitb16': 'dinov1/dino_vitbase16_pretrain.pth',
                               # 'dinov1_vits8': '/home/xuhuihui/workspace/dino/out/checkpoint0000.pth',}
                               'dinov1_vits8': 'dinov1/dino_deitsmall8_300ep_pretrain.pth', }
            state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[dino_name]), map_location='cpu')
            # state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[dino_name]), map_location='cpu')['teacher']
            # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            # self.ssl.load_state_dict(state_dict, strict=False) 
            self.ssl.load_state_dict(state_dict, strict=True) 
        if configs['freeze_pt']:
            for p in self.ssl.parameters():
                p.requires_grad_(False)  
        self.embed_dim = self.ssl.embed_dim
        self.patch_size = self.ssl.patch_embed.patch_size
        self.num_heads = dino_configs['num_heads']
        if os.environ.get('GO_TRAIN') == 'true':
            self.ssl = torch.compile(self.ssl)

    def forward(self, x, n=1):
        # b 3 h w, normalized

        rets = self.ssl(x, n=n)
        return rets


DINO_NAME_TO_CONFIGS = {
# def vit_small(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
    'dinov1_vits16': {
        'num_classes': 0,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    },
        # patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        # qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    'dinov1_vits8': {
        'num_classes': 0,
        'patch_size': 8,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    },
# def vit_base(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
    'dinov1_vitb8': {
        'num_classes': 0,
        'patch_size': 8,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),              
    },

    # model = VisionTransformer(
    #     patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    #     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    'dinov1_vitb16': {
        'num_classes': 0,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),              
    },
}
