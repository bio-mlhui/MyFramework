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
import models.backbone.dinos.dinov1.vision_transformer as vits
from detectron2.modeling import BACKBONE_REGISTRY

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
                               'dinov1_vitb8': 'dinov1/dino_vitbase8_pretrain.pth',}
            state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[dino_name]), map_location='cpu')
            self.ssl.load_state_dict(state_dict, strict=True) 
        if configs['freeze_pt']:
            for p in self.ssl.parameters():
                p.requires_grad_(False)  
        self.embed_dim = self.ssl.embed_dim
        self.patch_size = self.ssl.patch_embed.patch_size

    def get_intermediate_feats(self, x, masks=None):
        return self.ssl.get_intermediate_feats(x)

    def get_alignseg_feats(self, **kwargs):
        return self.ssl.get_alignseg_feats(**kwargs)

    def forward(self, x):
        return self.ssl.forward_features(x)

    # NestTensor?
    def forward_dino_ssl(self, x, last_hw, masks=None,): # b c h w -> ms: {'res2', 'res3', 'res4, 'res5}, reg: {'reg2', 'reg3', 'reg4', 'reg5'}
        batch_size, _, H, W = x.shape # b c h w
        patch_size = [H // self.ssl.patch_size, W // self.ssl.patch_size]
        x = self.ssl.patch_embed(x) # b c h w -> b hw c
        x = x.view(x.shape[0], patch_size[0], patch_size[1], x.shape[-1]).contiguous() # b h w c

        x = (x + last_hw).flatten(1, 2) # b hw c
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.ssl.mask_token.to(x.dtype).unsqueeze(0), x)    
        x_poses = self.ssl.interpolate_pos_encoding_hw(x, W, H) # b hw c
        x = x + x_poses
        
        clsssl_regs = torch.cat([self.ssl.cls_token + self.ssl.pos_embed[:, 0].unsqueeze(0), self.ssl.register_tokens], dim=1)
        registers = torch.cat([self.pt_task_regs, clsssl_regs], dim=1)
        x = torch.cat([registers.repeat(x.shape[0], 1, 1), x], dim=1) # b reg_hw c

        for blk in self.ssl.blocks:
            x = blk(x)
        x = self.ssl.norm(x)
        x = self.dino_same_stage(x)

        reg_feats, hw_feats = x.split([sum(self.task_num_regs), x.shape[1] - sum(self.task_num_regs)], dim=1)
        hw_feats = rearrange(hw_feats, 'b (h w) c -> b h w c', h=patch_size[0], w=patch_size[1])
            
        return (reg_feats, hw_feats)


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
}


from models.backbone.dinos.dinov1.vision_transformer import Block, PatchEmbed
from models.backbone.dinos.dinov1.utils import trunc_normal_
import math
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward_features(self, x):
        B, _, H, W = x.shape
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feats = x[:, 1:]
        feats = feats.reshape(B, H//self.patch_embed.patch_size, W//self.patch_embed.patch_size, -1).permute(0, 3, 1, 2)
        return feats


    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output




    # region alignseg
    def forward_backbone(self, x, last_self_attention=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)
        if last_self_attention:
            x, attn = x
        x = self.norm(x)
        if last_self_attention:
            return x, attn[:, :, 0, 1:]  # [B, heads, cls, cls-patch]
        return x
    
    def get_alignseg_feats(self, inputs, nmb_crops=(1,0), last_self_attention=False): # num_crops[1,2]
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = [1, ]  # for inference
        if sum(nmb_crops) > 1:
            # for training
            idx_crops.append(sum(nmb_crops))
        # [1, 3]
        assert len(idx_crops) <= 2, "Only supporting at most two different type of crops (global and local crops)"
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])
            _out = self.forward_backbone(_out, last_self_attention=last_self_attention)
            if last_self_attention:
                _out, _attn = _out
            spatial_tokens = _out[:, 1:]  # b hw c
            spatial_tokens = spatial_tokens.reshape(-1, self.embed_dim)  # [bhw, embed_dim]

            if start_idx == 0:
                output_spatial = spatial_tokens
                if last_self_attention:
                    # only keep 1st global crop attention
                    attentions = _attn
            else:
                output_spatial = torch.cat((output_spatial, spatial_tokens))
                if last_self_attention:
                    attentions = torch.cat((attentions, _attn))
            start_idx = end_idx

        result = output_spatial
        if last_self_attention:
            result = (result, attentions)
        return result
    # endregion