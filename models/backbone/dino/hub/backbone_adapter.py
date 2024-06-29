import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from models.encoder.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

import logging
import copy
import os
from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Dict, List, Optional
from models.backbone.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from einops import repeat, rearrange

from .comer_modules import CNN, CTIBlock
from detectron2.modeling import BACKBONE_REGISTRY
_logger = logging.getLogger(__name__)


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
}


@BACKBONE_REGISTRY.register()
class DinoReg_VitAdapter_NoDecoder(nn.Module):

    @property
    def device(self):
        return self.ssl.register_tokens.device
    

    def has_multiscale(self, configs,):  
        ms_d_model, proj_add_norm, proj_add_star_relu, proj_bias, proj_dropout = configs['d_model'], configs['proj_add_norm'], \
            configs['proj_add_star_relu'], configs['proj_bias'], configs['proj_dropout']
        input_proj_list = {}
        for huihui in self.multiscale_shapes.keys():
            input_proj_list[huihui]= nn.Sequential(
                partial(LayerNormGeneral, bias=False, eps=1e-6)(self.multiscale_shapes[huihui].dim) if proj_add_norm else nn.Identity(),
                nn.Linear(self.multiscale_shapes[huihui].dim, ms_d_model, bias=proj_bias),
                StarReLU() if proj_add_star_relu else nn.Identity(),
                nn.Dropout(proj_dropout) if proj_dropout != 0 else nn.Identity(),
            )
        self.input_projs = nn.ModuleDict(input_proj_list)

        num_encoder_layers = configs['num_encoder_layers']
        deform_scales, fpn_scales = configs['deform_scales'], configs['fpn_scales']
        drop_path_rate=configs['drop_path_rate']
        deform_configs, mlp_configs, fpn_configs = configs['deform_configs'], configs['mlp_configs'], configs['fpn_configs']
        self.ms_d_model = ms_d_model
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        
        deform_configs['reg_sequence_length'] = (self.dino_stage_idx - self.first_attn_stage_idx) * self.task_num_regs[0] + \
                                                (self.num_stages - self.dino_stage_idx) * (self.task_num_regs[0] + self.task_num_regs[1] + self.task_num_regs[2])
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        self.deform_layers = nn.ModuleList([MSDeformAttnTransformerEncoderLayer(d_model=ms_d_model,
                                                                                multiscale_shapes=self.multiscale_shapes, deform_scales=deform_scales, fpn_scales=fpn_scales, 
                                                                                fpn_configs=fpn_configs,
                                                                                
                                                                                num_feature_levels=len(self.deform_scales),
                                                                                deform_configs=deform_configs,
                                                                                mlp_configs=mlp_configs,
                                                                                drop_path=dp_rates[j]) for j in range(num_encoder_layers)])
        self.pos_2d = build_position_encoding('2d')
    

    def is_a_peft_dino(self, configs):
        dino_configs = DINO_NAME_TO_CONFIGS[configs['name']]
        
        self.ssl = DinoVisionTransformer(**dino_configs)
        name_to_pt_path = {'dinov2_vitb14_reg': 'dinov2/dinov2_vitb14_reg4_pretrain.pth',
                           'dinov2_vits14_reg': 'dinov2/dinov2_vits14_reg4_pretrain.pth'}
        state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[configs['name']]), map_location='cpu')
        self.ssl.load_state_dict(state_dict, strict=True) 
        if configs['freeze']:
            for p in self.ssl.parameters():
                p.requires_grad_(False)

        self.num_pt_registers = configs['num_pt_registers']
        self.pt_task_regs = nn.Embedding(self.num_pt_registers, self.ssl.embed_dim)
        self.pt_task_reg_poses = nn.Embedding(self.num_pt_registers, self.ssl.embed_dim)
        nn.init.normal_(self.pt_task_regs.weight, std=1e-6)
        trunc_normal_(self.pt_task_reg_poses.weight, std=0.02)

        self.task_num_regs = [self.num_pt_registers, self.ssl.cls_token.shape[1], self.ssl.register_tokens.shape[1]]
        self.num_registers = sum(self.task_num_regs)
        self.ssl_pretrain_size = (dino_configs['img_size'], dino_configs['img_size'])

    def __init__(self, configs, 
                 pretrain_size=224, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_CTI=True, pretrained=None,with_cp=False,
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 cnn_feature_interaction=True,
                 dim_ratio=6.0,
                 *args, **kwargs):
        super().__init__()
        self.is_a_peft_dino(configs['dino'])
        # self.has_multiscale(configs.pop('ms_configs'))

        ms_configs = configs['ms_configs']


        use_CTI_toC = ms_configs['use_CTI_toC'] 
        use_CTI_toV = ms_configs['use_CTI_toV']  
        conv_inplane = ms_configs['conv_inplane']  
        add_vit_feature = ms_configs['add_vit_feature'] 
        add_vit_feature = ms_configs['add_vit_feature'] 
        add_vit_feature = ms_configs['add_vit_feature'] 


        self.ms_fusion_deps = ms_configs['ms_fusion_deps']
        self.level_embed = nn.Embedding(3, self.ssl.embed_dim)
        torch.nn.init.normal_(self.level_embed.weight)
        norm_layer = nn.LayerNorm

        self.local_conv = CNN(inplanes=conv_inplane, embed_dim=self.ssl.embed_dim)
        self.ms_fusion_before_layers = nn.ModuleList([
            CTIBlock(dim=self.ssl.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                    init_values=init_values, 
                    drop_path=ms_configs['cti_drop_path'],
                    norm_layer=norm_layer, with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                    use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                    extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(self.ms_fusion_deps))
        ])
        self.ms_fusion_after_layers = nn.ModuleList([
            CTIBlock(dim=self.ssl.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                    init_values=init_values, 
                    drop_path=ms_configs['cti_drop_path'],
                    norm_layer=norm_layer, with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                    use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                    extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(self.ms_fusion_deps))
        ])

        self.up = nn.ConvTranspose2d(self.ssl.embed_dim, self.ssl.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.ssl.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        # decoder head things
        # logging.debug(f'ssl的总参数数量:{sum(p.numel() for p in self.ssl.parameters())}') 


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

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

    def get_deform_query_key_args(self, x):
        batch_size, _, H, W = x.shape

        # L 2
        res345_spatial_shapes = torch.as_tensor([(H // 8, W//8), (H//16, W//16), (H//32, W//32)]).long().to(self.device)
        dino_hw_spatial_shapes = torch.as_tensor([(H // 14, W//14)]).long().to(self.device)
        res345_valid_ratios = x.new_ones([batch_size, 3, 2]) # b L 2
        dino_hw_valid_ratios = x.new_ones([batch_size, 1, 2]) # b L 2
        
        # L
        res345_level_start_index = torch.cat((res345_spatial_shapes.new_zeros((1, )), res345_spatial_shapes.prod(1).cumsum(0)[:-1]))
        # L 2 -> b hw_sigma 1 2
        res345_reference_points = self.get_reference_points(dino_hw_spatial_shapes.long().to(self.device), res345_valid_ratios).to(self.device)

        dino_hw_level_start_index = dino_hw_spatial_shapes.new_zeros((1, ))
        dino_hw_reference_points = self.get_reference_points(res345_spatial_shapes, dino_hw_valid_ratios).to(self.device)
        
        
        return [res345_reference_points, res345_spatial_shapes, res345_level_start_index], \
        [dino_hw_reference_points, dino_hw_spatial_shapes, dino_hw_level_start_index]

    def forward_decoder(self, pt_regs, hw_feats, res345, res2):
        # pt_regs = self.ssl.norm(pt_reg_poses)
        # clsssl_regs = self.ssl.norm(clsssl_regs)
        # x = self.ssl.norm(x)  
        pass
        
    def forward(self, x, masks=None):
        batch_size, _, H, W = x.shape
        res345_deform_key_args, dino_deform_key_args = self.get_deform_query_key_args(x)

        res2, res3, res4, res5 = self.local_conv(x) # b h w c
        res345_poses = [self.ssl.interpolate_pos_encoding_hw2(src) + self.level_embed.weight[src_idx].contiguous() \
                             for src_idx, src in enumerate(res3, res4, res5)] # list[b h w c]
        res345 = torch.cat([res3.flatten(1,2), res4.flatten(1,2), res5.flatten(1,2)], dim=1) # b hw_sigma c
        res345_poses = torch.cat([pos.flatten(1,2) for pos in res345_poses], dim=1) # b hw_sigma c

        patch_size = [H//self.ssl.patch_size, W//self.ssl.patch_size]
        dino_hw_feats = self.ssl.patch_embed(x) # b hw/14 c
        dino_hw_feats = dino_hw_feats.view(batch_size, patch_size[0], patch_size[1], x.shape[-1]).contiguous() # b h w c
        if masks is not None:
            dino_hw_feats = torch.where(masks.unsqueeze(-1), self.ssl.mask_token.to(dino_hw_feats.dtype).unsqueeze(0), x)   
        dino_hw_feats_poses = self.ssl.interpolate_pos_encoding_hw(dino_hw_feats, W, H) # b hw c
        dino_hw_feats = dino_hw_feats.flatten(1,2).contiguous() + dino_hw_feats_poses
        num_hw_tokens = dino_hw_feats.shape[1]

        clsssl_regs = torch.cat([self.ssl.cls_token + self.ssl.pos_embed[:, 0].contiguous().unsqueeze(0), self.ssl.register_tokens], dim=1)
        clsssl_regs = repeat(clsssl_regs, '1 n c -> b n c',b=batch_size)
        num_clsssl_regs = clsssl_regs.shape[1]

        pt_regs = repeat(self.pt_task_regs.weight, 'n c -> b n c', b=batch_size)
        pt_reg_poses = repeat(self.pt_task_reg_poses.weight, 'n c -> b n c',b=batch_size)

        fusion_layer_outs = []
        # 0, 3, 6, 9
        for dep, blk in enumerate(self.ssl.blocks):
            if dep in self.ms_fusion_deps:
                res345 = res345 + res345_poses

                ms_fusion_idx = self.ms_fusion_deps.index(dep)
                pt_regs, clsssl_regs, dino_hw_feats = self.ms_fusion_before_layers[ms_fusion_idx](\
                        pt_regs=pt_regs, pt_reg_poses=pt_reg_poses, clsssl_regs=clsssl_regs, dino_hw_feats=dino_hw_feats, 
                        res345_feats=res345, res345_poses=res345_poses, res345_deform_key_args=res345_deform_key_args)

                blk_input = torch.cat([pt_regs, clsssl_regs, dino_hw_feats], dim=1)
                blk_input = blk(blk_input)
                pt_regs, clsssl_regs, dino_hw_feats = blk_input.split([self.num_pt_registers, num_clsssl_regs, num_hw_tokens], dim=1)

                res345, res2 = self.ms_fusion_after_layers[ms_fusion_idx](\
                    res345_feats=res345, res345_poses=res345_poses, dino_deform_key_args=dino_deform_key_args, res2=res2,
                    pt_regs=pt_regs, clsssl_regs=clsssl_regs, dino_hw_feats=dino_hw_feats,)

                # layer_prediction like a decoder
                fusion_layer_outs.append(self.forward_decoder(pt_regs=pt_regs, dino_hw_feats=dino_hw_feats, res345=res345, res2=res2))

            else:
                blk_input = torch.cat([pt_regs, clsssl_regs, dino_hw_feats], dim=1)
                blk_input = blk(blk_input)
                pt_regs, clsssl_regs, dino_hw_feats = blk_input.split([self.num_pt_registers, num_clsssl_regs, num_hw_tokens], dim=1)

    


        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONE_REGISTRY.register()
class DinoReg_VitAdapter_NoDecoder(nn.Module):
    @property
    def device(self):
        return self.ssl.register_tokens.device
    
    def has_multiscale(self, configs,):  
        ms_d_model, proj_add_norm, proj_add_star_relu, proj_bias, proj_dropout = configs['d_model'], configs['proj_add_norm'], \
            configs['proj_add_star_relu'], configs['proj_bias'], configs['proj_dropout']
        input_proj_list = {}
        for huihui in self.multiscale_shapes.keys():
            input_proj_list[huihui]= nn.Sequential(
                partial(LayerNormGeneral, bias=False, eps=1e-6)(self.multiscale_shapes[huihui].dim) if proj_add_norm else nn.Identity(),
                nn.Linear(self.multiscale_shapes[huihui].dim, ms_d_model, bias=proj_bias),
                StarReLU() if proj_add_star_relu else nn.Identity(),
                nn.Dropout(proj_dropout) if proj_dropout != 0 else nn.Identity(),
            )
        self.input_projs = nn.ModuleDict(input_proj_list)

        num_encoder_layers = configs['num_encoder_layers']
        deform_scales, fpn_scales = configs['deform_scales'], configs['fpn_scales']
        drop_path_rate=configs['drop_path_rate']
        deform_configs, mlp_configs, fpn_configs = configs['deform_configs'], configs['mlp_configs'], configs['fpn_configs']
        self.ms_d_model = ms_d_model
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        
        deform_configs['reg_sequence_length'] = (self.dino_stage_idx - self.first_attn_stage_idx) * self.task_num_regs[0] + \
                                                (self.num_stages - self.dino_stage_idx) * (self.task_num_regs[0] + self.task_num_regs[1] + self.task_num_regs[2])
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        self.deform_layers = nn.ModuleList([MSDeformAttnTransformerEncoderLayer(d_model=ms_d_model,
                                                                                multiscale_shapes=self.multiscale_shapes, deform_scales=deform_scales, fpn_scales=fpn_scales, 
                                                                                fpn_configs=fpn_configs,
                                                                                
                                                                                num_feature_levels=len(self.deform_scales),
                                                                                deform_configs=deform_configs,
                                                                                mlp_configs=mlp_configs,
                                                                                drop_path=dp_rates[j]) for j in range(num_encoder_layers)])
        self.pos_2d = build_position_encoding('2d')
    

    def is_a_peft_dino(self, configs):
        dino_configs = DINO_NAME_TO_CONFIGS[configs['name']]
        
        self.ssl = DinoVisionTransformer(**dino_configs)
        name_to_pt_path = {'dinov2_vitb14_reg': 'dinov2/dinov2_vitb14_reg4_pretrain.pth',
                           'dinov2_vits14_reg': 'dinov2/dinov2_vits14_reg4_pretrain.pth'}
        state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[configs['name']]), map_location='cpu')
        self.ssl.load_state_dict(state_dict, strict=True) 
        if configs['freeze']:
            for p in self.ssl.parameters():
                p.requires_grad_(False)

        self.num_pt_registers = configs['num_pt_registers']
        self.pt_task_regs = nn.Embedding(self.num_pt_registers, self.ssl.embed_dim)
        self.pt_task_reg_poses = nn.Embedding(self.num_pt_registers, self.ssl.embed_dim)
        nn.init.normal_(self.pt_task_regs.weight, std=1e-6)
        trunc_normal_(self.pt_task_reg_poses.weight, std=0.02)

        self.task_num_regs = [self.num_pt_registers, self.ssl.cls_token.shape[1], self.ssl.register_tokens.shape[1]]
        self.num_registers = sum(self.task_num_regs)
        self.ssl_pretrain_size = (dino_configs['img_size'], dino_configs['img_size'])

    def __init__(self, configs, 
                 pretrain_size=224, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_CTI=True, pretrained=None,with_cp=False,
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 cnn_feature_interaction=True,
                 dim_ratio=6.0,
                 *args, **kwargs):
        super().__init__()
        self.build_vit_backbone()
        # embed_dim
        # 
        self.is_a_peft_dino(configs['dino'])
        # self.has_multiscale(configs.pop('ms_configs'))

        ms_configs = configs['ms_configs']


        use_CTI_toC = ms_configs['use_CTI_toC'] 
        use_CTI_toV = ms_configs['use_CTI_toV']  
        conv_inplane = ms_configs['conv_inplane']  
        add_vit_feature = ms_configs['add_vit_feature'] 
        add_vit_feature = ms_configs['add_vit_feature'] 
        add_vit_feature = ms_configs['add_vit_feature'] 


        self.ms_fusion_deps = ms_configs['ms_fusion_deps']
        self.level_embed = nn.Embedding(3, self.ssl.embed_dim)
        torch.nn.init.normal_(self.level_embed.weight)
        norm_layer = nn.LayerNorm

        self.local_conv = CNN(inplanes=conv_inplane, embed_dim=self.ssl.embed_dim)
        self.ms_fusion_before_layers = nn.ModuleList([
            CTIBlock(dim=self.ssl.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                    init_values=init_values, 
                    drop_path=ms_configs['cti_drop_path'],
                    norm_layer=norm_layer, with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                    use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                    extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(self.ms_fusion_deps))
        ])
        self.ms_fusion_after_layers = nn.ModuleList([
            CTIBlock(dim=self.ssl.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                    init_values=init_values, 
                    drop_path=ms_configs['cti_drop_path'],
                    norm_layer=norm_layer, with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                    use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                    dim_ratio=dim_ratio,
                    cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                    extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(self.ms_fusion_deps))
        ])

        self.up = nn.ConvTranspose2d(self.ssl.embed_dim, self.ssl.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.ssl.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.ssl.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        # decoder head things
        # logging.debug(f'ssl的总参数数量:{sum(p.numel() for p in self.ssl.parameters())}') 


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

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

    def get_deform_query_key_args(self, x):
        batch_size, _, H, W = x.shape

        # L 2
        res345_spatial_shapes = torch.as_tensor([(H // 8, W//8), (H//16, W//16), (H//32, W//32)]).long().to(self.device)
        dino_hw_spatial_shapes = torch.as_tensor([(H // 14, W//14)]).long().to(self.device)
        res345_valid_ratios = x.new_ones([batch_size, 3, 2]) # b L 2
        dino_hw_valid_ratios = x.new_ones([batch_size, 1, 2]) # b L 2
        
        # L
        res345_level_start_index = torch.cat((res345_spatial_shapes.new_zeros((1, )), res345_spatial_shapes.prod(1).cumsum(0)[:-1]))
        # L 2 -> b hw_sigma 1 2
        res345_reference_points = self.get_reference_points(dino_hw_spatial_shapes.long().to(self.device), res345_valid_ratios).to(self.device)

        dino_hw_level_start_index = dino_hw_spatial_shapes.new_zeros((1, ))
        dino_hw_reference_points = self.get_reference_points(res345_spatial_shapes, dino_hw_valid_ratios).to(self.device)
        
        
        return [res345_reference_points, res345_spatial_shapes, res345_level_start_index], \
        [dino_hw_reference_points, dino_hw_spatial_shapes, dino_hw_level_start_index]

    def forward_decoder(self, pt_regs, hw_feats, res345, res2):
        # pt_regs = self.ssl.norm(pt_reg_poses)
        # clsssl_regs = self.ssl.norm(clsssl_regs)
        # x = self.ssl.norm(x)  
        pass
        
    def forward(self, x, masks=None):
        batch_size, _, H, W = x.shape
        res345_deform_key_args, dino_deform_key_args = self.get_deform_query_key_args(x)

        res2, res3, res4, res5 = self.local_conv(x) # b h w c
        res345_poses = [self.ssl.interpolate_pos_encoding_hw2(src) + self.level_embed.weight[src_idx].contiguous() \
                             for src_idx, src in enumerate(res3, res4, res5)] # list[b h w c]
        res345 = torch.cat([res3.flatten(1,2), res4.flatten(1,2), res5.flatten(1,2)], dim=1) # b hw_sigma c
        res345_poses = torch.cat([pos.flatten(1,2) for pos in res345_poses], dim=1) # b hw_sigma c

        patch_size = [H//self.ssl.patch_size, W//self.ssl.patch_size]
        dino_hw_feats = self.ssl.patch_embed(x) # b hw/14 c
        dino_hw_feats = dino_hw_feats.view(batch_size, patch_size[0], patch_size[1], x.shape[-1]).contiguous() # b h w c
        if masks is not None:
            dino_hw_feats = torch.where(masks.unsqueeze(-1), self.ssl.mask_token.to(dino_hw_feats.dtype).unsqueeze(0), x)   
        dino_hw_feats_poses = self.ssl.interpolate_pos_encoding_hw(dino_hw_feats, W, H) # b hw c
        dino_hw_feats = dino_hw_feats.flatten(1,2).contiguous() + dino_hw_feats_poses
        num_hw_tokens = dino_hw_feats.shape[1]

        clsssl_regs = torch.cat([self.ssl.cls_token + self.ssl.pos_embed[:, 0].contiguous().unsqueeze(0), self.ssl.register_tokens], dim=1)
        clsssl_regs = repeat(clsssl_regs, '1 n c -> b n c',b=batch_size)
        num_clsssl_regs = clsssl_regs.shape[1]

        pt_regs = repeat(self.pt_task_regs.weight, 'n c -> b n c', b=batch_size)
        pt_reg_poses = repeat(self.pt_task_reg_poses.weight, 'n c -> b n c',b=batch_size)

        fusion_layer_outs = []
        # 0, 3, 6, 9
        for dep, blk in enumerate(self.ssl.blocks):
            if dep in self.ms_fusion_deps:
                res345 = res345 + res345_poses

                ms_fusion_idx = self.ms_fusion_deps.index(dep)
                pt_regs, clsssl_regs, dino_hw_feats = self.ms_fusion_before_layers[ms_fusion_idx](\
                        pt_regs=pt_regs, pt_reg_poses=pt_reg_poses, clsssl_regs=clsssl_regs, dino_hw_feats=dino_hw_feats, 
                        res345_feats=res345, res345_poses=res345_poses, res345_deform_key_args=res345_deform_key_args)

                blk_input = torch.cat([pt_regs, clsssl_regs, dino_hw_feats], dim=1)
                blk_input = blk(blk_input)
                pt_regs, clsssl_regs, dino_hw_feats = blk_input.split([self.num_pt_registers, num_clsssl_regs, num_hw_tokens], dim=1)

                res345, res2 = self.ms_fusion_after_layers[ms_fusion_idx](\
                    res345_feats=res345, res345_poses=res345_poses, dino_deform_key_args=dino_deform_key_args, res2=res2,
                    pt_regs=pt_regs, clsssl_regs=clsssl_regs, dino_hw_feats=dino_hw_feats,)

                # layer_prediction like a decoder
                fusion_layer_outs.append(self.forward_decoder(pt_regs=pt_regs, dino_hw_feats=dino_hw_feats, res345=res345, res2=res2))

            else:
                blk_input = torch.cat([pt_regs, clsssl_regs, dino_hw_feats], dim=1)
                blk_input = blk(blk_input)
                pt_regs, clsssl_regs, dino_hw_feats = blk_input.split([self.num_pt_registers, num_clsssl_regs, num_hw_tokens], dim=1)

    


        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x
    
class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        drop_path_rate_list=None,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if drop_path_rate_list is not None:
            assert depth == len(drop_path_rate_list)
            dpr = drop_path_rate_list
        
        if ffn_layer == "mlp":
            logging.debug("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logging.debug("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logging.debug("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        elif ffn_layer == 'kan':
            logging.debug('using Kan layer as FFN')
            ffn_layer = KANLayer
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        # b 1_hw c
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def interpolate_pos_encoding_hw(self, x, w, h):
        # b h/14 w/14 c
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1) # b h/14 w/14 c
        return patch_pos_embed.to(previous_dtype)

    def interpolate_pos_encoding_hw2(self, x):
        # b h w c
        previous_dtype = x.dtype
        _, h0, w0, dim = x.shape
        N = self.pos_embed.shape[1] - 1
        if (h0 * w0 == N) and (h0 == w0):
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, 1:]
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).contiguous().to(previous_dtype) # b h w c
        return patch_pos_embed


    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x, masks=None):
        # b s c -> b s c
        for blk in self.blocks:
            x = blk(x)
        return x


from models.backbone.metaformer_build_tool import SepConv, Attention,  Attention_REG, \
    DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG, LayerNormWithoutBias, LayerNormGeneral, MetaFormerBlock,\
        DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG, DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG, StarReLU, DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTONE_REG
from models.backbone.metaformer_build_tool import Mlp as Meta_MLP
from models.backbone.metaformer_build_tool import MlpHead 
from timm.models.layers import trunc_normal_, DropPath
def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MSFusion_After_Layer(nn.Module):
    def __init__(self, d_model, after_configs) -> None:
        super().__init__()
        deform_configs = after_configs['deform']
        self.res345_cross_dino_deform = MSDeformAttn_with_GlobalRegisters(d_model=d_model, num_feature_levels=1, deform_configs=deform_configs)
        
        res345_self_configs = after_configs['self']
        # token mixer
        if res345_self_configs['name'] is not None:
            self.res345_self = None
        elif res345_self_configs['name'] == 'deform':
            self.res345_self = None
        elif res345_self_configs['name'] == 'conv':
            self.res345_self = None
        
        self.fpn = None
        self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        self.fpns = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1) \
                                          for _ in range(4)])
        
        # feature mixer
        self.mlp = None

        # cross attention masked attention
        # masked deform attention
        # token mixer


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

    def forward(self, src, pos, fpns, scale_to_hw_length, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # b reg_hw_sigma c： fpns: b h w c
        scale_hw_length = [haosen[0] * haosen[1] for haosen in scale_to_hw_length]
        
        src2 = self.norm1(src)
        fpns2 = [self.norm1(haosen) for haosen in fpns]
        
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reference_points, src2, spatial_shapes, level_start_index, padding_mask, sum(scale_hw_length), )
        
        hw_feats = src2[:, -sum(scale_hw_length):].contiguous().split(scale_hw_length, dim=1)
        fpn_and_deform_feats = [rearrange(hw_feats[haosen], 'b (h w) c -> b h w c',h=scale_to_hw_length[haosen][0], w=scale_to_hw_length[haosen][1])\
            for haosen in self.fpn_and_deform_idxs]
        fpns2 = self.forward_fpn(fpns2, fpn_and_deform_feats)
        
        src = src + self.dropout1(src2)
        fpns = [fpns[idx]+self.dropout1(fpns2[idx]) for idx in range(len(fpns))]
        
        src = src + self.dropout2(self.mlp(self.norm2(src)))
        
        fpns = [fpns[idx]+self.dropout2(self.mlp(self.norm2(fpns[idx]))) for idx in range(len(fpns))]

        return src, fpns

class MSFusion_Before_Layer(nn.Module):
    def __init__(self, d_model, before_configs) -> None:
        super().__init__()
        deform_configs = before_configs['deform']
        self.dino_cross_res345_deform = MSDeformAttn_with_GlobalRegisters(d_model=d_model, num_feature_levels=3, deform_configs=deform_configs)
        
        res345_self_configs = before_configs['self']
        # token mixer
        if res345_self_configs['name'] is not None:
            self.res345_self = None
        elif res345_self_configs['name'] == 'deform':
            self.res345_self = None
        elif res345_self_configs['name'] == 'conv':
            self.res345_self = None
        
        self.fpn = None
        self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        self.fpns = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1) \
                                          for _ in range(4)])
        
        # feature mixer
        self.mlp = None

        # cross attention masked attention
        # masked deform attention
        # token mixer


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

    def forward(self, src, pos, fpns, scale_to_hw_length, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # b reg_hw_sigma c： fpns: b h w c
        scale_hw_length = [haosen[0] * haosen[1] for haosen in scale_to_hw_length]
        
        src2 = self.norm1(src)
        fpns2 = [self.norm1(haosen) for haosen in fpns]
        
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reference_points, src2, spatial_shapes, level_start_index, padding_mask, sum(scale_hw_length), )
        
        hw_feats = src2[:, -sum(scale_hw_length):].contiguous().split(scale_hw_length, dim=1)
        fpn_and_deform_feats = [rearrange(hw_feats[haosen], 'b (h w) c -> b h w c',h=scale_to_hw_length[haosen][0], w=scale_to_hw_length[haosen][1])\
            for haosen in self.fpn_and_deform_idxs]
        fpns2 = self.forward_fpn(fpns2, fpn_and_deform_feats)
        
        src = src + self.dropout1(src2)
        fpns = [fpns[idx]+self.dropout1(fpns2[idx]) for idx in range(len(fpns))]
        
        src = src + self.dropout2(self.mlp(self.norm2(src)))
        
        fpns = [fpns[idx]+self.dropout2(self.mlp(self.norm2(fpns[idx]))) for idx in range(len(fpns))]

        return src, fpns


class Instance_Detection_Head(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,):
        pass


class SemanticSegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,):
        pass



