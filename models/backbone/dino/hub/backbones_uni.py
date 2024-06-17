# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Union
import torch.nn as nn
import torch
import os
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name
class Weights(Enum):
    LVD142M = "LVD142M"
from detectron2.modeling import BACKBONE_REGISTRY
import torch.nn.functional as F
import logging
import detectron2.utils.comm as comm
 
def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import dino as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model

def dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)

def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from models.backbone.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from kan import KANLayer

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
    
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, LoraModel
from detectron2.modeling import META_ARCH_REGISTRY
from models.backbone.metaformer_build_tool import SepConv, Attention,  Attention_REG, \
    DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG, LayerNormWithoutBias, LayerNormGeneral, MetaFormerBlock, DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG, DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG
from models.backbone.metaformer_build_tool import Mlp as Meta_MLP
@BACKBONE_REGISTRY.register()
class Dinov2_LORA_REG(nn.Module):
    # 假设: 在 8/阶段
    # def _load_from_state_dict(
    #     self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    # ):
    #     """
    #     partially load the token embeddings
    #     """
    #     with torch.no_grad():
    #         # state_dict
    #         reg_keys = [key for key in state_dict.keys() if 'register_tokens' in key]
    #         assert len(reg_keys) == 1
    #         reg_keys = reg_keys[0]
    #         reg_token_weights = state_dict[reg_keys] # 1 4 dim
    #         original_number = reg_token_weights.shape[1]
    #         # repeat
    #         if self.num_register_tokens != original_number:
    #             repeats = self.num_register_tokens // original_number
    #             reg_token_weights = reg_token_weights.repeat(1, repeats, 1) # 1 

    #         state_dict[reg_keys] = reg_token_weights  
        
    #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def has_a_peft_ssl(self, ssl_configs, ssl_freeze, ssl_pt_path, ssl_out_dino_dim, lora_configs):
        self.ssl = DinoVisionTransformer(**ssl_configs)
        state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), ssl_pt_path), map_location='cpu')
        self.ssl.load_state_dict(state_dict, strict=True) 
        self.use_dino_norm = ssl_out_dino_dim
        if ssl_freeze:
            for p in self.ssl.parameters():
                p.requires_grad_(False)
        # ssl_finetune
        # peft_config = LoraConfig(
        #     r=8,
        #     target_modules=None, # list[str]
        #     lora_alpha = 8,
        #     lora_dropout = 0,
        #     fan_in_fan_out = True, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        #     bias="none", # If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation
        #     use_rslora= False, # scaling factor设置成 alpha/r 还是 alpha/sqrt(r)
        #     modules_to_save = None, # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint
        #     init_lora_weights='pissa', # How to initialize the weights of the adapter layers
        #     layers_to_transform= None, # list[int] If a list of ints is passed, it will apply the adapter to the layer indices that are specified in this list. 
        #     layers_pattern= None, #  The layer pattern name, used only if layers_to_transform is different from None.
        #     rank_pattern = None, # the mapping from layer names or regexp expression to ranks which are different from the default rank specified by r.
        #     alpha_pattern = None, # The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha.
        #     megatron_config = None, #  The TransformerConfig arguments for Megatron.
        #     megatron_core="megatron.core", #  The core module from Megatron to use
        #     loftq_config = dict, #  The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights and initialize Lora layers. Also pass init_lora_weights='loftq'. Note that you should not pass a quantized model in this case, as LoftQ will quantize the model itself.
        #     use_dora= False, # Weight-Decomposed Low-Rank Adaptation
        #     layer_replication= None #  Build a new stack of layers by stacking the original model layers according to the ranges specified. This allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will all have separate LoRA adapters attached to them.
        # )
        # self.ssl = LoraModel(self.ssl, config=peft_config, adapter_name='default')

    def has_metaformer(self, 
                       configs,
                       mlps=Meta_MLP,
                       norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                       layer_scale_init_values=None,               
                 ):
        meta_configs= configs.pop('meta_configs')
        ssl_configs= meta_configs.pop('ssl_configs')
        name_to_configs = {
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
        dino_configs, ssl_pt_path, ssl_freeze, ssl_out_dino_dim, ssl_lora_configs = \
            name_to_configs[ssl_configs['dino_name']], ssl_configs['pt_path'], ssl_configs['freeze'], ssl_configs['out_dino_norm'], ssl_configs['lora_configs']
        
        depths, dims, token_mixers, drop_path_rate, downsample_layer_name, self.first_attn_stage_idx = meta_configs.pop('depths'), meta_configs.pop('dims'), meta_configs.pop('token_mixers'),\
            meta_configs.pop('drop_path_rate'),  meta_configs.pop('downsample_layer_name'), meta_configs.pop('first_attn_stage_idx')
        res_scale_init_values = meta_configs.pop('res_scale_init_values')
        if downsample_layer_name == 'four_stage_last_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG
        elif downsample_layer_name == 'four_stage_lasttwo_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG
        elif downsample_layer_name == 'five_stage_lasttwo_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG
        else:
            raise ValueError()
        
        dino_stage_idx, num_stage = depths.index(None), len(depths)
        dims[dino_stage_idx], depths[dino_stage_idx] = dino_configs['embed_dim'], dino_configs['depth']
        self.dims = dims
        name_to_token_mixers = {'conv': SepConv, 'attn_32': Attention, 'first_attn_32': partial(Attention_REG, reg_cls=self),\
            'iden': nn.Identity, 'attn_64': partial(Attention, head_dim=64), 'first_attn_64': partial(Attention_REG, reg_cls=self, head_dim=64),  }
        
        down_dims = [3] + dims
        self.downsample_layers = nn.ModuleList([downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)])
        
        assert len(token_mixers) == num_stage
        depth_token_mixers = []
        for stage_idx, (haosen_dep, haosen_mixer) in enumerate(zip(depths, token_mixers)):
            if isinstance(haosen_mixer, str):
                depth_token_mixers.extend([name_to_token_mixers[haosen_mixer]] * haosen_dep)
            elif isinstance(haosen_mixer, list):
                for hhsen_mixer, hhsen_depth in haosen_mixer:
                    assert isinstance(hhsen_mixer, str) and isinstance(hhsen_depth, int)
                    depth_token_mixers.extend([name_to_token_mixers[hhsen_mixer]] * hhsen_depth)
            elif haosen_mixer is None:
                depth_token_mixers.extend([None] * haosen_dep)
            else:
                raise ValueError()
        assert len(depth_token_mixers) == sum(depths)
        
        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage
        cur = 0
        self.before_stages = nn.ModuleList()
        self.after_stages = nn.ModuleList()
        for i in range(num_stage):
            if i != dino_stage_idx:
                stage = nn.Sequential(
                    *[MetaFormerBlock(dim=dims[i],
                    token_mixer=depth_token_mixers[cur+j],
                    mlp=mlps[i],
                    norm_layer=norm_layers[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i],
                    ) for j in range(depths[i])]
                )
                if i < dino_stage_idx:
                    self.before_stages.append(stage)
                elif i > dino_stage_idx:
                    self.after_stages.append(stage)   
            else:
                dino_configs['drop_path_rate_list'] = dp_rates[cur: (cur+depths[i])]
                self.has_a_peft_ssl(ssl_configs=dino_configs, ssl_freeze=ssl_freeze, ssl_pt_path=ssl_pt_path, lora_configs=ssl_lora_configs,
                                    ssl_out_dino_dim=ssl_out_dino_dim)
                self.alias_conv = nn.Conv2d(self.ssl.embed_dim, self.ssl.embed_dim, kernel_size=3, padding=1,)
                self.alias_norm = partial(LayerNormGeneral, bias=False, eps=1e-6)(self.ssl.embed_dim)
            cur += depths[i]
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.before_stages.apply(_init_weights)
        self.after_stages.apply(_init_weights)
        self.downsample_layers.apply(_init_weights)
        self.dino_stage_idx = dino_stage_idx
        stage_to_level_embed = [None] * self.first_attn_stage_idx
        for haosen in range(self.first_attn_stage_idx, len(depths)):
            level_embed = nn.Embedding(1, self.dims[haosen])
            torch.nn.init.normal_(level_embed.weight)
            stage_to_level_embed.append(level_embed)
        
        self.stage_to_level_embed = nn.ModuleList(stage_to_level_embed)
        
    def __init__(self, configs,):
        super().__init__()
        self.has_metaformer(configs=configs)    
        
        from models.backbone.utils import ImageMultiscale_Shape
        multiscale_shapes = {}
        for idx, _ in enumerate(self.before_stages):
            multiscale_shapes[f'res{idx+2}'] = ImageMultiscale_Shape(2**(idx+2), self.dims[idx])
        multiscale_shapes[f'res{len(self.before_stages)+2}'] = ImageMultiscale_Shape(self.ssl.patch_size, self.ssl.embed_dim) 
        for idx, _ in enumerate(self.after_stages):
            multiscale_shapes[f'res{len(self.before_stages)+1+idx+2}'] = ImageMultiscale_Shape(self.ssl.patch_size * (2**(idx+1)), self.dims[len(self.before_stages)+idx+1])        
        self.multiscale_shapes = multiscale_shapes
        self.max_stride = self.ssl.patch_size * (2**(len(self.after_stages)))
        task_configs = configs.pop('task_configs')
        self.local_tasks, self.global_tasks = configs.pop('local_tasks'), configs.pop('global_tasks')
        self.task_to_num_regs = {'cls': 1, 'ssl': self.ssl.num_register_tokens}
        # ssl的pos
        self.ssl_reg_pos = nn.Parameter(torch.zeros(1, self.ssl.num_register_tokens, self.ssl.embed_dim))
        
        # local_registers
        self.sem_seg_reg = nn.Parameter(torch.zeros(1, task_configs['sem_seg']['num_registers'], self.dims[self.first_attn_stage_idx]))
        self.sem_seg_reg_pos = nn.Parameter(torch.zeros(1, task_configs['sem_seg']['num_registers'], self.dims[self.first_attn_stage_idx]))
        self.task_to_num_regs['sem_seg'] = task_configs['sem_seg']['num_registers']
        self.ins_det_reg = nn.Parameter(torch.zeros(1, task_configs['ins_det']['num_registers'], self.dims[self.first_attn_stage_idx]))
        self.ins_det_reg_pos = nn.Parameter(torch.zeros(1, task_configs['ins_det']['num_registers'], self.dims[self.first_attn_stage_idx]))
        self.task_to_num_regs['ins_det'] = task_configs['ins_det']['num_registers']
        
        self.task_to_reg_ptr = {'sem_seg': self.sem_seg_reg, 'ins_det': self.ins_det_reg, 'cls': self.ssl.cls_token, 'ssl': self.ssl.register_tokens}
        self.task_to_reg_pos_ptr = {'sem_seg': self.sem_seg_reg_pos, 'ins_det': self.ins_det_reg_pos, 'ssl': self.ssl_reg_pos}

        trunc_normal_(self.ssl_reg_pos, std=0.02)
        trunc_normal_(self.sem_seg_reg_pos, std=0.02)
        nn.init.normal_(self.sem_seg_reg, std=1e-6)
        trunc_normal_(self.ins_det_reg_pos, std=0.02)
        nn.init.normal_(self.ins_det_reg, std=1e-6)
       
        # head
        # 和数据集相关?
        # task_name_to_head_cls = {'cls': ClassificationHead, 'sem': SemanticSegmentationHead, 'ins': Instance_Detection_Head}
        # for t_name, t_config in task_configs.items():
        #     # task_heads[t_name] = task_name_to_head_cls[t_name](t_config)
        # self.task_heads = nn.ModuleDict(task_heads)
        ms_configs = configs.pop('ms_configs')
        ms_configs['deform_configs']['cls_ssl_num_scales'] = len(self.downsample_layers) - self.dino_stage_idx
        ms_configs['deform_configs']['tasks'] = self.local_tasks + self.global_tasks
        self.ms_fusion = META_ARCH_REGISTRY.get(ms_configs['name'])(configs=ms_configs,
                                                                    multiscale_shapes=multiscale_shapes,
                                                                    task_to_num_regs=self.task_to_num_regs)
        if comm.is_main_process():
            logging.debug(f'SSL的总参数数量:{sum(p.numel() for p in self.ssl.parameters())}')
            logging.debug(f'MS的总参数数量:{sum(p.numel() for p in self.ms_fusion.parameters())}')
            logging.debug(f'before的总参数数量:{sum(p.numel() for p in self.before_stages.parameters())}')
            logging.debug(f'after的总参数数量:{sum(p.numel() for p in self.after_stages.parameters())}')
            logging.debug(f'downsample的总参数数量:{sum(p.numel() for p in self.downsample_layers.parameters())}')
            logging.debug(f'ssl_reg_pos的总参数数量:{sum(p.numel() for p in [self.ssl_reg_pos,])}')
            logging.debug(f'task_registers的总参数数量:{sum(p.numel() for p in [self.ins_det_reg, self.sem_seg_reg])}')
            logging.debug(f'task_registers_pos的总参数数量:{sum(p.numel() for p in [self.ins_det_reg_pos, self.sem_seg_reg_pos])}')
            logging.debug(f'alias_conv_norm的总参数数量:{sum(p.numel() for p in self.alias_conv.parameters())}') # 5M
            logging.debug(f'alias_conv_norm的总参数数量:{sum(p.numel() for p in self.alias_norm.parameters())}')
            pass
    def get_local_registers(self,):
        register_toks = torch.cat([self.task_to_reg_ptr[haosen]  for haosen in self.local_tasks], dim=1) # 1 s c
        # 1 c -> 1 1 c,  1 s c
        register_tok_poses = torch.cat([self.task_to_reg_pos_ptr[haosen] for haosen in self.local_tasks], dim=1)
        return register_toks, register_tok_poses
    
    def get_global_registers(self, ):
        register_toks = torch.cat([self.task_to_reg_ptr[haosen]  for haosen in self.global_tasks], dim=1) # 1 s c
        # 1 c -> 1 1 c,  1 s c
        register_tok_poses = torch.cat([self.task_to_reg_pos_ptr[haosen] if haosen != 'cls' else self.ssl.pos_embed[:, 0].unsqueeze(0)\
            for haosen in self.global_tasks], dim=1) 
        return  register_toks,  register_tok_poses  

    def get_first_attn_level_embed(self,):
        return self.stage_to_level_embed[self.first_attn_stage_idx].weight

    def prepare_tokens_with_masks(self, x, masks=None):
        x, task_registers = x['x'], x['task_registers']  # 1 s c
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # b s c
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    task_registers.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )   # b 1+4+s+hw c
        return x

    def forward_before(self, x):
        ret = []
        for i in range(len(self.before_stages)):
            x = self.downsample_layers[i](x)
            x = self.before_stages[i](x)
            ret.append(x.contiguous()) # b hw c / b reg_hw c
        return ret
    
    def forward_after(self, registers, x):
        ret = []
        for i in range(len(self.after_stages)):
            registers, x = self.downsample_layers[i+self.dino_stage_idx+1](registers, x)
            _, H, W, _ = x.shape
            input = torch.cat([registers, x.flatten(1,2)], dim=1) # b reg_hw c
            input = input + self.stage_to_level_embed[i+self.dino_stage_idx+1].weight
            
            input = self.after_stages[i](input)
            registers, x = input.split([registers.shape[1], H*W], dim=1)
            x = x.view(x.shape[0], H, W, -1)
            ret.append((registers, x))
        return ret

    # NestTensor?
    def forward_dino_ssl(self, x, local_regs, last_x, masks=None,): # b c h w -> ms: {'res2', 'res3', 'res4, 'res5}, reg: {'reg2', 'reg3', 'reg4', 'reg5'}
        batch_size, _, H, W = x.shape # b c h w
        patch_size = [H // self.ssl.patch_size, W // self.ssl.patch_size]
        x = self.ssl.patch_embed(x) # b c h w -> b hw c
        x = x.view(x.shape[0], patch_size[0], patch_size[1], x.shape[-1]) # b h w c
 
        last_x = F.interpolate(last_x.permute(0, 3, 1, 2), patch_size, mode='bilinear', align_corners=False).contiguous() # b c h w
        last_x = self.alias_norm(self.alias_conv(last_x).permute(0, 2, 3, 1)) # b h w c
        # x = self.alias_norm(self.alias_conv(x + last_x).permute(0, 2, 3, 1).flatten(1,2).contiguous())
        x = (x + last_x).flatten(1, 2) # b hw c
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.ssl.mask_token.to(x.dtype).unsqueeze(0), x)    
        x_poses = self.ssl.interpolate_pos_encoding_hw(x, W, H) # b hw c
        x = x + x_poses
        x = x + self.stage_to_level_embed[self.dino_stage_idx].weight
                
        global_regs, global_reg_poses = self.get_global_registers()
        global_regs = global_regs + global_reg_poses
        x = torch.cat([local_regs, global_regs, x], dim=1) # b reg_hw c
        
        for blk in self.ssl.blocks:
            x = blk(x)
        if self.use_dino_norm:
            x = self.ssl.norm(x)

        reg, x = x.split([x.shape[1] - patch_size[0]*patch_size[1], patch_size[0]*patch_size[1]], dim=1) # b reg_hw c -> b reg c; b hw c
        x = x.view(x.shape[0], patch_size[0], patch_size[1], -1).contiguous()
        return reg, x
    
    
    def forward(self, x, masks=None):
        
        masks = None # b t h w
        x = x # b c t h w
        # masks = masks.squeeze(1).flatten(1) # b hw
        x = x.squeeze(2) # b c h w
        batch_size, _, H, W = x.shape
        # stage1
        ret = []
        ret.extend( self.forward_before(x)) # # [b h/4 w/4 c, b reg_h/8*w/8 2c]
        assert self.first_attn_stage_idx == 1 and len(self.before_stages) == 2
        last_feat = ret[-1] # b reg_hw c
        
        num_local_regs = sum([self.task_to_num_regs[haosen] for haosen in self.local_tasks])
        # reg, hw
        local_regs, last_x = last_feat.split([num_local_regs, last_feat.shape[1] - num_local_regs], dim=1)
        before_stride = 4 * (2 ** (len(self.before_stages) - 1))
        last_x = last_x.view(batch_size, H//before_stride, W//before_stride, -1).contiguous()
        ret[-1] = (local_regs, last_x) # b s c, b h w c
        
        local_regs, last_x = self.downsample_layers[len(self.before_stages)](local_regs, last_x) 
        registers, x = self.forward_dino_ssl(x, local_regs=local_regs, last_x=last_x, masks=masks) # b s c, b h w c
        ret.append((registers, x.contiguous()))
        
        after_feats = self.forward_after(registers=registers, x=x)
        ret.extend(after_feats)
        
        output = {}
        for haosen in range(self.first_attn_stage_idx):
            assert isinstance(ret[haosen], torch.Tensor)
            output[f'res{haosen+2}'] = {'hw': ret[haosen]}
        for haosen in range(self.first_attn_stage_idx, self.dino_stage_idx):
            stage_output = {'hw': ret[haosen][1]}
            reg_feats = ret[haosen][0].split([self.task_to_num_regs[haosen] for haosen in self.local_tasks], dim=1)
            for idx, ltask in enumerate(self.local_tasks):
                stage_output[ltask] = reg_feats[idx]
            output[f'res{haosen+2}'] = stage_output
        for haosen in range(self.dino_stage_idx, len(ret)):
            stage_output = {'hw': ret[haosen][1]}
            reg_feats = ret[haosen][0].split([self.task_to_num_regs[haosen] for haosen in self.local_tasks+self.global_tasks], dim=1)
            for idx, ltask in enumerate(self.local_tasks+self.global_tasks):
                stage_output[ltask] = reg_feats[idx]
            output[f'res{haosen+2}'] = stage_output

        ret = self.ms_fusion(output)
        pass


class Instance_Detection_Head(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,):
        pass

    def compute_loss(self, ):
        pass


class SemanticSegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,):
        pass

    def compute_loss(self, ):
        pass


class ClassificationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,):
        pass

    def compute_loss(self, ):
        pass


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
        # b hw c
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
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim) # b hw c
        return patch_pos_embed.to(previous_dtype)


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


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



