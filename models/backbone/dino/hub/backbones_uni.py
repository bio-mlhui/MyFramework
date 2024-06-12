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

import logging

 
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

@BACKBONE_REGISTRY.register()
class Dinov2_LORA_REG(nn.Module):
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
    
    def has_a_peft_ssl(self, 
                       configs,
                       peft_configs):
        arch_name = configs.pop('arch_name', 'vit_large')
        img_size = configs.pop('img_size', 518)
        patch_size = configs.pop('patch_size', 14)
        init_values = configs.pop('init_values', 1.0)
        ffn_layer = configs.pop('ffn_layer', 'mlp')
        block_chunks = configs.pop('block_chunks', 0)
        num_register_tokens = configs.pop('num_register_tokens', 0)
        interpolate_antialias = configs.pop('interpolate_antialias', False)
        interpolate_offset = configs.pop('interpolate_offset', 0.1)
        pretrained = configs.pop('pretrained', True)
        weights = configs.pop('weights', 'LVD142M')
        freeze_ssl = configs.pop('freeze_ssl', False)   
        pt_path = configs.pop('pt_path')

        # dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        # 'arch_name': 'vit_giant2',
        # 'img_size': 518,
        # 'patch_size': 14,
        # 'init_values': 1.0,
        # 'ffn_layer': 'swiglufused',
        # 'block_chunks': 0,
        # 'num_register_tokens': 4,
        # 'interpolate_antialias': True,
        # 'interpolate_offset': 0.0,
        # 'pretrained': True,
        # 'weights': 'LVD142M',
        # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        # 'interpolate_antialias': False,
        # 'interpolate_offset': 0.1,   
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
        vit_kwargs.update(**configs)
        arch_name_to_configs = {
            'vit_small': {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'mlp_ratio': 4,
                'block_fn': partial(Block, attn_class=MemEffAttention),
            },
            'vit_base': {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4,
                'block_fn': partial(Block, attn_class=MemEffAttention),
            },
            'vit_large': {
                'embed_dim': 1024,
                'depth': 24,
                'num_heads': 16,
                'mlp_ratio': 4,
                'block_fn': partial(Block, attn_class=MemEffAttention),
            },
            'vit_giant2': {
                'embed_dim': 1536,
                'depth': 40,
                'num_heads': 24,
                'mlp_ratio': 4,
                'block_fn': partial(Block, attn_class=MemEffAttention),
            },

        }
        vit_kwargs.update(arch_name_to_configs[arch_name])
        self.ssl = DinoVisionTransformer(**vit_kwargs)
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        if model_full_name not in pt_path:
            logging.warning('pt_path对应的预训练模型 和 现在的模型不匹配')
        state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), f'{pt_path}'), map_location='cpu')
        self.ssl.load_state_dict(state_dict, strict=True) 
        if freeze_ssl:
            for p in self.ssl.parameters():
                p.requires_grad_(False)
        # ssl_finetune
        peft_config = LoraConfig(
            r=8,
            target_modules=None, # list[str]
            lora_alpha = 8,
            lora_dropout = 0,
            fan_in_fan_out = True, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            bias="none", # If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation
            use_rslora= False, # scaling factor设置成 alpha/r 还是 alpha/sqrt(r)
            modules_to_save = None, # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint
            init_lora_weights='pissa', # How to initialize the weights of the adapter layers
            layers_to_transform= None, # list[int] If a list of ints is passed, it will apply the adapter to the layer indices that are specified in this list. 
            layers_pattern= None, #  The layer pattern name, used only if layers_to_transform is different from None.
            rank_pattern = None, # the mapping from layer names or regexp expression to ranks which are different from the default rank specified by r.
            alpha_pattern = None, # The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha.
            megatron_config = None, #  The TransformerConfig arguments for Megatron.
            megatron_core="megatron.core", #  The core module from Megatron to use
            loftq_config = dict, #  The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights and initialize Lora layers. Also pass init_lora_weights='loftq'. Note that you should not pass a quantized model in this case, as LoftQ will quantize the model itself.
            use_dora= False, # Weight-Decomposed Low-Rank Adaptation
            layer_replication= None #  Build a new stack of layers by stacking the original model layers according to the ranges specified. This allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will all have separate LoRA adapters attached to them.
        )
        self.ssl = LoraModel(self.ssl, config=peft_config, adapter_name='default')

    def __init__(self,
            configs,
        ):
        super().__init__()
        ssl_configs = configs.pop('ssl_configs')
        lora_configs = configs.pop('lora_configs')
        self.has_a_peft_ssl(ssl_configs, lora_configs)
        ms_configs = configs.pop('ms_configs')
        task_configs = configs.pop('task_configs')

        self.input_tasks = task_configs.keys()


        # ms_fusion: multiscale fm_reg -> multiscale fm_reg
        # self.ms_fusion = META_ARCH_REGISTRY.get(ms_configs['name'])(ms_configs)

        task_heads = {}
        task_registers = {}
        task_name_to_head_cls = {'cls': ClassificationHead, 'sem': SemanticSegmentationHead, 'ins': Instance_Detection_Head}
        for t_name, t_config in task_configs.items():
            task_registers[t_name] = nn.Parameter(torch.zeros(1, t_config['num_registers'], self.ssl.embed_dim))
            task_heads[t_name] = task_name_to_head_cls[t_name](t_config)  

        # 需要数据集的参数
        self.task_registers = nn.ModuleDict(task_registers)
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x,): # b c h w -> ms: {'res2', 'res3', 'res4, 'res5}, reg: {'reg2', 'reg3', 'reg4', 'reg5'}
        batch_size, _, H, W = x.shape
        task_registers = torch.cat([self.task_registers[t_name] for t_name in self.input_tasks], dim=1) # 1 s_t c
        
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

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

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

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



