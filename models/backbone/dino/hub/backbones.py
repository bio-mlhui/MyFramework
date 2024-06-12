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

from .. import dino_model as vits


@BACKBONE_REGISTRY.register()
class DinoV2(nn.Module):
    def __init__(self,
            configs,
            # arch_name: str = "vit_large",
            # img_size: int = 518,
            # patch_size: int = 14,
            # init_values: float = 1.0,
            # ffn_layer: str = "mlp",
            # block_chunks: int = 0,
            # num_register_tokens: int = 0,
            # interpolate_antialias: bool = False,
            # interpolate_offset: float = 0.1,
            # pretrained: bool = True,
            # weights: Union[Weights, str] = Weights.LVD142M,
            # **kwargs,
        ):
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

        freeze = configs.pop('freeze', False)
        super().__init__()
        
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
        model = vits.__dict__[arch_name](**vit_kwargs)

        if pretrained:
            model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
            state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), f'{model_base_name}_{model_full_name}_pretrain.pth'), map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
        if freeze:
            for p in model.parameters():
                p.requires_grad_(False)
        self.model = model
        return model
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    


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
