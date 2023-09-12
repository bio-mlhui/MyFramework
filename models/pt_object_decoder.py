from typing import Any, Optional
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from einops import repeat, reduce, rearrange
from util.misc import NestedTensor
import matplotlib.pyplot as plt
import copy
import torch_geometric.nn as geo_nn
from torch_geometric.data import Batch
from scipy.optimize import linear_sum_assignment

###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
from .position_encoding import build_position_encoding
from .model_utils import find_scale_from_multiscales, find_scales_from_multiscales, pad_1d_feats, \
    register_model, get_optimizer, get_total_grad_norm,\
        visualization_for_AMR_V0, zero_module, _get_clones
from .layers_unimodal_attention import FeatureResizer, CrossAttentionLayer, MLP, SelfAttentionLayer, FFNLayer 
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
import pycocotools.mask as mask_util
import util.box_ops as box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride
from functools import partial


_Pt_Vis_Model_entrypoints = {}
def register_Pt_Vis_Model(fn):
    Pt_Vis_Model_name = fn.__name__
    _Pt_Vis_Model_entrypoints[Pt_Vis_Model_name] = fn

    return fn
def Pt_Vis_Model_entrypoint(Pt_Vis_Model_name):
    try:
        return _Pt_Vis_Model_entrypoints[Pt_Vis_Model_name]
    except KeyError as e:
        print(f'RVOS moel {Pt_Vis_Model_name} not found')


class UniNext_Vid_Vis_PtModel(nn.Module):
    def __init__(self, 
                 pt_path,
                 out_channel,) -> None:
        super().__init__()
        self.out_channel = out_channel
        self.model = UNINEXT_VID()

    def forward(self, video):
        return {'object_embeds': None,
                'object_box': None,
                'object_masks': None}
    
@register_Pt_Vis_Model
def dvis_vid(configs):
    pass

