# Modify for sample points visualization
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_

from ..functions import MSDeformAttnFunction
from detectron2.modeling import META_ARCH_REGISTRY

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
from einops import rearrange, repeat


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Simplify_S6_feat2offset(nn.Module):
    def __init__(
        self,
        d_inner,
        d_state=16,     # 默认hidden_state是16
        dt_rank="auto", # 默认是d_model // 16
        d_conv=1,       # 默认没有预先的conv1d
        d_conv_bias=True,
        use_Dskip=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        """
        input:  b s d_inner; s是一堆embeddings + query point 
        output: b s d_inner;  没有res链接, 我们想要的是d_inner到d_inner 但是input是feature, output是offset和weight
        s形成一个rnn
        """
        super().__init__()
        self.activation = "silu"
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        if self.d_conv > 1:
            self.act = StarReLU()
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=d_conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, 
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True,)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError 
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.zeros(self.d_inner).float() * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.use_Dskip = use_Dskip
        if self.use_Dskip:
            self.D = nn.Parameter(torch.ones(self.d_inner).float())  # Keep in fp32
            self.D._no_weight_decay = True
        else:
            self.D = None

        self._reset_parameters()
    
    def _reset_parameters(self,):
        constant_(self.x_proj.weight.data, 0.)
        if self.use_Dskip:
            constant_(self.D, 0.)


    def forward(self, x):
        # b d_inner L
        batch, dim, seqlen = x.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if self.d_conv > 1:
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        return selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            None,
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

# deformssv7
@META_ARCH_REGISTRY.register()
class MSDeform_OnOffset_v1(nn.Module):
    def __init__(self, scan_configs=None,):
        super().__init__()
        self.d_model = scan_configs['d_model']
        self.n_levels = scan_configs['n_levels']
        self.n_heads = scan_configs['n_heads']
        self.n_points = scan_configs['enc_n_points']

        # 每个head有不同的offset embeds, 但是每个head共享同一个s6; pixel-level的操作就别用embeds了
        # b nq c -> b nq head L k off_dim 
        self.offset_dim = scan_configs['offset_dim']
        self.offset_embeds = nn.Linear(self.d_model, self.n_heads * self.n_levels * self.n_points * self.offset_dim, bias=False)
        if scan_configs['offset_act_name'] == 'star_relu':
            self.offset_act = StarReLU()
        elif scan_configs['offset_act_name'] == 'silu':
            self.offset_act = nn.SiLU()
        elif scan_configs['offset_act_name'] == 'none':
            self.offset_act = nn.Identity()
        else:
            raise ValueError()
        
        # b*nq*head LK off_dim -> b*nq*head LK off_dim
        self.offset_scan = Simplify_S6_feat2offset(d_inner=self.offset_dim,
                                                   d_state=scan_configs['d_state'],
                                                   dt_rank=scan_configs['dt_rank'],
                                                   d_conv=scan_configs['d_conv'],
                                                   d_conv_bias=scan_configs['d_conv_bias'],
                                                   use_Dskip=scan_configs['use_Dskip'],)
        # b*nq head L k off_dim -> b*nq head L k 2
        self.sampling_offsets_weight = nn.Parameter(torch.zeros([self.n_heads, self.n_levels, self.n_points, self.offset_dim, 2]))
        # b*nq head L k 2
        self.sampling_offsets_bias = nn.Parameter(torch.zeros([self.n_heads, self.n_levels, self.n_points, 2]))
        # b*nq head L K off_dim -> b*nq head L k
        self.attention_weights_weight = nn.Parameter(torch.zeros([self.n_heads, self.n_levels, self.n_points, self.offset_dim, 1]))
        # b*nq head L k 1
        self.attention_weights_bias = nn.Parameter(torch.zeros([self.n_heads, self.n_levels, self.n_points]))

        self.im2col_step = 512

        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(self.d_model, self.n_heads))
        _d_per_head = self.d_model // self.n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")  

        self.value_proj = nn.Linear(self.d_model, self.d_model) 
        self.output_proj = nn.Linear(self.d_model, self.d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_embeds.weight.data, 0)
        self.offset_scan._reset_parameters()
        constant_(self.sampling_offsets_weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets_bias = nn.Parameter(grid_init)
        constant_(self.attention_weights_weight.data, 0.)
        constant_(self.attention_weights_bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        # b hw_sigma d -> b hw_sigma head*head_dim
        value = self.value_proj(input_flatten)

        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # b nq c -> b nq head*off_dim*L*K -> b*nq*head off_dim LK
        offset_embeds = self.offset_act(self.offset_embeds(query))
        offset_embeds = rearrange(offset_embeds, 'b nq (head L K off_dim) -> (b nq) head L K 1 off_dim',
                                  head=self.n_heads, L=self.n_levels, K=self.n_points,nq=Len_q)
        # b*nq head L K 1 off_dim  @ head L K off_dim 2  
        sampling_offsets = (offset_embeds @ self.sampling_offsets_weight).squeeze(4) + self.sampling_offsets_bias  # b*nq head L K 2
        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # b*nq head L k 1 off_dim @ head L K off_dim 1 -> b*nq head L K
        attention_weights = (offset_embeds @ self.attention_weights_weight).squeeze((4, 5)) + self.attention_weights_bias # b*nq head L k 1
        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output, sampling_locations, attention_weights

