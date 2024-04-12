# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from detectron2.modeling import META_ARCH_REGISTRY
class Simplify_S6_ScanFeat(nn.Module):
    def __init__(
        self,
        d_inner,
        d_state=16,
        dt_rank="auto",
        d_conv=1,
        d_conv_bias=True,
        
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        """
        by default, dt_rank = d_inner // 16
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank

        self.d_conv = d_conv
        if self.d_conv > 1:
            self.act = nn.SiLU()
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

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
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

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True
        

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
        return selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

import warnings
import math
from torch.nn.init import xavier_uniform_, constant_
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

@META_ARCH_REGISTRY.register()
class MSDeform_OnScan_v1(nn.Module):
    def __init__(self, 
                 scan_configs,
                 ):
        super().__init__()
        d_model = scan_configs['d_model']
        n_levels = scan_configs['n_levels']
        n_heads = scan_configs['n_heads']
        n_points = scan_configs['enc_n_points']
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
            
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points 

        # b nq c -> b nq head*L*K*2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        value_d_conv = scan_configs['value_d_conv'] if 'value_d_conv' in scan_configs else 3
        self.value_d_conv = value_d_conv
        
        # b hw_sigma c -> b hw_sigma head head_dim
        if value_d_conv == 1:
            self.value_proj = nn.Linear(d_model, d_model)
        else:
            self.value_proj = nn.Conv2d(
                in_channels=self.d_model,
                out_channels=self.d_model, # head * head_dim
                bias=True,
                kernel_size=value_d_conv,
                groups=self.d_model,
                padding= (value_d_conv - 1) // 2,
            )
        
        # b Nq c -> b Nq head L k 2 -> b Nq head*L*(K+1) c
        self.head_dim = d_model // self.n_heads
        self.s6_scan = Simplify_S6_ScanFeat(d_inner = self.head_dim,
                                            d_state = scan_configs['d_state'],
                                            dt_rank= scan_configs['dt_rank'],
                                            d_conv= scan_configs['d_conv'],
                                            d_conv_bias = scan_configs['d_conv_bias'],)
        # b Nq head L 1 c -> b Nq head c -> b Nq d_model
        self.output_proj = nn.Linear(d_model, d_model) # head * head_dim -> d_model
        self._reset_parameters()
        
        
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, 
                query, # b Nq c
                reference_points, # # b Nq L 2;
                input_flatten, # b hw_sigma c
                input_spatial_shapes, # # L, 2; 
                input_level_start_index, # # L, int
                input_padding_mask=None, 
                ):
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
        B, Nq, Dim = query.shape 
        B, HW_sigma, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == HW_sigma

        # b Nq d_model -> b Nq head L K 2
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)
        # b Nq head L k 2
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # b Nq head L k+1 2
        sampling_locations = torch.cat([sampling_locations, 
                                        reference_points[:, :, None, :, None, :].repeat(1, 1, self.n_heads, 1, 1, 1)], dim=-2)
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        
        if self.value_d_conv == 1:
            input_flatten = self.value_proj(input_flatten)
            
        # b hw_sigma d_model -> list[b hw d_model], L
        value_list = input_flatten.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b d_model hw_sigma
            value_l_ = value_list[lid_].permute(0, 2, 1).contiguous().view(B, -1, H_, W_)
            if self.value_d_conv > 1:
                value_l_ = self.value_proj(value_l_)
            # b d_model H W -> b*head head_dim H W
            value_l_ = value_l_.view(B*self.n_heads, -1, H_, W_).contiguous() 
            # b Nq head k+1 2 -> b*head Nq k+1 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1).contiguous()
            # b*head d_inner Nq k+1
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                                mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # b*head d_inner Nq L k+1 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b head) d_inn Nq L k -> (b Nq) d_inn (head L k)',b=B, head=self.n_heads,
                                                                                            d_inn=self.head_dim, L=self.n_levels, k=self.n_points+1) # b scan d l
        
        # b*nq d_inn head*L(k+1)
        ss_input = self.s6_scan(ss_input)
        ss_input = ss_input.view(B, Nq, -1, self.n_heads, self.n_levels, self.n_points + 1).contiguous()
        # b nq d_inn head L -> b nq head d_inn
        ss_input = ss_input[..., -1].contiguous().sum(4) 
        ss_input = ss_input.view(B, Nq, -1).contiguous() # b Nq scan*c
        ss_input = self.output_proj(ss_input)
        return ss_input, sampling_locations, None


