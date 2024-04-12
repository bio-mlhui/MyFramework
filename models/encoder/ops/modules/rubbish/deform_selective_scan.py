
from models.layers.position_encoding import build_position_encoding
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from einops import rearrange


import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from detectron2.utils import comm
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# triton cross scan, 2x speed than pytorch implementation =========================
# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb; pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops



# b h w c, b h w -> Sigma K c -> Sigma K c
# K 没有限制, 每个token的deformable offsets
class MSDeformSS_v1(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs # scan 2
        self.n_points = n_points # K 8
        # scan L K 2
        self.sampling_offsets = nn.Linear(d_model, n_scan_dircs * n_levels * n_points * 2)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, n_scan_dircs * d_model // n_scan_dircs)
        self.output_proj = nn.Linear(d_model, d_model)
        d_inner = d_model // n_scan_dircs
        d_conv = scan_configs['d_conv']

        if d_conv > 1:
            factory_kwargs = {"device": None, "dtype": None}
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=True,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.d_conv = d_conv
        self.build_scan(d_scan=d_model // n_scan_dircs,
                        **scan_configs)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_scan_dircs, dtype=torch.float32) * (2.0 * math.pi / self.n_scan_dircs)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_scan_dircs, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
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

        # b hw_sigma c
        value = self.value_proj(input_flatten) 
        # b hw_sigma scan c
        value = value.view(B, HW_sigma, self.n_scan_dircs, self.d_model // self.n_scan_dircs)
        # b Nq c -> b Nq scan L K 2
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_scan_dircs, self.n_levels, self.n_points, 2)
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # b Nq 1 L 1 2 + b Nq scan L K 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            if self.d_conv > 1:
                value_l_ = self.conv2d(value_l_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # b*scan c Nq L K -> b*scan 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b scan) c Nq L k -> (b Nq) scan c (L k)',b=B, scan=self.n_scan_dircs) # b scan d l

        # b Nq c -> b*Nq scan c
        summarize_token = self.query_proj(query).view(B * Nq, self.n_scan_dircs, -1)

        ss_input = torch.cat([ss_input, summarize_token.unsqueeze(-1)], dim=-1) # b_Nq scan c LK+1

        ss_input = self.forward_ss4(ss_input) 
        summarize_token = ss_input[:, :, :, -1].contiguous() # b_Nq scan c
        summarize_token = summarize_token.view(B, Nq, -1) # b Nq scan*c
        output = self.output_proj(summarize_token)
        return output, sampling_locations, None

    # b k c l的输入
    def build_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        
        self.out_norm_type = 'v0'
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v3=partial(self.forward_ss4_v1, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_ss4 = FORWARD_TYPES.get(forward_type, None)
        k_group = self.n_scan_dircs

        self.act: nn.Module = act_layer()
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)


    def forward_ss4_v1(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead

    # b hw_sigma head c
    # list[h w]
    
    N_, S_, M_, D_ = value.shape
    # B, Nq, Head, L, K, 2
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # 0-1 -> -1, 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # b hw head c -> b hw head*c -> b head*c hw -> b*head c h w
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        
        # b nq head K 2 -> b head nq k 2 -> b*head nq k 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # b*head c h w; b*head nq k 2 -> b*head nq c k
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # b*head nq c L k -> b*head nq c L*k -> b*head nq c 1 -> b head c Nq
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous() # B NQ HEAD C




class MSDeformSS_v2(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs # scan 2
        self.n_points = n_points # K 8
        # scan L K 2
        d_inner = d_model // n_scan_dircs
        self.offset_embeds = nn.Embedding(n_scan_dircs * n_levels * n_points, d_inner)
        self.offset_rnn = Mamba(d_model=d_inner,
                                expand=scan_configs['offset_expand'],
                                d_state=scan_configs['d_state'],
                                d_conv=scan_configs['d_conv'])
        self.sampling_offsets = nn.Linear(d_inner, 2)
        
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, n_scan_dircs * d_model // n_scan_dircs)
        self.output_proj = nn.Linear(d_model, d_model)
        d_inner = d_model // n_scan_dircs
        d_conv = scan_configs['d_conv']

        if d_conv > 1:
            factory_kwargs = {"device": None, "dtype": None}
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=True,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.d_conv = d_conv
        self.build_scan(d_scan=d_model // n_scan_dircs,
                        **scan_configs)

    def _reset_parameters(self):
        constant_(self.offset_embeds.weight.data, 0.)
        # thetas = torch.arange(self.n_scan_dircs, dtype=torch.float32) * (2.0 * math.pi / self.n_scan_dircs)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_scan_dircs, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
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

        # b hw_sigma c
        value = self.value_proj(input_flatten) 
        # b hw_sigma scan c
        value = value.view(B, HW_sigma, self.n_scan_dircs, self.d_model // self.n_scan_dircs)


        # b Nq c -> b*Nq scan c
        summarize_token = self.query_proj(query).view(B * Nq, self.n_scan_dircs, -1)
        offset_embeds = repeat(self.offset_embeds.weight, '(scan L k) d -> (b Nq scan) (L k) d', 
                                b=B, Nq=Nq, scan=self.n_scan_dircs, L=self.n_levels, k=self.n_points)
        # b*nq*scan 1+LK c 
        offset_embeds = torch.cat([summarize_token.flatten(0, 1).unsqueeze(1), offset_embeds], dim=1)
        offset_embeds = self.offset_rnn(offset_embeds) # b L d
        offset_embeds = offset_embeds[:, 1:].contiguous() # b*nq*scan LK c
        sampling_offsets = self.sampling_offsets(offset_embeds).view(B, Nq, self.n_scan_dircs, self.n_levels, self.n_points, 2)
        # b Nq c -> b Nq scan L K 2
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # b Nq 1 L 1 2 + b Nq scan L K 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            if self.d_conv > 1:
                value_l_ = self.conv2d(value_l_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # b*scan c Nq L K -> b*scan 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b scan) c Nq L k -> (b Nq) scan c (L k)',b=B, scan=self.n_scan_dircs) # b scan d l



        ss_input = torch.cat([ss_input, summarize_token.unsqueeze(-1)], dim=-1) # b_Nq scan c LK+1

        ss_input = self.forward_ss4(ss_input) 
        summarize_token = ss_input[:, :, :, -1].contiguous() # b_Nq scan c
        summarize_token = summarize_token.view(B, Nq, -1) # b Nq scan*c
        output = self.output_proj(summarize_token)
        return output, sampling_locations, None

    # b k c l的输入
    def build_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        
        self.out_norm_type = 'v0'
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v3=partial(self.forward_ss4_v1, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_ss4 = FORWARD_TYPES.get(forward_type, None)
        k_group = self.n_scan_dircs

        self.act: nn.Module = act_layer()
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)


    def forward_ss4_v1(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D



# 4下, hw,
# rnn offset
# 每个dir一个dim
class MSDeformSS_v4(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 **kwargs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs 
        assert n_scan_dircs == 4
        self.n_points = n_points #        
        # self.query_proj = nn.Linear(d_model, self.dirc_dim) # 每个query 生成rnn的summarize token; 每个路径的summarize token(不)一样
        # self.value_proj = nn.Linear(d_model, self.dirc_dim) # multiscale 生成路径的value值, 每个路径的value(不)一样
        
        self.offset_embeds = nn.Embedding(n_scan_dircs * n_levels * n_points, d_model) # 每个路径有不同的offset query points;
        # self.offset_rnn = Mamba(d_model=d_model, # 不同路径的summarize token解码出不同的offset points
        #                         expand=scan_configs['offset_expand'],
        #                         d_state=scan_configs['d_state'],
        #                         d_conv=scan_configs['d_conv'])
        # self.offset_rnn.out_proj = nn.Linear(self.d_model, 2, bias=bias, **factory_kwargs)
        d_conv = scan_configs['d_conv']

        if d_conv > 1:
            factory_kwargs = {"device": None, "dtype": None}
            self.conv2d = nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                groups=d_model,
                bias=True,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.d_conv = d_conv
        self.build_offset_scan(d_scan=d_model,
                               **scan_configs)
        self.build_scan(d_scan=d_model,
                        **scan_configs)

    def _reset_parameters(self):
        constant_(self.offset_embeds.weight.data, 0.)
        # thetas = torch.arange(self.n_scan_dircs, dtype=torch.float32) * (2.0 * math.pi / self.n_scan_dircs)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_scan_dircs, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        # xavier_uniform_(self.query_proj.weight.data)
        # constant_(self.query_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

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

        # b hw_sigma scan c
        value = input_flatten.unsqueeze(2).repeat(1, 1, self.n_scan_dircs, 1)
        # b Nq c -> b Nq scan c -> b*Nq scan c
        summarize_token = query.unsqueeze(2).repeat(1, 1, self.n_scan_dircs, 1).flatten(0, 1)  # b*Nq scan c
        
        # scan L k d -> b*Nq scan L k d
        offset_embeds = repeat(self.offset_embeds.weight, '(scan L k) d -> (b Nq) scan L k d', 
                                b=B, Nq=Nq, scan=self.n_scan_dircs, L=self.n_levels, k=self.n_points)
        # b*nq scan L 1 c ; b*nq scan L k d; 
        offset_embeds = torch.cat([summarize_token.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.n_levels, 1, 1),
                                   offset_embeds], dim=3) 
        # b*nq scan L 1+k d -> b*nq scan L(1+k) d -> b*nq scan d L(1+k)
        offset_embeds = offset_embeds.flatten(2, 3).permute(0, 1, 3, 2).contiguous()
        # b*nq scan d L(1+k) -> b nq scan 2 L 1+k
        offset_embeds = self.forward_offset_scan(offset_embeds).view(B, Nq, self.n_scan_dircs, -1, self.n_levels, 1+self.n_points) # b L (1+k) d
        # b nq scan 2 L k -> b nq scan L k 2
        sampling_offsets = offset_embeds[..., 1:].contiguous().permute(0, 1, 2, 4, 5, 3).contiguous()
        sampling_offsets = sampling_offsets.unbind(2) # list[b nq L k 2]
        # L 2
        offset_normalizer_wh = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # W H
        # b nq L k 2 / 1 1 L 1 2 +  b Nq L 1 2
        sampling_locations_wh = reference_points[:, :, :, None, :] \
                                + sampling_offsets[0] / offset_normalizer_wh[None, None, :, None, :]

        offset_normalizer_iwh = torch.stack([-1 * input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # -W H
        sampling_locations_iwh = reference_points[:, :, :, None, :] \
                                + sampling_offsets[1] / offset_normalizer_iwh[None, None, :, None, :]

        offset_normalizer_wih = torch.stack([input_spatial_shapes[..., 1], -1 * input_spatial_shapes[..., 0]], -1) # W -H
        sampling_locations_wih = reference_points[:, :, :, None, :] \
                                + sampling_offsets[2] / offset_normalizer_wih[None, None, :, None, :]

        offset_normalizer_iwih = torch.stack([-1 * input_spatial_shapes[..., 1], -1 * input_spatial_shapes[..., 0]], -1) # -W -H
        sampling_locations_iwih = reference_points[:, :, :, None, :] \
                                + sampling_offsets[3] / offset_normalizer_iwih[None, None, :, None, :]
        
        # list[b nq L k 2] -> b nq scan l k 2
        sampling_locations = torch.stack([sampling_locations_wh, sampling_locations_iwh, sampling_locations_wih, sampling_locations_iwih],
                                         dim=2) # b Nq scan L k 2
        
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            if self.d_conv > 1:
                value_l_ = self.conv2d(value_l_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # list[b*scan c Nq K] -> b*scan c Nq L k 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b scan) c Nq L k -> (b Nq) scan c L k',b=B, scan=self.n_scan_dircs)
        # b*nq scan c L k
        # b*Nq scan c -> b*nq scan c 1 1 -> b*nq scan c L k+1
        ss_input = torch.cat([ss_input, summarize_token.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,self.n_levels, 1),], dim=-1)
        ss_input = ss_input.flatten(-2).contiguous() # b*nq scan c L(k+1)
        ss_input = self.forward_ss4(ss_input) 
        ss_input = ss_input.view(B, Nq, self.n_scan_dircs,-1, self.n_levels, self.n_points+1)
        output = ss_input[..., -1].contiguous().sum(-1).sum(2) # b nq scan c L -> b nq scan c  -> b nq c
        # output = self.output_proj(summarize_token)
        return output, sampling_locations, None



    def build_offset_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
         
        self.disable_z = True
        self.forward_offset_scan = partial(self.forward_ss4_offset_scan, force_fp32=True, SelectiveScan=SelectiveScanOflex)
        k_group = self.n_scan_dircs
        # x proj ============================
        self.offset_x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.offset_x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.offset_x_proj], dim=0)) # (K, N, inner)
        del self.offset_x_proj

        # dt proj ============================
        self.offset_dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.offset_dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.offset_dt_projs], dim=0)) # (K, inner, rank)
        self.offset_dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.offset_dt_projs], dim=0)) # (K, inner)
        del self.offset_dt_projs
        # A, D =======================================
        self.offset_A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.offset_Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        
        # out_proj
        self.out_proj = nn.Linear(d_scan, 2, bias=True)


    def forward_ss4_offset_scan(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.offset_x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.offset_dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.offset_dt_projs_bias # 
        A_logs = self.offset_A_logs # scan * d_innter state
        Ds = self.offset_Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape # b*nq scan d L(1+k)
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L).permute(0, 1, 3, 2) # b k l d
        ys = self.out_proj(ys) # b k l 2
        ys = ys.permute(0, 1, 3, 2).contiguous() # b k 2 l
        return ys

    # b k c l的输入
    def build_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        
        self.out_norm_type = 'v0'
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v3=partial(self.forward_ss4_v1, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_ss4 = FORWARD_TYPES.get(forward_type, None)
        k_group = self.n_scan_dircs

        self.act: nn.Module = act_layer()
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)


    def forward_ss4_v1(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


# v5
# 只有Offsets是用了Mamba, features还是用attention
class MSDeformSS_v5(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 **kwargs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs
        self.n_points = n_points 
        from models.layers.utils import zero_module
        self.offset_embeds = zero_module(nn.Embedding(n_scan_dircs * n_levels * n_points, d_model)) # 每个路径有不同的offset query points;
        self.offset_scan = Mamba(d_model=d_model, expand=1, use_fast_path=False) # # 多个scan共享
        self.offset_scan.out_proj = nn.Linear(d_model, 3, bias=True)
        # self.build_offset_scan(d_scan=d_model,
        #                        **scan_configs)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()
        
        
    def _reset_parameters(self):
        pass
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

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

        value = input_flatten
        # b hw_sigma scan c
        value = value.unsqueeze(2).repeat(1, 1, self.n_scan_dircs, 1)
        # b Nq c -> b Nq scan c -> b*Nq scan c
        summarize_token = query.unsqueeze(2).repeat(1, 1, self.n_scan_dircs, 1).flatten(0, 1)  # b*Nq scan c
        
        # scan L k d -> b*Nq scan L k d
        offset_embeds = repeat(self.offset_embeds.weight, '(scan L k) d -> (b Nq) scan L k d', 
                                b=B, Nq=Nq, scan=self.n_scan_dircs, L=self.n_levels, k=self.n_points)
        # b*nq scan L 1 c ; b*nq scan L k d; 
        offset_embeds = torch.cat([summarize_token.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.n_levels, 1, 1),
                                   offset_embeds], dim=3) 
        
        # b*nq scan L 1+k d -> b*nq scan L(1+k) d 
        offset_embeds = offset_embeds.flatten(2, 3).contiguous()
        # b*nq scan L(1+k) d -> b*nq*scan L(1+k) d -> b nq scan L (1+k) d
        offset_embeds = self.offset_scan(offset_embeds.flatten(0, 1).contiguous()).view(B, Nq, self.n_scan_dircs, self.n_levels, 1+self.n_points, -1)
        # b nq scan L 1+k 3 -> b nq scan L k 3
        offset_embeds = offset_embeds[:, :, :, :, 1:].contiguous()
        # b nq scan L k 2, b nq scan L k
        sampling_offsets, attention_weights = offset_embeds[..., :2].contiguous(), offset_embeds[..., 2].contiguous()

        # L 2
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # W H
        # b nq scan L k 2 / 1 1 1 L 1 2 +  b nq 1 L 1 2 -> b nq scan L k 2
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
     
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # list[b*scan c Nq k] -> b*scan c Nq L k -> b Nq c scan L k -> b Nq c scan*L*K
        sampling_values = torch.stack(sampling_value_list, dim=-2).view(B, self.n_scan_dircs, -1, Nq, self.n_levels, self.n_points).permute(0, 3, 2, 1, 4, 5).flatten(3)
        # b nq scan l k -> b nq scan*l*k -> b nq 1 scan*l*k -> b nq c
        output = (sampling_values * (attention_weights.flatten(2).unsqueeze(2).softmax(-1))).sum(-1)
        output = output # b nq c
        return output, sampling_locations, None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D



# None
class MSDeformSS_none(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 **kwargs,
                 ):
        super().__init__()


    def _reset_parameters(self):
        pass

    def forward(self, 
                query, # b Nq c
                reference_points, # # b Nq L 2;
                input_flatten, # b hw_sigma c
                input_spatial_shapes, # # L, 2; 
                input_level_start_index, # # L, int
                input_padding_mask=None, 
                ):
        return query, None, None

from mamba_ssm import Mamba

# b h w c, b h w -> Sigma K c -> Sigma K c
# K 没有限制, 每个token的deformable offsets
class MSDeformSS_v3(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs # scan 
        self.n_points = n_points # K 
        # d_model
        d_inner = d_model
        self.query_proj = nn.Linear(d_model, n_scan_dircs * d_inner)
        self.offset_embeds = nn.Embedding(n_scan_dircs * n_levels * n_points, d_inner)
        self.offset_rnn = Mamba(d_model=d_inner,
                                expand=scan_configs['offset_expand'],
                                d_state=scan_configs['d_state'],
                                d_conv=scan_configs['d_conv'])
        self.sampling_offsets = nn.Linear(d_inner, 2)

        self.value_proj = nn.Linear(d_model, n_scan_dircs * d_inner)

        self.output_proj = nn.Linear(n_scan_dircs * d_inner, d_model)
        d_conv = scan_configs['d_conv']

        if d_conv > 1:
            factory_kwargs = {"device": None, "dtype": None}
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=True,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.d_conv = d_conv
        self.build_scan(d_scan=d_inner,
                        **scan_configs)
        self.d_inner = d_inner

    def _reset_parameters(self):
        constant_(self.offset_embeds.weight.data, 0.)
        # constant_(self.offset_rnn)
        # thetas = torch.arange(self.n_scan_dircs, dtype=torch.float32) * (2.0 * math.pi / self.n_scan_dircs)
        # # scan 2
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # # scan L K 2
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_scan_dircs, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
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

        # b hw_sigma c
        value = self.value_proj(input_flatten) 
        # b hw_sigma scan c
        value = value.view(B, HW_sigma, self.n_scan_dircs, self.d_inner)
        # b Nq c -> b*Nq scan c
        summarize_token = self.query_proj(query).view(B * Nq, self.n_scan_dircs, -1)

        offset_embeds = repeat(self.offset_embeds.weight, '(scan L k) d -> (b Nq scan) (L k) d', 
                                b=B, Nq=Nq, scan=self.n_scan_dircs, L=self.n_levels, k=self.n_points)
        # b*nq*scan 1+LK c 
        offset_embeds = torch.cat([summarize_token.flatten(0, 1).unsqueeze(1), offset_embeds], dim=1)
        offset_embeds = self.offset_rnn(offset_embeds) # b L d
        offset_embeds = offset_embeds[:, 1:].contiguous() # b*nq*scan LK c
        sampling_offsets = self.sampling_offsets(offset_embeds).view(B, Nq, self.n_scan_dircs, self.n_levels, self.n_points, 2)
        # b Nq c -> b Nq scan L K 2
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # b Nq 1 L 1 2 + b Nq scan L K 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            if self.d_conv > 1:
                value_l_ = self.conv2d(value_l_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # b*scan c Nq L K -> b*scan 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b scan) c Nq L k -> (b Nq) scan c (L k)',b=B, scan=self.n_scan_dircs) # b scan d l

        ss_input = torch.cat([ss_input, summarize_token.unsqueeze(-1)], dim=-1) # b_Nq scan c LK+1

        ss_input = self.forward_ss4(ss_input) 
        summarize_token = ss_input[:, :, :, -1].contiguous() # b_Nq scan c
        summarize_token = summarize_token.view(B, Nq, -1) # b Nq scan*c
        output = self.output_proj(summarize_token)
        return output, sampling_locations, None

    # b k c l的输入
    def build_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        
        self.out_norm_type = 'v0'
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v3=partial(self.forward_ss4_v1, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_ss4 = FORWARD_TYPES.get(forward_type, None)
        k_group = self.n_scan_dircs

        self.act: nn.Module = act_layer()
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)


    def forward_ss4_v1(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D



class MSDeformSS_vNone(nn.Module):
    def __init__(self, 
                 d_model,
                 n_levels,
                 n_scan_dircs,
                 n_points,
                 scan_configs,
                 ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_scan_dircs = n_scan_dircs # scan 2
        self.n_points = n_points # K 8
        # scan L K 2
        self.offset_embeds = nn.Embedding(n_scan_dircs * n_levels * n_points, d_model)
        self.offset_rnn = Mamba(d_model=d_model)
        
        self.sampling_offsets = nn.Linear(d_model, n_scan_dircs * n_levels * n_points * 2)
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, n_scan_dircs * d_model // n_scan_dircs)
        self.output_proj = nn.Linear(d_model, d_model)
        d_inner = d_model // n_scan_dircs
        d_conv = scan_configs['d_conv']

        if d_conv > 1:
            factory_kwargs = {"device": None, "dtype": None}
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=True,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        self.d_conv = d_conv
        self.build_scan(d_scan=d_model // n_scan_dircs,
                        **scan_configs)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_scan_dircs, dtype=torch.float32) * (2.0 * math.pi / self.n_scan_dircs)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_scan_dircs, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
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

        # b hw_sigma c
        value = self.value_proj(input_flatten) 
        # b hw_sigma scan c
        value = value.view(B, HW_sigma, self.n_scan_dircs, self.d_model // self.n_scan_dircs)
        # b Nq c -> b*Nq scan c
        summarize_token = self.query_proj(query).view(B * Nq, self.n_scan_dircs, -1)

        offset_embeds = repeat(self.offset_embeds.weight, '(scan L k) d -> (b Nq scan) 1 d (L k)', 
                                b=B, Nq=Nq, scan=self.n_scan_dircs, L=self.n_levels, k=self.n_points)
        offset_inputs = torch.cat([summarize_token.flatten(0, 1), offset_embeds])

        
        # b Nq c -> b Nq scan L K 2
        sampling_offsets = self.sampling_offsets(query).view(B, Nq, self.n_scan_dircs, self.n_levels, self.n_points, 2)
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # b Nq 1 L 1 2 + b Nq scan L K 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b hw_sigma scan c -> list[b hw scan c], L
        value_list = value.split([H_ * W_ for H_, W_ in input_spatial_shapes], dim=1)
        # b Nq scan L K 2
        sampling_grids = 2 * sampling_locations - 1 # [0, 1] -> [-1, 1]
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            # b hw scan c -> b hw scan*c -> b scan*c hw -> b*scan c h w
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(B*self.n_scan_dircs, -1, H_, W_)
            if self.d_conv > 1:
                value_l_ = self.conv2d(value_l_)
            # b Nq scan K 2 -> b scan Nq K 2 -> b*scan Nq K 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # b*scan c Nq K
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)  
        
        # b*scan c Nq L K -> b*scan 
        ss_input = torch.stack(sampling_value_list, dim=-2)
        ss_input = rearrange(ss_input, '(b scan) c Nq L k -> (b Nq) scan c (L k)',b=B, scan=self.n_scan_dircs) # b scan d l

        ss_input = torch.cat([ss_input, summarize_token.unsqueeze(-1)], dim=-1) # b_Nq scan c LK+1

        ss_input = self.forward_ss4(ss_input) 
        summarize_token = ss_input[:, :, :, -1].contiguous() # b_Nq scan c
        summarize_token = summarize_token.view(B, Nq, -1) # b Nq scan*c
        output = self.output_proj(summarize_token)
        return output, sampling_locations, None

    # b k c l的输入
    def build_scan(self,
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        
        self.out_norm_type = 'v0'
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v3=partial(self.forward_ss4_v1, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        self.forward_ss4 = FORWARD_TYPES.get(forward_type, None)
        k_group = self.n_scan_dircs

        self.act: nn.Module = act_layer()
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)


    def forward_ss4_v1(self, ss4_input,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys



    def build_offset_ss(self, 
            d_scan=None,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3, # < 2 means no conv 
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v3noz",
            channel_first=False,
            **kwargs,  
        ):
        factory_kwargs = {"device": None, "dtype": None}
        d_inner = d_scan
        dt_rank = math.ceil(d_scan / 16) if dt_rank == "auto" else dt_rank
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.forward_offset_ss = partial(self.forward_offset_ss, force_fp32=True, SelectiveScan=SelectiveScanOflex)
        k_group = 1
        self.act: nn.Module = act_layer()
        # x proj ============================
        self.offset_x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.offset_x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.offset_x_proj], dim=0)) # (K, N, inner)
        del self.offset_x_proj

        # dt proj ============================
        self.offset_dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.offset_dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.offset_dt_projs], dim=0)) # (K, inner, rank)
        self.offset_dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.offset_dt_projs], dim=0)) # (K, inner)
        del self.offset_dt_projs
        # A, D =======================================
        self.offset_A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.offset_Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        pass


    def forward_offset_ss_v1(self, ss4_input, # b_Nq scan L k d_inner ->
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=True, # True: input fp32, deformable 
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
                ): # x的输入维度是d_model // scan_dirs = d_inner
        x_proj_weight = self.x_proj_weight # scan rank+state+state  c
        dt_projs_weight = self.dt_projs_weight  # scan d_innter rank
        dt_projs_bias = self.dt_projs_bias # 
        A_logs = self.A_logs # scan * d_innter state
        Ds = self.Ds

        # b_Nq scan c LK+1
        B, K, _, L = ss4_input.shape
        D, N = A_logs.shape #  K*d_inner state
        K, D, R = dt_projs_weight.shape  # scan  _ rank

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", ss4_input, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)   

        xs = ss4_input.view(B, -1, L) # b_nq scan*d_inner LK+1
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, L)
        return ys


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
