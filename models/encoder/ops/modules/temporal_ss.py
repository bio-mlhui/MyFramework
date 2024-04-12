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
from torch.nn.init import xavier_uniform_, constant_

from mamba_ssm import Mamba
from einops import rearrange, reduce, repeat

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.modeling import META_ARCH_REGISTRY

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from models.layers.position_encoding import build_position_encoding
class SS1D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=2, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True) # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True) # (K=2, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, T = x.shape
        L = T
        K = 2

        # b c T -> b c T, b c T' -> b 2 c T
        xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1) # b 2 c T

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0].contiguous(), torch.flip(out_y[:, 1], dims=[-1]).contiguous() # b d t


    def forward(self, x: torch.Tensor, **kwargs): # bhw t c
        B, T, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # b t d

        x = x.permute(0, 2, 1).contiguous() # b d t
        x = self.act(self.conv1d(x)) # b d t
        y1, y2 = self.forward_core(x) # B d t
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = y.permute(0, 2, 1).contiguous() # b t d
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

@META_ARCH_REGISTRY.register()
class SS1D_Temporal_Multiscale(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        d_model = configs['d_model']
        self.homo = SS1D(d_model=configs['d_model'],
                        d_state=configs['d_state'] if 'd_state' in configs else 16,
                        d_conv=configs['d_conv'] if 'd_conv' in configs else 3,
                        expand=configs['expand'] if 'expand' in configs else 2,
                        dt_rank=configs['dt_rank'] if 'dt_rank' in configs else 'auto',
                        dt_min=configs['dt_min'] if 'dt_min' in configs else 0.001,
                        dt_max=configs['dt_max'] if 'dt_max' in configs else 0.1,
                        dt_init=configs['dt_init'] if 'dt_init' in configs else 'random',
                        dt_scale=configs['dt_scale'] if 'dt_scale' in configs else 1.0,
                        dt_init_floor=configs['dt_init_floor'] if 'dt_init_floor' in configs else 1e-4,
                        dropout=configs['dropout'] if 'dropout' in configs else 0,
                        conv_bias=configs['conv_bias'] if 'conv_bias' in configs else True,
                        bias=configs['bias'] if 'bias' in configs else False,
                        )
        
        # self.input_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.pos_1d = build_position_encoding(position_embedding_name='1d')

        # xavier_uniform_(self.input_proj.weight.data)
        # constant_(self.input_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

    def forward(self, 
                query=None,  # bt hw_sigma c
                video_aux_dict=None,
                batch_size=None,
                **kwargs
                ):
        if video_aux_dict is not None:
            nf = video_aux_dict['nf']
            batch_size = query.shape[0] // video_aux_dict['nf']
        else:
            assert batch_size is not None
            nf = query.shape[0] // batch_size
        # query = self.input_proj(query) # b t hw_sigma c
        query = rearrange(query, '(b t) hw c -> (b hw) t c',t=nf, b=batch_size)
        poses = self.pos_1d(mask=torch.zeros_like(query[..., 0]).bool(), hidden_dim=query.shape[-1]).permute(0, 2, 1).contiguous()
        query = query + poses
        query = self.homo(query) # b t c
        query = rearrange(query, '(b hw) t c -> (b t) hw c',b=batch_size)
        return query, None, None


@META_ARCH_REGISTRY.register()
class SelfAttn_Temporal_Multiscale(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        d_model = configs['d_model']
        self.homo = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=8,
                                          dropout=0.)
        
        # self.input_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.pos_1d = build_position_encoding(position_embedding_name='1d')

        # xavier_uniform_(self.input_proj.weight.data)
        # constant_(self.input_proj.bias.data, 0.)output_proj
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self..bias.data, 0.)

    def forward(self, 
                query=None,  # bt hw_sigma c
                video_aux_dict=None,
                **kwargs
                ):
        nf = video_aux_dict['nf']
        batch_size = query.shape[0] // video_aux_dict['nf']
        # query = self.input_proj(query) # b t hw_sigma c
        query = rearrange(query, '(b t) hw c -> (b hw) t c',t=nf, b=batch_size)
        poses = self.pos_1d(mask=torch.zeros_like(query[..., 0]).bool(), hidden_dim=query.shape[-1]).permute(0, 2, 1).contiguous()
        query = query + poses
        query = self.homo(query) # b t c
        query = rearrange(query, '(b hw) t c -> (b t) hw c',b=batch_size)
        return query, None, None


# masked mamba
@META_ARCH_REGISTRY.register()
class SS1D_Temporal_MaskedObjSequence(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        d_model = configs['d_model']
        self.homo = SS1D(d_model=configs['d_model'],
                        d_state=configs['d_state'] if 'd_state' in configs else 16,
                        d_conv=configs['d_conv'] if 'd_conv' in configs else 3,
                        expand=configs['expand'] if 'expand' in configs else 2,
                        dt_rank=configs['dt_rank'] if 'dt_rank' in configs else 'auto',
                        dt_min=configs['dt_min'] if 'dt_min' in configs else 0.001,
                        dt_max=configs['dt_max'] if 'dt_max' in configs else 0.1,
                        dt_init=configs['dt_init'] if 'dt_init' in configs else 'random',
                        dt_scale=configs['dt_scale'] if 'dt_scale' in configs else 1.0,
                        dt_init_floor=configs['dt_init_floor'] if 'dt_init_floor' in configs else 1e-4,
                        dropout=configs['dropout'] if 'dropout' in configs else 0,
                        conv_bias=configs['conv_bias'] if 'conv_bias' in configs else True,
                        bias=configs['bias'] if 'bias' in configs else False,
                        )
        
        self.pos_1d = build_position_encoding(position_embedding_name='1d')

        self.version = configs['version']
        self.k_points = configs['k_points']

    @torch.no_grad()
    def get_temporal_mask(self, 
                          mask_features=None,  # b c t h w
                          temporal_query_feats=None,  # b nq c
                          spatial_shapes=None,
                          level_start_idx=None,
                          query_norm=None,
                          mask_mlp=None ): 
        # b c t h w, b nq c
        temporal_query_feats = query_norm[0](temporal_query_feats) # b nq c
        mask_embeds = mask_mlp[0](temporal_query_feats)  # b n c
        mask_logits = torch.einsum("bqc,bcthw->bqthw", mask_embeds, mask_features) 
        batch_size, nq, nf, H, W = mask_logits.shape
            
        if self.version == 1:
            # b n t h w
            mask_logits = mask_logits.flatten(3) # b n t hw
            temporal_mask = torch.zeros_like(mask_logits)
            topk, indices = torch.topk(mask_logits, self.k_points, dim =-1) # value, idx; b n t k
            y_coords = indices // W # b n t k
            x_coords = indices % W # b n t k
            abs_coords = torch.stack([y_coords, x_coords], dim=4) # b n t k 2
            rel_coords = abs_coords.float() / torch.tensor([H, W], device=mask_logits.device).float() # b n t k 2
            # neural tangent kernel
            scale_seq_coordinates = []
            for haosen in spatial_shapes:
                haosen_h, haosen_w = haosen
                scale_abs_coords = (rel_coords *haosen).long() # b n t k 2
                scale_abs_coords_seq = scale_abs_coords[..., 0] * haosen_w + scale_abs_coords[..., 1] # b n t k
                scale_seq_coordinates.append(scale_abs_coords_seq)
            return scale_seq_coordinates    

        elif self.version == 2:
            # b n t h w
            mask_logits = mask_logits.flatten(3) # b n t hw
            topk, indices = torch.topk(mask_logits, self.k_points, dim =-1) # value, idx; b n t k
            y_coords = indices // W # b n t k
            x_coords = indices % W # b n t k
            abs_coords = torch.stack([y_coords, x_coords], dim=4) # b n t k 2
            rel_coords = abs_coords.float() / torch.tensor([H, W], device=mask_logits.device).float() # b n t k 2
            # neural tangent kernel
            scale_seq_coordinates = []
            for haosen, start_idx in zip(spatial_shapes, level_start_idx):
                haosen_h, haosen_w = haosen
                scale_abs_coords = (rel_coords *haosen).long() # b n t k 2
                scale_abs_coords_seq = scale_abs_coords[..., 0] * haosen_w + scale_abs_coords[..., 1] # b n t k
                scale_seq_coordinates.append(scale_abs_coords_seq + start_idx)
            scale_seq_coordinates = torch.stack(scale_seq_coordinates, dim=3).flatten(3) # b n t L k -> b n t Lk
            return scale_seq_coordinates 

        else:
            raise ValueError() 
        # if self.version == 0:
        #     mask_logits = mask_logits.sigmoid() > 0.5 
        #     mask_logits = mask_logits.flatten(0, 1) # bn t h w

        #     temporal_mask = []
        #     for haosen in spatial_shapes:
        #         mask_logits = F.interpolate(mask_logits, size=haosen, mode='nearest').flatten(2) #bn t hw
        #         temporal_mask.append(mask_logits)
        #     temporal_mask = torch.cat(temporal_mask, dim=-1) # bn t hw_sigma
        
        #     temporal_mask = rearrange(temporal_mask, '(b n) t s -> b n t s',b=batch_size, n=nq)

        #     temporal_mask[torch.where(temporal_mask.sum(-1) == temporal_mask.shape[-1])] = False 
        #     return temporal_mask

        # elif self.version == 2:
        #     k=4
        #     # b n t h w
        #     mask_logits = mask_logits.flatten(3) # b n t hw
        #     temporal_mask = torch.zeros_like(mask_logits)
        #     topk, indices = torch.topk(temporal_mask, k, dim =-1) # value, idx; b n t k
        #     y_coords = indices // W # b n t k
        #     x_coords = indices % W # b n t k

        #     abs_coords = torch.stack([y_coords, x_coords], dim=3) # b n t k 2
        #     rel_coords = abs_coords.float() / torch.tensor([H, W], device=mask_logits.device).float() # b n t k 2
        #     return rel_coords 

    def forward(self, 
                query=None,  # bt hw_sigma c
                spatial_shapes=None,
                level_start_index=None,
                mask_features=None, # b c t h w
                temporal_query_feats=None, # b nq c
                query_norm = None,
                mask_mlp = None,
                **kwargs):
        
        hw_sigma = query.shape[1]
        batch_size, _, nf, *_ = mask_features.shape
        nq = temporal_query_feats.shape[1]
        query = rearrange(query, '(b t) hw c -> (b hw) t c',t=nf, b=batch_size).contiguous()
        poses = self.pos_1d(mask=torch.zeros_like(query[..., 0]).bool(), hidden_dim=query.shape[-1]).permute(0, 2, 1).contiguous()
        query = query + poses # (b hw) t c

        temporal_mask = self.get_temporal_mask(mask_features=mask_features, 
                                               temporal_query_feats=temporal_query_feats,
                                               spatial_shapes=spatial_shapes,
                                               query_norm=query_norm,
                                               mask_mlp=mask_mlp,
                                               level_start_idx=level_start_index)  

        if self.version == 1:
            # list[b n t k], 32, 16, 8, hw的index
            inputs = []
            query = rearrange(query, '(b hw) t c -> b t hw c',b=batch_size, hw=hw_sigma)
            for idx_mask, haosen, start_idx in zip(temporal_mask, spatial_shapes, level_start_index):
                cardib = query[:, :, start_idx:(start_idx + haosen[0] * haosen[1])].contiguous() # b t hw c
                cardib = repeat(cardib, 'b t hw c -> b n t hw c', n=nq).contiguous()
                idx_mask = repeat(idx_mask, 'b n t k -> b n t k c', c=cardib.shape[-1])
                sampled_points = torch.gather(cardib, dim=3, index=idx_mask).contiguous() # b n t k c
                inputs.append(sampled_points)
            
            inputs = torch.stack(inputs, dim=0) # L b n t k c
            L, _, nq, _, kk, _ = inputs.shape
            inputs = inputs.flatten(0, 2).flatten(1, 2).contiguous() #  Lbn tk c
            inputs = self.homo(inputs)
            inputs = rearrange(inputs, '(L b n) (t k) c -> b n t (L k) c',L=L,b=batch_size,n=nq,t=nf,k=kk)
            # list[b n t k] -> b n t (L k), hw_sigma的index
            temporal_mask = [tm + start_idx for tm, start_idx in zip(temporal_mask, level_start_index)]
            temporal_mask = torch.stack(temporal_mask, 3).flatten(3).unsqueeze(-1).repeat(1,1,1,1,inputs.shape[-1]) # b n t Lk c

            residual = torch.zeros_like(query)[:, None].repeat(1, nq, 1, 1, 1) # b n t hw_sigma c
            # The backward pass is implemented only for src.shape == index.shape
            assert inputs.shape == temporal_mask.shape
            residual.scatter_add_(dim=3, index=temporal_mask, src=inputs)
            residual = residual.sum(1) # b t hw_sigma c
            query = query + residual
            query = query.flatten(0, 1).contiguous()
            return query, None, None
        
        elif self.version == 2:
            # b n t LK
            inputs = []
            query = rearrange(query, '(b hw) t c -> b t hw c',b=batch_size, hw=hw_sigma)
            repeat_query = repeat(query, 'b t hw c -> b n t hw c',n=nq).contiguous()
            temporal_mask = temporal_mask.unsqueeze(-1).repeat(1,1,1,1,query.shape[-1]) # b n t Lk c
            sampled_points = torch.gather(repeat_query, dim=3, index=temporal_mask).contiguous() # b n t k c
            L = len(spatial_shapes)
            kk = sampled_points.shape[3] // L
            sampled_points = rearrange(sampled_points, 'b n t (L K) c -> (L b n) (t K) c',L=len(spatial_shapes)).contiguous()
            sampled_points = self.homo(sampled_points)
            sampled_points = rearrange(sampled_points, '(L b n) (t k) c -> b n t (L k) c',L=L,b=batch_size,n=nq,t=nf,k=kk)

            residual = torch.zeros_like(repeat_query) # b n t hw_sigma c
            # The backward pass is implemented only for src.shape == index.shape
            assert sampled_points.shape == temporal_mask.shape
            residual.scatter_add_(dim=3, index=temporal_mask, src=sampled_points)
            residual = residual.sum(1) # b t hw_sigma c
            query = query + residual
            query = query.flatten(0, 1).contiguous() # bt hw_siamga c
            return query, None, None
        else:
            raise ValueError()
        
        # scatter, where, gather, index_fill

        # elif self.version == 0:  
 
        #     query = repeat(query, '(b hw) t c -> (b nf) (t hw) c', nf=nf).contiguous()
        #     temporal_mask = temporal_mask.flatten(0, 1).flatten(1) # bn thw

        #     max_len = max([haosen.int().sum() for haosen in temporal_mask])
        #     # list[s c], bn
        #     inputs = [F.pad(query[idx, haosen], pad=[0, 0, 0, max_len-len(haosen)], value=0.) for idx, haosen in enumerate(temporal_mask)]
        #     query = self.homo(query) # b t c
        #     query = rearrange(query, '(b hw) t c -> (b t) hw c',b=batch_size)
        #     return query, None, None




        # elif version == 2:
        #     temporal_mask = rearrange(temporal_mask, 'b n t k 2 -> (b n t) k 2')
        #     inputs = []
        #     query = rearrange(query, '(b hw) t c -> b t hw c',b=batch_size, hw=hw_sigma)
        #     for haosen, haosen_2 in zip(spatial_shapes, level_start_index):
        #         cardib = query[:, :, haosen_2:(haosen_2 + haosen[0] * haosen[1])].contiguous() # b t hw c
        #         cardib = repeat(cardib, 'b t (h w) c -> (b n t) c h w',h=haosen[0], w=haosen[1], n=nf).contiguous()

        #         # n c h w, n p 2 -> n c p
        #         sampled_points = point_sample(cardib, point_coords=temporal_mask).contiguous() # bnt c k
        #         inputs.append(sampled_points)
        #     for shape in spatial_shapes:

