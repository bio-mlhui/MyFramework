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

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 512

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

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
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
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

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
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




class MSDeformAttn_with_GlobalRegisters(nn.Module):
    def __init__(self, 
                 d_model=256,
                 task_to_num_regs=None,
                 num_feature_levels=None,
                 deform_configs = None,):
        n_levels = num_feature_levels
        headdim_first = deform_configs['headdim_first']
        n_points = deform_configs['enc_n_points']
        local_global_lambda = deform_configs['local_global_lambda']
        if headdim_first:
            headdim = deform_configs['head_dim'] 
            assert d_model % headdim == 0          
            n_heads = d_model // headdim
        else:
            n_heads = deform_configs['n_heads']
            assert d_model % n_heads == 0
            headdim = d_model // n_heads
        if not _is_power_of_2(headdim):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                        "which is more efficient in our CUDA implementation.")         
        super().__init__()
        self.im2col_step = 512
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points

        # key = hw的情况
        # hw_reg c -> hw_reg head L 4 2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # hw_reg c -> hw_reg head L 4
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # key = register的情况
        self.task_to_num_regs = task_to_num_regs
        # hw_reg c -> hw_reg head L REGs
        self.cls_weights = nn.Parameter(torch.zeros([d_model, n_heads, n_levels, self.task_to_num_regs['cls']]))
        self.cls_bias = nn.Parameter(torch.zeros([n_heads, n_levels, self.task_to_num_regs['cls']]))
        self.ssl_weights = nn.Parameter(torch.zeros([d_model, n_heads, n_levels, self.task_to_num_regs['ssl']]))
        self.ssl_bias = nn.Parameter(torch.zeros([n_heads, n_levels, self.task_to_num_regs['ssl']]))
        self.sem_seg_weights = nn.Parameter(torch.zeros([d_model, n_heads, n_levels, self.task_to_num_regs['sem_seg']]))
        self.sem_seg_bias = nn.Parameter(torch.zeros([n_heads, n_levels, self.task_to_num_regs['sem_seg']]))
        self.ins_det_weights = nn.Parameter(torch.zeros([d_model, n_heads, n_levels, self.task_to_num_regs['ins_det']]))
        self.ins_det_bias = nn.Parameter(torch.zeros([n_heads, n_levels, self.task_to_num_regs['ins_det']]))
        self.local_global_lambda = local_global_lambda
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
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
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        
        constant_(self.cls_weights.data, 0.)
        constant_(self.cls_bias.data, 0.)
        constant_(self.ssl_weights.data, 0.)
        constant_(self.ssl_bias.data, 0.)
        constant_(self.sem_seg_weights.data, 0.)
        constant_(self.sem_seg_bias.data, 0.)
        constant_(self.ins_det_weights.data, 0.)
        constant_(self.ins_det_bias.data, 0.)
        
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

        # reg_hw c -> reg_hw head dim
        value = self.value_proj(input_flatten) 
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        split_lens = list(zip([self.num_registers] * len(input_spatial_shapes), 
                              [(haosen[0] * haosen[1]) for haosen in input_spatial_shapes]))
        assert sum(split_lens) == Len_q
        reg_hw_scale_values = value.split(split_lens)
        reg_values, hw_values = torch.cat(reg_hw_scale_values[0::2], dim=1), torch.cat(reg_hw_scale_values[1::2], dim=1)
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == hw_values.shape[1]

        # key == hw, local self-awareness
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # h, w的max值
            # b reg_hw_sigma L 2 -> b reg_hw_sigma 1 L 1 2 + b reg_hw_sigma head L N 2 / 
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :] # offset/max_h  + 相对坐标, 那么offset输出的是绝对坐标
        # elif reference_points.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #                          + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b reg_hw_sigma c
        output = MSDeformAttnFunction.apply(
            hw_values, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        # key == reg, global self-awareness
        # b reg_hw_sigma head L*REG
        register_weights = self.register_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.num_registers)
        register_weights = F.softmax(register_weights, -1)
        # b reg_hw_sigma head dim
        output_reg_value = torch.einsum('bshi,bihc->bshc', register_weights, reg_values)
        output_reg_value = output_reg_value.flatten(2).contiguous()


        # sum
        output = self.local_global_lambda[0] * output + self.local_global_lambda[1] * output_reg_value

        output = self.output_proj(output)

        return output, sampling_locations, attention_weights
