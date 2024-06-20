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

class MSDeformAttn_with_GlobalRegisters_v0(nn.Module):
    def __init__(self, 
                 d_model=256,
                 task_to_num_regs=None,
                 num_feature_levels=None,
                 deform_configs = None,):
        n_levels = num_feature_levels
        headdim_first = deform_configs['headdim_first']
        n_points = deform_configs['enc_n_points']
        local_global_lambda = deform_configs['local_global_lambda']
        task_to_num_regs = deform_configs['task_to_num_regs']
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
        self.task_to_num_scales = task_to_num_scales
        
        # ssl_num_scales
        self.cls_ssl_num_scales = deform_configs['cls_ssl_num_scales']
        self.scale_linear_cls = nn.Linear(d_model, n_heads*self.task_to_num_regs['cls']*self.cls_ssl_num_scales)
        self.scale_linear_ssl = nn.Linear(d_model, n_heads*self.task_to_num_regs['ssl']*self.cls_ssl_num_scales)
        # task
        self.scale_linear_sem_seg = nn.Linear(d_model, n_heads*self.task_to_num_regs['sem_seg']*n_levels)
        self.scale_linear_ins_det = nn.Linear(d_model, n_heads*self.task_to_num_regs['ins_det']*n_levels)
        
        self.task_to_linear = {'cls': self.scale_linear_cls, 'ssl': self.scale_linear_ssl, 'sem_seg': self.scale_linear_sem_seg, 'ins_det': self.scale_linear_ins_det}
        
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
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
                
        constant_(self.scale_linear_cls.weight, 0.)
        constant_(self.scale_linear_cls.bias, 0.)
        constant_(self.scale_linear_ssl.weight, 0.)
        constant_(self.scale_linear_ssl.bias, 0.)
        constant_(self.scale_linear_sem_seg.weight, 0.)
        constant_(self.scale_linear_sem_seg.bias, 0.)
        constant_(self.scale_linear_ins_det.weight, 0.)
        constant_(self.scale_linear_ins_det.bias, 0.) 
               
    def forward(self, query, key_splits, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
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
        
        value = self.value_proj(input_flatten) 
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # reg_key, hw_key
        assert key_splits[-1][0] == 'hw'
        reg_values, hw_values = value.split([value.shape[1] - key_splits[-1][-1], key_splits[-1][-1]], dim=1)

        # key == hw, local self-awareness
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # h, w的max值, L 2
            # b Nq L 2 -> b Nq 1 L 1 2 + b Nq head L N 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :] # offset/max_h  + 相对坐标, 那么offset输出的是绝对坐标
        # elif reference_points.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #                          + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b Nq c
        hw_key_info = MSDeformAttnFunction.apply(hw_values, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        # key == reg, global self-awareness
        task_weight_liear, task_weight_bias = self.get_task_linear_proj(key_splits) # head reg c, head reg
        reg_key_weights = torch.einsum('bNc,hRc->bNhR', query, task_weight_liear) + task_weight_bias # b N h REG
        reg_key_weights = F.softmax(reg_key_weights, -1)
        # b reg_hw_sigma head dim
        reg_key_info = torch.einsum('bNhR,bRhc -> bNhc', reg_key_weights, reg_values)
        reg_key_info = reg_key_info.flatten(2) # b Nq c

        # sum
        output = self.local_global_lambda[0] * hw_key_info + self.local_global_lambda[1] * reg_key_info
        output = self.output_proj(output)

        return output

    def get_task_linear_proj(self,key_splits):
        # ('name', scale * SIN_REG)
        linear_weights = []
        linear_bias = []
        for task_name, length in key_splits[:-1]: # no hw
            wei, bias = self.task_to_linear[task_name].weight, self.task_to_linear[task_name].bias # head * scale * SIN_REG, in
            assert (wei.shape[0]  == length * self.n_heads) and ( bias.shape[0] == self.n_heads *length)
            wei = rearrange(wei, '(head scale REG) c -> head (scale REG) c', head=self.n_heads, REG=self.task_to_num_regs[task_name])
            bias = rearrange(bias, '(head scale REG) -> head (scale REG)', head=self.n_heads, REG=self.task_to_num_regs[task_name])
            linear_weights.append(wei)
            linear_bias.append(bias)
        linear_weights = torch.cat(linear_weights, dim=1) # head reg_sigma c 
        linear_bias = torch.cat(linear_bias, dim=1) # head reg_siga
        return linear_weights, linear_bias


from einops import rearrange

class MSDeformAttn_with_GlobalRegisters(nn.Module):
    def __init__(self, 
                 d_model=256,
                 num_feature_levels=None,
                 deform_configs = None,):
        n_levels = num_feature_levels
        headdim_first = deform_configs['headdim_first']
        n_points = deform_configs['enc_n_points']
        hw_reg_lambda = deform_configs['hw_reg_lambda']
        reg_sequence_length = deform_configs['reg_sequence_length']
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
        self.reg_sequence_length = reg_sequence_length
        self.reg_linear_weights = nn.Linear(d_model, n_heads * reg_sequence_length)
        
        self.hw_reg_lambda = hw_reg_lambda
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
                
        constant_(self.reg_linear_weights.weight, 0.)
        constant_(self.reg_linear_weights.bias, 0.)
               
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, hw_sigma=None):
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
        
        value = self.value_proj(input_flatten) 
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads).contiguous()

        reg_values, hw_values = value.split([value.shape[1]-hw_sigma, hw_sigma], dim=1)

        # key == hw, local self-awareness
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            # L 2
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # h, w的max值, L 2
            # b Nq L 2 -> b Nq 1 L 1 2 + b Nq head L N 2 / 1 1 1 L 1 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :] # offset/max_h  + 相对坐标, 那么offset输出的是绝对坐标
        # elif reference_points.shape[-1] == 4:
        #     sampling_locations = reference_points[:, :, None, :, None, :2] \
        #                          + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # b Nq c
        hw_key_info = MSDeformAttnFunction.apply(hw_values, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        # # key == reg, global self-awareness
        # reg_key_weights = self.reg_linear_weights(query).view(N, Len_q, self.n_heads, 1, -1) # b nq c -> b Nq head 1 reg
        # assert reg_key_weights.shape[-1] == reg_values.shape[1]
        # reg_key_weights = F.softmax(reg_key_weights, -1)
        
        # reg_values = reg_values.permute(0, 2, 1, 3).unsqueeze(1) # b 1 head reg c
        # reg_key_info = reg_key_weights @ reg_values # b nq head 1 reg @ b 1 head reg c
        # # reg_key_info = torch.einsum('bNhR,bRhc -> bNhc', reg_key_weights, reg_values)
        # reg_key_info = reg_key_info.flatten(2) # b nq head 1 c -> b nq head*c


        # key == reg, global self-awareness
        reg_key_weights = self.reg_linear_weights(query).view(N, Len_q, self.n_heads, -1).softmax(-1) # b nq c -> b Nq head reg
        assert reg_key_weights.shape[-1] == reg_values.shape[1]
        reg_key_info = torch.einsum('bNhR,bRhc -> bNhc', reg_key_weights, reg_values).flatten(2) 
        # sum
        output = self.hw_reg_lambda[0] * hw_key_info + self.hw_reg_lambda[1] * reg_key_info

        # sum
        output = self.hw_reg_lambda[0] * hw_key_info + self.hw_reg_lambda[1] * reg_key_info
        output = self.output_proj(output)

        return output

