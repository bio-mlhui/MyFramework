# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# 没有noise的，也就是原生mask2former

import logging
import math
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d


from einops import repeat
from util.debug import *
from .encoder_multiscale import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from .encoder_video import VideoSwinTransformer, VideoResNet
# from .backbone import VideoSwinTransformerBackbone, ResNetBackbone
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from .position_encoding import build_position_encoding
from util.misc import NestedTensor
def aligned_bilinear(tensor, factor):
    """
    Input:  
        - tensor: tbn 1 h w
        - factor: 1
    """
    assert tensor.dim() == 4 
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]
def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    """
    Input: 
        - params: 
            T(TBN num_params)
        - channels: dynamic_mask_channels
        - weight_num / bias_num
            list[2048, 64, 8], #dynamic_conv_layers
    OUtput:
        - weight_splits: list[conv1_weight: T(tbn * hidden, num, 1, 1)]
        - bias_split: list[conv1_bias: T(tbn * hidden)]
    """
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    # list[conv1_weight: T(tbn num=in*hidden), conv1_bias: T(tbn hidden) , conv2_weight: T(tbn num=hidden*hidden), conv2_bias: T(tbn num=hidden)]
    # conv3_weight: T(tbn num=hidden*1), conv3_bias: T(tbn num=1)
    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers] # list[conv1_weight: T(tbn num), conv2_weight: T(tbn num)]
    bias_splits = params_splits[num_layers:] # list[conv1_weight: T(tbn num), conv2_weight: T(tbn num)]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1) # tbn*hidden, in/hidden, 1, 1
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels) # tbn*hidden
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1) # tbn * 1, hidden, 1, 1
            bias_splits[l] = bias_splits[l].reshape(num_insts) # tbn * 1

    return weight_splits, bias_splits
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        


        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        

        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     ):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)  # n b d

        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # input: b
        # output: b dim
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        float_attn_mask: bool = False,
        learn_features: bool = False,
        only1: bool = False,
        dynamic_conv: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            float_attn_mask: 使用bool attention mask 还是 float attention mask
            learn_features: 是否除了text sentence再学一个query feat
        """
        super().__init__()
        # # positional encoding
        # N_steps = hidden_dim // 2
        # self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.learn_features = learn_features
        if learn_features:
            # learnable query features
            self.query_feat = zero_module(nn.Embedding(num_queries, hidden_dim))
            
        # learnable query p.e.
        self.query_embed = zero_module(nn.Embedding(num_queries, hidden_dim))

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        # self.num_feature_levels = 2
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        self.box_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.float_attn_mask = float_attn_mask
        self.only1 = only1
        
        self.dynamic_conv = dynamic_conv
        if self.dynamic_conv:
            self.controller_layers = 3
            self.dynamic_mask_channels = 8
            weight_nums, bias_nums = [], [] #conv1_weights_num, conv2_weights_num, conv3_weights3_num
            self.in_channels = mask_dim
            for l in range(self.controller_layers):
                if l == 0:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                    bias_nums.append(self.dynamic_mask_channels)
                elif l == self.controller_layers - 1:
                    weight_nums.append(self.dynamic_mask_channels * 1) # output layer c -> 1
                    bias_nums.append(1)
                else:
                    weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                    bias_nums.append(self.dynamic_mask_channels)
            self.weight_nums = weight_nums
            self.bias_nums = bias_nums
            self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        
            controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3) #生成dynamic convolution的参数，以object queries为输入
            for layer in controller.layers:
                nn.init.zeros_(layer.bias)
                nn.init.xavier_uniform_(layer.weight)
            self.controller = controller
            self.mask_out_stride = 4
            self.mask_feat_stride = 4
        else:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    
    def forward(self,
                conditionals):
        """
        Input:
            - conditionals:
                - multiscales: list[T(tb c hi wi)]  # 32x 16x 8x , 相同维度
                - masks: list[T(tb hi wi)]
                - poses: list[T(tb c hi wi)]
                - stride4: T(tb c hi wi) # 4x
                - text_sentence: T(tb c)
                - nf
                - batch_size
        Output:
            out:
                - pred_logits: predictions of the classes
                    T(b t n classes)
                - pred_masks: predictions of masks
                    T(b t n h w)
                - pred_boxes:
                    T(b t n 4)
                - aux_outputs
        """
        multiscales, masks, multiscale_poses, mask_features, text_sentence,\
            nf, batch_size = conditionals
    
        assert len(multiscales) == self.num_feature_levels # 3
        

        srcs = [] # list[hw_i tb c]
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(multiscales[i].shape[-2:])
            # tb c h w -> tb c hw -> hw tb c
            srcs.append(self.input_proj[i](multiscales[i]).flatten(2) + self.level_embed.weight[i][None,:,None])
            poses.append(multiscale_poses[i].flatten(2).permute(2, 0, 1)) # hw tb c
            srcs[-1] = srcs[-1].permute(2, 0, 1) # hw tb c

            # masks[i] = masks[i].flatten(1) # b (h w)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, nf*batch_size, 1) # n tb c
        
        output = repeat(text_sentence, 'tb c -> n tb c',n=self.num_queries) # n tb c
        if self.learn_features:
            output += self.query_feat.weight.unsqueeze(1).repeat(1, nf*batch_size, 1)
   
        predictions_class = [] # list[T(tb n class, real)], init -> 32x -> 16x -> 8x
        predictions_mask = [] # list[T(tb n H W, real)], 
        predictions_box = []
        # 可以加一个predictions_box

        # 初始的mask preditions, class predictions, attention mask
        # (tb n class, real), (tb n H W, real), (tb n 4, real), (tb*head n hw, 1是不想attend的位置, 0是想attend的位置，和mhsa的接口一致)
        # (tb n class, real), (tb n H W, real), (tb n 4, real), (tb*head n hw, float, 有正负, real)
        outputs_class, outputs_mask, outputs_box, attn_mask = self.forward_prediction_heads(
                                                output, mask_features, attn_mask_target_size=size_list[0])
        outputs_class = rearrange(outputs_class, '(t b) n c -> b t n c',t=nf,b=batch_size) # real
        outputs_mask = rearrange(outputs_mask, '(t b) n h w -> b t n h w',t=nf, b=batch_size) # real
        outputs_box = rearrange(outputs_box, '(t b) n d -> b t n d',t=nf,b=batch_size)  # [0,1]
        
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_box.append(outputs_box)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # 如果这个query的mask预测为全空，即attn_mask都是1，那么放弃加mask?
            # For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            # For a float mask, the mask values will be added to the attention weight.
            if not self.float_attn_mask:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # n tb c
                memory=srcs[level_index], # hw tb c
                memory_mask=attn_mask, # tb*head n hw
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses[level_index],  # hw tb c
                query_pos=query_embed, # n tb c
            )

            output = self.transformer_self_attention_layers[i](
                output, # n tb c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n tb c
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output # n tb c
            )
            
            # (tb n classes, real), (tb n H W, real), (tb*head n hw, 1是不想attend的位置, 0是想attend的位置，和mhsa的接口一致)
                # attention mask 只有predictions
            outputs_class, outputs_mask, outputs_box, attn_mask = self.forward_prediction_heads(
                                                            output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],)

            outputs_class = rearrange(outputs_class, '(t b) n c -> b t n c',t=nf,b=batch_size)
            outputs_mask = rearrange(outputs_mask, '(t b) n h w -> b t n h w',t=nf, b=batch_size)
            outputs_box = rearrange(outputs_box, '(t b) n d -> b t n d',t=nf,b=batch_size)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_box.append(outputs_box)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1], # b t n classes
            'pred_masks': predictions_mask[-1], # b t n H W
            'pred_boxes': predictions_box[-1],
            'aux_outputs': self._set_aux_loss(  # 
                predictions_class, predictions_mask, predictions_box
            ) # list[{'pred_logits':T(b t n H W), 'pred_masks':T(b t n H W)}], 9个，包含最初始的
        }
        return out
    
    def dynamic_mask(self, mask_features, mask_head_params):
        """
        Input:
            - mask_features:
                1 TBNc h w
            - mask_head_params:
                TBN d
        Output:
            - mask logits:
                Tb n h w
        """
        device = mask_features.device

        weights, biases = parse_dynamic_params(params=mask_head_params, channels=self.dynamic_mask_channels, weight_nums=self.weight_nums, bias_nums=self.bias_nums)
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0]) # 1 tbn*1 h w
        mask_logits = rearrange(mask_logits, '1 tbn h w -> tbn 1 h w')
        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = rearrange(mask_logits, '(tb n) 1 h w -> tb n h w', n=self.num_queries)
        return mask_logits
        
    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features: 1 tbn*in_channel h w 
        :param weights: [w0, w1, ...]: conv1: T(tbn * (out), (in), 1, 1)
        :param bias: [b0, b1, ...]: conv1_bais: T(tbn*(out))
        :param num_insts: tbn
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x,  # 1 tbn*in h w
                w,  # tbn*out, in, 1, 1 (out_channels, in_channels/groups, kH, kW)
                bias=b, # tbn*out
                stride=1, padding=0,
                groups=num_insts #tbn
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """
        Input:
            - output: current object queries
                T(n tb c)
            - mask_features:
                T(tb c H W)
            - attn_mask_target_size:
                list[h, w]
            - sampled_time:
                T(tb )
        Output:

        """
        decoder_output = self.decoder_norm(output)  # n tb c
        decoder_output = decoder_output.transpose(0, 1)  # tb n c
        outputs_class = self.class_embed(decoder_output)  # tb n 2
        outputs_box = self.box_embed(decoder_output).sigmoid() # tb n 4
        
        if self.dynamic_conv:
            dynamic_mask_head_params = self.controller(decoder_output) # tb n num_params
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, 'tb n num -> (tb n) num')
            mask_features = repeat(mask_features, 'tb c h w -> 1 (tb n c) h w',n=self.num_queries)
            outputs_mask = self.dynamic_mask(mask_features, dynamic_mask_head_params) # tb n h w
        else:
            mask_embed = self.mask_embed(decoder_output)  # tb n d
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # tb n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # tb n h w, real
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # a True value indicates that the corresponding position is not allowed to attend
        # tb n h w -> tb 1 n hw -> tb head n hw -> tb*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_class, outputs_mask, outputs_box, attn_mask
    
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boxes):
        """
        Input:
            - output_class:
                list[T(tb n classes)]
            - outputs_seg_masks:
                list[T(tb n H W)]
            - outputs_boxes:
                list[T(tb n 4)]
        """
        return [
            {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1],  outputs_boxes[:-1])
        ]


class FpnHead(nn.Module):
    """放大8x特征图, 加上4x特征图, 维度保持不变"""
    def __init__(self, dim ) -> None:
        super().__init__()
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.adapter = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(32, dim)
    
    def forward(self, small_feat, large_feat):
        """
        - small_feat: b c h w ( 8x )
        - large_feat: b c h w ( 4x )
        """
        x = self.interpolate(small_feat)
        x = x + self.adapter(large_feat)
        x = self.conv(x)
        x = self.norm(x)
        return x

class FeatureResize2D(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout=0., do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.conv = nn.Conv2d(input_feat_size, output_feat_size, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(32, output_feat_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, encoder_features):
        x = self.conv(encoder_features)
        if self.do_ln:
            x = self.norm(x)
        x = self.act(x)
        output = self.dropout(x)
        return output  


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


class DeformFusionWrapper(nn.Module):
    def __init__(self, 
                 deform_encoder,
                 num_feature_levels,
                 d_model) -> None:
        super().__init__()
        self.deform_encoder = deform_encoder
        self.level_embed = nn.Embedding(num_feature_levels, d_model)
        self.num_feature_levels = num_feature_levels
        
    def get_valid_ratio(self, mask):
        """
        Input:
            - mask:
                TB h w
        Output:
            - int
        """
        _, H, W = mask.shape
        # T(TB, )
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # T(TB, )
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # T(TB, 2)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def forward(self, srcs, masks, pos_embeds, text_word, text_padding_mask, text_pos,):
        """
        Input:
            - srcs:
                List[T(TB c hi wi)] # 8x 16x 32x 64x
            - masks: masks of srcs
                T(TB hi wi)
            - pos_embeds:
                List[T(TB c hi wi)]
            - text_word:
                T(s b d)
            - text_padding_mask:
                T(b s)
            - text_pos:
                T(s b d)
        """
        src_flatten = []
        mask_flattn = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)): # 8x -> 16x -> 32x -> 64x
            TB, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = rearrange(src, 'TB c h w -> TB (h w) c')
            mask = rearrange(mask, 'TB h w -> TB (h w)')
            pos_embed = rearrange(pos_embed, 'TB c h w -> TB (h w) c')
            lvl_pos_embed = pos_embed + self.level_embed.weight[lvl][None, None, :]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
            src_flatten.append(src)
            mask_flattn.append(mask)
            
        # TB \sigma(hi wi) c
        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flattn, dim=1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        
        # #levels, 2
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # (0, h0*wo, h1*w1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # TB num_levels 2
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # TB (h_sigma, w_sigma) c
        memory = self.deform_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)
                            #   text_word=text_word,
                            #   text_padding_mask=text_padding_mask,
                            #   text_pos=text_pos)
        
        memory_features = []
        spatial_index = 0
        for lvl in range(self.num_feature_levels - 1): # 8x 16x 32x
            h, w = spatial_shapes[lvl]
            
            memory_lvl = memory[:, spatial_index: (spatial_index + h*w), :].contiguous()
            memory_lvl = rearrange(memory_lvl, 'TB (h w) c -> TB c h w',h=h, w=w)
            memory_features.append(memory_lvl)
            spatial_index += h*w
            
        return memory_features  

# with deformable encoder
class Multimodal_encoder_deformable(nn.Module):
    """
    a u-shape encoder, composed of a spatial decoder and a backbone
    """
    def __init__(self,
                 
                 backbone_name,
                 backbone_pretrained, 
                 backbone_pretrained_path, 
                 train_backbone, 
                 running_mode,
                 
                 d_model,
                 num_encoder_layers,
                 nheads,
                 npoints,
                 fuse_text_multiples,
                 
                 text_encoder_type='roberta-base',
                 freeze_text_encoder=True,
                 
                 dilation=None,) -> None:
        super().__init__()
        # 96 80 142
        # 192 40 71
        # 384 20 36
        self.vid_backbone = self.build_backbone(
            backbone_name=backbone_name,
            backbone_pretrained=backbone_pretrained,
            backbone_pretrained_path=backbone_pretrained_path,
            train_backbone=train_backbone,
            running_mode=running_mode,
            dilation=dilation #这个参数的两个backbone不相同
        )
        
        bb_out_channels = self.vid_backbone.layer_output_channels
        input_proj_list = []
        for ch in bb_out_channels[-3:]:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(ch, d_model, kernel_size=1, bias=True),
                    nn.GroupNorm(32, d_model)
                )
            )
        input_proj_list.append(nn.Sequential(
            nn.Conv2d(ch, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model)
        ))
        
        self.stride4_proj = nn.Sequential(
            nn.Conv2d(bb_out_channels[0], d_model, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(32, d_model)
        )
        
        self.vid_proj = nn.ModuleList(input_proj_list)
        for proj in self.vid_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.xavier_uniform_(self.stride4_proj[0].weight, gain=1)
        nn.init.constant_(self.stride4_proj[0].bias, 0)
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        encoder = DeformableTransformerEncoder(
            DeformableTransformerEncoderLayer(
                d_model=d_model,
                d_ffn=2048,
                dropout=0.,
                activation='relu',
                n_levels=4,
                n_heads=nheads,
                n_points=npoints,
            ),
            num_encoder_layers
        )
        if fuse_text_multiples:
            encoder.fusion_module = self.fusion_module
        self.transformer_encoder = DeformFusionWrapper(encoder, num_feature_levels=4, d_model=d_model)
        
        self.text_encoder = RobertaModel.from_pretrained('/home/xhh/pt/roberta_base_model.pth')
        # self.text_encoder.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
        self.tokenizer = RobertaTokenizerFast.from_pretrained('/home/xhh/pt/roberta_tokenizer_fast_base.pth')
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=d_model,
            dropout=0.1,
        )

        self.vid_pos_embed = build_position_encoding(position_embedding_name='2d')
        self.text_pos_embed = build_position_encoding(position_embedding_name='1d')

        
        self.fpn_head = FpnHead(dim=d_model)
    
    def forward(self, vids, texts, valid_indices, exist_queries=None):
        """Inputs
            - vids: NT(t b 3 H W)
            - texts: list[str], b
            - valid_indices: None (default to output hidden features of all frames) 
                        or a 1D int tensor
            - exist_queries:
                list[list[str], ni], b
            Ouputs:
            - hidden_features: t b out_dim h w

        """ 
        batch_size = len(texts)
        nf = len(valid_indices)
        device = vids.tensors.device
        
        # list[NT(t b ci hi wi)], 4
        backbone_out = self.vid_backbone(vids)
        if valid_indices is not None:
            for layer_out in backbone_out:
                layer_out.tensors = layer_out.tensors.index_select(0, valid_indices)
                layer_out.mask = layer_out.mask.index_select(0, valid_indices)
                
        for layer_out in backbone_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (t b) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> (t b) h w')
        
        # NT(b s d_model), T(b d_model)
        text_word_features, text_sentence = self.forward_text(texts, device=device)    
        text_feature, text_pad_mask = text_word_features.decompose()
        text_feature = rearrange(text_feature, 'b s d -> s b d')
        text_pos = self.text_pos_embed(text_pad_mask, hidden_dim=text_feature.shape[2])
        text_pos = rearrange(text_pos, 'b d s -> s b d')
        
        srcs = [] # 8x -> 16x -> 32x -> 64x
        masks = []
        poses = []
        for lvl, feat in enumerate(backbone_out[-3:]):
            src, mask = feat.decompose() # 8x -> 16x -> 32x
            src_proj_l = self.vid_proj[lvl](src)
            # TB c h w
            pos = self.vid_pos_embed(mask, hidden_dim=src_proj_l.shape[1])
            
            *_, h, w = src_proj_l.shape
            src_proj_l = rearrange(src_proj_l, '(t b) c h w -> (t h w) b c',t=nf,b=batch_size)
            src_proj_l = self.fusion_module(tgt=src_proj_l,
                                            memory=text_feature,
                                            memory_key_padding_mask=text_pad_mask,
                                            pos=text_pos,
                                            query_pos=None)
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (t b) c h w',t=nf, h=h,w=w)
            
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos)
        
        src = self.vid_proj[3](backbone_out[-1].tensors)
        *_,h,w=src.shape
        mask = vids.mask.index_select(0, valid_indices)
        mask = rearrange(mask, 't b h w -> (t b) 1 h w')
        mask = F.interpolate(mask.float(), size=src.shape[-2:]).bool()[:,0,:,:]
        pos = self.vid_pos_embed(mask, hidden_dim=src.shape[1])
        
        src = rearrange(src, '(t b) c h w -> (t h w) b c',b=batch_size, t=nf)
        src = self.fusion_module(tgt=src,
                                 memory=text_feature,
                                 memory_key_padding_mask=text_pad_mask,
                                 pos=text_pos,
                                 query_pos=None)
        src = rearrange(src, '(t h w) b c -> (t b) c h w', t=nf, h=h,w=w)
        srcs.append(src)
        masks.append(mask)
        poses.append(pos)
        
        # List[T(TB d_model hi wi)], 32x 16x 8x
        vid_memory = self.transformer_encoder(srcs, masks, poses, 
                                              text_word=text_feature, text_padding_mask=text_pad_mask, text_pos=text_pos)[::-1]
        masks = masks[:3][::-1]
        poses = poses[:3][::-1]

        stride8 = vid_memory[-1]
        stride4 = self.stride4_proj(backbone_out[0].tensors)
        stride4 = self.fpn_head(stride8, stride4)
    
        text_sentence = repeat(text_sentence, 'b c -> (t b) c',t=nf)
        return vid_memory, masks, poses, stride4, text_sentence, nf, batch_size
        # return vid_memory[1:], masks[1:], poses[1:], stride4, text_sentence, nf, batch_size

    def build_backbone(self,
                       backbone_name,
                        backbone_pretrained,
                        backbone_pretrained_path,
                        train_backbone,
                        running_mode,
                        dilation):
        if backbone_name == 'swin-t':
            return VideoSwinTransformer(backbone_pretrained=backbone_pretrained,
                                                backbone_pretrained_path=backbone_pretrained_path,
                                                train_backbone=train_backbone,
                                                running_mode=running_mode)
        elif 'resnet' in backbone_name:
            return VideoResNet(backbone_name=backbone_name,
                                  train_backbone=train_backbone,
                                   dilation=dilation)
        else:
            raise NotImplementedError(f'error: backbone "{backbone_name}" is not supported')
    
    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]
            text_features = encoded_text.last_hidden_state 
            text_features = self.txt_proj(text_features)    
            text_masks = text_attention_mask              
            text_features = NestedTensor(text_features, text_masks) # NestedTensor

            text_sentence_features = encoded_text.pooler_output  
            text_sentence_features = self.txt_proj(text_sentence_features)  
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output



def build_mask2former_transformer_decoder(model_configs):
    dataset_name = model_configs['dataset_profile']
    if model_configs['binary_classification']:
        num_classes = 1
    else:
        if dataset_name == 'ytvos':
            num_classes = 65 
        elif dataset_name == 'davis':
            num_classes = 78
        elif dataset_name == 'a2d' or dataset_name == 'jhmdb':
            num_classes = 1
        else: 
            num_classes = 91 # for coco
            
    multimodal_encoder = Multimodal_encoder_deformable(
        backbone_name=model_configs['backbone_name'],
        backbone_pretrained=model_configs['backbone_pretrained'],
        backbone_pretrained_path=model_configs['backbone_pretrained_path'],
        train_backbone=model_configs['backbone_train'],
        running_mode=model_configs['backbone_running_mode'],
        
        d_model=model_configs['d_model'],
        num_encoder_layers=model_configs['transformer_encoder_nlayers'],
        nheads=model_configs['transformer_encoder_nheads'],
        npoints=model_configs['transformer_encoder_npoints'],
        fuse_text_multiples=model_configs['transformer_encoder_fuse_text_multiples'],
        
        text_encoder_type='roberta-base',
        freeze_text_encoder=True,
    )    
    
    mask_decoder = MultiScaleMaskedTransformerDecoder(
        in_channels=model_configs['d_model'],
        num_classes=num_classes,
        hidden_dim=model_configs['d_model'],
        num_queries=model_configs['transformer_decoder_nqueries'],
        nheads=model_configs['transformer_decoder_nheads'],
        dim_feedforward=model_configs['transformer_decoder_ffd'],
        dec_layers=model_configs['transformer_decoder_nlayers'],
        pre_norm=model_configs['transformer_decoder_pre_norm'],
        mask_dim=model_configs['dynamic_conv_mask_dim'],
        enforce_input_project=model_configs['transformer_decoder_enforce_proj'],
        float_attn_mask=model_configs['transformer_decoder_float_attn_mask'],
        learn_features=model_configs['transformer_decoder_learn_features'],
        only1=model_configs['transformer_decoder_only1'],
        dynamic_conv= False
    )
    
    no_box = True
    if not no_box:
        losses = ['labels', 'boxes', 'masks']
        set_cost_bbox = model_configs['set_cost_box']
        set_cost_giou = model_configs['set_cost_giou']
        weight_dict = {}
        weight_dict['loss_ce'] = model_configs['cls_loss_coef']
        weight_dict['loss_bbox'] = model_configs['bbox_loss_coef']
        weight_dict['loss_giou'] = model_configs['giou_loss_coef']
        weight_dict['loss_mask'] = model_configs['mask_loss_coef']
        weight_dict['loss_dice'] = model_configs['dice_loss_coef']
    else:
        # losses = ['labels', 'masks', 'attns']
        losses = ['labels', 'masks']
        set_cost_bbox = 0
        set_cost_giou = 0
        weight_dict = {}
        weight_dict['loss_ce'] = model_configs['cls_loss_coef']
        weight_dict['loss_mask'] = model_configs['mask_loss_coef']
        weight_dict['loss_dice'] = model_configs['dice_loss_coef']
        # weight_dict['loss_attn'] = model_configs['attn_loss_coef']  # 
    from .t51_matcher import build_matcher
    matcher = build_matcher(
        binary=model_configs['binary_classification'],
        dataset_profile=dataset_name,
        set_cost_class=model_configs['set_cost_class'],
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        set_cost_mask=model_configs['set_cost_mask'],
        set_cost_dice=model_configs['set_cost_dice'],
        mask_loss_type=model_configs['mask_loss_type'] if 'mask_loss_type' in model_configs else 'focal'
    )
    
    from .t51_criterion import SetCriterion
    criterion = SetCriterion(
            num_classes, 
            matcher=matcher,
            weight_dict=weight_dict, 
            eos_coef=model_configs['eos_coef'],
            losses=losses,
            focal_alpha=model_configs['focal_alpha'],
            mask_loss_type=model_configs['mask_loss_type'] if 'mask_loss_type' in model_configs else 'focal'
        )
    
    return multimodal_encoder, mask_decoder, criterion

import sys
from util.misc import get_total_grad_norm

class Wrapper(nn.Module):
    def __init__(self,
                 multimodal_encoder,
                 mask_decoder,
                 criterion,

                **kwargs):
        super().__init__()
        
        self.multimodal_encoder = multimodal_encoder
        self.mask_decoder = mask_decoder
        self.criterion = criterion
    
    def forward(self, samples, valid_indices, text_queries, targets, text_auxiliary):
        """Input
            - samples: 
                NT(t b 3 H W)
            - valid_indices:
                T(t_valid, )
            - text_queries:
                list[str], b
            - targets: 
                list[list[dict], b], t_valid
            Output:
            - diffusion loss
        """
        conditionals = self.multimodal_encoder(samples, text_queries, valid_indices, text_auxiliary)
        
        out = self.mask_decoder(conditionals)
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
        
        # if H!=h or W!=w:
        #     model_logits = F.interpolate(model_logits, size=(H, W), mode='bilinear')
        losses = self.criterion(out, targets)
        
        weight_dict = self.criterion.weight_dict
        
        loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(losses)
            sys.exit(1)
        loss.backward()

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in losses.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)

        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

    @torch.no_grad()
    def sample(self, samples, valid_indices, text_queries, text_auxiliary):
        """
        Input
            - samples: 
                NT(t b 3 h w)
            - valid_indices:
                T(t_valid, )
            - text_queries:
                list[str], b
        Output:
            - sampled_masks:
                'pred_masks': [-, +] / {-1, 1}
                    T(t b n h w), 
                'pred_is_referred': 
                    T(t b n 2)
            }
        """
        
        conditionals = self.multimodal_encoder(samples, text_queries, valid_indices, exist_queries=None)
        out = self.mask_decoder(conditionals)
        # pred_logits: b t n classes, real
        # pred_boxes: b t n 4, [0, 1]
        # pred_masks: b t n h w, real
        # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]

        output = {}
        output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
        output['pred_is_referred'] = rearrange(out['pred_logits'], 'b t n c -> t b n c')

        return output
    
from .models_rvos import register_rvos_model, get_optimizer
__all__ = ['mask2former']
@register_rvos_model
def mask2former(device, model_configs):
    configs = vars(model_configs)
        
    multimodal_encoder, mask_decoder, criterion = build_mask2former_transformer_decoder(configs)
    # multimodal_encoder, mask_decoder, criterion = build_mask2former_deformable_decoder(model_configs)
    
    model = Wrapper(
        multimodal_encoder=multimodal_encoder,
        mask_decoder=mask_decoder,
        criterion=criterion,
    )

    model = model.to(device)
    
    model.to(device)

    optmization_configs = model_configs.optimization
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.vid_backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
        "lr": optmization_configs.text_backbone_lr}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

    return model, optimizer 
        