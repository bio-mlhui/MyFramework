import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange, reduce
from functools import partial
from .position_encoding import build_position_encoding
from util.misc import NestedTensor

from typing import Any, Optional
from torch import Tensor
import math
import copy

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class VideoDivSelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_norm = nn.LayerNorm(d_model)
            
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.spatial_norm = nn.LayerNorm(d_model)
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

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     nf: Optional[int] = None,
                     h: Optional[int] = None,
                     w: Optional[int] = None,
                     ):
        """
        tgt: (t h w) b c
        """
        # rearrange
        tgt = rearrange(tgt, '(t h w) b c -> t (h w b) c',t=nf,h=h,w=w)
        query_pos = rearrange(query_pos, '(t h w) b c -> t (h w b) c',t=nf,h=h,w=w)
        if tgt_mask is not None:
            tgt_mask = rearrange(tgt_mask, 'b_head (t h w) k-> (h w b_head) t k',t=nf,h=h,w=w)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = rearrange(tgt_key_padding_mask, 'b (t h w) -> (h w b) t',t=nf,h=h,w=w)
        # attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.time_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.time_norm(tgt)
        
        # rearrange            
        tgt = rearrange(tgt, 't (h w b) c -> (h w) (t b) c',h=h,w=w)
        query_pos = rearrange(query_pos, 't (h w b) c -> (h w) (t b) c',h=h,w=w)
        if tgt_mask is not None:
            tgt_mask = rearrange(tgt_mask, '(h w b_head) t k-> (t b_head) (h w) k',h=h,w=w)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = rearrange(tgt_key_padding_mask, '(h w b) t -> (t b) (h w)',h=h,w=w)
                  
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.spatial_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.spatial_norm(tgt) 
         
        tgt = rearrange(tgt, '(h w) (t b) c -> (t h w) b c',h=h,w=w,t=nf)      
        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    nf: Optional[int] = None,
                    h: Optional[int] = None,
                    w: Optional[int] = None,):
        
        # rearrange
        tgt = self.time_norm(tgt)
        tgt = rearrange(tgt, '(t h w) b c -> t (h w b) c',t=nf,h=h,w=w)
        query_pos = rearrange(query_pos, '(t h w) b c -> t (h w b) c',t=nf,h=h,w=w)
        if tgt_mask is not None:
            tgt_mask = rearrange(tgt_mask, 'b_head (t h w) k-> (h w b_head) t k',t=nf,h=h,w=w)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = rearrange(tgt_key_padding_mask, 'b (t h w) -> (h w b) t',t=nf,h=h,w=w)
        # attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.time_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
 
        # rearrange 
        tgt = self.spatial_norm(tgt)           
        tgt = rearrange(tgt, 't (h w b) c -> (h w) (t b) c',h=h,w=w)
        query_pos = rearrange(query_pos, 't (h w b) c -> (h w) (t b) c',h=h,w=w)
        if tgt_mask is not None:
            tgt_mask = rearrange(tgt_mask, '(h w b_head) t k-> (t b_head) (h w) k',h=h,w=w)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = rearrange(tgt_key_padding_mask, '(h w b) t -> (t b) (h w)',h=h,w=w)
                  
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.spatial_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)   
         
        tgt = rearrange(tgt, '(h w) (t b) c -> (t h w) b c',h=h,w=w,t=nf)     
        return tgt


    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                nf: Optional[int] = None,
                h: Optional[int] = None,
                w: Optional[int] = None,):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos, nf, h, w)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos, nf, h, w)


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

class CrossSelfFFN_Module(nn.Module):
    def __init__(self,  d_model, 
                 nhead, dropout=0.0,
                 activation="relu", normalize_before=False): 
        super().__init__()
        
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.ffn = FFNLayer(d_model, nhead, dropout, activation, normalize_before)
    
    def forward(self,
                query_feats,
                query_poses,
                scale_feats,
                scale_poses,
                
                query_padding_mask=None,
                scale_padding_mask=None,
                cross_attn_mask=None,
                self_attn_mask=None):
        
        output = self.cross_attn(
            tgt=query_feats,  # n b c
            memory=scale_feats, # thw b c
            memory_mask=cross_attn_mask, # bh n thw
            memory_key_padding_mask=scale_padding_mask,  # here we do not apply masking on padded region
            pos=scale_poses,  # thw b c
            query_pos=query_poses, # n b c
        )

        output = self.self_attn(
            output, # n b c
            tgt_mask=self_attn_mask,
            tgt_key_padding_mask=query_padding_mask, # b n 
            query_pos=query_poses, # n b c
        )
        output = self.ffn(
            output # n b c
        )
        return output
        

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 1, dim_head = 256):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

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

def _get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    if activation == 'none':
        return nn.Identity()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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

class FeatureResizer_MultiLayer(nn.Module):
    def __init__(self, 
                 input_feat_size, hidden_size, output_feat_size, 
                 num_layers,

                 dropout, 
                 do_ln,
                 activation
                 ):
        super().__init__()
        self.do_ln = do_ln
        self.num_layers = num_layers
        h = [hidden_size] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(n, k),
                           _get_activation_layer(activation),
                           nn.Dropout(dropout))
                for n, k in zip([input_feat_size] + h, h + [output_feat_size])])
        
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if self.do_ln:
            x = self.layer_norm(x)
        return x
        
class ObjectGraph_SelfAttention_Layer(nn.Module):
    """
    输入一堆object queries, 他们各自当前卷积video mask, 对这些object queries做self-attention
    """
    def __init__(self, ) -> None:
        super().__init__()