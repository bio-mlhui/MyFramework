
import torch
import torch.nn as nn
from .position_encoding import build_position_encoding
_fusion_entrypoints = {}
def register_fusion(fn):
    fusion_name = fn.__name__
    _fusion_entrypoints[fusion_name] = fn

    return fn
def fusion_entrypoint(fusion_name):
    try:
        return _fusion_entrypoints[fusion_name]
    except KeyError as e:
        print(f'RVOS moel {fusion_name} not found')

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
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask) # b tgt src, float, 0,1
        tgt = tgt * tgt2
        return tgt, attn_weights

  
class NoFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
                query_feat, amr_feats, amr_pad_masks,
                text_pad_masks=None,
                text_feats=None):
        return query_feat, amr_feats, text_feats
    
@register_fusion
def no_fusion(configs):
    return NoFusion()

# b nq c, b s c, 只转换query
class VidQuery_Text_v1(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.cross_module = VisionLanguageFusionModule(d_model=configs['d_model'],
                                                       nhead=configs['nhead'],
                                                       dropout=configs['dropout'])
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
        # amr shortest path positional embedding

    def forward(self, 
                query_feat, amr_feats, amr_pad_masks,
                text_pad_masks=None,
                text_feats=None):
        memory = amr_feats.clone() # b v+e_max c
        memory_key_padding_mask = amr_pad_masks # b v+e_max
        memory_pos = torch.zeros_like(amr_feats) 

        if text_feats is not None:
            assert text_pad_masks is not None
            text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            memory = torch.cat([memory, text_feats], dim=1)
            memory_key_padding_mask = torch.cat([memory_key_padding_mask, text_pad_masks], dim=1)
            memory_pos = torch.cat([memory_pos,text_pos], dim=1)
        query_feat =  self.cross_module(tgt=query_feat.permute(1,0,2),
                                        memory=memory.permute(1,0,2), 
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        pos=memory_pos.permute(1,0,2), 
                                        query_pos=None)[0]
        return query_feat.permute(1,0,2), amr_feats, text_feats
    
@register_fusion
def vidquery_text_v1(configs):
    return VidQuery_Text_v1(configs)


# b nq c, b s c, 只转换query
class VidQuery_Text_v3(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.cross_module = VisionLanguageFusionModule(d_model=configs['d_model'],
                                                       nhead=configs['nhead'],
                                                       dropout=configs['dropout'])
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
        # amr shortest path positional embedding

    def forward(self, 
                query_feat, amr_feats, amr_pad_masks,
                text_pad_masks=None,
                text_feats=None):
        memory = query_feat.clone() # b nq c

        tgt = amr_feats
        tgt_pos = torch.zeros_like(tgt)
        lamr = amr_feats.shape[1]
        if text_feats is not None:
            ltext = text_feats.shape[1]
            text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            tgt = torch.cat([tgt, text_feats], dim=1)
            tgt_pos = torch.cat([tgt_pos,text_pos], dim=1)
        tgt =  self.cross_module(tgt=tgt.permute(1,0,2),
                                memory=memory.permute(1,0,2), 
                                memory_key_padding_mask=None,
                                pos=None, 
                                query_pos=tgt_pos.permute(1,0,2))[0]
        tgt = tgt.permute(1,0,2)
        amr_feats, text_feats = tgt.split([lamr, ltext], dim=1)
        return query_feat, amr_feats, text_feats
    
@register_fusion
def vidquery_text_v3(configs):
    return VidQuery_Text_v3(configs)

from .transformer import _get_activation_fn
import copy
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
class TransformerEncoderLayerDot(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src * self.dropout2(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src * self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src * self.dropout2(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src * self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# b nq c, b s c, 两个都转换
from .transformer import TransformerEncoder, TransformerEncoderLayer
class VidQuery_Text_v2(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.self_module = TransformerEncoder(TransformerEncoderLayerDot(d_model=configs['d_model'],
                                                                      nheads=configs['nheads'],
                                                                      dim_feedforward=configs['dim_ff'],
                                                                      dropout=configs['dropout'],
                                                                      activation=configs['act']), 
                                                num_layers=configs['num_layers'])
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
    
    def forward(self, 
                query_feat, amr_feats, amr_pad_masks,
                text_pad_masks=None,
                text_feats=None):
        lquery, lamr = query_feat.shape[1], amr_feats.shape[1]
        src = torch.cat([query_feat, amr_feats], dim=1)
        src_pad_mask = torch.cat([torch.zeros_like(query_feat[:, :, 0]).bool(), amr_pad_masks], dim=1)
        src_pos = torch.zeros_like(src)

        if text_feats is not None:
            assert text_pad_masks is not None
            text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            src = torch.cat([src, text_feats], dim=1)
            src_pad_mask = torch.cat([src_pad_mask, text_pad_masks], dim=1)
            src_pos = torch.cat([src_pos, text_pos], dim=1)
            ltext = text_feats.shape[1]
        else:
            ltext = 0

        src =  self.self_module(src=src.permute(1,0,2),
                                mask=None,
                                src_key_padding_mask=src_pad_mask,
                                pos=src_pos.permute(1,0,2))
        src = src.permute(1,0,2)
        return src.split([lquery, lamr, ltext], dim=1)


@register_fusion
def vidquery_text_v2(configs):
    return VidQuery_Text_v2(configs)


class perFrameVideo_Text_v1(nn.Module):
    def __init__(self, d_model, nhead, dropout) -> None:
        super().__init__()
        self.cross_module = VisionLanguageFusionModule(d_model=d_model,
                                                                nhead=nhead,
                                                                dropout=dropout)
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
        # amr shortest path positional embedding
    
    def forward(self, video_feats, video_poses,
                text_feats, text_pad_masks,
                amr_feats, amr_pad_masks):
        # dict, bt c h w
        # text_feats: b s c
        # amr_feats: b v+e_max c
        BT = video_feats[0].shape[0]
        B = len(text_feats)
        T = BT // B
        text_feats = repeat(text_feats, 'b s c -> (b t) s c', t=T)
        text_pad_masks = repeat(text_pad_masks, 'b s -> (b t) s',t=T)
        amr_feats = repeat(amr_feats, 'b s c -> (b t) s c', t=T)
        amr_pad_masks = repeat(amr_pad_masks, 'b s -> (b t) s',t=T)

        text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0,2,1) # b s c
        amr_pos = torch.zeros_like(amr_feats)
        memory = torch.cat([text_feats, amr_feats], dim=1)   
        memory_pad_masks = torch.cat([text_pad_masks, amr_pad_masks], dim=1)
        memory_pos = torch.cat([text_pos, amr_pos], dim=1)

        for lvl, (tgt_feat, tgt_pos) in enumerate(zip(video_feats, video_poses)):
            tgt_pos = tgt_pos.flatten(2).permute(0, 2, 1) # bt c h w -> bt c hw -> bt hw c
            h, w = tgt_feat.shape[-2:]
            tgt_feat = tgt_feat.flatten(2).permute(0,2,1) # bt hw c
            tgt_feat = self.cross_module(tgt=tgt_feat.permute(1,0,2),
                                    memory=memory.permute(1,0,2), 
                                    memory_key_padding_mask=memory_pad_masks,
                                    pos=memory_pos.permute(1,0,2), 
                                    query_pos=tgt_pos.permute(1,0,2))[0]
            video_feats[lvl] = rearrange(tgt_feat, '(h w) bt c-> bt c h w',h=h,w=w)

        return video_feats

@register_fusion
def perFrameVideo_text_v1(configs):
    return perFrameVideo_Text_v1(d_model=configs['d_model'],
                                 nhead=configs['nhead'],
                                 dropout=configs['dropout'])