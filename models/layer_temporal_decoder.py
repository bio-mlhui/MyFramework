from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat, rearrange

_temporal_decoder_entrypoints = {}
def register_temporal_decoder(fn):
    temporal_decoder_name = fn.__name__
    _temporal_decoder_entrypoints[temporal_decoder_name] = fn

    return fn
def temporal_decoder_entrypoint(temporal_decoder_name):
    try:
        return _temporal_decoder_entrypoints[temporal_decoder_name]
    except KeyError as e:
        print(f'RVOS moel {temporal_decoder_name} not found')
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
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
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
                     query_pos: Optional[Tensor] = None):
        tgt2, cross_weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt, cross_weight

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2, cross_weight = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt, cross_weight

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
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

import copy
from models.transformer import _get_clones
from models.layers_unimodal_attention import zero_module

class VITA(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_queries: int,
        num_classes: int,
        num_frame_queries: int,
        enc_layers: int,
        dec_layers: int,
        enc_window_size: int,
        order,
        mask_feat_proj,

        nheads: int = 8,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_project: bool = True,

    ):
        super().__init__()
        assert enforce_input_project
        self.order = order
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.enc_layers = enc_layers
        self.window_size = enc_window_size

        self.early_fusion = None
        self.early_fusion_add_pos = None
        self.layer_fusion_add_pos = None
        self.layer_fusion_modules = [None] * self.enc_layers
        self.enc_self_attn = nn.ModuleList()
        self.enc_ffn = nn.ModuleList()

        for _ in range(self.enc_layers):
            self.enc_self_attn.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                ),
            )
            self.enc_ffn.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

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

        self.mask_feat_proj_name = mask_feat_proj.pop('name')
        if self.mask_feat_proj_name == 'conv2d':
            self.vita_mask_features = Conv2d(**mask_feat_proj)
        elif self.mask_feat_proj_name == 'conv3d':
            raise NotImplementedError()
            self.vita_mask_features = nn.Conv3d(**mask_feat_proj)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self._reset_parameters()
        self.mask_out_stride = 4
        self.mask_threshold=0.5

    def _reset_parameters(self):
        weight_init.c2_xavier_fill(self.vita_mask_features)
        zero_module(self.query_embed)
        zero_module(self.query_feat)

    def hack_fusion(self, 
                    fusion_module,
                    early_fusion,
                    early_fusion_deep_copy, 
                    early_fusion_add_pos,
                    encoder_layer_ref_self,
                    encoder_layer_deep_copy,
                    encoder_layer_add_pos,):
        self.early_fusion_add_pos = early_fusion_add_pos
        if early_fusion:
            if early_fusion_deep_copy:
                self.early_fusion = nn.ModuleList([copy.deepcopy(fusion_module)])
            else:
                self.early_fusion = [fusion_module] # do not save in checkpoint
        else:
            self.early_fusion = [None]

        self.layer_fusion_ref_self = encoder_layer_ref_self
        self.layer_fusion_add_pos = encoder_layer_add_pos
        if encoder_layer_ref_self == 'after':
            if encoder_layer_deep_copy:
                self.layer_fusion_modules = _get_clones(fusion_module, self.enc_layers)
            else:
                self.layer_fusion_modules = [fusion_module] * self.enc_layers
        elif encoder_layer_ref_self == 'before':
            raise ValueError()
        else:
            assert encoder_layer_ref_self == None
            self.layer_fusion_modules = [None] * self.enc_layers

    def forward_decoder(self,
                            frame_query, # t_nqf LB c
                            mask_features, # LB t c h w
                            B, T, L,nqf):
        src = frame_query   # t_nqf LB c
        # nq L*B c
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L*B, 1) 
        output = self.query_feat.weight.unsqueeze(1).repeat(1, L*B, 1)

        decoder_outputs = []
        cross_weight_by_layer = []
        for i in range(self.num_layers):
            if self.order == 'cross_self_lln':
                output, cross_weight = self.transformer_cross_attention_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # LB nq t_nqf
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

            elif self.order == 'self_cross_lln':
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                output, cross_weight = self.transformer_cross_attention_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # LB nq t_nqf
            
            output = self.transformer_ffn_layers[i](
                output
            )                
            cross_weight_by_layer.append(rearrange(cross_weight, '(L b) nq (t nqf) -> L b nq t nqf',t=T,nqf=nqf, L=L,b=B))
            dec_out = self.decoder_norm(output) # nq LB c
            decoder_outputs.append(rearrange(dec_out, 'nq (L b) c -> L b nq c', L=L))

        mask_embeds = [self.mask_embed(dec_o) for dec_o in decoder_outputs] # L b nq c
        pred_cls = [self.class_embed(dec_o) for dec_o in decoder_outputs] # L b nq class+1
        mask_features = self.vita_mask_features(mask_features.flatten(0,1)) # l_b_t c h w
        mask_features = rearrange(mask_features, '(L b t) c h w -> L b t c h w',L=L,b=B,t=T)
        pred_masks_by_layer = [torch.einsum('lbnc,lbtchw->lbnthw', mask_e, mask_features) for mask_e in mask_embeds]
        
        out = {
            'temporal_queries': torch.stack(decoder_outputs,dim=1), # L D b nq c
            'pred_masks': torch.stack(pred_masks_by_layer, dim=1), # L D b nq t h w
            'pred_logits': torch.stack(pred_cls), # L D b nq class+1
            'frame_queries':rearrange(src, '(t nqf) (L b) c -> L b t nqf c',t=T,nqf=nqf,L=L,b=B), # L b t nqf c
            'cross_attn_weights': torch.stack(cross_weight_by_layer, dim=1), # L D b nq t nqf
        }        
        return out 


    def forward(self,
                frame_query_by_layer, # list[b t nq c]
                mask_features, # bt c h w
                amrs=None,  
                amr_token_feats=None,  # b s c
                amr_token_seg_ids=None,
                text_feats=None, 
                text_pad_masks=None):
        # list[b t nq c]
        # 只用最后一层
        if not self.training:
            frame_query = [frame_query_by_layer[-1]] # b t nq c
        else:
            frame_query = frame_query_by_layer[:-3] # 训练的时候用后三层
        B = len(amrs)
        L = len(frame_query)
        mask_features = repeat(mask_features, '(b T) c h w -> (L b) T c h w', L=L,b=B) # lb t c h w
        frame_query = torch.stack(frame_query, dim=0).flatten(0, 1) # lb t nq c
        _, T, nqf, _ = frame_query.shape
        # repeat amr by L times
        repeated_amrs = []
        for _ in range(L):
            for idx in range(B):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (L b) s c', L=L)
        text_feats = repeat(text_feats, 'b s c -> (L b) s c',L=L)
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (L b) s', L=L)
        text_pad_masks = repeat(text_pad_masks, 'b s -> (L b) s',L=L)
        frame_query = self.input_proj_dec(frame_query)
        if self.early_fusion is not None:
            frame_query, amr_token_feats, text_feats = self.early_fusion[0](
                                                                frame_queries=frame_query,
                                                                is_video_frame_query=True,
                                                                amrs=repeated_amrs, 
                                                                time_pad = frame_query.new_zeros([L * B, T]).bool(), 
                                                                amr_text_add_pos=self.early_fusion_add_pos,
                                                                amr_token_feats=amr_token_feats,
                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                text_feats=text_feats, 
                                                                text_pad_masks=text_pad_masks)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous() # t nq lb c

        if (self.window_size != 0) and self.window_size < T:
            pad = int(ceil(T / self.window_size)) * self.window_size - T # 让window能
            _T = pad + T
            frame_query = F.pad(frame_query, (0,0,0,0,0,0,0,pad))  # t_pad
            enc_mask = frame_query.new_ones(L*B, _T).bool()        # lb t_pad
            enc_mask[:, :T] = False
        else:
            enc_mask = frame_query.new_zeros([L*B, T]).bool()

        # t nq LB c
        frame_query, amr_token_feats, text_feats = self.encode_frame_query(frame_query, 
                                                                            enc_mask,
                                                                            amrs=amrs, 
                                                                            amr_token_feats=amr_token_feats,
                                                                            amr_token_seg_ids=amr_token_seg_ids, 
                                                                            text_feats=text_feats, 
                                                                            text_pad_masks=text_pad_masks)
        frame_query = frame_query[:T].flatten(0,1)              # tnq LB c

        ret =  self.forward_decoder(frame_query=frame_query, # L D b nq [class/t h w/t nqf]
                                    mask_features=mask_features, # LB t c h w
                                    B=B,T=T,nqf=nqf,L=L)
        return ret, rearrange(amr_token_feats, '(L b) s c -> L b s c',L=L,b=B), \
                        rearrange(text_feats, '(L b) s c -> L b s c',L=L,b=B)

    def encode_frame_query(self, frame_query, attn_mask,
                                amrs=None, 
                                amr_token_feats=None, 
                                amr_token_seg_ids=None,
                                text_feats=None, 
                                text_pad_masks=None):
        """
        _t nq LB c, LB _t
        b s c, b s
        """
        t_pad = frame_query.shape[0]
        if self.window_size == 0 or (self.window_size >= t_pad):
            return_shape = frame_query.shape 
            frame_query = frame_query.flatten(0, 1)
            for i in range(self.enc_layers):
                frame_query = self.enc_self_attn[i](frame_query) # (t nqf) b c
                if self.layer_fusion_modules[i] is not None:
                    frame_query = rearrange(frame_query, '(t nq) b c -> b t nq c',t=t_pad)
                    frame_query, amr_token_feats, text_feats = self.layer_fusion_modules[i](frame_queries=frame_query,
                                                                    is_video_frame_query=True,
                                                                    time_pad = attn_mask, # b t
                                                                    amrs=amrs, 
                                                                    amr_text_add_pos=self.layer_fusion_add_pos,
                                                                    amr_token_feats=amr_token_feats,
                                                                    amr_token_seg_ids=amr_token_seg_ids, 
                                                                    text_feats=text_feats, 
                                                                    text_pad_masks=text_pad_masks)
                    frame_query = rearrange(frame_query, 'b t nq c -> (t nq) b c')
                frame_query = self.enc_ffn[i](frame_query)
            frame_query = frame_query.view(return_shape)
            return frame_query, amr_token_feats, text_feats
        else:
            T, fQ, LB, C = frame_query.shape
            W = self.window_size
            Nw = T // W
            half_W = int(ceil(W / 2))  # 滑动一半

            # b t -> bN w nq -> bN w_nq
            window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)
            # b t -> b N t' t'
            _attn_mask  = torch.roll(attn_mask, half_W, 1)
            _attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
            _attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
            _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1) # Padding size肯定小于window size
            _attn_mask[:, 0, :half_W, half_W:] = True
            _attn_mask[:, 0, half_W:, :half_W] = True # 开头不关注结尾，结尾不关注开头
            _attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.enc_layers):
                if layer_idx % 2 == 0:
                    frame_query = self._window_attn(frame_query, window_mask, layer_idx,)
                else:
                    frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx,)
                if self.layer_fusion_modules[layer_idx] is not None:
                    frame_query = rearrange(frame_query, 't nq b c -> b t nq c')
                    frame_query, amr_token_feats, text_feats = self.layer_fusion_modules[layer_idx](frame_queries=frame_query, # b t nq c
                                                                    is_video_frame_query=True,
                                                                    time_pad = attn_mask, # b t
                                                                    amrs=amrs, 
                                                                    amr_text_add_pos=self.layer_fusion_add_pos,
                                                                    amr_token_feats=amr_token_feats,
                                                                    amr_token_seg_ids=amr_token_seg_ids, 
                                                                    text_feats=text_feats, 
                                                                    text_pad_masks=text_pad_masks)
                    frame_query = rearrange(frame_query, 'b t nq c -> t nq b c')
                frame_query = self.enc_ffn[layer_idx](frame_query)
            return frame_query, amr_token_feats, text_feats

    def _window_attn(self, frame_query, attn_mask, layer_idx):
        # t nq b c
        T, fQ, LB, C = frame_query.shape
        # bN t'
        W = self.window_size
        Nw = T // W

        # t nq b c -> N t' nq b c -> t' nq b N c -> t'nq bN c
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_query

    def _shift_window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_query = torch.roll(frame_query, -half_W, 0)

        return frame_query
    
@register_temporal_decoder
def vita(configs, pt_dir):
    return VITA(hidden_dim=configs['d_model'],
                num_frame_queries=configs['n_fqueries'],
                num_queries=configs['nqueries'],
                enc_layers=configs['enc_layers'],
                dec_layers=configs['dec_layers'],
                enc_window_size=configs['swin_window'],
                order=configs['order'],
                mask_feat_proj=configs['mask_feat_proj'],
                num_classes=configs['num_classes'])

