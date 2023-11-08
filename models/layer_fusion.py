
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

# video query + text  
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


class perFrameMultiscaleFlatten_Text_nofusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, flatten_multiscale, flatten_multiscale_pos,
                text_feats=None, text_pad_masks=None,
                amr_feats=None, amr_pad_masks=None):
        return flatten_multiscale

@register_fusion
def perFrameVideoFlatten_text_nofusion(configs):
    return perFrameMultiscaleFlatten_Text_nofusion()

# dot_cross
class perFrameMultiscaleFlatten_Text_v1(nn.Module):
    def __init__(self, d_model, nheads, dropout) -> None:
        super().__init__()
        self.cross_module = VisionLanguageFusionModule(d_model=d_model,
                                                        nhead=nheads,
                                                        dropout=dropout)
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, flatten_multiscale, flatten_multiscale_pos,
                text_feats=None, text_pad_masks=None,
                amr_feats=None, amr_pad_masks=None):
        assert (amr_feats is not None) and (amr_pad_masks is not None)
        # b s c
        output = flatten_multiscale
        memory = amr_feats.clone() # b v+e_max c
        memory_key_padding_mask = amr_pad_masks # b v+e_max
        memory_pos = torch.zeros_like(amr_feats) 

        if text_feats is not None:
            assert text_pad_masks is not None
            text_pos = self.text1d_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            memory = torch.cat([memory, text_feats], dim=1)
            memory_key_padding_mask = torch.cat([memory_key_padding_mask, text_pad_masks], dim=1)
            memory_pos = torch.cat([memory_pos,text_pos], dim=1)

        output = self.cross_module(tgt=output.permute(1,0,2),
                                    memory=memory.permute(1,0,2), 
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=memory_pos.permute(1,0,2), 
                                    query_pos=flatten_multiscale_pos.permute(1,0,2))[0]
        return output.permute(1,0,2)

@register_fusion
def perFrameVideoFlatten_text_v1(configs):
    return perFrameMultiscaleFlatten_Text_v1(d_model=configs['d_model'],
                                             nheads=configs['nheads'],
                                             dropout=configs['dropout'])


class CommonCrossAttentionWeights(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                  dropout=0.0,
                  dot_or_add='dot'):
        super().__init__()

        assert d_model % nheads == 0
        head_dim = d_model // nheads
        self.scale = head_dim ** -0.5
        self.nheads = nheads
        self.head_dim = head_dim
        self.dot_or_add = dot_or_add

        self.amr_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
        self.amr_v_emb = nn.Linear(d_model, nheads * head_dim, bias=False)

        self.vis_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
        self.vis_v_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
  
        self.visual_out = nn.Sequential( nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.amr_out = nn.Sequential( nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, amr_feats, amr_pad_mask, visual_feats):
        amr_pad_mask = repeat(amr_pad_mask, 'b s -> (b h) s',h=self.nheads)
        # b s c, b s, b s c
        amr_feats_qk = self.amr_emb(amr_feats)
        visual_feats_qk = self.vis_emb(visual_feats)
        amr_feats_qk, visual_feats_qk = map(lambda t: rearrange(t, 'b s (h c) -> (b h) s c', h=self.nheads), (amr_feats_qk, visual_feats_qk))

        amr_values = self.amr_v_emb(amr_feats)
        visual_values = self.vis_v_emb(visual_feats)
        amr_values, visual_values = map(lambda t: rearrange(t, 'b s (h c) -> (b h) s c', h=self.nheads), (amr_values, visual_values))

        common_weights = torch.einsum('bnc,bmc->bnm', amr_feats_qk, visual_feats_qk) * self.scale

        # amr cross vis:
        amr_vis_weights = common_weights.softmax(-1) # b s_amr s_vis
        amr_feats_2 = amr_vis_weights @ visual_values
        amr_feats_2 = rearrange(amr_feats_2, '(b h) s c -> b s (h c)', h=self.nheads)
        amr_feats_2 = self.amr_out(amr_feats_2)

        # visual cross text:
        vis_text_weights = common_weights.permute(0, 2, 1) # bh s_vis s_amr
        vis_text_weights.masked_fill_(amr_pad_mask.unsqueeze(1), torch.finfo(common_weights.dtype).min)
        vis_text_weights = vis_text_weights.softmax(dim=-1)
        visual_feats_2 = vis_text_weights @ amr_values
        visual_feats_2 = rearrange(visual_feats_2, '(b h) s c -> b s (h c)', h=self.nheads)
        visual_feats_2 = self.visual_out(visual_feats_2)

        if self.dot_or_add == 'dot':
            return amr_feats_2 * amr_feats, visual_feats * visual_feats_2
        elif self.dot_or_add =='add':
            return amr_feats_2 + amr_feats, visual_feats + visual_feats_2
        elif self.dot_or_add == 'dot_add':
            return amr_feats + (amr_feats * amr_feats_2), visual_feats + (visual_feats * visual_feats_2)

class VisionLanguageFusionModule_v2(nn.Module):
    def __init__(self, d_model, nhead, 
                 dropout=0.0, dot_or_add='dot'):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dot_or_add = dot_or_add
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
        if self.dot_or_add == 'add':
            return tgt + tgt2, attn_weights
        else:
            return tgt * tgt2, attn_weights

# multiscale + text
class Visual_AMRText_SeqSeq(nn.Module):
    def __init__(self, 
                 d_model,
                nheads=8, 
                dropout=0.0,
                amr_pos_type=['lap', 'depth'],
                text_pos_type=['sin', 'learned'],
                use_text=False,
                transform_amr_text=False,
                dot_or_add='dot',) -> None:
        super().__init__()
        self.d_model = d_model
        self.amr_pos_type = amr_pos_type
        self.text_pos_type = text_pos_type
        self.transform_amr_text = transform_amr_text             # 这一层内部全部转换text, visual,
                                                                     # 但是transoform amr text控制是否把内部的text用起来
        self.use_text = use_text
        
        if self.transform_amr_text:
            self.multiscale_text_module = CommonCrossAttentionWeights(d_model=d_model,
                                                                       nheads=nheads,
                                                                         dropout=dropout,
                                                                      dot_or_add=dot_or_add)
        else:
            self.multiscale_text_module = VisionLanguageFusionModule_v2(d_model=d_model,
                                                                     nhead=nheads,
                                                                     dropout=dropout,
                                                                     dot_or_add=dot_or_add,)
        if 'depth' in amr_pos_type:
            # 最大深度
            self.depth_amr_pos = nn.Embedding(50, embedding_dim=d_model)
        if 'learned' in text_pos_type:
            self.lrn_text_pos = nn.Embedding(512, embedding_dim=512)
        elif 'sin' in self.text_pos_type:
            self.sin_text_pos = build_position_encoding(position_embedding_name='1d')

    def amr_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        if self.amr_pos_type == 'lap':
            import scipy as sp
            import dgl
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(g.number_of_nodes()) - N * A * N

            # # Eigenvectors with numpy
            # EigVal, EigVec = np.linalg.eig(L.toarray())
            # idx = EigVal.argsort() # increasing order
            # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
            # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

            # Eigenvectors with scipy
            #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
            EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
            EigVec = EigVec[:, EigVal.argsort()] # increasing order
            g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

            return g

    def get_amr_poses(self, amrs=None, amr_token_feats=None, amr_token_seg_ids=None):
        # b (v+e)_max c
        amr_poses = torch.zeros_like(amr_token_feats)
        if 'lap' in self.amr_pos_type:
            raise NotImplementedError()
        return amr_poses

    def get_text_poses(self, text_feats=None, text_pad_masks=None):
        # b s c
        text_poses = torch.zeros_like(text_feats)
        if 'sin' in self.text_pos_type:
            sin_pos = self.sin_text_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            text_poses += sin_pos
        return text_poses

    def forward_video_multiscale(self,
                multiscale_feats, multiscale_poses, multiscale_is_flattened,
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None):
        B = len(amrs)
        if multiscale_is_flattened:
            # bt \sigmaHE c
            assert len(multiscale_feats.shape) == 3
            BT = multiscale_is_flattened.shape[0]
            multiscale_feats = multiscale_feats + multiscale_poses # bt s c
        else:
            # list[bt c h w]
            BT = multiscale_feats[0].shape[0]
            scale_sizes = [mf.shape[-2:] for mf in multiscale_feats]
            scale_length = [sum(mf.shape[-2:]) for mf in multiscale_feats]
            flattened_multiscale = torch.cat([mf.flatten(2) for mf in multiscale_feats], dim=-1) # bt c \sigmaHW
            flattened_poses = torch.cat([mf.flatten(2) for mf in multiscale_poses], dim=-1)
            flattened_multiscale = flattened_multiscale + flattened_poses
            multiscale_feats = flattened_multiscale.permute(0, 2, 1) # bt s c

        T = BT // B
        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = repeat(amr_token_feats, 'b s c -> (b t) s c', t=T)    
        memory_pad_masks = repeat(amr_token_seg_ids==0, 'b s -> (b t) s',t=T)
        memory_poses = repeat(amr_poses, 'b s c -> (b t) s c', t=T)

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, 
                                             text_pad_masks=text_pad_masks)
            memory = torch.cat([memory, 
                                repeat(text_feats, 'b s c -> (b t) s c', t=T)], dim=1)
            
            memory_pad_masks = torch.cat([memory_pad_masks, 
                                          repeat(text_pad_masks, 'b s -> (b t) s',t=T)], dim=1)
            
            memory_poses = torch.cat([memory_poses, repeat(text_poses, 'b s c -> (b t) s c', t=T)])
        
        memory = memory + memory_poses # bt s c

        if self.transform_amr_text:
            memory, multiscale_feats = self.multiscale_text_module(amr_feats=memory,
                                                                   amr_pad_mask=memory_pad_masks, 
                                                                   visual_feats=multiscale_feats)
        else:
            multiscale_feats = self.multiscale_text_module(tgt=multiscale_feats.permute(1,0,2),
                                                            memory=memory.permute(1,0,2), 
                                                            memory_key_padding_mask=memory_pad_masks,
                                                            pos=None, 
                                                            query_pos=multiscale_feats.permute(1,0,2))[0]
            multiscale_feats = multiscale_feats.permute(1, 0, 2)
        
        if not multiscale_is_flattened:
            multiscale_feats = multiscale_feats.split(scale_length, dim=-1) # list[bt c hw]
            multiscale_feats = [rearrange(mf, 'bt c (h w) -> bt c h w', h=sz[0], w=sz[1]) for mf, sz in zip(multiscale_feats, scale_sizes)]

        if not self.transform_amr_text:
            return multiscale_feats, amr_token_feats, text_feats
        else:
            amr_length = amr_token_feats.shape[1]
            if self.use_text:
                amr_token_feats = rearrange(memory[:, :amr_length], '(b t) s c -> b t s c', b=B, T=T)
                text_feats = rearrange(memory[:, amr_length:], '(b t) s c -> b t s c', b=B, T=T)
                amr_token_feats = amr_token_feats.mean(1)
                text_feats = text_feats.mean(1)
            else:
                amr_token_feats = memory
                amr_token_feats = amr_token_feats.mean(1)
            # bt -> b
            return multiscale_feats, amr_token_feats, text_feats


    def forward_video_frameQuery(self,
                frame_queries, # b t nq c
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None):

        B, T, nq, _ = frame_queries.shape

        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = amr_token_feats.clone(),    
        memory_pad_masks = amr_token_seg_ids==0
        memory_poses = amr_poses

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, 
                                             text_pad_masks=text_pad_masks)
            memory = torch.cat([memory,text_feats], dim=1)
            memory_pad_masks = torch.cat([memory_pad_masks, text_pad_masks], dim=1)
            memory_poses = torch.cat([memory_poses, text_poses])
        
        memory = memory + memory_poses # b s c
        frame_queries = frame_queries.flatten(1, 2) # b tnq c
        if self.transform_amr_text:
            memory, frame_queries = self.query_text_module(memory=memory,
                                                        memory_pad_masks=memory_pad_masks,
                                                        visual_feats=frame_queries)
        else:
            frame_queries = self.query_text_module(tgt=frame_queries.permute(1,0,2),
                                                        memory=memory.permute(1,0,2), 
                                                        memory_key_padding_mask=memory_pad_masks,
                                                        pos=None, 
                                                        query_pos=None)[0]
            frame_queries = frame_queries.permute(1,0,2)

        frame_queries = rearrange(frame_queries, 'b (t nq) c -> b t nq c',t=T, nq=nq)
        if not self.transform_amr_text:
            return frame_queries, amr_token_feats, text_feats
        else:
            amr_length = amr_token_feats.shape[1]
            if self.use_text:
                amr_token_feats = memory[:, :amr_length]
                text_feats = memory[:, amr_length:]
            else:
                amr_token_feats = memory
            return frame_queries, amr_token_feats, text_feats


    def forward_image_multiscale(self,
                multiscale_feats, multiscale_poses, multiscale_is_flattened,
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None): # 假设multiscale肯定转换
        B = len(amrs)
        if multiscale_is_flattened:
            # b \sigmaHE c
            assert len(multiscale_feats.shape) == 3
            multiscale_feats = multiscale_feats + multiscale_poses # b s c
        else:
            # list[b c h w]
            scale_sizes = [mf.shape[-2:] for mf in multiscale_feats]
            scale_length = [(mf.shape[-2:][0] * mf.shape[-2:][1]) for mf in multiscale_feats]
            flattened_multiscale = torch.cat([mf.flatten(2) for mf in multiscale_feats], dim=-1) # b c \sigmaHW
            flattened_poses = torch.cat([mf.flatten(2) for mf in multiscale_poses], dim=-1)
            flattened_multiscale = flattened_multiscale + flattened_poses
            multiscale_feats = flattened_multiscale.permute(0, 2, 1) # b s c
        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = amr_token_feats.clone()   
        memory_pad_masks = amr_token_seg_ids==0
        memory_poses = amr_poses

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, text_pad_masks=text_pad_masks)
            memory = torch.cat([memory, text_feats], dim=1)
            memory_pad_masks = torch.cat([memory_pad_masks, text_pad_masks], dim=1)
            memory_poses = torch.cat([memory_poses, text_poses], dim=1)
        
        memory = memory + memory_poses # b s c

        if self.transform_amr_text:
            memory, multiscale_feats = self.multiscale_text_module(amr_feats=memory,
                                                                   amr_pad_mask=memory_pad_masks, 
                                                                   visual_feats=multiscale_feats) # b s c
        else:
            multiscale_feats = self.multiscale_text_module(tgt=multiscale_feats.permute(1,0,2),
                                                            memory=memory.permute(1,0,2), 
                                                            memory_key_padding_mask=memory_pad_masks,
                                                            pos=None, 
                                                            query_pos=multiscale_feats.permute(1,0,2))[0]
            multiscale_feats = multiscale_feats.permute(1, 0, 2)
        
        if not multiscale_is_flattened:
            multiscale_feats = multiscale_feats.permute(0, 2, 1) # b c s
            multiscale_feats = multiscale_feats.split(scale_length, dim=-1) # list[bt c hw]
            multiscale_feats = [rearrange(mf, 'bt c (h w) -> bt c h w', h=sz[0], w=sz[1]) for mf, sz in zip(multiscale_feats, scale_sizes)]

        if not self.transform_amr_text:
            return multiscale_feats, amr_token_feats, text_feats
        else:
            amr_length = amr_token_feats.shape[1]
            if self.use_text:
                amr_token_feats = memory[:, :amr_length]
                text_feats = memory[:, amr_length:]
            else:
                amr_token_feats = memory
            # bt -> b
            return multiscale_feats, amr_token_feats, text_feats



    def forward(self,
                amrs, amr_token_feats, amr_token_seg_ids,
                is_image_multiscale=False,
                is_video_multiscale=False,
                is_video_frame_query=False,
                text_feats=None, text_pad_masks=None,

                frame_queries=None, # b t nq c
                multiscale_feats=None, multiscale_poses=None, multiscale_is_flattened=None,
                ):
        if is_video_multiscale:
            return self.forward_video_multiscale(multiscale_feats=multiscale_feats,
                                           multiscale_poses=multiscale_poses,
                                           multiscale_is_flattened=multiscale_is_flattened,
                                           amrs=amrs, 
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            text_pad_masks=text_pad_masks)
        elif is_image_multiscale:
            return self.forward_image_multiscale(multiscale_feats=multiscale_feats,
                                           multiscale_poses=multiscale_poses,
                                           multiscale_is_flattened=multiscale_is_flattened,
                                           amrs=amrs, 
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            text_pad_masks=text_pad_masks)
        elif is_video_frame_query:
            return self.forward_video_frameQuery(frame_queries=frame_queries,
                                            amrs=amrs, 
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            text_pad_masks=text_pad_masks)
        pass

@register_fusion
def visual_amrtext_seqseq(configs):
    return Visual_AMRText_SeqSeq(d_model=configs['d_model'],
                                    nheads=configs['nheads'],
                                    dropout=configs['dropout'],
                                    amr_pos_type=configs['amr_pos_type'],
                                    text_pos_type=configs['text_pos_type'],
                                    use_text=configs['use_text'],
                                    transform_amr_text=configs['transform_amr_text'],
                                    dot_or_add=configs['dot_or_add'],)

class perFrameQuery_AMRText(nn.Module):
    def __init__(self, 
                 d_model,
                nheads=8, 
                dropout=0.0,
                no_fusion=False,
                amr_pos_type=['lap', 'depth'],
                text_pos_type=['sin', 'learned'],
                use_text=False,
                transform_amr_text=False,
                dot_or_add='dot') -> None:
        super().__init__()
        self.d_model = d_model
        self.no_fusion = no_fusion
        if no_fusion:
            return
        self.amr_pos_type = amr_pos_type
        self.text_pos_type = text_pos_type
        self.transform_amr_text = transform_amr_text # 是否
        self.use_text = use_text
        if self.transform_amr_text:
            self.query_text_module = CommonCrossAttentionWeights(d_model=d_model, nhead=nheads, dropout=dropout,
                                                                      dot_or_add=dot_or_add)
        else:
            self.query_text_module = VisionLanguageFusionModule(d_model=d_model,
                                                                     nhead=nheads,
                                                                     dropout=dropout)
        if 'depth' in amr_pos_type:
            # 最大深度
            self.depth_amr_pos = nn.Embedding(50, embedding_dim=d_model)
        if 'learned' in text_pos_type:
            self.lrn_text_pos = nn.Embedding(512, embedding_dim=512)
        elif 'sin' in self.text_pos_type:
            self.sin_text_pos = build_position_encoding(position_embedding_name='1d')

    def amr_positional_encoding(self, g, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        if self.amr_pos_type == 'lap':
            import scipy as sp
            import dgl
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(g.number_of_nodes()) - N * A * N

            # # Eigenvectors with numpy
            # EigVal, EigVec = np.linalg.eig(L.toarray())
            # idx = EigVal.argsort() # increasing order
            # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
            # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

            # Eigenvectors with scipy
            #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
            EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
            EigVec = EigVec[:, EigVal.argsort()] # increasing order
            g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

            return g

    def get_amr_poses(self, amrs=None, amr_token_feats=None, amr_token_seg_ids=None):
        # b (v+e)_max c
        amr_poses = torch.zeros_like(amr_token_feats)
        if 'lap' in self.amr_pos_type:
            raise NotImplementedError()
        return amr_poses

    def get_text_poses(self, text_feats=None, text_pad_masks=None):
        # b s c
        text_poses = torch.zeros_like(text_feats)
        if 'sin' in self.text_pos_type:
            sin_pos = self.sin_text_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            text_poses += sin_pos
        return text_poses

    def forward(self,
                frame_queries, # b t nq c
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None):
        if self.no_fusion:
            return frame_queries, text_feats, amr_token_feats

        B, T, nq, _ = frame_queries.shape

        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = amr_token_feats.clone(),    
        memory_pad_masks = amr_token_seg_ids==0
        memory_poses = amr_poses

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, 
                                             text_pad_masks=text_pad_masks)
            memory = torch.cat([memory,text_feats], dim=1)
            memory_pad_masks = torch.cat([memory_pad_masks, text_pad_masks], dim=1)
            memory_poses = torch.cat([memory_poses, text_poses])
        
        memory = memory + memory_poses # b s c
        frame_queries = frame_queries.flatten(1, 2) # b tnq c
        if self.transform_amr_text:
            memory, frame_queries = self.query_text_module(memory=memory,
                                                        memory_pad_masks=memory_pad_masks,
                                                        visual_feats=frame_queries)
        else:
            frame_queries = self.query_text_module(tgt=frame_queries.permute(1,0,2),
                                                        memory=memory.permute(1,0,2), 
                                                        memory_key_padding_mask=memory_pad_masks,
                                                        pos=None, 
                                                        query_pos=None)[0]
            frame_queries = frame_queries.permute(1,0,2)

        frame_queries = rearrange(frame_queries, 'b (t nq) c -> b t nq c',t=T, nq=nq)
        if not self.transform_amr_text:
            return frame_queries, amr_token_feats, text_feats
        else:
            amr_length = amr_token_feats.shape[1]
            if self.use_text:
                amr_token_feats = memory[:, :amr_length]
                text_feats = memory[:, amr_length:]
            else:
                amr_token_feats = memory
            return frame_queries, amr_token_feats, text_feats


@register_fusion
def perFrameQuery_amr_text(configs):
    return perFrameQuery_AMRText(d_model=configs['d_model'],
                                    nheads=configs['nheads'],
                                    dropout=configs['dropout'],
                                    no_fusion=configs['no_fusion'],
                                    amr_pos_type=configs['amr_pos_type'],
                                    text_pos_type=configs['text_pos_type'],
                                    use_text=configs['use_text'],
                                    transform_amr_text=configs['transform_amr_text'],
                                    dot_or_add=configs['dot_or_add'])




# mutlscale + video query
class Fpn2D_multiple(nn.Module):
    def __init__(self, dim, cascaded_scales) -> None:
        """
        cascaded_scales: ['1','4'],  ['1','16'], ['1','32']
        """
        super().__init__()
        self.initial_adapter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(32, dim)
        )
        # from small to big
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert len(cascaded_scales) > 1
        for (temporal_stride, spatial_stride), (next_temporal_stride, next_spatial_stride) \
            in zip(cascaded_scales[:-1], cascaded_scales[1:]):
            assert temporal_stride == next_temporal_stride, 'the temporal stride must be the same for the FPN 2D'
            scale_factor = spatial_stride//next_spatial_stride
            assert scale_factor >= 2
            self.adapters.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.upsamples.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
            self.convs.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(32, dim))
    
    def forward(self, multiscales, multiscales_poses, mask_feats):
        """ bt c h w"""
        cascaded_feats = multiscales + [mask_feats]
        new_fused_feats = []
        new_fused_feats.append(self.initial_adapter(cascaded_feats[0]))
        for idx, large_feat in enumerate(cascaded_feats[1:]): # from small map to large map 
            small_feats = new_fused_feats[-1]
            large_feat = self.adapters[idx](large_feat)
            large_feat += self.upsamples[idx](small_feats) 
            large_feat = self.convs[idx](large_feat)
            large_feat = self.norms[idx](large_feat)

            new_fused_feats.append(large_feat)
        return new_fused_feats[:-1], new_fused_feats[-1]
from torch_geometric.nn.inits import glorot
class perFrameMultiscale_VideoQuery_Text_v1(nn.Module):
    def __init__(self, 
                 d_model,
                 query_proj,
                 mask_feats_proj,
                 multiscale_proj,
                 fusion_dot,
                 fpn,
                 ) -> None:
        super().__init__()
        # proj
        assert query_proj.pop('name') == 'linear'
        self.query_proj = nn.Linear(**query_proj)
        assert mask_feats_proj.pop('name') == 'conv2d'
        self.mask_feats_proj = nn.Conv2d(**mask_feats_proj)
        self.multiscale_proj = nn.ModuleList()
        for proj_config in multiscale_proj:
            assert proj_config.pop('name') == 'conv2d'
            self.multiscale_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                        nn.GroupNorm(32, d_model)))
        self.multiscale_2d_pos = build_position_encoding(position_embedding_name='2d')
        # fusion dot
        assert fusion_dot.pop('name') == 'VisionLanguageFusionModule'
        self.cross_module = VisionLanguageFusionModule(d_model=fusion_dot['d_model'],
                                                    nhead=fusion_dot['nhead'],
                                                    dropout=fusion_dot['dropout'])
        # fpn
        self.fpn = Fpn2D_multiple(dim=fpn['d_model'],
                                  cascaded_scales=fpn['cascaded_scales'])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def proj_query_feats(self, multiscale_feats, mask_feats, video_queries):
        # list[bt c h w]
        # b t c h w
        # b nq c
        batch_size, nf, *_ =  mask_feats.shape
        nq = video_queries.shape[1]
        multiscales = [] # bt c h w
        multiscales_poses = [] # bt c h w
        for lvl, feat in enumerate(multiscale_feats):  
            # bt c h w
            src_proj_l = self.multiscale_proj[lvl](feat.clone())
            pad_mask = torch.zeros_like(feat[:, 0]).bool() # bt h w
            pos = self.multiscale_2d_pos(pad_mask, hidden_dim=src_proj_l.shape[1]) # bt c h w
            multiscales.append(src_proj_l)
            multiscales_poses.append(pos)
        mask_feats = self.mask_feats_proj(mask_feats.flatten(0,1)) # bt c h w
        video_queries = self.query_proj(video_queries)
        return multiscales, multiscales_poses, mask_feats, video_queries
    
    def fusion_with_text(self, multiscale_feats, multiscales_poses, 
                         video_queries, 
                         text_feats, text_pad_masks,amr_feats, amr_pad_masks):
        BT = multiscale_feats[0].shape[0]
        B = len(text_feats)
        T = BT // B
        memory = torch.cat([amr_feats, text_feats], dim=1)
        memory_pad_mask = torch.cat([amr_pad_masks, text_pad_masks], dim=1)
        perFrame_memory = repeat(memory, 'b s c -> (b t) s c',t=T)
        perFrame_memory_pad_mask = repeat(memory_pad_mask, 'b s -> (b t) s',t=T)
        fused_multiscales = []
        for lvl, (feat, poses) in enumerate(zip(multiscale_feats, multiscales_poses)):
            _, _, h, w = feat.shape
            feat = rearrange(feat, 'bt c h w -> (h w) bt c')
            poses = rearrange(poses, 'bt c h w -> (h w) bt c')
            feat, attn_weight = self.cross_module(tgt=feat,
                                                    memory=perFrame_memory.permute(1,0,2), 
                                                    memory_key_padding_mask=perFrame_memory_pad_mask,
                                                    pos=None, query_pos=poses)
            fused_multiscales.append(rearrange(feat, '(h w) bt c -> bt c h w',h=h,w=w))
        
        video_queries = self.cross_module(tgt=video_queries.permute(1,0,2),
                                        memory=memory.permute(1,0,2), 
                                        memory_key_padding_mask=memory_pad_mask,
                                        pos=None, query_pos=None)[0]
        video_queries = video_queries.permute(1,0,2)

        return fused_multiscales, video_queries

    def forward(self, multiscale_feats, mask_feats,
                video_queries,
                text_feats=None, text_pad_masks=None,
                amr_feats=None, amr_pad_masks=None):
        # list[bt c h w]
        # list[b t c h w]
        # b nq c
        # proj 32, 16, 8, 4, query
        B, T, *_ = mask_feats.shape
        BT = multiscale_feats[0].shape[0]
        assert BT == (B * T)
        multiscale_feats, multiscales_poses, mask_feats,video_queries = self.proj_query_feats(multiscale_feats, mask_feats,video_queries)
        # fusion dot
        multiscale_feats, video_queries = self.fusion_with_text(multiscale_feats, multiscales_poses,
                                                                video_queries,text_feats, text_pad_masks,amr_feats, amr_pad_masks)
        # fpn 32, 16, 8, 4; get 4
        multiscale_feats, mask_feats = self.fpn(multiscale_feats, multiscales_poses, mask_feats)
        mask_feats = rearrange(mask_feats, '(b t) c h w -> b t c h w', b=B, t=T)
        # conv query with 4, get mask
        pred_masks = torch.einsum('btchw,bnc->btnhw',mask_feats, video_queries)
        # output query, pred_masks
        return {'obj_queries': video_queries, 'pred_masks': pred_masks}

@register_fusion
def perFrameMultiscale_VideoQuery_text_v1(configs):
    return perFrameMultiscale_VideoQuery_Text_v1(d_model=configs['d_model'],
                                                 query_proj=configs['query_proj'],
                                                 mask_feats_proj=configs['mask_feats_proj'],
                                                 multiscale_proj=configs['multiscale_proj'],
                                                 fusion_dot=configs['fusion_dot'],
                                                 fpn=configs['fpn'])

class perFrameMultiscale_VideoQuery_Text_v2(nn.Module):
    def __init__(self, 
                 d_model,
                 query_proj,
                 mask_feats_proj,
                 fusion_dot,
                 fpn,
                 ) -> None:
        super().__init__()
        # proj
        assert query_proj.pop('name') == 'linear'
        self.query_proj = nn.Linear(**query_proj)
        assert mask_feats_proj.pop('name') == 'conv2d'
        self.mask_feats_proj = nn.Conv2d(**mask_feats_proj)
        self.multiscale_2d_pos = build_position_encoding(position_embedding_name='2d')
        # fusion dot
        assert fusion_dot.pop('name') == 'VisionLanguageFusionModule'
        self.cross_module = VisionLanguageFusionModule(d_model=fusion_dot['d_model'],
                                                    nhead=fusion_dot['nhead'],
                                                    dropout=fusion_dot['dropout'])
        # fpn
        self.fpn = Fpn2D_multiple(dim=fpn['d_model'],
                                  cascaded_scales=fpn['cascaded_scales'])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def proj_query_feats(self, multiscale_feats, mask_feats, video_queries):
        # list[bt c h w]
        # b t c h w
        # b nq c
        batch_size, nf, *_ =  mask_feats.shape
        nq = video_queries.shape[1]
        multiscales = [] # bt c h w
        multiscales_poses = [] # bt c h w
        for lvl, feat in enumerate(multiscale_feats):  
            # bt c h w
            src_proj_l = self.multiscale_proj[lvl](feat.clone())
            pad_mask = torch.zeros_like(feat[:, 0]).bool() # bt h w
            pos = self.multiscale_2d_pos(pad_mask, hidden_dim=src_proj_l.shape[1]) # bt c h w
            multiscales.append(src_proj_l)
            multiscales_poses.append(pos)
        mask_feats = self.mask_feats_proj(mask_feats.flatten(0,1)) # bt c h w
        video_queries = self.query_proj(video_queries)
        return multiscales, multiscales_poses, mask_feats, video_queries
    
    def fusion_with_text(self, multiscale_feats, multiscales_poses, 
                         video_queries, 
                         text_feats, text_pad_masks,amr_feats, amr_pad_masks):
        BT = multiscale_feats[0].shape[0]
        B = len(text_feats)
        T = BT // B
        memory = torch.cat([amr_feats, text_feats], dim=1)
        memory_pad_mask = torch.cat([amr_pad_masks, text_pad_masks], dim=1)
        perFrame_memory = repeat(memory, 'b s c -> (b t) s c',t=T)
        perFrame_memory_pad_mask = repeat(memory_pad_mask, 'b s -> (b t) s',t=T)
        fused_multiscales = []
        for lvl, (feat, poses) in enumerate(zip(multiscale_feats, multiscales_poses)):
            _, _, h, w = feat.shape
            feat = rearrange(feat, 'bt c h w -> (h w) bt c')
            poses = rearrange(poses, 'bt c h w -> (h w) bt c')
            feat, attn_weight = self.cross_module(tgt=feat,
                                                    memory=perFrame_memory.permute(1,0,2), 
                                                    memory_key_padding_mask=perFrame_memory_pad_mask,
                                                    pos=None, query_pos=poses)
            fused_multiscales.append(rearrange(feat, '(h w) bt c -> bt c h w',h=h,w=w))
        
        video_queries = self.cross_module(tgt=video_queries.permute(1,0,2),
                                        memory=memory.permute(1,0,2), 
                                        memory_key_padding_mask=memory_pad_mask,
                                        pos=None, query_pos=None)[0]
        video_queries = video_queries.permute(1,0,2)

        return fused_multiscales, video_queries

    def forward(self, multiscale_feats, mask_feats,
                video_queries,
                text_feats=None, text_pad_masks=None,
                amr_feats=None, amr_pad_masks=None):
        # list[bt c h w]
        # list[b t c h w]
        # b nq c
        # proj 32, 16, 8, 4, query
        B, T, *_ = mask_feats.shape
        BT = multiscale_feats[0].shape[0]
        assert BT == (B * T)
        multiscale_feats, multiscales_poses, mask_feats,video_queries = self.proj_query_feats(multiscale_feats, mask_feats,video_queries)
        # fusion dot
        multiscale_feats, video_queries = self.fusion_with_text(multiscale_feats, multiscales_poses,
                                                                video_queries,text_feats, text_pad_masks,amr_feats, amr_pad_masks)
        # fpn 32, 16, 8, 4; get 4
        multiscale_feats, mask_feats = self.fpn(multiscale_feats, multiscales_poses, mask_feats)
        mask_feats = rearrange(mask_feats, '(b t) c h w -> b t c h w', b=B, t=T)
        # conv query with 4, get mask
        pred_masks = torch.einsum('btchw,bnc->btnhw',mask_feats, video_queries)
        # output query, pred_masks
        return {'obj_queries': video_queries, 'pred_masks': pred_masks}

@register_fusion
def perFrameMultiscale_VideoQuery_text_v2(configs):
    return perFrameMultiscale_VideoQuery_Text_v2(d_model=configs['d_model'],
                                                 query_proj=configs['query_proj'],
                                                 mask_feats_proj=configs['mask_feats_proj'],
                                                 multiscale_proj=configs['multiscale_proj'],
                                                 fusion_dot=configs['fusion_dot'],
                                                 fpn=configs['fpn'])