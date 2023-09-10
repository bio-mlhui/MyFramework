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
from .layers_unimodal_attention import CrossAttentionLayer, FFNLayer, SelfAttentionLayer


# attention layers需要知道在做哪个任务, 因为有些layers对于某些任务不用计算
# 并且假设每次forward attention layers只做一个任务

_multimodal_attention_encoder_entrypoints = {}
def register_multimodal_attention_encoder(fn):
    selfattn_encoder_name = fn.__name__
    _multimodal_attention_encoder_entrypoints[selfattn_encoder_name] = fn
    return fn

def multimodal_attention_encoder_entrypoints(selfattn_encoder_name):
    try:
        return _multimodal_attention_encoder_entrypoints[selfattn_encoder_name]
    except KeyError as e:
        print(f'Self-Attention Encoder {selfattn_encoder_name} not found')



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class VideoText_NoFusion(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    
    def forward(self,
                video_feat, video_pad_mask, video_pos,
                text_feat, text_pad_mask, attention_mask,
                mlm=False, mvm=False, refer=False):
        
        return video_feat, text_feat
    
@register_multimodal_attention_encoder
def videotext_no_fusion(configs, d_model):
    return VideoText_NoFusion()

class VideoText_Seperate_ConcateSelf(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                 dim_feedforward,
                 dropout,
                 num_layers=6) -> None:
        super().__init__()
        from .transformer import TransformerEncoder, TransformerEncoderLayer
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model,
                                    nheads=nheads,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout,
                                    activation='relu',
                                    normalize_before=False),
            num_layers=num_layers
        )
    def forward(self,
        video_feat, video_pad_mask, video_pos,
        text_feat, text_pad_mask, attention_mask,
        mlm=False, mvm=False, refer=False):
        """
        video_feat: b t c h w
        video_pad_mask: b t h w
        video_pos: b t c h w
        text_feat: b s c
        text_pad_mask: b s
        attention
        """
        pass

"""
video特征cross text的特征
text特征cross video的特征
"""
class VideoText_SeperateCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout,
                 # important
                 num_layers) -> None:
        super().__init__()
        self.video_cross_text_attn = nn.ModuleList()
        self.video_cross_text_lln = nn.ModuleList()
        for _ in range(num_layers):
            self.video_cross_text_attn.append(CrossAttentionLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dropout=dropout,))
            self.video_cross_text_lln.append(FFNLayer(d_model=d_model,
                                                      dropout=dropout))
            
        self.num_heads = nhead
        self.num_layers = num_layers

        self.text_cross_video_attn = nn.ModuleList()
        self.text_cross_video_lln = nn.ModuleList()
        for _ in range(num_layers):
            self.text_cross_video_attn.append(CrossAttentionLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dropout=dropout,))
            self.text_cross_video_lln.append(FFNLayer(d_model=d_model,
                                                      dropout=dropout))
    
    def forward(self,
                video_feat, video_pad_mask, video_pos,
                text_feat, text_pad_mask, attention_mask,
                mlm=False, mvm=False, refer=False):
        assert (mlm and (not mvm) and (not refer)) or (mvm and (not mlm) and (not refer)) or (refer and (not mlm) and (not mvm))

        batch_size, nf, _, h, w = video_feat.shape
        token_max = text_feat.shape[1] # b s c
        video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
        video_pad_mask = rearrange(video_pad_mask, 'b t h w -> b (t h w)')
        video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
        text_feat = rearrange(text_feat, 'b s c -> s b c')

        if mvm or refer: 
            if attention_mask is not None:
                vid_cross_text_am = repeat(attention_mask, 'b thw s -> (b head) thw s', head=self.num_heads)
            else:
                vid_cross_text_am = None

            fused_video_feats = video_feat.clone()
            for i in range(self.num_layers):
                fused_video_feats = self.video_cross_text_attn[i](
                    tgt=fused_video_feats, 
                    memory=text_feat.clone(),
                    memory_mask=vid_cross_text_am,
                    memory_key_padding_mask=text_pad_mask.clone(),
                    pos= None,
                    query_pos=video_pos.clone())
                
                fused_video_feats = self.video_cross_text_lln[i](
                    fused_video_feats
                )
            fused_video_feats = rearrange(fused_video_feats, '(t h w) b c -> b t c h w', t=nf, h=h,w=w)
            if mvm:
                return fused_video_feats

        if mlm or refer:
            if attention_mask is not None:
                text_cross_vid_am = repeat(attention_mask, 'b s thw -> (b head) s thw', head=self.num_heads)
            else:
                text_cross_vid_am = None

            fused_text_feats = text_feat.clone()
            for i in range(self.num_layers):
                fused_text_feats = self.text_cross_video_attn[i](
                    tgt=fused_text_feats,  # s b c
                    memory=video_feat.clone(),  # (t h w) b c
                    memory_mask= text_cross_vid_am,  # b*h s thw
                    memory_key_padding_mask=None, # no key pad
                    pos=video_pos.clone(),
                    query_pos=None
                )
                fused_text_feats = self.text_cross_video_lln[i](
                    fused_text_feats
                )

            fused_text_feats = rearrange(fused_text_feats, 's b c -> b s c')
            if mlm:
                return fused_text_feats
            
        if refer:
            return fused_video_feats, fused_text_feats
        
        raise ValueError()

@register_multimodal_attention_encoder
def videotext_seperate_cross(configs, d_model):
    return VideoText_SeperateCrossAttention(d_model=d_model,
                                       nhead=configs.nhead,
                                       dropout=configs.dropout,
                                        # important
                                       num_layers=configs.num_layers,)

"""
video特征cross text的特征
text特征cross video的特征 之后再加上video
"""
class VideoText_Seperate_VideoCross_TextCrossSelf(nn.Module):
    def __init__(self, d_model, nhead, dropout,
                 # important
                 num_video_cross_text_layers,
                 num_text_cross_video_layers) -> None:
        super().__init__()
        self.video_cross_text_attn = nn.ModuleList()
        self.video_cross_text_lln = nn.ModuleList()
        for _ in range(num_video_cross_text_layers):
            self.video_cross_text_attn.append(CrossAttentionLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dropout=dropout,))
            self.video_cross_text_lln.append(FFNLayer(d_model=d_model,
                                                      dropout=dropout))
            
        self.num_heads = nhead
        self.num_video_cross_text_layers = num_video_cross_text_layers


        self.text_cross_video_attn = nn.ModuleList()
        self.text_self_attn = nn.ModuleList()
        self.text_cross_video_lln = nn.ModuleList()
        for _ in range(num_text_cross_video_layers):
            self.text_cross_video_attn.append(CrossAttentionLayer(d_model=d_model,
                                                         nhead=nhead,
                                                         dropout=dropout,))
            self.text_self_attn.append(SelfAttentionLayer(d_model=d_model,
                                                          nhead=nhead,dropout=dropout))
            self.text_cross_video_lln.append(FFNLayer(d_model=d_model,
                                                      dropout=dropout))
        self.num_text_cross_video_layers = num_text_cross_video_layers
    def forward(self,
                video_feat, video_pad_mask, video_pos,
                text_feat, text_pad_mask, attention_mask,
                mlm=False, mvm=False, refer=False):
        assert (mlm and (not mvm) and (not refer)) or (mvm and (not mlm) and (not refer)) or (refer and (not mlm) and (not mvm))

        batch_size, nf, _, h, w = video_feat.shape
        token_max = text_feat.shape[1] # b s c
        video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
        video_pad_mask = rearrange(video_pad_mask, 'b t h w -> b (t h w)')
        video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
        text_feat = rearrange(text_feat, 'b s c -> s b c')

        if mvm or refer: 
            if attention_mask is not None:
                vid_cross_text_am = repeat(attention_mask, 'b thw s -> (b head) thw s', head=self.num_heads)
            else:
                vid_cross_text_am = None

            fused_video_feats = video_feat.clone()
            for i in range(self.num_video_cross_text_layers):
                fused_video_feats = self.video_cross_text_attn[i](
                    tgt=fused_video_feats, 
                    memory=text_feat.clone(),
                    memory_mask=vid_cross_text_am,
                    memory_key_padding_mask=text_pad_mask.clone(),
                    pos= None,
                    query_pos=video_pos.clone())
                
                fused_video_feats = self.video_cross_text_lln[i](
                    fused_video_feats
                )
            fused_video_feats = rearrange(fused_video_feats, '(t h w) b c -> b t c h w', t=nf, h=h,w=w)
            if mvm:
                return fused_video_feats

        if mlm or refer:
            if attention_mask is not None:
                text_cross_vid_am = repeat(attention_mask, 'b s thw -> (b head) s thw', head=self.num_heads)
            else:
                text_cross_vid_am = None

            fused_text_feats = text_feat.clone()
            for i in range(self.num_text_cross_video_layers):
                fused_text_feats = self.text_cross_video_attn[i](
                    tgt=fused_text_feats,  # s b c
                    memory=video_feat.clone(),  # (t h w) b c
                    memory_mask= text_cross_vid_am,  # b*h s thw
                    memory_key_padding_mask=None, # no key pad
                    pos=video_pos.clone(),
                    query_pos=None
                )
                fused_text_feats = self.text_self_attn[i](
                    tgt=fused_text_feats,
                     tgt_mask=None,
                     tgt_key_padding_mask=text_pad_mask.clone(),
                     query_pos=None,
                )
                fused_text_feats = self.text_cross_video_lln[i](
                    fused_text_feats
                )

            fused_text_feats = rearrange(fused_text_feats, 's b c -> b s c')
            if mlm:
                return fused_text_feats
            
        if refer:
            return fused_video_feats, fused_text_feats
        
        raise ValueError()

@register_multimodal_attention_encoder
def videotext_seperate_videocross_textcrossself(configs, d_model):
    return VideoText_Seperate_VideoCross_TextCrossSelf(d_model=d_model,
                                       nhead=configs.nhead,
                                       dropout=configs.dropout,
                                        # important
                                        num_text_cross_video_layers=configs.num_text_cross_video_layers,
                                        num_video_cross_text_layers=configs.num_video_cross_text_layers)

@register_multimodal_attention_encoder
def neighborhood(configs, d_model):
    return 



def build_sin2d_pos(b, h, w, dim, normalize=True, temperature=10000, scale = 2 * math.pi):
    
    num_pos_feats = dim // 2

    not_mask = torch.ones([b, h, w]).float()
    
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0,3,1,2)
    return pos # b c h w

def build_sin1d_pos(b, s, dim, normalize=True, temperature=10000, scale = 2 * math.pi):
    num_pos_feats = dim
    not_mask = torch.ones([b, s]).float()
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, T]
    if normalize:
        eps = 1e-6
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats).float()
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t  # [B, T, C]
    # n,c,t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = pos_x.permute(0, 2, 1)    # [B, C, T]
    return pos


class QueryVideoText_DividedSelfAttention(nn.Module):
    def __init__(self,
                 d_model, 
                 nheads,
                 dropout,
                 normalize_before=False,
                 
                 repeated_spatial_fuse_strategy=None, # mean/multiply/sin_pos/learn
                 max_h=None, max_w=None,
                 repeated_temporal_fuse_strategy=None, # 
                 max_len=None,
                 
                 first='temporal',):
        
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.time_norm = nn.LayerNorm(d_model)
            
        self.spatial_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)

        self.spatial_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.nheads = nheads
        self.normalize_before = normalize_before
        self.first = first
        
        
        self.repeated_spatial_fuse_strategy = repeated_spatial_fuse_strategy
        self.repeated_temporal_fuse_strategy = repeated_temporal_fuse_strategy
        
        if repeated_spatial_fuse_strategy == 'sin_pos':
            self.sin_spatial_pos = build_position_encoding(position_embedding_name='2d')
        elif repeated_spatial_fuse_strategy == 'learn':
            self.lrn_spatial_pos = nn.Parameter(data=torch.zeros(max_h, max_w, d_model), requires_grad=True)

        if repeated_temporal_fuse_strategy == 'sin_pos':
            self.sin_temporal_pos = build_position_encoding(position_embedding_name='1d')
        elif repeated_temporal_fuse_strategy == 'learn':
            self.lrn_temporal_pos = nn.Parameter(data=torch.zeros(max_len, d_model),requires_grad=True)
            
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
       

    def forward(self,
                query_feats, query_pos,
                video_feats, video_pos,
                text_feats, text_pad_mask,
                self_attention_mask,
                ):
        
        if self.first == 'spatial':
            query_feats, video_feats, text_feats = self.forward_spatial_attention(
                video_feats=video_feats,
                video_pos=video_pos,
                object_queries=query_feats,
                object_query_pos=query_pos,
                text_tokens=text_feats,
                text_pad_mask=text_pad_mask,
                self_attention_mask=self_attention_mask
            )
            
            query_feats, video_feats, text_feats = self.forward_temporal_attention(
                video_feats=video_feats,
                video_pos=video_pos,
                object_queries=query_feats,
                object_query_pos=query_pos,
                text_tokens=text_feats,
                text_pad_mask=text_pad_mask,
                self_attention_mask=self_attention_mask                
            )
        elif self.first == 'temporal':
            query_feats, video_feats, text_feats = self.forward_temporal_attention(
                video_feats=video_feats,
                video_pos=video_pos,
                object_queries=query_feats,
                object_query_pos=query_pos,
                text_tokens=text_feats,
                text_pad_mask=text_pad_mask,
                self_attention_mask=self_attention_mask
            )
            
            query_feats, video_feats, text_feats = self.forward_spatial_attention(
                video_feats=video_feats,
                video_pos=video_pos,
                object_queries=query_feats,
                object_query_pos=query_pos,
                text_tokens=text_feats,
                text_pad_mask=text_pad_mask,
                self_attention_mask=self_attention_mask                
            )            
            

    def rearrange_temporal_self_attention_masks(self, self_attention_masks,
                                                nq, max, nf, h, w, bs, device):
        # bhw nq+nf+max nq+nf+max
        output = torch.ones([bs*h*w, nq+nf+max, nq+nf+max], device=device).bool()
        
        query_video_sam = rearrange(self_attention_masks[0], 'b nq (t h w) -> (b h w) nq t',t=nf, h=h,w=w)
        token_video_sam = rearrange(self_attention_masks[1], 'b max (t h w) -> (b h w) max t',t=nf, h=h, w=w)
        
        output[:, :nq, nq:(nq+nf)] = query_video_sam
        output[:, (nq+nf):, nq:(nq+nf)] = token_video_sam
        
        
        video_query_sam = rearrange(self_attention_masks[2], 'b (t h w) nq -> (b h w) t nq', t=nf, h=h, w=w)
        token_query_sam = repeat(self_attention_masks[3], 'b max nq -> (b h w) max nq',h=h,w=w)
        output[:, nq:(nq+nf), :nq] = video_query_sam
        output[:, (nq+nf):, :nq] = token_query_sam
        
        
        video_token_sam = rearrange(self_attention_masks[4], 'b (t h w) max -> (b h w) t max',t=nf,h=h,w=w)
        query_token_sam = repeat(self_attention_masks[5], 'b nq max -> (b h w) nq max',h=h,w=w)
        
        output[:, nq:(nq+nf), (nq+nf):] = video_token_sam
        output[:, :nq, (nq+nf):] = query_token_sam
        
        return output
       
    def forward_temporal_attention(self,
                               video_feats, video_pos,
                               object_queries, object_query_pos,
                               text_tokens, text_pos, text_pad_mask,
                               self_attention_mask):
        """
        video_feats: b t c h w
        
        object_queries: b nq c
        
        text_tokens: b max c
        
        self_attention_masks: [b nq (thw); b max (thw); 
                                b thw nq; b max nq;
                                b thw max; b nq max]
        
        """
        bs, nf, _,  h, w = video_feats.shape[-2:]
        nq = object_queries.shape[1]
        max = text_tokens.shape[1]
        
        if self_attention_mask is not None:
            sam = self.rearrange_temporal_self_attention_masks(self_attention_mask,
                                                               nq=nq, max=max, nf=nf, h=h,w=w, bs=bs, device=video_feats.device)
            # (b h w) (nq+t+max) (nq+t+max)
            sam = repeat(sam, 'bhw s1 s2 -> (bhw heads) s1 s2', heads=self.nheads,)
        else:
            sam = None
        
        video_feats = rearrange(video_feats, 'b t c h w -> t (b h w) c')
        video_pos = rearrange(video_pos, 'b t c h w -> t (b h w) c')
        video_pad_mask = torch.zeros([video_feats.shape[1], video_feats.shape[0]], 
                                     device=video_feats.device).bool() # (b h w) t
        
        object_queries = repeat(object_queries, 'b nq c -> nq (b h w) c', h=h, w=w)
        object_query_pos = repeat(object_query_pos, 'b nq c -> nq (b h w) c',h=h, w=w)
        object_query_pad_mask = torch.zeros([object_queries.shape[1], object_queries.shape[0]], 
                                            device=object_queries.device).bool() # (b h w) nq
        
        text_tokens = repeat(text_tokens, 'b s c -> s (b h w) c', h=h, w=w)
        text_pos = repeat(text_pos, 'b s c -> s (b h w) c', h=h, w=w)
        text_pad_mask = repeat(text_pad_mask, 'b s -> (b h w) s', h=h, w=w)
        
        query = torch.cat([object_queries, video_feats, text_tokens], dim=0) 
        
        query_pos = torch.cat([object_query_pos, video_pos, text_pos], dim=0)
        
        query_pad_mask = torch.cat([video_pad_mask, object_query_pad_mask, text_pad_mask], dim=1)
        
        q = k = self.with_pos_embed(query, query_pos), 
        query2, weights = self.time_attn(query=q,
                              key=k,
                              value=query, attn_mask=sam,
                              key_padding_mask=query_pad_mask,
                              )
        query = self.time_norm(query + self.dropout(query2))
        

        object_out = query[:nq, ...]  # nq bhw c
        video_out = query[nq:(nq+nf), ...] # t bhw c
        token_out = query[(nq+nf):, ...] # max bhw c
        
        # (b t) hw c
        # (b t) nq c
        # (b t) max c
        
        object_out = self.fuse_repeated_spatial(object_out, b=bs, h=h,w=w)
        object_out = rearrange(object_out, 's b c -> b s c')
        
        token_out = self.fuse_repeated_spatial(token_out, b=bs, h=h, w=w)
        token_out = rearrange(token_out, 's b c -> b s c')
        
        video_out = rearrange(video_out, 't (b h w) c -> b t c h w', b=bs, h=h, w=w)
        
        return object_out, video_out, token_out

    def fuse_repeated_spatial(self, out, b, h, w):
        """
        s (b h w) c
        """ 
        if self.repeated_spatial_fuse_strategy == 'mean':
            return reduce(out, 's (b h w) c -> s b c', h=h,w=w, reduction='mean')
        
        elif self.repeated_spatial_fuse_strategy == 'sin_pos':
            pos = build_sin2d_pos(b=b, h=h,w=w,dim=out.shape[-1]).to(device=out.device) # b c h w
            pos = repeat(pos, 'b c h w -> s (b h w) c', s=out.shape[0])
            
            out = out * pos
            return reduce(out, 's (b h w) c -> s b c', h=h,w=w, reduction='mean')
        
        elif self.repeated_spatial_fuse_strategy == 'learn':
            pos = repeat(self.lrn_spatial_pos[:h, :w, :], 'h w c -> s (b h w) c',s=out.shape[0], b=b)
            out = out * pos
            return reduce(out, 's (b h w) c -> s b c', h=h,w=w, reduction='mean')            

        else:
            raise ValueError()


    def rearrange_spatial_self_attention_masks(self, self_attention_masks,
                                       nq, max, nf, h, w, bs, device):
        #bt (nq+hw+max) (nq+hw+max)
        hw = h*w
        output = torch.ones([bs*nf, nq+hw+max, nq+hw+max], device=device).bool()
        
        query_video_sam = rearrange(self_attention_masks[0], 'b nq (t h w) -> (b t) nq (h w)',t=nf, h=h,w=w)
        token_video_sam = rearrange(self_attention_masks[1], 'b max (t h w) -> (b t) max (h w)', t=nf, h=h, w=w)
        
        output[:, :nq, nq:(nq+hw)] = query_video_sam
        output[:, (nq+hw):, nq:(nq+hw)] = token_video_sam
        
        
        video_query_sam = rearrange(self_attention_masks[2], 'b (t h w) nq -> (b t) (h w) nq', t=nf, h=h, w=w)
        token_query_sam = repeat(self_attention_masks[3], 'b max nq -> (b t) max nq',t=nf)
        output[:, nq:(nq+hw) ,:nq] = video_query_sam
        output[:, (nq+hw):, :nq] = token_query_sam
        
        
        video_token_sam = rearrange(self_attention_masks[4], 'b (t h w) max -> (b t) (h w) max',t=nf,h=h,w=w)
        query_token_sam = repeat(self_attention_masks[5], 'b nq max -> (b t) nq max',t=nf)
        
        output[:, nq:(nq+hw), (nq+hw):] = video_token_sam
        output[:, :nq, (nq+hw):] = query_token_sam
        
        return output
 
    def forward_spatial_attention(self,
                               video_feats, video_pos,
                               object_queries, object_query_pos,
                               text_tokens, text_pos, text_pad_mask,
                               self_attention_mask):
        """
        video_feats: b t c h w
        
        object_queries: b nq c
        
        text_tokens: b max c
        
        self_attention_masks: [b nq (thw); b max (thw); 
                                b thw nq; b max nq;
                                b thw max; b nq max]
        
        """
        bs, nf, _,  h, w = video_feats.shape[-2:]
        nq = object_queries.shape[1]
        max = text_tokens.shape[1]
        
        if self_attention_mask is not None:
            sam = self.rearrange_spatial_self_attention_masks(self_attention_mask,
                                                               nq=nq, max=max, nf=nf, h=h,w=w, bs=bs, device=video_feats.device)
            # (b t) (nq+hw+max) (nq+hw+max)
            sam = repeat(sam, 'bt s1 s2 -> (bt heads) s1 s2', heads=self.nheads,)
        else:
            sam = None
        
        video_feats = rearrange(video_feats, 'b t c h w -> (h w) (b t) c')
        video_pos = rearrange(video_pos, 'b t c h w ->(h w) (b t) c')
        video_pad_mask = torch.zeros([video_feats.shape[1], video_feats.shape[0]], 
                                     device=video_feats.device).bool() # (b t) (h w)
        
        object_queries = repeat(object_queries, 'b nq c -> nq (b t) c', t=nf)
        object_query_pos = repeat(object_query_pos, 'b nq c -> nq (b t) c', t=nf)
        object_query_pad_mask = torch.zeros([object_queries.shape[1], object_queries.shape[0]], 
                                            device=object_queries.device).bool() # (b t) nq
        
        text_tokens = repeat(text_tokens, 'b s c -> s (b t) c', t=nf)
        text_pos = repeat(text_pos, 'b s c -> s (b t) c', t=nf)
        text_pad_mask = repeat(text_pad_mask, 'b s -> (b t) s', t=nf)
        
        query = torch.cat([object_queries, video_feats, text_tokens], dim=0) 
        
        query_pos = torch.cat([object_query_pos, video_pos, text_pos], dim=0)
        
        query_pad_mask = torch.cat([video_pad_mask, object_query_pad_mask, text_pad_mask], dim=1)
        
        q = k = self.with_pos_embed(query, query_pos), 
        query2, weights = self.time_attn(query=q,
                              key=k,
                              value=query, attn_mask=sam,
                              key_padding_mask=query_pad_mask)
    
        query = self.spatial_norm(query + self.dropout(query2))
        

        object_out = query[:nq, ...]  # nq bt c
        video_out = query[nq:(nq+h*w), ...] #  hw bt c
        token_out = query[(nq+nf):, ...] # max bt c
        
        object_out = self.fuse_repeated_temporal(object_out, b=bs, h=h,w=w)
        object_out = rearrange(object_out, 's b c -> b s c')
        token_out = self.fuse_repeated_temporal(token_out, b=bs, h=h, w=w)
        token_out = rearrange(token_out, 's b c -> b s c')
        
        video_out = rearrange(video_out, '(h w) (b t) c -> b t c h w', b=bs, h=h, w=w, t=nf)
        
        return object_out, video_out, token_out
  
    def fuse_repeated_temporal(self, out, b, t):
        """
        s (b t) c
        """ 
        if self.repeated_temporal_fuse_strategy == 'mean':
            return reduce(out, 's (b t) c -> s b c', t, reduction='mean')
        
        elif self.repeated_temporal_fuse_strategy == 'sin_pos':
            pos = build_sin1d_pos(b=b, t=t, dim=out.shape[-1])
            pos = repeat(pos, 'b c t -> s (b t) c', s=out.shape[0])
            
            out = out * pos
            return reduce(out, 's (b t) c -> s b c', t, reduction='mean')
        
        elif self.repeated_spatial_fuse_strategy == 'learn':
            pos = repeat(self.lrn_temporal_pos[:t, :], 't c -> s (b t) c',s=out.shape[0], b=b)
            out = out * pos
            return reduce(out, 's (b t) c -> s b c', t, reduction='mean')            

        else:
            raise ValueError()     
        
class QueryVideoText_MotionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
      
class QueryVideoText_SwinAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class VideoQueryText_BottleNeck_Attention(nn.Module):
    pass


class DivVideo_Text_SelfAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 nheads, 
                 dropout=0.0,
                 pre_norm=False):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.time_norm = nn.LayerNorm(d_model)
            
        self.spatial_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.spatial_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.normalize_before = pre_norm

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                video_feat, video_pad_mask, video_pos,
                text_feat, text_pad_mask, attention_mask=None):
        """
        video_feat: b t c h w
        video_pad_mask: b t h w
        video_pos: b t c h w
        text_feat: b s c
        text_pad_mask: b s
        attentioN_mask: b (t h w) s
        """
        # rearrange
        batch_size, nf, _, h, w = video_feat.shape
        F.multi_head_attention_forward
        video_feat = rearrange(video_feat, 'b t c h w -> t (h w b) c')
        video_pad_mask = rearrange(video_pad_mask, 'b t h w -> (h w b) t')
        video_pos = rearrange(video_pos, 'b t c h w -> t (h w b) c')
        
        text_feat = repeat(text_feat, 'b s c -> s (h w b) c', h=h, w=w)
        text_pad_mask = repeat(text_pad_mask, 'b s -> s (h w b)',h=h, w=w)
        
        if attention_mask is not None:
            attention_mask = rearrange(attention_mask, 'b (t h w) s -> (h w b) t s',t=nf,h=h,w=w)
            attention_mask = None

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
                    query_pos: Optional[Tensor] = None):
        
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


    def forward(self,
                video_feat, video_pad_mask, video_pos,
                text_feat, text_pad_mask, attention_mask=None):

        pass

@register_multimodal_attention_encoder
def videotext_divtemporal(configs, d_model):
    attn_layer = DivVideo_Text_SelfAttention(
        d_model=d_model,
        nhead=configs['nheads'],
        dropout=configs['dropout'],
        activation=configs['activation'],
        pre_norm=configs['pre_norm'],
    )
    return _get_clones(attn_layer, N=configs['num_layers'])
