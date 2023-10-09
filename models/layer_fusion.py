
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
                text_pad_mask=None,
                text_feats=None):
        memory = amr_feats
        memory_key_padding_mask = amr_pad_masks
        memory_pos = torch.zeros_like(amr_feats)

        if text_feats is not None:
            assert text_pad_mask is not None
            text_pos = self.text1d_pos(text_pad_mask, hidden_dim=text_feats.shape[-1])
            memory = torch.cat([memory, text_feats], dim=1)
            memory_key_padding_mask = torch.cat([memory_key_padding_mask, text_pad_mask], dim=1)
            memory_pos = torch.cat([memory_pos,text_pos], dim=1)

        text_pos = self.text1d_pos(text_pad_mask, hidden_dim=text_feats.shape[-1])
        query_feat =  self.cross_module(tgt=query_feat.permute(1,0,2),
                                        memory=memory.permute(1,0,2), 
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        pos=memory_pos.permute(1,0,2), 
                                        query_pos=None)[0]
        return query_feat.permute(1,0,2), amr_feats, text_feats
    
@register_fusion
def vidquery_text_v1(configs):
    return VidQuery_Text_v1(configs)


# b nq c, b s c, 两个都转换
from .transformer import TransformerEncoder, TransformerEncoderLayer
class VidQuery_Text_v2(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.self_module = TransformerEncoder(TransformerEncoderLayer(d_model=configs['d_model'],
                                                                      nheads=configs['nheads'],
                                                                      dim_feedforward=configs['dim_ff'],
                                                                      dropout=configs['dropout'],
                                                                      activation=configs['act']), 
                                                num_layers=configs['num_layers'])
        self.text1d_pos = build_position_encoding(position_embedding_name='1d')
    
    def forward(self, 
                query_feat, amr_feats, amr_pad_masks,
                text_pad_mask=None,
                text_feats=None):
        lquery, lamr = query_feat.shape[1], amr_feats.shape[1]
        src = torch.cat([query_feat, amr_feats], dim=1)
        src_pad_mask = torch.cat([torch.zeros_like(query_feat[:, :, 0]).bool(), amr_pad_masks], dim=1)
        src_pos = torch.zeros_like(src)

        if text_feats is not None:
            assert text_pad_mask is not None
            text_pos = self.text1d_pos(text_pad_mask, hidden_dim=text_feats.shape[-1])
            src = torch.cat([src, text_feats], dim=1)
            src_pad_mask = torch.cat([src_pad_mask, text_pad_mask], dim=1)
            src_pos = torch.cat([src_pos, text_pos], dim=1)
            ltext = text_feats.shape[1]
        else:
            ltext = 0

        src =  self.self_module(src=src.permute(1,0,2),
                                mask=None,
                                src_key_padding_mask=src_pad_mask,
                                pos=src_pos.permute(1,0,2))[0]
        src = src.permute(1,0,2)

        return src.split([lquery, lamr, ltext], dim=1)


@register_fusion
def vidquery_text_v2(configs):
    return VidQuery_Text_v2(configs)