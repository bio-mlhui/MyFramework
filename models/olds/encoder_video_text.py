import torch
from einops import rearrange, repeat, reduce
from typing import Any, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from .layers_unimodal_attention import *



def build_video_clip(configs):
    return None

_video_text_encoder_entrypoints = {}

def register_video_text_encoder(fn):
    video_text_encoder_name = fn.__name__
    _video_text_encoder_entrypoints[video_text_encoder_name] = fn
    return fn

def video_text_encoder_entrypoints(video_text_encoder_name):
    try:
        return _video_text_encoder_entrypoints[video_text_encoder_name]
    except KeyError as e:
        print(f'video_text_encoder {video_text_encoder_name} not found')

# return1: (video_feats, video_pad, video_pos, stride4), (video_masked_gt,)
# return2: (tokenized_feats,), (text_feats, text_pad_mask), (text_masked_gt)


class VisualTextFeats_Queries(nn.Module):
    def __init__(self, 
                 num_queries,
                 learn_feat,
                 num_layers,
                 d_model,
                 
                 scale_encoder_configs,) -> None:
        super().__init__()
        
        self.query_feats = nn.Embedding(num_queries, embedding_dim=d_model)
        self.query_pos = nn.Embedding(num_queries, embedding_dim=d_model)
        
        self.query_type_embed = nn.Embedding(3, embedding_dim=d_model, scale_grad_by_freq=True)
            
        self.layers = nn.ModuleList()  
        
        # 每个模态的encoder只负责MLM/MVM, 不负责Mask Referring Modelling, 所以要在这个类中实现MRM
        self.mask_embed_decoder = MLP(d_model, d_model, d_model, 3)
        
        self.build_scale_encoder(scale_encoder_configs, d_model=d_model)
    
    def build_scale_encoder(self, configs, d_model):
        pass
        
    
    def forward(self, visual_feats, visual_pos, visual_pad_mask,
                text_feats, text_pad_mask,
                ):
        # return final outputs, 
        pass
    
    def forward_mqm(self,
                    query_outputs, 
                    mask_feats,
                    gt_labels, ):
        """
        query_outputs: b nq c h w
        mask_feats: b t c h w 
        gt_labels: list[ni t h w], b
        """

@register_video_text_encoder
def visualtextfeats_queries(configs, d_model):
    return None




