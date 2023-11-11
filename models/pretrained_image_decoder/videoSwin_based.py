from typing import Any, Optional
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from einops import repeat, reduce, rearrange
from util.misc import NestedTensor
import matplotlib.pyplot as plt
import copy
import torch_geometric.nn as geo_nn
from torch_geometric.data import Batch
from scipy.optimize import linear_sum_assignment

###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
from models.position_encoding import build_position_encoding
from models.model_utils import find_scale_from_multiscales, find_scales_from_multiscales, pad_1d_feats, \
    register_model, get_optimizer, get_total_grad_norm,\
        visualization_for_AMR_V0, zero_module, _get_clones
from models.layers_unimodal_attention import FeatureResizer, CrossAttentionLayer, MLP, SelfAttentionLayer, FFNLayer 
from models.transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from models.transformer import TransformerEncoder, TransformerEncoderLayer
import pycocotools.mask as mask_util
import util.box_ops as box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride, nested_tensor_from_tensor_list_with_stride
from functools import partial

from .msdeformattn import MSDeformAttnPixelDecoder_fusionText
class VideoSwin_ObjDecoder(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 num_classes = None,
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                parsing_encoder={},

                refdecoder={}) -> None:
        super().__init__()

        self.pt_dir = pt_dir
        self.d_model = d_model
        self.max_stride = max_stride
        # video encoder
        from models.video_swin import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=os.path.join(pt_dir, swint_pretrained_path),
                                                running_mode=swint_runnning_mode)
        if swint_freeze:
            for p in self.video_swint.parameters():
                p.requires_grad_(False) 
                 
        assert len(video_projs) == len(video_feat_scales)
        self.video_feat_scales = video_feat_scales
        backbone_channels, backbone_scales = self.video_swint.get_desc()
        assert len(backbone_channels) == len(backbone_scales)
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        from models.position_encoding import build_position_encoding
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        

        self.deform_multiscale_2dencoder = MSDeformAttnPixelDecoder_fusionText(**parsing_encoder)
        self.decoder_query_feats = zero_module(nn.Embedding(refdecoder['nqueries']),d_model)
        self.decoder_query_embed = zero_module(nn.Embedding(refdecoder['nqueries'], d_model))
        self.decoder_used_scales = refdecoder['used_scales']
        self.decoder_conved_scale = refdecoder['conved_scale']
        self.decoder_nlayers = refdecoder['nlayers']
        self.decoder_nqueries = refdecoder['nqueries']
        self.decoder_level_embed = nn.Embedding(len(self.decoder_used_scales), d_model)
        cross_layer = refdecoder['cross_layer']
        assert cross_layer.pop('name') == 'cross_attention'
        self.decoder_cross_video_layers = _get_clones(CrossAttentionLayer(**cross_layer),
                                                                   self.decoder_nlayers)
        self.decoder_nheads = cross_layer['nhead']
        self_layer = refdecoder['self_layer']
        assert self_layer.pop('name') == 'self_attention'
        self.decoder_self_layers = _get_clones(SelfAttentionLayer(**self_layer),
                                                            self.decoder_nlayers)  
        ffn_layer = refdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                            self.decoder_nlayers) 
        # norm, mask out, box, cls, mask
        self.decoder_label_embed = nn.Linear(d_model, num_classes + 1)
        self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.decoder_mask_out_stride = refdecoder['mask_out_stride']
        self.decoder_mask_threshold = refdecoder['mask_threshold']
        self.fusion_module = None
        self.fusion_add_pos = None

    def init_parameters(self,): 
        for proj in self.video_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


    def encode_video(self, samples):
        bb_out = self.video_swint(samples)  
        nf, batch_size, *_ = bb_out[0].tensors.shape
        orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src.clone())
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(orig_pad_mask.float(),
                                             size=src_proj_l.shape[-2:],mode='nearest') > 0.5
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses
    
    def forward_refdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_label = self.decoder_label_embed(decoder_output)  # bt n 2
        outputs_box = self.decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_label, outputs_mask, outputs_box, attn_mask, decoder_output
 
    def forward(self, samples,
                      amrs=None, amr_token_feats=None, amr_token_seg_ids=None, text_feats=None, text_pad_masks=None):
        batch_size = len(amrs)
        samples.tensors = rearrange(samples.tensors, '(b t) c h w -> t b c h w',b=batch_size)
        samples.mask = rearrange(samples.mask, '(b t) h w -> t b h w',b=batch_size)
        nf, batch_size = samples.tensors.shape[0], samples.tensors.shape[1]
        bt = nf * batch_size
        device = samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        
        if self.fusion_module is not None:
            multiscales, amr_token_feats, text_feats = self.fusion_module(multiscale_feats=multiscales, 
                                                        multiscale_poses=multiscales_poses,
                                                        multiscale_is_flattened=False,
                                                        is_video_multiscale=True,
                                                        amrs=amrs, 
                                                        amr_text_add_pos=self.fusion_add_pos,
                                                        amr_token_feats=amr_token_feats,
                                                        amr_token_seg_ids=amr_token_seg_ids, 
                                                        text_feats=text_feats, 
                                                        text_pad_masks=text_pad_masks)
            
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        mask_features_2d, mask_features_3d, _, multiscales, amr_token_feats, text_feats \
            = self.deform_multiscale_2dencoder.forward_features(multiscales,
                                                                amrs=amrs, 
                                                                amr_token_feats=amr_token_feats,
                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                text_feats=text_feats, 
                                                                text_pad_masks=text_pad_masks)
        
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in multiscales]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in multiscales_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in multiscales_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c

        decoder_outputs = []
        decoder_pred_masks = []
        decoder_pred_logits = []
        decoder_pred_boxes = []

        out_labels, out_mask, out_box, attn_mask, dec_query_out = self.forward_refdecoder_heads(output, mask_features_2d, attn_mask_target_size=size_list[0])
        decoder_outputs.append(dec_query_out)
        decoder_pred_masks.append(out_mask)
        decoder_pred_logits.append(out_labels)
        decoder_pred_boxes.append(out_box)

        for i in range(self.decoder_nlayers):
            level_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.decoder_ffn_layers[i](
                output # n bt c
            )
            out_labels, out_mask, out_box, attn_mask, dec_query_out = self.forward_refdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_outputs.append(dec_query_out)
            decoder_pred_masks.append(out_mask)
            decoder_pred_logits.append(out_labels)
            decoder_pred_boxes.append(out_box)


        return {'obj_queries': decoder_outputs,
                'pred_masks': decoder_pred_masks,
                'multiscale_feats': None,
                'mask_features': mask_features_3d}

from .pt_obj_2D_decoder import register_pt_obj_2d_decoder

@register_pt_obj_2d_decoder
def vswin(configs, pt_dir):
    return VideoSwin_ObjDecoder(
        d_model=configs['d_model'],
        max_stride=configs['max_stride'],
        pt_dir=pt_dir,
        num_classes=configs['num_classes'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swin_freeze'],
        video_projs=configs['video_projs'],
        parsing_encoder=configs['parsing_encoder'],
        refdecoder=configs['decoder'],
    )
