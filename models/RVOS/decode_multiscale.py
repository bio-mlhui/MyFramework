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

import logging
###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
import data_schedule.utils.box_ops as box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride
from functools import partial
from util.misc import to_device
from models.utils.visualize_amr import save_model_output
from models.registry import register_model

###########################################################################
# amr without variable
###########################################################################

__all__ = ['text_v0', 'text_v0linamr']


###########################################################################
# 相比v0, 用object decoder
###########################################################################
# 改成3d proj
# 改成matching
# mrbart的encoding训练
# proj变成多层
# 改成MTTR的那种融合方式, 因为如果只关注32x的feature, 能够使用(thw) parsing, 而不是2d parsing
# 加上LLN
# temporal queries
# parsing encoder, 考虑temporal信息
###########################################################################
# 在fusion之后, parsing encoder, decoder, matching都是单帧
# text_sentence的特征加到每个query上
class Text_V0(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                roberta_freeze = True,
                text_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 768,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
            
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_refer': 2,
                             'refdecoder_giou': 2,
                             'refdecoder_bbox': 5,
                            # 现在的模型只有decoder有loss
                            # 其他的module是否有loss
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                'refer_class_weight': [1, 0.1],
                                                'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5, 'box': 5, 'giou': 2 },
                                                },
                },
                refdecoder={ 
                    'nqueries': 5,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        
        self.d_model = d_model
        self.max_stride = max_stride
        # video encoder
        from .video_swin import VideoSwinTransformer
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
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        from transformers import RobertaModel, RobertaTokenizerFast
        self.roberta = RobertaModel.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        if roberta_freeze:
            for p in self.roberta.parameters():
                p.requires_grad_(False)
        
        assert text_proj.pop('name') == 'FeatureResizer'
        self.txt_proj = FeatureResizer(**text_proj)
        self.text_pos_embed = build_position_encoding(position_embedding_name='1d')
        
        assert fusion.pop('name') == 'VisionLanguageFusionModule'
        self.cross_product = VisionLanguageFusionModule(**fusion)

        assert parsing_encoder.pop('name') == 'deform_video_2d_fpn'
        self.deform_multiscale_2dencoder = DeformVideo2D_with_FPN(**parsing_encoder)

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
        self.decoder_refer_embed = nn.Linear(d_model, 2)
        self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.decoder_mask_out_stride = refdecoder['mask_out_stride']
        self.decoder_mask_threshold = refdecoder['mask_threshold']
 
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

    # # 2d pos
    # def encode_video(self, samples):
    #     bb_out = self.video_swint(samples)  
    #     nf, batch_size, *_ = bb_out[0].tensors.shape
    #     orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
    #     for layer_out in bb_out:
    #         layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
    #         layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
    #     multiscales = []
    #     multiscales_pad_masks = []
    #     multiscales_poses = []
    #     for lvl, feat in enumerate(bb_out): 
    #         src, pad_mask = feat.decompose() 
    #         src_proj_l = self.video_proj[lvl](src.clone())
    #         src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
    #         multiscales.append(src_proj_l)
    #         multiscales_pad_masks.append(pad_mask)
    #         pos_2d = self.video_2d_pos(pad_mask.flatten(0, 1), hidden_dim=src_proj_l.shape[2])
    #         pos_2d = rearrange(pos_2d, '(b t) c h w -> b t c h w', b=batch_size, t=nf)
    #         multiscales_poses.append(pos_2d)
    #         if lvl == (len(bb_out) - 1):
    #             for idx in range(lvl+1, len(self.video_proj)):
    #                 src_proj_l = self.video_proj[idx](src.clone())
    #                 src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
    #                 pad_mask = F.interpolate(orig_pad_mask.float(),
    #                                          size=src_proj_l.shape[-2:],mode='nearest') > 0.5
    #                 multiscales.append(src_proj_l)
    #                 multiscales_pad_masks.append(pad_mask)
    #                 pos_2d = self.video_2d_pos(pad_mask.flatten(0, 1), hidden_dim=src_proj_l.shape[2])
    #                 pos_2d = rearrange(pos_2d, '(b t) c h w -> b t c h w', b=batch_size, t=nf)
    #                 multiscales_poses.append(pos_2d)
    #     return multiscales, multiscales_pad_masks, multiscales_poses
   
    def encode_text(self, text_queries, device):
        tokenized = self.roberta_tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.roberta(**tokenized)
        # encoded_text.last_hidden_state: [batch_size, length, 768]
        # encoded_text.pooler_output: [batch_size, 768]
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # text_attention_mask: [batch_size, length]
        text_features = encoded_text.last_hidden_state 
        text_features = self.txt_proj(text_features)    
        text_masks = text_attention_mask              

        text_sentence_features = encoded_text.pooler_output  
        text_sentence_features = self.txt_proj(text_sentence_features)  
        # max b c, b max, b c
        return text_features.permute(1,0,2), text_masks, text_sentence_features
    
    def get_refdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features
    
    def forward_refdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_refer = self.decoder_refer_embed(decoder_output)  # bt n 2
        outputs_box = self.decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_refer, outputs_mask, outputs_box, attn_mask
 
    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """
        samples: t b c h w, t b h w
        frame_has_ann_by_batch: list[t, True/False], b
        """
        check_visualize = {} 
        device = samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # 抽文本的特征 max b c,  b max, b c 
        token_feats, token_pad_masks, token_sentence_feats = self.encode_text(text_queries, auxiliary, device)
        token_pos = self.text_pos_embed(token_pad_masks, hidden_dim=token_feats.shape[-1]).permute(2, 0, 1)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=token_feats, 
                                                    memory_key_padding_mask=token_pad_masks,
                                                    pos=token_pos, 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} fusion attention weights'] = attn_weight # b thw s, float, 0, 1
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> list[t c h w] -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b c -> bT c
        token_sentence_feats = repeat(token_sentence_feats, 'b c -> (b t) c', t=nf)[perFrame_has_ann]
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer
        
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = repeat(token_sentence_feats, 'bt c -> n bt c', n=self.decoder_nqueries,) # n bt c

        decoder_layer_preds = {}
        out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
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
            out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }

        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize } 
    

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w, bt' n
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)['refdecoder_refseg']
        last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        out_masks_logits =  last_layer_preds['pred_mask_logits'] 
        out_prob = last_layer_preds['pred_refer_logits'].softmax(dim=-1)
        # bt' n h w
        query_pred_masks = F.interpolate(out_masks_logits, scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False) 
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' n
        query_pred_is_referred_prob = out_prob[..., 0]
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
        
    
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bT n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        
        refseg_src = model_outs['refdecoder_refseg']
        
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
            
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

                
    # task loss
    def refdecoder_refseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        refer_class_weight = self.tasks['refdecoder_refseg']['refer_class_weight']
        matching_costs = self.tasks['refdecoder_refseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[t] -> bt
        target_valid = torch.cat([t["valid"][t['referent_idx']] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        

        loss_value = {'refdecoder_mask': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_bbox': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_giou': torch.tensor(0, device=target_valid.device).float(),
                      'refdecoder_dice': torch.tensor(0, device=target_valid.device).float(), 'refdecoder_refer': torch.tensor(0, device=target_valid.device).float(),}

        for i in range(-1, self.decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = Text_V0.refdecoder_matching(layer_pred, targets, matching_costs, refer_class_weight, self.decoder_mask_out_stride)
                if loss_weight['refdecoder_mask'] != 0 or loss_weight['refdecoder_dice'] !=0:
                    masks_losses = Text_V0.refdecoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['refdecoder_bbox'] != 0 or loss_weight['refdecoder_giou'] !=0:
                    boxes_losses = Text_V0.refdecoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['refdecoder_refer'] != 0:
                    refer_losses = Text_V0.refdecoder_refer_loss(layer_pred, targets, layer_matching_indices, refer_class_weight)
                    for k in refer_losses.keys():
                        loss_value[k] += layer_weight * refer_losses[k]
        return loss_value         

    @staticmethod
    def refdecoder_refer_loss(outputs, targets, indices, refer_class_weight):
        """
        indices: [[], []], bt
        """
        src_logits = outputs['pred_refer_logits']  # bt n 2
        bt, nq, _ = src_logits.shape # bt n 2
        
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool() 
        target_classes = torch.ones([bt, nq], device=src_logits.device).long() # bt n
        
        for batch_idx in range(bt):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        refer_class_weight = torch.tensor(refer_class_weight).to(src_logits)
        # btn 2, btn
        loss_ce = F.cross_entropy(src_logits.flatten(0,1), target_classes.flatten(), refer_class_weight)
        losses = {'refdecoder_refer': loss_ce}

        return losses
    
    @staticmethod
    def refdecoder_boxes_loss(outputs, targets, indices, num_boxes): 
        # list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        src_boxes = outputs['pred_box_logits'].sigmoid()  # bt n 4
        # list[4] -> bt 4
        src_boxes = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0) 
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0).to(src_boxes)  
        
        src_boxes = src_boxes[is_consistent]  # bt 4
        target_boxes = target_boxes[is_consistent] # bt 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['refdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['refdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    @staticmethod
    def refdecoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        # list[n t] -> list[t] -> bt
        is_consistent = torch.cat([t['valid'][t['referent_idx']] for t in targets]).bool()
        
        src_masks = outputs["pred_mask_logits"]  # bT n h w  
        # list[h w] -> bT h w
        src_masks = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)
        
        target_masks = torch.zeros_like(src_masks) # bT h w
        # list[n t h w] -> list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0).to(src_masks) # list[t h w] -> bt h w

        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        
        src_masks = src_masks[is_consistent].flatten(1) # bt hw
        target_masks = target_masks[is_consistent].flatten(1) # bt hw
        
        losses = {
            "refdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "refdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    @staticmethod
    @torch.no_grad()
    def refdecoder_matching(outputs, targets, matching_costs, refer_class_weight, decoder_mask_out_stride):
        src_refer_prob = outputs["pred_refer_logits"].softmax(dim=-1) # bt n 2
        src_boxes = outputs["pred_box_logits"].sigmoid()   # bt n 4
        src_masks_logits = outputs["pred_mask_logits"]  # bt n h w
        bt, nq, h, w = src_masks_logits.shape 

        # list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0)
        target_masks = target_masks.to(src_masks_logits)
        start = int(decoder_mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, start::decoder_mask_out_stride, start::decoder_mask_out_stride] 
        assert target_masks.size(1) * decoder_mask_out_stride == im_h
        assert target_masks.size(2) * decoder_mask_out_stride == im_w
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes'][t['referent_idx']] for t in targets], dim=0) 
        # list[t] -> bt 
        is_valid = torch.cat([t['valid'][t['referent_idx']] for t in targets], dim=0).bool()

        indices = [] 
        for i in range(bt):
            out_prob = src_refer_prob[i] # n 2
            out_bbox = src_boxes[i]  # n 4
            out_mask = src_masks_logits[i]  # n h w

            tgt_bbox = target_boxes[i].unsqueeze(0) # 1 4
            tgt_mask = target_masks[i].unsqueeze(0) # 1 h w
            tgt_valid = is_valid[i]    # True/False
            
            tgt_is_referred = (~tgt_valid).long()  # 1/0

            
            cost_refer = -out_prob[:, [tgt_is_referred]] # n 1

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 1
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # n hw : 1 hw -> n 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = matching_costs['refer'] * cost_refer +\
                matching_costs['bbox'] * cost_bbox + \
                matching_costs['giou'] * cost_giou + \
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        return indices


@register_model
def text_v0(device, configs):
    model = Text_V0(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("roberta" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 


class Text_v0linamr(Text_V0):
    def __init__(self, 
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                roberta_freeze = True,
                linamrbart_freeze=True,
                text_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 768,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                how_to_encode_linamr='encoder decoder',
                linamr_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                linamr_text_sentence_level_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1792, # + 768 = 1792
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'nlayers': 6,},
            
                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_refer': 2,
                             'refdecoder_giou': 2,
                             'refdecoder_bbox': 5,
                            # 现在的模型只有decoder有loss
                            # 其他的module是否有loss
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                'refer_class_weight': [1, 0.1],
                                                'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5, 'box': 5, 'giou': 2 },
                                                },
                },
                refdecoder={ 
                    'nqueries': 5,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },
                ) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)
        
        # from transformers import RobertaModel, RobertaTokenizerFast
        # self.roberta = RobertaModel.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        # self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(pt_dir, 'roberta_base'))
        # if roberta_freeze:
        #     for p in self.roberta.parameters():
        #         p.requires_grad_(False)
        
        # 
        # self.txt_proj = FeatureResizer(**text_proj)
        # self.text_pos_embed = build_position_encoding(position_embedding_name='1d')
    
        from .amr_utils.utils import BartForConditionalGeneration
        self.linamr_model = BartForConditionalGeneration.from_pretrained(os.path.join(pt_dir, 'amr', 'AMRBART_pretrain'))
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.linamr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_dir, 'amr', 'AMRBART_pretrain'))
        if linamrbart_freeze:
            for p in self.linamr_model.parameters():
                p.requires_grad_(False)
        
        assert linamr_proj.pop('name') == 'FeatureResizer'
        self.linamr_proj = FeatureResizer(**linamr_proj)
        assert linamr_text_sentence_level_proj.pop('name') == 'FeatureResizer'
        self.linamr_text_sentence_level_proj = FeatureResizer(**linamr_text_sentence_level_proj)
        self.how_to_encode_linamr = how_to_encode_linamr
    
    def linamr_model_forward(self, model_inputs, device):
        # input_ids: <s> text </s>
        # srcEtgt_ids: <s> text </s> <g> <MASK> </g>
        # Esrctgt_ids: <s> <MASK> </s> <g> amr </g>
        # labels: amr </g>
        # joint_ids: <s> text </s> <g> amr </g>
        if self.how_to_encode_linamr == 'encoder':
            # Esrctgt, label
            bart_input = model_inputs["Esrctgt_ids"] # b max
            attention_mask = bart_input.ne(self.linamr_tokenizer.pad_token_id).int() 
            bart_input = bart_input.to(device) # <s> <MASK> </s> <g> amr </g> pad
            attention_mask = attention_mask.to(device) # 0代表padding的位置
            # <s> <MASK> </s> <g> amr </g> pad
            encoder_outputs = self.linamr_model.model.encoder(
                input_ids=bart_input,
                attention_mask=attention_mask,
            ).last_hidden_state
            amr_embeds = encoder_outputs[:, 3:]
            amr_pad_masks = ~(attention_mask[:, 3:].bool())
            amr_sentence_level_embed = amr_embeds[:, 0] # b c
            return amr_embeds, amr_pad_masks, amr_sentence_level_embed 
        
        elif self.how_to_encode_linamr == 'encoder decoder':
            # <s> <MASK> </s> <g> amr </g> pad
            bart_input = model_inputs["Esrctgt_ids"] # b max
            attention_mask = bart_input.ne(self.linamr_tokenizer.pad_token_id).int()      
            # amr </g> pad pad
            labels = model_inputs["labels"] # b max
            
            dec_input = labels.new_zeros(labels.size(0), labels.size(1))
            # <g> amr </g> pad -> amr </g> pad pad
            dec_input[:, 1:] = labels[:, :-1].clone()
            dec_input[:, 0] = self.linamr_tokenizer.amr_bos_token_id 
 
            decoder_input_pad_mask = (dec_input == -100) 
            dec_input.masked_fill_(decoder_input_pad_mask, self.linamr_tokenizer.pad_token_id)
            
            bart_input = bart_input.to(device) # <s> <MASK> </s> <g> amr </g> pad
            attention_mask = attention_mask.to(device) # 0代表padding的位置
            labels = labels.to(device) # amr </g> -100
            dec_input = dec_input.to(device) # <g> amr </g> pad
            # self.tokenizer.decode([self.model.lm_head(decoder_output[0][i]).argmax().item() for i in range(len(decoder_output[0]))])
            # amr </g> pad
            amr_embeds = self.linamr_model(input_ids=bart_input,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=dec_input,
                                    labels=labels)
            amr_embeds_pad_mask = decoder_input_pad_mask[:, 1:]
            amr_embeds_pad_mask = F.pad(amr_embeds_pad_mask.float(), [0, 1], value=1.0).bool()
            return amr_embeds, amr_embeds_pad_mask
        
        elif self.how_to_encode_linamr == 'amr+text_encoder amr_decoder':
            # joint, label
            pass
        elif self.how_to_encode_linamr == 'amr+text_encoder amr+text_decoder':
            bart_input = model_inputs["joint_ids"]
            seg_ids = model_inputs['seg_ids'] # 0: text, 1: graph
            labels = model_inputs["joint_ids"].clone()
            labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)
            labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
            dec_input = model_inputs["joint_ids"].clone()
            dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()          # attention mask
            
            # text </s> <g> amr </g>
            decoder_output = self.linamr_model(input_ids=bart_input,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=dec_input,
                                        labels=labels).decoder_hidden_states
            decoder_output = self.text_proj(decoder_output)
            text_feat = decoder_output
            
            return decoder_output, meta_dict['each_token_length'], text_feat, None
     

    def encode_text(self, text_queries, auxiliary, device):
        tokenized = self.roberta_tokenizer.batch_encode_plus(text_queries, padding="longest", return_tensors="pt").to(device)
        encoded_text = self.roberta(**tokenized)
        # encoded_text.last_hidden_state: [batch_size, length, 768]
        # encoded_text.pooler_output: [batch_size, 768]
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # text_attention_mask: [batch_size, length]
        text_features = encoded_text.last_hidden_state 
        text_features = self.txt_proj(text_features)  
        text_masks = text_attention_mask              

        text_sentence_features = encoded_text.pooler_output  
        # max b c, b max, b c
        text_seq2seq_feats = text_features.permute(1,0,2) # max b c
        text_seq2seq_pad_masks = text_masks # b max
        text_seq2seq_sent_feats = text_sentence_features  # b c
        
        
        # dict["input_ids", "labels", "joint_ids"]
        linamr_feats, linamr_pad_masks, linamr_sentence_feats \
            = self.linamr_model_forward(auxiliary['model_inputs'], device=device)
                
        linamr_feats = self.linamr_proj(linamr_feats).permute(1,0,2) # max b c
        
        # s b c, b s, b c
        return torch.cat([text_seq2seq_feats, linamr_feats], dim=0),\
            torch.cat([text_seq2seq_pad_masks, linamr_pad_masks], dim=1),\
                self.linamr_text_sentence_level_proj(torch.cat([text_seq2seq_sent_feats, linamr_sentence_feats], dim=1))
        

@register_model
def text_v0linamr(device, configs):
    model = Text_v0linamr(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        linamr_proj=configs['linamr_proj'],
        linamr_text_sentence_level_proj=configs['linamr_text_sentence_level_proj'],
        how_to_encode_linamr=configs['how_to_encode_linamr'],
        linamrbart_freeze=configs['linamrbart_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if (("roberta" in n) or ("linamr_model" in n)) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])
    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.debug('你没用任何scheduler')
        print('你没用任何scheduler')
        return model, optimizer, None, None
    
    if sch_conf['name'] == 'polynomial_split':
        from models.model_utils import polynomial_decay_lambda
        group_names = ['model', 'vbb', 'text']
        poly_lambdas = []
        for gname in group_names:
            g_poly_conf = sch_conf[gname]
            poly_lambdas.append(partial(polynomial_decay_lambda, initial_learning_rate=g_poly_conf['initial_learning_rate'],
                                                                        end_learning_rate=g_poly_conf['end_learning_rate'],
                                                                        decay_steps=g_poly_conf['decay_steps'],
                                                                        power=g_poly_conf['power']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=poly_lambdas,
                                                      last_epoch=-1,)
        return model, optimizer, scheduler, sch_conf['unit']
    elif sch_conf['name'] == 'polynomial_freezebb':
        from models.model_utils import polynomial_decay_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(polynomial_decay_lambda, 
                                                                        initial_learning_rate=sch_conf['initial_learning_rate'],
                                                                        end_learning_rate=sch_conf['end_learning_rate'],
                                                                        decay_steps=sch_conf['decay_steps'],
                                                                        power=sch_conf['power'],
                                                                        ),
                                                      last_epoch=-1,
                                                      verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'polynomial':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                                        total_iters=sch_conf['total_iters'],
                                                        power=sch_conf['power'],
                                                        last_epoch=-1,
                                                        verbose=True)
        return model, optimizer, scheduler, sch_conf['unit']

    elif sch_conf['name'] == 'multistep_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=sch_conf['milestones'],
                                                        gamma=sch_conf['gamma'],
                                                        verbose=True), 
        return model, optimizer, scheduler, sch_conf['unit']
    
    elif sch_conf['name'] == 'invert_sqrt':
        from models.model_utils import inverse_sqrt_warmup_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=partial(inverse_sqrt_warmup_lambda,
                                                                         num_warmup_steps=sch_conf['num_warmup_steps'],
                                                                         num_training_steps=sch_conf['num_training_steps']), last_epoch=-1)
        return model, optimizer, scheduler, sch_conf['unit']
    else:
        raise ValueError()


