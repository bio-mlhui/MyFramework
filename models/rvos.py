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
import logging
###########################################################################
# 共享的module, # b n t h w; b t c h w
###########################################################################
from .position_encoding import build_position_encoding
from .model_utils import *
from .layers_unimodal_attention import FeatureResizer, CrossAttentionLayer, MLP, SelfAttentionLayer, FFNLayer 
from .transformer_deformable import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from .transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones
import util.box_ops as box_ops
from util.misc import get_world_size, is_dist_avail_and_initialized, nested_tensor_from_videos_list_with_stride, nested_tensor_from_tensor_list_with_stride
from functools import partial

###########################################################################
# amr without variable
###########################################################################

class AMR_Grounding_2DObj(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                 work_dir=None,
                 mode=None,
                loss_weight={},
                tasks = { 'objdecoder':{}},
                pixel_mean = [0.485, 0.456, 0.406],
                pixel_std = [0.229, 0.224, 0.225],
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                # obj decoder
                 obj_decoder = {
                     'name':None,
                     'path': None,
                     'freeze': True,
                 },
                reason_module={},
                temporal_decoder = {},
                fusion={},
                use_we=False,
                loss_type='object',
                word_embedding_random=False,
                ) -> None:
        super().__init__()
        self.use_we = use_we
        self.loss_type = loss_type
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.mode = mode
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if word_embedding_random:
            for p in self.amrbart_wordEmbedding.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        amr_proj_name = amrtext_wordEmbedding_proj.pop('name')
        if amr_proj_name == 'FeatureResizer':
            self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        elif amr_proj_name == 'linear':
            self.amrtext_wordEmbedding_proj = nn.Linear(**amrtext_wordEmbedding_proj)
        # self.amrtext_wordEmbedding_3c_to_c = nn.Linear(1024 * 3, 1024)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        from models.pretrained_image_decoder import pt_obj_2d_decoder_entrypoint
        create_obj_decoder = pt_obj_2d_decoder_entrypoint(obj_decoder['name']) # 里面有load image segmentation
        self.obj_decoder = create_obj_decoder(obj_decoder, pt_dir, work_dir)
        self.obj_decoder_num_layers = self.obj_decoder.num_layers # 10层  # load checkpoing
        self.obj_decoder_mask_out_stride = self.obj_decoder.mask_out_stride
        self.obj_decoder_mask_threshold = self.obj_decoder.mask_threshold
    
        if mode == 'rios测试预训练imseg':
            return
        
        # 和text相关
        logging.info('你在做和text有关的任务')
        print('你在做和text有关的任务')
        from models.layer_fusion import fusion_entrypoint
        
        create_fusion = fusion_entrypoint(fusion['name'])
        fusion_module = create_fusion(fusion)
        self.fusion_module = fusion_module
        # hack obj decoder fusion
        self.obj_decoder.sem_seg_head.pixel_decoder.hack_fusion(fusion_module=self.fusion_module,
                                                                early_fusion=fusion['deform_early'],
                                                                early_fusion_deep_copy=fusion['deform_early_dcopy'],
                                                                early_add_pos=fusion['deform_add_pos'] if 'deform_add_pos' in fusion else True,
                                                                encoder_layer_ref_self=fusion['deform_layer'],
                                                                encoder_layer_deep_copy=fusion['deform_layer_dcopy'],
                                                                encoder_layer_add_pos=fusion['deform_layer_add_pos'] if 'deform_layer_add_pos' in fusion else True)
        from .layer_graph import graphLayer_entrypoint
        create_reason_module = graphLayer_entrypoint(reason_module['graph']['name'])
        self.reason_module = create_reason_module(reason_module['graph'])
        if mode == '只训练rios':
            self.reason_2d_choose = reason_module['2d_choose_who']
            self.reason_2d_layer_if_reason =  self.tasks['2d_layer_if_reason'] # obj_decoder的每层是否reason
            assert len(self.tasks['objdecoder']['loss_layer_weights']) == self.obj_decoder_num_layers
            assert self.reason_2d_layer_if_reason[-1]
            assert len(self.reason_2d_layer_if_reason) == self.obj_decoder_num_layers

        elif mode == 'rios之后rvos' or mode == '只训练rvos' or mode == 'joint':
            self.reason_2d_choose = reason_module['2d_choose_who']
            self.reason_2d_layer_if_reason =  self.tasks['2d_layer_if_reason'] # obj_decoder的每层是否reason
            assert len(self.reason_2d_layer_if_reason) == self.obj_decoder_num_layers
            assert len(self.tasks['objdecoder']['loss_layer_weights']) == self.obj_decoder_num_layers
            from .layer_temporal_decoder import temporal_decoder_entrypoint
            create_temporal_decoder = temporal_decoder_entrypoint(temporal_decoder['name'])
            self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir)
            self.temporal_decoder_num_layers = self.temporal_decoder.num_layers
            self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
            self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
            self.temporal_decoder.hack_fusion(fusion_module,
                                                early_fusion=fusion['swin_early'],
                                                early_fusion_deep_copy=fusion['swin_early_dcopy'], 
                                                early_fusion_add_pos=fusion['swin_early_add_pos'],
                                                encoder_layer_ref_self=fusion['swin_layer'],
                                                encoder_layer_deep_copy=fusion['swin_layer_dcopy'],
                                                encoder_layer_add_pos=fusion['swin_early_add_pos'],)
            self.reason_3d_choose = reason_module['3d_choose_who']
            self.reason_3d_layer_if_reason = self.tasks['3d_layer_if_reason'] # decoder的每层是否reason
            assert self.reason_3d_layer_if_reason[-1]
            assert len(self.tasks['temporal_decoder']['loss_layer_weights']) == self.temporal_decoder.used_layers\
                                                                         * self.temporal_decoder_num_layers # 训练的时候用后三层
            assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers * self.temporal_decoder.used_layers
        else:
            return
    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']

        if self.use_we:
            if not self.training:
                we_size = len(self.amrbart_wordEmbedding.weight)
                global_we = self.amrtext_wordEmbedding_proj(self.amrbart_wordEmbedding.weight)
                global_we = repeat(global_we, 's c -> b s c',b=batch_size)
                global_seg_ids = (amr_token_seg_ids.new_ones([batch_size, we_size]) * 2).int()
            else:
                global_we = text_auxiliary['all_concept_roles'] # b mmax
                global_seg_ids = global_we.new_ones([batch_size, global_we.shape[1]]) * 2 # b mmax
                acc_pad = text_auxiliary['all_concept_roles_pad'] # b max
                global_seg_ids.masked_fill_(acc_pad, 0)
                global_we = self.amrtext_wordEmbedding_proj(self.amrbart_wordEmbedding(global_we))
        else:
            global_we = None
            global_seg_ids = None
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments, global_we, global_seg_ids
    

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, targets=None, visualize_dir=False):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        if len(samples.tensors.shape) == 5:
            # t b c h w
            nf, batch_size, *_ = samples.tensors.shape
            samples.tensors = rearrange(samples.tensors, 't b c h w -> (b t) c h w')
            samples.mask = rearrange(samples.mask, 't b h w -> (b t) h w')
        elif len(samples.tensors.shape) == 4:
            # b 3 h w
            batch_size = samples.tensors.shape[0]
        else:
            raise ValueError()
        device = samples.tensors.device
        if text_queries is not None:
            amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments, global_we, global_we_seg_ids,\
                  = self.encode_text(text_queries, auxiliary, device) 
        else:
            text_feats, text_pad_masks, amr_token_feats, amr_token_seg_ids = None, None, None, None

        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        if self.use_we:
            obj_decoder_output, _, _ = self.obj_decoder(samples,
                                                                    amrs=[None] * len(amrs), 
                                                                    amr_token_feats=global_we,
                                                                    amr_token_seg_ids=global_we_seg_ids, 
                                                                    text_feats=None, 
                                                                    text_pad_masks=None)
        else:
            obj_decoder_output, amr_token_feats, text_feats = self.obj_decoder(samples,
                                                                                amrs= amrs, 
                                                                                amr_token_feats=amr_token_feats,
                                                                                amr_token_seg_ids=amr_token_seg_ids, 
                                                                                text_feats=text_feats, 
                                                                                text_pad_masks=text_pad_masks)
        obj_queries_by_layer, pred_masks_by_layer, multiscale_feats, \
                            mask_features= obj_decoder_output['obj_queries'], obj_decoder_output['pred_masks'],\
                                                                    obj_decoder_output['multiscale_feats'], obj_decoder_output['mask_features'] # b nq c

        if self.mode == 'rios测试预训练imseg':
            return {'objdecoder': {'pred_masks': pred_masks_by_layer,}}
        elif self.mode == '只训练rios':  
            grounding_score_by_layer = []
            for layer_idx, obj_queries in enumerate(obj_queries_by_layer): 
                if self.reason_2d_layer_if_reason[layer_idx]:
                    if self.use_we:
                        amr_token_feats = contexualized_amr_feats
                    grounding_score = self.reason_module(obj_queries=obj_queries, 
                                                        amrs=amrs,
                                                        amr_token_feats=amr_token_feats,
                                                        amr_token_seg_ids=amr_token_seg_ids,
                                                        node_alignments=node_alignments,
                                                        text_feats=text_feats,
                                                        is_2d=True,
                                                        text_pad_masks=text_pad_masks) # list[vi nq]
                    if self.reason_2d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(grounding_score)
                else:
                    grounding_score_by_layer.append(None)
            return {'objdecoder': {'pred_masks': pred_masks_by_layer, # b nq h w
                                   'reason_2d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint': 
            #可能会有2d的loss计算
            repeated_amrs = []
            for idx in range(batch_size):
                for _ in range(nf):
                    repeated_amrs.append(copy.deepcopy(amrs[idx]))
            grounding_score_2d_by_layer = []
            for layer_idx, obj_queries in enumerate(obj_queries_by_layer): 
                if self.reason_2d_layer_if_reason[layer_idx]:
                    grounding_score_2d = self.reason_module(obj_queries=obj_queries.clone(), 
                                                        amrs=repeated_amrs,
                                                        amr_token_feats=repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf),
                                                        amr_token_seg_ids=repeat(amr_token_seg_ids, 'b s -> (b t) s',t=nf),
                                                        node_alignments=node_alignments,
                                                        text_feats=repeat(text_feats, 'b s c -> (b t) s c', t=nf),
                                                        is_2d=True, is_3d=False,
                                                        text_pad_masks=repeat(text_pad_masks,'b s -> (b t) s', t=nf))
                    if self.reason_2d_choose == '第一个':
                        grounding_score_2d = torch.stack([lg[0] for lg in grounding_score_2d], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_2d_by_layer.append(grounding_score_2d)
                else:
                    grounding_score_2d_by_layer.append(None)

            obj_queries_by_layer = [rearrange(obj_q, '(b t) nq c -> b t nq c',b=batch_size,t=nf) for obj_q in obj_queries_by_layer] 
            # L b s c, L b s c
            if self.use_we:
                temporal_decoder_output, _, _ = self.temporal_decoder(frame_query_by_layer=obj_queries_by_layer, # list[b t nq c]
                                                                                            mask_features=mask_features, # bt c h w
                                                                                            amrs=[None] * len(amrs), 
                                                                                            amr_token_feats=global_we,
                                                                                            amr_token_seg_ids=global_we_seg_ids, 
                                                                                            text_feats=None, 
                                                                                            text_pad_masks=None)
            else:
                temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(frame_query_by_layer=obj_queries_by_layer, # list[b t nq c]
                                                                                            mask_features=mask_features, # bt c h w
                                                                                            amrs= amrs, 
                                                                                            amr_token_feats=amr_token_feats,
                                                                                            amr_token_seg_ids=amr_token_seg_ids, 
                                                                                            text_feats=text_feats, 
                                                                                            text_pad_masks=text_pad_masks)
            # L D b nq c
            # L b t nqf c
            # L D b nq t nqf
            # L D b nq t h w
            # L D b nq class+1
            temporal_queries_by_layer, frame_queries_memory, cross_attn_weights_by_layer, \
              temporal_pred_masks_by_layer, temporal_pred_logits_by_layer,\
                = temporal_decoder_output['temporal_queries'], temporal_decoder_output['frame_queries'], \
                                                                temporal_decoder_output['cross_attn_weights'],\
                                                                temporal_decoder_output['pred_masks'], temporal_decoder_output['pred_logits']
            D = temporal_queries_by_layer.shape[1]
            L = temporal_queries_by_layer.shape[0]
            if self.use_we:
                amr_token_feats = repeat(amr_token_feats, 'b s c -> L D b s c',L=L, D=D)
                text_feats = repeat(text_feats, 'b s c -> L D b s c', L=L, D=D)
            else:
                amr_token_feats = repeat(amr_token_feats, 'L b s c -> L D b s c',D=D)
                text_feats = repeat(text_feats, 'L b s c -> L D b s c', D=D)
            frame_queries_memory = repeat(frame_queries_memory, 'L b t nqf c -> L D b t nqf c',D=D)
            # region
            # repeated_amrs = []
            # for idx in range(batch_size):
            #     for _ in range(nf):
            #         repeated_amrs.append(copy.deepcopy(amrs[idx]))
            # spatial_grounding_score = self.reason_module(obj_queries=frame_queries_memory.flatten(0, 1), # bt nq c 
            #                                             amrs=repeated_amrs,
            #                                             amr_token_feats=repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf),
            #                                             amr_token_seg_ids=repeat(amr_token_seg_ids, 'b s -> (b t) s',t=nf),
            #                                             node_alignments=node_alignments,
            #                                             text_feats=repeat(text_feats, 'b s c -> (b t) s c', t=nf),
            #                                             is_2d=True, is_3d=False,
            #                                             text_pad_masks=repeat(text_pad_masks,'b s -> (b t) s', t=nf),
            #                                         ) # list[vi nqf], batch_size * T
            # # list[vi nqf], b * T -> list[list[vi nqf], T], b
            # spg_by_batch = []
            # for idx in range(batch_size):
            #     spg_by_batch.append(spatial_grounding_score[idx*nf:(idx+1)*nf])
            # spg_scores = [torch.stack(sbb, dim=1) for sbb in spg_by_batch] # list[Vi T nqf]
            # endregion
            grounding_score_by_layer = []
            for layer_idx, (temporal_queries, cross_attn_weights, amr_tok_feat, txt_feat, frame_query_mem) in \
                enumerate(zip(temporal_queries_by_layer.flatten(0,1),
                            cross_attn_weights_by_layer.flatten(0,1), 
                            amr_token_feats.flatten(0,1), text_feats.flatten(0,1), frame_queries_memory.flatten(0,1))):
                if self.reason_3d_layer_if_reason[layer_idx]:
                    grounding_score = self.reason_module(temporal_queries=temporal_queries,  # b nq c
                                                            frame_queries=frame_query_mem, # b t nqf c
                                                            frame_queries_grounding_score=None, 
                                                             cross_attn_weights=cross_attn_weights,  # # b nq t nqf
                                                             is_3d=True, is_2d=False,
                                                             amrs=amrs,
                                                             amr_token_feats=amr_tok_feat,
                                                             amr_token_seg_ids=amr_token_seg_ids,
                                                             node_alignments=node_alignments,
                                                             text_feats=txt_feat,
                                                             text_pad_masks=text_pad_masks) # list[vi nq]
                    # 可视化所有object query的mask
                    if visualize_dir:
                        save_model_output(videos=samples.tensors,
                                        text_query=text_queries[0], 
                                        amr=auxiliary['amrs'][0],
                                        amr_tree_string=auxiliary['amr_tree_strings'][0], 
                                        directory=visualize_dir,
                                        pred_masks=temporal_pred_masks_by_layer.flatten(0,1)[-1][0],
                                        scores=grounding_score[0],)
                    if self.reason_3d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(grounding_score)
                else:
                    grounding_score_by_layer.append(None)
            return {'temporal_decoder': {'pred_masks': temporal_pred_masks_by_layer.flatten(0,1), # list[b nq t h w]
                                         'pred_logits': temporal_pred_logits_by_layer.flatten(0,1), # list[b nq class+1]
                                         'reason_3d': grounding_score_by_layer}, # list[b nq]
                    'objdecoder': {'pred_masks': pred_masks_by_layer, # list[bt nq h w]
                                   'reason_2d': grounding_score_2d_by_layer} ,} # list[None] / list[bt nq]

    def forward_rios(self, samples, text_queries, auxiliary, targets, visualize=False):
        # samples: list[3 h w] -> b 3 H W
        # targets: list[{'masks': ni h w}]
        # loss有 n个物体的mask loss + reason choose loss
        images = [(x - self.pixel_mean) / self.pixel_std for x in samples]
        samples = nested_tensor_from_tensor_list_with_stride(images, max_stride=self.max_stride)
        batch_size, _, H, W = samples.tensors.shape
        new_targets = self.rios_targets_handler(targets, pad_H=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary, targets=new_targets) 
        if self.loss_type == 'referent':
            loss_value_dict = self.objdecoder_loss_referent(model_outs, new_targets)
        else:
            loss_value_dict = self.objdecoder_loss(model_outs, new_targets)

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

    @torch.no_grad()
    def sample_rios(self,samples, text_queries, auxiliary, targets, visualize=False):
        images = [(x - self.pixel_mean) / self.pixel_std for x in samples]
        samples = nested_tensor_from_tensor_list_with_stride(images, max_stride=self.max_stride)
        batch_size, _, H, W = samples.tensors.shape
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)['objdecoder']
        # b nq h w
        out_mask_logits = decoder_layer_preds['pred_masks'][-1]
        if self.mode == 'rios测试预训练imseg': 
            new_targets = self.rios_targets_handler(targets, pad_H=H, pad_W=W) # 缩放gt mask到4x
            matching_result = self.objdecoder_matching(out_mask_logits, new_targets)
            gt_referent_idx = new_targets['gt_referent_idx']
            # list[h w] -> b h w
            out_mask_logits = torch.stack([out_mask[src_idx[tgt_idx.tolist().index(gt_ref_idx)]]
                                            for out_mask, gt_ref_idx, (src_idx, tgt_idx) in zip(out_mask_logits, gt_referent_idx, matching_result, )], dim=0)
        else:
            ref_last_layer_gscore = decoder_layer_preds['reason_2d'][-1] # b nq
            argmax_query_idx = ref_last_layer_gscore.argmax(-1) # b
            # b h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
        # b 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # b 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], b
        size_after_aug = [] #list[h,w], b
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
        return {
            'query_pred_masks': processed_pred_masks, # [n h w], batch
            'query_pred_is_referred_prob': query_pred_is_referred_prob, # [n], batch
        }
    

    def rios_targets_handler(self, targets, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = []
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n h w -> n H W
            tgt_masks.append(F.pad(targets[idx]['masks'].float().unsqueeze(0), pad=(0, pad_W-w, 0, pad_H-h)).bool()[0])
        for btc_idx in range(batch_size):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        isvalid = [targets[idx]['valid'] for idx in range(batch_size)] # list[ni], b
        return {
            'masks': tgt_masks,
            'gt_referent_idx': gt_referent_idx,
            'isvalid': isvalid
        }

    def objdecoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = sum([t.int().sum() for t in isvalid])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss(layer_gscore_output, matching_indices, targets)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, matching_indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss(self, layer_gscore_output, matching_indices,  targets):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        num_refs = (ref_is_valid.int().sum())
        match_as_gt_indices = [] # list[int], bt
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}


    def objdecoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = len(isvalid)
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss_referent(layer_gscore_output, matching_indices, targets, num_objs)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching_referent(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # lis[int], batch
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # list[int], b
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[[J]] for t, J in zip(tgt_masks, referent_idx)], dim=0) # n_simga h w
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss_referent(self, layer_gscore_output, matching_indices,  targets, num_refs):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        assert ref_is_valid.any()

        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}
    
 
    def forward_rvos(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        
        rvos_targets = self.rvos_targets_handler(targets, pad_T=T, pad_H=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        # 可能会有object decoder的loss
        if self.loss_type == 'object':
            loss_value_dict = self.temporal_decoder_loss(model_outs, rvos_targets)
        elif self.loss_type == 'referent':
            loss_value_dict = self.temporal_decoder_loss_referent(model_outs, rvos_targets)

        video_rios_targets = self.video_rios_targets_handler(targets, pad_T=T, pad_H=H, pad_W=W)
        has_ann = video_rios_targets['has_ann'] # bT
        obj_pred_masks = model_outs['objdecoder']['pred_masks'] # list[bt nq h w]
        obj_pred_masks = [opm[has_ann] for opm in obj_pred_masks]
        model_outs['objdecoder']['pred_masks'] = obj_pred_masks

        obj_gscores = model_outs['objdecoder']['reason_2d'] # list[bT nq/None]
        new_obj_gscores = []
        for og in obj_gscores:
            if og is None:
                new_obj_gscores.append(None)
            else:
                new_obj_gscores.append(og[has_ann]) # bt nq -> bt' nq
        model_outs['objdecoder']['reason_2d'] = new_obj_gscores
        if self.loss_type == 'object':
            loss_value_dict.update(self.objdecoder_loss(model_outs, video_rios_targets))
        elif self.loss_type == 'referent':
            loss_value_dict.update(self.objdecoder_loss_referent(model_outs, video_rios_targets))

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

    @torch.no_grad()
    def sample_rvos(self, samples, text_queries, auxiliary, targets, visualize_dir=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride) # targets[0]['masks']
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, visualize_dir=visualize_dir,)
        # b nq T h w -> b T nq h w
        out_mask_logits = decoder_layer_preds['temporal_decoder']['pred_masks'][-1].permute(0,2,1,3,4)
        if self.mode == '测试rvos bound':
            raise NotImplementedError()
            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            ref_last_layer_gscore = decoder_layer_preds['temporal_decoder']['reason_3d'][-1]  # b nq
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[:, max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()] # bT -> bt' h w
        # bt' 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.temporal_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.temporal_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # list[(h w)],batch -> list[(h w)], bt'
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        assert len(query_pred_masks) == len(size_original)
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0].bool()
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt' n -> list[n], bt' -> list[n t'], b
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


    def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
        labels = [t['class_labels'] for t in targets] # list[ni], batch
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
        # list[ni]
        is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> list[T]
        perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

        return {
            'labels': labels, 
            'masks': tgt_masks,
            'is_valid': is_valid,
            'referent_idx': gt_referent_idx,
            'has_ann': perFrame_has_ann
        }

    def video_rios_targets_handler(self, targets, pad_T, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w
            
        # list[n h w], bt'
        rep_tgt_masks = []
        for btc_idx in range(batch_size):
            t_m = tgt_masks[btc_idx].split(1, dim=1)
            t_m = [tm.squeeze(1) for tm in t_m]
            rep_tgt_masks.extend(t_m)
        rep_is_valid = [rtm.flatten(1).any(-1) for rtm in rep_tgt_masks] # list[ni], bt'
        num_anns_by_batch = [tm.shape[1] for tm in tgt_masks] # list[int]
        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        rep_gt_referent_idx = [] # list[int], bt'
        for btc_idx in range(batch_size):
            rep_gt_referent_idx.extend([gt_referent_idx[btc_idx]] * num_anns_by_batch[btc_idx])
                

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> bT
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann])
        # 有annotation并且是valid的

        return {
            'masks': rep_tgt_masks,
            'gt_referent_idx': rep_gt_referent_idx,
            'isvalid': rep_is_valid,
            'has_ann': perFrame_has_ann
        }

    def temporal_decoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idxs = targets['referent_idx'] # list[int]

        num_objs = sum([tm.flatten(0,1).flatten(1).any(-1).int().sum() for tm in tgt_masks])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        num_refs = len(tgt_masks) # 
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_class': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers   
        out_logits = model_outs['temporal_decoder']['pred_logits'] # list[b nq class+1]
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_out_logits, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                    out_logits,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching(layer_mask_output, layer_out_logits, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if self.loss_weight['tempdecoder_class'] != 0:
                    class_losses = self.temporal_decoder_classes_loss(layer_out_logits, targets, matching_indices, num_objs)
                    for k in class_losses.keys():
                        loss_value[k] += layer_loss_weight * class_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, out_class_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        tgt_classes = targets['labels'] # list[ni]
        src_masks_logits = out_mask_logits  # b nq T h w
        src_class_logits = out_class_logits # b nq class+1

        batch_size, nq, T, h, w = src_masks_logits.shape 
        src_class_probs = src_class_logits.softmax(-1) # b nq class+1
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_class_prob = src_class_probs[i] # nq class+1

            tgt_cls = tgt_classes[i] # ni
            cost_class = - out_class_prob[:, tgt_cls] # nq ni

            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W

            scores = []
            for ann_t in range(out_mask.shape[1]):
                out_t_mask = out_mask[:, ann_t] # nq h w
                tgt_t_mask = tgt_mask[:, ann_t] # ni h w
                c_mask = batch_sigmoid_ce_loss(out_t_mask.flatten(1), tgt_t_mask.flatten(1)) # nq ni
                c_dice = batch_dice_loss(out_t_mask.flatten(1), tgt_t_mask.flatten(1)) # nq ni

                t_cost =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * c_mask + \
                    self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * c_dice
                scores.append(t_cost)
            scores = torch.stack(scores, dim=0).mean(0) # n nq ni -> nq ni
            C = scores + self.tasks['temporal_decoder']['objseg_matching_costs']['class'] * cost_class
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_classes_loss(self, layer_out_logits, targets, matching_indices, num_objs):
        # b nq class+1, 
        target_labels = targets['labels'] #list[ni], batch
        # t_sigma
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, matching_indices)]) # n_sigma
    
        idx = get_src_permutation_idx(matching_indices)
        # b nq 充满背景类别
        target_classes = torch.full(
            layer_out_logits.shape[:2], layer_out_logits.shape[2] -1, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o
        class_weights = torch.ones(layer_out_logits.shape[2]).float() # class+1
        class_weights[-1] = self.tasks['temporal_decoder']['eos_weight']

        loss_ce = F.cross_entropy(layer_out_logits.transpose(1,2), target_classes, weight=class_weights.to(layer_out_logits))
        losses = {"tempdecoder_class": loss_ce}
        return losses

    def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
        has_anntation = targets['has_ann'] # list[T]
        # is_valid = targets['is_valid'] # list[ni]

        # b nq T H W -> list[ni t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[ni t' H W], b 
        tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# list[ni_t' h w]
        tgt_masks = torch.cat([tm.flatten(0,1) for tm in tgt_masks],dim=0) # list[ni_t' h w]
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_objs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_objs),
        }
        return losses    

    def temporal_decoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int]
        # list[ni t' h w] -> list[t' hw]
        num_refs = sum([tm[ref_idx].flatten(1).any(-1).int().sum() for tm, ref_idx in zip(tgt_masks, referent_idx)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()

        num_refs_video = len(tgt_masks)
        num_refs_video = torch.as_tensor([num_refs_video], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs_video)
        num_refs_video = torch.clamp(num_refs_video / get_world_size(), min=1).item()

        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers 
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_refs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss_referent(layer_gscore_output, targets, matching_indices, num_refs, num_refs_video)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching_referent(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int], batch
        src_masks_logits = out_mask_logits  # b nq T h w

        batch_size, nq, T, h, w = src_masks_logits.shape
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 t' H W
            tgt_mask = tgt_mask.to(out_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1

            C =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice
            C = C.cpu()
            indices.append(linear_sum_assignment(C))            
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_refs):
        has_anntation = targets['has_ann'] # list[T]
        ref_idx = targets['referent_idx'] # list[int], batch
        # b nq T H W -> list[1 t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[1 t' H W], b 
        tgt_masks = [t[[J]] for t, J in zip(targets['masks'], ref_idx)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# 1_t'_sigma h w
        tgt_masks = torch.cat([tm.flatten(0, 1) for tm in tgt_masks],dim=0) # 1_t'_sigma h w
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
        }
        return losses    

    def temporal_reason_loss_referent(self, layer_gscore_output, targets, matching_indices, num_refs, num_refs_video):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs_video}


    def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, global_num_refs):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        num_refs = (ref_is_valid.int().sum())
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = tgt_idx.tolist().index(ref_idx)
            match_as_gt_idx = src_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs}

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        # 把pixel mean, pixel std加上去
        # samples -> NestedTensor
        if self.mode == 'rios测试预训练imseg':
            raise ValueError() # 只能以evaluate形式运行
        elif self.mode == '只训练rios':
            return self.forward_rios(samples, text_queries, auxiliary, targets)
        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint':
            return self.forward_rvos(samples, text_queries, auxiliary, targets) 
        else:
            raise ValueError()

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize_dir=False):
        if self.mode == 'rios测试预训练imseg':
            return self.sample_rios(samples, text_queries=None, auxiliary=None, targets=targets)
        elif self.mode == '只训练rios':
            return self.sample_rios(samples, text_queries, auxiliary, targets)
        elif self.mode == '只训练rvos' or self.mode == 'rios之后rvos' or self.mode == 'joint':
            return self.sample_rvos(samples, text_queries, auxiliary, targets,visualize_dir=visualize_dir) 
        else:
            raise ValueError()

@register_model
def amr_grounding_2dobj(device, configs):
    model = AMR_Grounding_2DObj(
        d_model=configs['d_model'],
        max_stride=configs['max_stride'],
        use_we=configs['use_we'] if 'use_we' in configs else False,
        pt_dir=configs['pt_dir'],
        work_dir=configs['work_dir'],
        mode=configs['mode'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        pixel_mean=configs['pixel_mean'],
        pixel_std=configs['pixel_std'],
        word_embedding_random=configs['word_embedding_random'] if 'word_embedding_random' in configs else False,
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],

        obj_decoder=configs['obj_decoder'],
        reason_module=configs['reason_module'], 
        temporal_decoder=configs['temporal_decoder'],
        fusion=configs['fusion'],
        loss_type=configs['loss_type'] if 'loss_type' in configs else 'object'
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder.backbone" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder.backbone" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] 
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.info('你没用任何scheduler')
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
                                                        verbose=True)
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


class AMR_Grounding_2DObj_MeVis(AMR_Grounding_2DObj):
    def __init__(self, d_model=256, max_stride=32, pt_dir='/home/xhh/pt', work_dir=None, mode=None, loss_weight={}, tasks={ 'objdecoder': {} }, pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225], amrbart_wordEmbedding_freeze=True, amrtext_wordEmbedding_proj={ 'name': 'FeatureResizer','input_feat_size': 1024,'output_feat_size': 256,'dropout': 0,'do_ln': True }, obj_decoder={ 'name': None,'path': None,'freeze': True }, reason_module={}, temporal_decoder={}, fusion={}, use_we=False, loss_type='object') -> None:
        super().__init__(d_model, max_stride, pt_dir, work_dir, mode, loss_weight, tasks, pixel_mean, pixel_std, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, obj_decoder, reason_module, temporal_decoder, fusion, use_we, loss_type)

        assert 'sigmoid' in reason_module['name']


    def objdecoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['objdecoder']['loss_layer_weights']
        isvalid = targets['isvalid'] #list[ni], batch
        device = isvalid[0].device
        num_objs = len(isvalid)
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_reason': torch.tensor(0, device=device).float(),}
        
        out_mask_logits = model_outs['objdecoder']['pred_masks'] # list[b nq H W], num_layers
        out_gscores = model_outs['objdecoder']['reason_2d'] # list[b ni], num_layers     
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.objdecoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['objdecoder_mask'] != 0 or self.loss_weight['objdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.objdecoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['objdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_2d_loss = self.ref_choose_2d_loss_referent(layer_gscore_output, matching_indices, targets, num_objs)
                    for k in reason_2d_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_2d_loss[k]
        return loss_value      

    @torch.no_grad()
    def objdecoder_matching_referent(self, out_mask_logits, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # lis[int], batch
        src_masks_logits = out_mask_logits  # b nq h w
        batch_size, nq, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 H W
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.tasks['objdecoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['objdecoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def objdecoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_boxes):
        # b nq H W
        # list[ni H W], b
        tgt_masks = targets['masks']
        referent_idx = targets['gt_referent_idx'] # list[int], b
        src_masks = torch.cat([t[J] for t, (J, _) in zip(out_mask_logits, matching_indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[[J]] for t, J in zip(tgt_masks, referent_idx)], dim=0) # n_simga h w
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
            "objdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_boxes),
        }
        return losses    

    def ref_choose_2d_loss_referent(self, layer_gscore_output, matching_indices,  targets, num_refs):
        is_valid = targets['isvalid'] # list[ni], batch
        referent_idx = targets['gt_referent_idx'] # list[int], batch
        ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
        assert ref_is_valid.any()

        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'objdecoder_reason': choose_loss.sum() / num_refs}
    

    def temporal_decoder_loss_referent(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int]
        # list[ni t' h w] -> list[t' hw]
        num_refs = sum([tm[ref_idx].flatten(1).any(-1).int().sum() for tm, ref_idx in zip(tgt_masks, referent_idx)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()

        num_refs_video = len(tgt_masks)
        num_refs_video = torch.as_tensor([num_refs_video], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs_video)
        num_refs_video = torch.clamp(num_refs_video / get_world_size(), min=1).item()

        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers 
        assert len(loss_layer_weights) == len(out_mask_logits)
        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores,
                                                                                                     loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching_referent(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss_referent(layer_mask_output, targets, matching_indices, num_refs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss_referent(layer_gscore_output, targets, matching_indices, num_refs, num_refs_video)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching_referent(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idx = targets['referent_idx'] # list[int], batch
        src_masks_logits = out_mask_logits  # b nq T h w

        batch_size, nq, T, h, w = src_masks_logits.shape
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i][[referent_idx[i]]].to(out_mask) # 1 t' H W
            tgt_mask = tgt_mask.to(out_mask)
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1

            C =  self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice
            C = C.cpu()
            indices.append(linear_sum_assignment(C))            
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss_referent(self, out_mask_logits, targets, matching_indices, num_refs):
        has_anntation = targets['has_ann'] # list[T]
        ref_idx = targets['referent_idx'] # list[int], batch
        # b nq T H W -> list[1 t' H W]
        src_masks = [t[J][:, has_ann.bool()] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]

        # list[1 t' H W], b 
        tgt_masks = [t[[J]] for t, J in zip(targets['masks'], ref_idx)]
        
        src_masks = torch.cat([sm.flatten(0, 1) for sm in src_masks],dim=0)# 1_t'_sigma h w
        tgt_masks = torch.cat([tm.flatten(0, 1) for tm in tgt_masks],dim=0) # 1_t'_sigma h w
        tgt_masks = tgt_masks.to(src_masks)
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        losses = {
            "tempdecoder_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
            "tempdecoder_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_refs),
        }
        return losses    

    def temporal_reason_loss_referent(self, layer_gscore_output, targets, matching_indices, num_refs, num_refs_video):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'] # list[ni]
        ref_is_valid = torch.tensor([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]).bool().to(self.device) # b
        match_as_gt_indices = [J[0] for (J, _) in matching_indices] # list[int], b
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs_video}


class AMR_Grounding_3DObj(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                 work_dir=None,
                 mode=None,
                loss_weight={},
                tasks = { 'tempdecoder':{}},
                pixel_mean = [0.485, 0.456, 0.406],
                pixel_std = [0.229, 0.224, 0.225],
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {},
                 temporal_decoder = {},
                reason_module_3d={},
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.mode = mode
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        from models.pretrained_video_instance_decoder import pt_3d_obj_decoder_entrypoint
        create_temporal_decoder = pt_3d_obj_decoder_entrypoint(temporal_decoder['name'])
        self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir, work_dir)
        self.temporal_decoder_num_layers = self.temporal_decoder.num_layers # 4层
        self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
        self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
        if mode == 'rvos测试预训练vis':
            pass
        elif mode == '只训练rvos':
            self.reason_3d_choose = reason_module_3d['choose_who']
            self.reason_3d_layer_if_reason = reason_module_3d['layer_if_reason'] # obj_decoder的每层是否reason
            assert self.reason_3d_layer_if_reason[-1]
            assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers
            from .layer_graph import graphLayer_entrypoint
            create_reason_module = graphLayer_entrypoint(reason_module_3d['graph']['name'])
            self.reason_module_3d = create_reason_module(reason_module_3d['graph'])
        else:
            raise ValueError()
        
    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_ = samples.tensors.shape
        device = samples.tensors.device

        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        # TODO: 添加text 特征变换
        temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(samples, 
                                                                                     
                                                                                    amrs=amrs, 
                                                                                    amr_token_feats=amr_token_feats,
                                                                                    amr_token_seg_ids=amr_token_seg_ids, 
                                                                                    text_feats=text_feats, 
                                                                                    text_pad_masks=text_pad_masks)
         # l b nq c, l b t nqf c, l b nq T nqf
         # l b t nq h w,
        temporal_queries_by_layer, frame_queries_by_layer, cross_attn_weights_by_layer,\
              pred_masks_by_layer, multiscale_feats = temporal_decoder_output['video_queries'], temporal_decoder_output['frame_queries'], \
                                                            temporal_decoder_output['cross_attn_weights'],\
                                                         temporal_decoder_output['pred_masks'], temporal_decoder_output['multiscale_feats']

        if self.mode == 'rvos测试预训练vis':
            return {'tempdecoder': {'pred_masks': pred_masks_by_layer,}}
        elif self.mode == '只训练rvos':  
            grounding_score_by_layer = []
            for layer_idx, (frame_queries, temporal_queries, cross_attn_weights) in enumerate(zip(frame_queries_by_layer, 
                                                                                                  temporal_queries_by_layer, 
                                                                                                  cross_attn_weights_by_layer)):
                if self.reason_3d_layer_if_reason[layer_idx]:
                    grounding_score = self.reason_module_3d(temporal_queries=temporal_queries, 
                                                            frame_queries=frame_queries,
                                                             cross_attn_weights=cross_attn_weights, 
                                                             amrs=amrs,
                                                             amr_token_feats=amr_token_feats,
                                                             amr_token_seg_ids=amr_token_seg_ids,
                                                             node_alignments=node_alignments,
                                                             text_feats=text_feats,
                                                             text_pad_masks=text_pad_masks) # list[vi nq]
                    
                    if self.reason_3d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(torch.stack(grounding_score, dim=0))
                else:
                    grounding_score_by_layer.append(None)

            return {'temporal_decoder': {'pred_masks': pred_masks_by_layer, # list[b nq h w]
                                   'reason_3d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
        # b nq T h w
        out_mask_logits = decoder_layer_preds['objdecoder_objseg'].permute(0,2,1,3,4)
        if self.is_pretraining_seg:
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            # list[n t h w]
            tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
            for btc_idx in range(batch_size):
                start = int(self.obj_decoder_mask_out_stride // 2)
                im_h, im_w = tgt_masks[btc_idx].shape[-2:]
                tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
                assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
                assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
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

        new_targets = self.rvos_targets_handler(targets, pad_h=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        loss_value_dict = self.temporal_decoder_loss(model_outs, new_targets)

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

    def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
        # list[ni]
        is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> list[T]
        perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

        return {
            'masks': tgt_masks,
            'is_valid': is_valid,
            'referent_idx': gt_referent_idx,
            'has_ann': perFrame_has_ann
        }

    def temporal_decoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idxs = targets['referent_idx'] # list[int]
        is_valid = targets['is_valid'] # list[ni]
        num_objs = sum([is_v.int().sum() for is_v in is_valid])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        num_refs = sum([is_v[ref_idx].int() for is_v, ref_idx in zip(is_valid, referent_idxs)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers     

        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        src_masks_logits = out_mask_logits  # b nq T h w
        batch_size, nq, T, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni

            C = self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
        has_anntation = targets['has_ann'].bool() # list[T]
        is_valid = targets['is_valid'].bool() # list[ni]
        # b nq T H W -> list[ni t' H W]
        src_masks = [t[J][:, has_ann] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]
        
        # list[ni t' H W], b 
        tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        masks_losses = torch.cat([self.binary_cross_entropy_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)
        dice_losses = torch.cat([self.dice_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)

        losses = {
            "tempdecoder_mask": masks_losses.sum() / num_objs,
            "tempdecoder_dice": dice_losses.sum() / num_objs,
        }
        return losses    

    def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss


    def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, num_refs):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'].bool() # list[ni]
        ref_is_valid = torch.cat([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]) # b
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = tgt_idx.tolist().index(ref_idx)
            match_as_gt_idx = src_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs}


@register_model
def amr_grounding_3dobj(device, configs):
    model = AMR_Grounding_3DObj(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        work_dir=configs['work_dir'],
        max_stride=configs['max_stride'],
        pixel_mean=configs['pixel_mean'],
        pixel_std=configs['pixel_std'],
        mode=configs['mode'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        temporal_decoder=configs['temporal_decoder'],
        reason_module_3d=configs['reason_module_3d']
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("obj_decoder.backbone" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("obj_decoder.backbone" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] 
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    sch_conf = configs['optimization']['scheduler']
    if sch_conf['name'] == 'MultiStepLR':
        logging.info('你没用任何scheduler')
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
        logging.info('你没用任何scheduler')
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


