from typing import Any, Optional, List, Dict, Set
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as Trans_F
from einops import repeat, reduce, rearrange
from utils.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from functools import partial
from utils.misc import to_device
from models.utils.visualize_amr import save_model_output
from models.registry import register_model
from data_schedule.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from models.optimization.scheduler import build_scheduler 
from data_schedule import build_schedule
from detectron2.config import configurable
from models.registry import register_model
import detectron2.utils.comm as comm
import copy
from models.optimization.utils import get_total_grad_norm
from models.optimization.optimizer import get_optimizer
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from models.backbone.utils import VideoMultiscale_Shape

class Decode_FrameQuery(nn.Module):
    def __init__(self,
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.loss_weight = configs['model']['loss_weights']
        video_text_backbone_configs = configs['model']['video_text_backbone'] 
        frame_encoder_configs = configs['model']['frame_encoder']
        frame_decoder_configs = configs['model']['frame_decoder']
        temporal_encoder_configs = configs['model']['temporal_encoder']
        temporal_decoder_configs = configs['model']['temporal_decoder']

        self.fusion_module =  META_ARCH_REGISTRY.get(configs['model']['fusion_module']['name'])(configs['model']['fusion_module'])

        video_text_backbone_cls = BACKBONE_REGISTRY.get(video_text_backbone_configs['name'])
        self.video_text_backbone = video_text_backbone_cls(video_text_backbone_configs)
        self.max_stride = self.video_text_backbone.max_stride

        frame_encoder_cls = META_ARCH_REGISTRY.get(frame_encoder_configs['name'])
        self.frame_encoder = frame_encoder_cls(frame_encoder_configs,
                                                multiscale_shapes=self.video_text_backbone.multiscale_shapes,
                                                text_dim=self.video_text_backbone.text_dim,
                                                fusion_module=self.fusion_module)
        
        same_dim_multiscale_shapes = VideoMultiscale_Shape.set_multiscale_same_dim(shape_by_dim=self.video_text_backbone.multiscale_shapes,
                                                                                   same_dim=configs['model']['fusion_module']['d_model'])  
        
        frame_decoder_cls = META_ARCH_REGISTRY.get(frame_decoder_configs['name'])
        self.frame_decoder = frame_decoder_cls(frame_decoder_configs, multiscale_shapes=same_dim_multiscale_shapes)
        
        temporal_encoder_cls = META_ARCH_REGISTRY.get(temporal_encoder_configs['name'])
        self.temporal_encoder = temporal_encoder_cls(temporal_encoder_configs, fusion_module=self.fusion_module)
        
        temporal_decoder_cls = META_ARCH_REGISTRY.get(temporal_decoder_configs['name'])
        self.temporal_decoder = temporal_decoder_cls(temporal_decoder_configs, multiscale_shapes=same_dim_multiscale_shapes,)
 
    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['weight_decay']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'backbone':None, 'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_text_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if log_lr_group_idx['backbone'] is None:
                        log_lr_group_idx['backbone'] = len(params)

                else:
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                                     
                # pos_embed, norm, embedding的weight decay特殊对待
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    logging.debug(f'setting weight decay of {module_name}.{module_param_name} to zero')
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
   
        return params, log_lr_group_idx

    def model_preds(self, videos, text_dict):
        # 抽特征
        multiscales, text_inputs = self.video_text_backbone(videos=videos,  text_dict=text_dict)  # b c t h w
        # 对比学习
        # self.video_text_backbone.compute_loss(multiscales, text_inputs)

        multiscales, text_inputs = self.frame_encoder(multiscales=multiscales, text_inputs=text_inputs)

        # b c t h w -> {'queries': list[b t nqf c], 'pred_masks': b t n h w}
        frame_decoder_output = self.frame_decoder(multiscales=multiscales)
        
        frame_query_by_layer = [query for query in frame_decoder_output[-3:]] # list[b t nqf c]
        frame_query_by_layer = [haosen['queries'] for haosen in frame_query_by_layer]
        # lb t nqf c
        text_inputs = text_inputs.repeat(3)
        frame_queries, text_inputs = self.temporal_encoder(frame_query_by_layer=frame_query_by_layer,  
                                                           text_inputs=text_inputs)
        
        # Lb t nqf c -> {'queries': list[Lb nq c], 'predictions': }
        temporal_decoder_output = self.temporal_decoder(frame_queries=frame_queries, 
                                                        text_inputs=text_inputs, 
                                                        multiscales=multiscales)
        return temporal_decoder_output

    def forward(self, batch_dict):
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # 0-1, float, b t 3 h w
        videos = (videos - self.pixel_mean) / self.pixel_std
        text_dict = to_device(batch_dict.pop('refer_dict'), self.device)
        targets = to_device(batch_dict.pop('targets'), self.device)
        frame_targets = to_device(batch_dict.pop('frame_targets'), self.device)
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch

        batch_size, T, _, H, W = videos.shape
        
        temporal_decoder_output = self.model_preds(videos, text_dict)
        # loss_value_dict = self.frame_decoder.compute_loss(frame_decoder_output, frame_targets)
        loss_value_dict = self.temporal_decoder.compute_loss(temporal_decoder_output, targets)

        loss_value_dict = {key: loss_value_dict[key] for key in list(self.loss_weight.keys())}
        # gradient_norm = get_total_grad_norm(self.model.parameters(), norm_type=2)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        videos = (videos - self.pixel_mean) / self.pixel_std
        text_dict = to_device(batch_dict.pop('refer_dict'), self.device)
        assert len(videos) == 1
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]

        batch_size, T, _, H, W = videos.shape

        if T > 24:
            split_outputs= []
            start = 0
            while start < T:
                split_vid = videos[:, start:(start+24)]
                split_dec_output = self.model_preds(split_vid, text_dict) 
                split_outputs.append(split_dec_output[-1])
                start = start + 24
            decoder_output = {}
            decoder_output['pred_masks'] = torch.cat([haosen['pred_masks']  for haosen in split_outputs], dim=1)
            decoder_output['pred_class'] = torch.cat([haosen['pred_class']  for haosen in split_outputs],dim=1)
                
        else:
            decoder_output = self.model_preds(videos, text_dict) 
            # 如果是List的话, 那么取最后一层
            if isinstance(decoder_output, list):
                decoder_output = decoder_output[-1]

        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear') > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    @torch.no_grad()
    def tta_sample(self, asembles, visualize_dir=None):
        # test time agumentation 
        preds_by_aug = []
        for sample in asembles:
            preds_by_aug.append(self.sample(asembles, visualize_dir=visualize_dir))
        return preds_by_aug


@register_model
def decode_frame_query(configs, device):
    model =  Decode_FrameQuery(configs)
    from .aux_mapper import AUXMapper_v1
    model.to(device)
    params_group, log_lr_group_idx = Decode_FrameQuery.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    train_samplers, train_loaders, eval_function, dataset_features = build_schedule(configs, 
                                                                                    model_input_mapper.mapper, 
                                                                                    partial(model_input_mapper.collate, max_stride=model.max_stride))

    # dataset_specific initialization

    return model, optimizer, scheduler,  train_samplers, train_loaders, log_lr_group_idx, eval_function


# model_aux_mapper(mode=train/evaluate, )
# model_aux_collate_fn(mode=train/evaluate, )

# class AMR_Grounding_3DObj(nn.Module):
#     def __init__(self, 
#                  d_model=256,
#                  max_stride=32,
#                  pt_dir='/home/xhh/pt',
#                  work_dir=None,
#                  mode=None,
#                 loss_weight={},
#                 tasks = { 'tempdecoder':{}},
#                 pixel_mean = [0.485, 0.456, 0.406],
#                 pixel_std = [0.229, 0.224, 0.225],
#                 # amrtext
#                 amrbart_wordEmbedding_freeze=True,
#                 amrtext_wordEmbedding_proj = {},
#                  temporal_decoder = {},
#                 reason_module_3d={},
#                 ) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.loss_weight = loss_weight
#         self.tasks = tasks
#         self.pt_dir = pt_dir
#         self.max_stride = max_stride
#         self.mode = mode
#         from .amr_utils.utils import BartForConditionalGeneration
#         AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
#         self.amrbart_wordEmbedding = AMRBart.model.shared
#         if amrbart_wordEmbedding_freeze:
#             for p in self.amrbart_wordEmbedding.parameters():
#                 p.requires_grad_(False) 
#         assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
#         self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)

#         self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

#         from models.pretrained_video_instance_decoder import pt_3d_obj_decoder_entrypoint
#         create_temporal_decoder = pt_3d_obj_decoder_entrypoint(temporal_decoder['name'])
#         self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir, work_dir)
#         self.temporal_decoder_num_layers = self.temporal_decoder.num_layers # 4层
#         self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
#         self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
#         if mode == 'rvos测试预训练vis':
#             pass
#         elif mode == '只训练rvos':
#             self.reason_3d_choose = reason_module_3d['choose_who']
#             self.reason_3d_layer_if_reason = reason_module_3d['layer_if_reason'] # obj_decoder的每层是否reason
#             assert self.reason_3d_layer_if_reason[-1]
#             assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers
#             from .layer_graph import graphLayer_entrypoint
#             create_reason_module = graphLayer_entrypoint(reason_module_3d['graph']['name'])
#             self.reason_module_3d = create_reason_module(reason_module_3d['graph'])
#         else:
#             raise ValueError()
        
#     @property
#     def device(self):
#         return self.pixel_mean.device

#     def encode_text(self, text_queries, text_auxiliary, device):
#         amrs = text_auxiliary['amrs'] # list[Graph]
#         batch_size = len(amrs)
#         text_tokens = text_auxiliary['text_token_ids'] # b smax
#         text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
#         text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
#         text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
#         text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
#         for batch_idx in range(batch_size):
#             text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
#         text_feats, text_pad_masks = pad_1d_feats(text_feats)       

#         amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
#         amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
#         amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
#         amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
#         amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
#         # list[list[ti c]] -> list[Vi+Ei c]
#         amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
#         for batch_idx in range(batch_size):
#             amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
#         amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
#         assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
#         assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
#         node_alignments = text_auxiliary['node_alignments']
#         return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

#     def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
#         """ text_auxiliary
#         'amrs': list[T(2 E_i)]
#         'seg_ids': b (V+E)max
#         'token_splits': list[list[int]]
#         'tokens_ids': b max
#         """
#         # 你想visualize的东西
#         check_visualize = {} 
#         nf, batch_size, *_ = samples.tensors.shape
#         device = samples.tensors.device

#         amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
#         # list[bt nq c], num_layers,  obj_queries
#         # list[bt nq h w], num_layers,  pred_masks
#         # TODO: 添加text 特征变换
#         temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(samples, 
                                                                                     
#                                                                                     amrs=amrs, 
#                                                                                     amr_token_feats=amr_token_feats,
#                                                                                     amr_token_seg_ids=amr_token_seg_ids, 
#                                                                                     text_feats=text_feats, 
#                                                                                     text_pad_masks=text_pad_masks)
#          # l b nq c, l b t nqf c, l b nq T nqf
#          # l b t nq h w,
#         temporal_queries_by_layer, frame_queries_by_layer, cross_attn_weights_by_layer,\
#               pred_masks_by_layer, multiscale_feats = temporal_decoder_output['video_queries'], temporal_decoder_output['frame_queries'], \
#                                                             temporal_decoder_output['cross_attn_weights'],\
#                                                          temporal_decoder_output['pred_masks'], temporal_decoder_output['multiscale_feats']

#         if self.mode == 'rvos测试预训练vis':
#             return {'tempdecoder': {'pred_masks': pred_masks_by_layer,}}
#         elif self.mode == '只训练rvos':  
#             grounding_score_by_layer = []
#             for layer_idx, (frame_queries, temporal_queries, cross_attn_weights) in enumerate(zip(frame_queries_by_layer, 
#                                                                                                   temporal_queries_by_layer, 
#                                                                                                   cross_attn_weights_by_layer)):
#                 if self.reason_3d_layer_if_reason[layer_idx]:
#                     grounding_score = self.reason_module_3d(temporal_queries=temporal_queries, 
#                                                             frame_queries=frame_queries,
#                                                              cross_attn_weights=cross_attn_weights, 
#                                                              amrs=amrs,
#                                                              amr_token_feats=amr_token_feats,
#                                                              amr_token_seg_ids=amr_token_seg_ids,
#                                                              node_alignments=node_alignments,
#                                                              text_feats=text_feats,
#                                                              text_pad_masks=text_pad_masks) # list[vi nq]
                    
#                     if self.reason_3d_choose == '第一个':
#                         grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
#                     else:
#                         raise ValueError()
#                     grounding_score_by_layer.append(torch.stack(grounding_score, dim=0))
#                 else:
#                     grounding_score_by_layer.append(None)

#             return {'temporal_decoder': {'pred_masks': pred_masks_by_layer, # list[b nq h w]
#                                    'reason_3d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

#     @torch.no_grad()
#     def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
#         if not isinstance(samples, NestedTensor):
#             samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
#         T, batch_size, _, H, W = samples.tensors.shape
#         perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
#         ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
#         perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
#         decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
#         # b nq T h w
#         out_mask_logits = decoder_layer_preds['objdecoder_objseg'].permute(0,2,1,3,4)
#         if self.is_pretraining_seg:
#             for idx in range(batch_size):
#                 h, w = targets[idx]['masks'].shape[-2:]
#                 # n t h w -> n t H W
#                 targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
#             # list[n t h w]
#             tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
#             for btc_idx in range(batch_size):
#                 start = int(self.obj_decoder_mask_out_stride // 2)
#                 im_h, im_w = tgt_masks[btc_idx].shape[-2:]
#                 tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
#                 assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
#                 assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

#             gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
#             _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
#             # list[t h w] -> b t h w
#             out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
#                                             for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
#                                                                                                          perFrame_has_ann)], dim=0)
#             out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
#         else:
#             refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
#             ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
#             ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
#             argmax_query_idx = ref_last_layer_gscore.argmax(-1)
#             # b T h w
#             out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
#             out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
#         # # # bt 1 h w
#         query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
#                                          scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
#         query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
#         # bt' 1
#         query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
#         size_original = [] #list[h,w], bt'
#         size_after_aug = [] #list[h,w], bt'
        
#         # 保证没有temporal增强
#         for bth_idx in range(batch_size):
#             size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
#             size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
#         processed_pred_masks = []
#         for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
#             f_mask_h, f_mask_w = after_aug_size  
#             f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
#             if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
#                 # n h w -> 1 n h w -> n h w
#                 f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
#             processed_pred_masks.append(f_pred_masks) # n h w
            
#         # list[n h w], bt -> list[n t' h w], b
#         by_batch_preds = []
#         by_batch_preds_probs = []
#         cnt = 0
#         # bt n -> list[n], bt -> list[n t'], b
#         query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
#         query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
#         for bth_idx in range(batch_size):
#             by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
#             by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
#             cnt += ann_number_by_batch[bth_idx]
#         assert cnt == len(processed_pred_masks)
#         return {
#             'query_pred_masks': by_batch_preds, # [n t' h w], batch
#             'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
#         }
  

#     def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
#         if not isinstance(samples, NestedTensor):
#             samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
#         T, batch_size, _, H, W = samples.tensors.shape

#         new_targets = self.rvos_targets_handler(targets, pad_h=H, pad_W=W)
#         model_outs = self.model_outputs(samples, text_queries, auxiliary) 
#         loss_value_dict = self.temporal_decoder_loss(model_outs, new_targets)

#         loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
#         if not math.isfinite(loss.item()):
#             print("Loss is {}, stopping training".format(loss.item()))
#             print(loss)
#             sys.exit(1)
#         loss.backward()
#         loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
#         loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
#         grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
#         return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

#     def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
#         batch_size = len(targets)
#         tgt_masks = [] 
#         # list[ni t' h w] -> list[ni t' H W]
#         for idx in range(batch_size):
#             _, _, h, w = targets[idx]['masks'].shape # ni t' h w
#             tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

#         for btc_idx in range(batch_size):
#             start = int(self.temporal_decoder_mask_out_stride // 2)
#             im_h, im_w = tgt_masks[btc_idx].shape[-2:]
#             tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
#             assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
#             assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

#         gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
#         # list[ni]
#         is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

#         perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
#         # list[t_video_i] -> list[T]
#         perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

#         return {
#             'masks': tgt_masks,
#             'is_valid': is_valid,
#             'referent_idx': gt_referent_idx,
#             'has_ann': perFrame_has_ann
#         }

#     def temporal_decoder_loss(self, model_outs, targets):
#         loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
#         tgt_masks = targets['masks'] # list[ni t' H W]
#         referent_idxs = targets['referent_idx'] # list[int]
#         is_valid = targets['is_valid'] # list[ni]
#         num_objs = sum([is_v.int().sum() for is_v in is_valid])
#         num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_objs)
#         num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

#         num_refs = sum([is_v[ref_idx].int() for is_v, ref_idx in zip(is_valid, referent_idxs)])
#         num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_refs)
#         num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
#         loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
#                       'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
#                       'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
#         out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
#         out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers     

#         for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
#             if layer_loss_weight != 0:
#                 matching_indices = self.temporal_decoder_matching(layer_mask_output, targets)
#                 if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
#                     assert layer_mask_output is not None
#                     masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
#                     for k in masks_losses.keys():
#                         loss_value[k] += layer_loss_weight * masks_losses[k]
#                 if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
#                     reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
#                     for k in reason_loss.keys():
#                         loss_value[k] += layer_loss_weight * reason_loss[k]
#         return loss_value     

#     @torch.no_grad()
#     def temporal_decoder_matching(self, out_mask_logits, targets):
#         perFrame_has_ann = targets['has_ann'] # list[T]
#         tgt_masks = targets['masks'] # list[ni t' H W]
#         src_masks_logits = out_mask_logits  # b nq T h w
#         batch_size, nq, T, h, w = src_masks_logits.shape 
#         indices = [] 
#         for i in range(batch_size):
#             out_mask = src_masks_logits[i]  # nq T h w
#             out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
#             tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W
            
#             cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni
#             cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni

#             C = self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
#                 self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice 
#             C = C.cpu()
#             indices.append(linear_sum_assignment(C))
            
#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ]

#     def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
#         has_anntation = targets['has_ann'].bool() # list[T]
#         is_valid = targets['is_valid'].bool() # list[ni]
#         # b nq T H W -> list[ni t' H W]
#         src_masks = [t[J][:, has_ann] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]
        
#         # list[ni t' H W], b 
#         tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
#         # 每个视频可能有不同的annotation数量
#         # list[ni] -> n_sigma
#         masks_losses = torch.cat([self.binary_cross_entropy_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)
#         dice_losses = torch.cat([self.dice_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)

#         losses = {
#             "tempdecoder_mask": masks_losses.sum() / num_objs,
#             "tempdecoder_dice": dice_losses.sum() / num_objs,
#         }
#         return losses    

#     def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
#         # n T h w, n t h w, T, -> list[cross_entropy], n
#         src_masks = src_masks.flatten(1) # n thw
#         tgt_masks = tgt_masks.flatten(1) # n thw

#         ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
#         ce_loss = ce_loss.mean(-1) # n
#         return ce_loss
    
#     def dice_mask_loss(self, src_masks, tgt_masks):
#         # n T h w, n t h w, -> n
#         src_masks = src_masks.flatten(1) # n thw
#         tgt_masks = tgt_masks.flatten(1).float() # n thw

#         src_masks = src_masks.sigmoid()
#         numerator = 2 * ((src_masks * tgt_masks).sum(1))
#         denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
#         loss = 1 - (numerator + 1) / (denominator + 1)
#         return loss


#     def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, num_refs):
#         # b nq
#         referent_idx = targets['referent_idx'] # list[int], batch
#         is_valid = targets['is_valid'].bool() # list[ni]
#         ref_is_valid = torch.cat([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]) # b
#         match_as_gt_indices = [] # list[int], b
#         for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
#             sel_idx = tgt_idx.tolist().index(ref_idx)
#             match_as_gt_idx = src_idx[sel_idx]
#             match_as_gt_indices.append(match_as_gt_idx.item())
#         match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
#         choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
#         return {'tempdecoder_reason': choose_loss.sum() / num_refs}


