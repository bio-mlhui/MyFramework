from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from einops import repeat, rearrange
from models.layers.decoder_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, CrossAttentionLayer_ww
from models.layers.anyc_trans import MLP
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from scipy.optimize import linear_sum_assignment

from models.layers.utils import zero_module, _get_clones
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting


@META_ARCH_REGISTRY.register()
class FrameQuery_Refer(nn.Module):
    def __init__(self, 
                 configs,):
        loss_config = configs['loss']
        super().__init__(loss_config)

        attn_configs = configs['attn']
        inputs_projs = configs['inputs_projs'] # None/dict
        self.num_heads = attn_configs['nheads']
        self.nlayers = configs['nlayers']
        self.nqueries = configs['nqueries']
        reason_configs = configs['reason_module']
        self.mask_scale = configs['mask_scale']
        self.order = configs['order']
        d_model = configs['d_model']
        num_classes = configs['num_classes']
        self.cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            activation=attn_configs['activation'],
                                                            normalize_before=attn_configs['normalize_before']),
                                        self.nlayers)
        self.self_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=attn_configs['nheads'],
                                                          dropout=0.0,
                                                          activation=attn_configs['activation'],
                                                          normalize_before=attn_configs['normalize_before']),
                                       self.nlayers)  
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=attn_configs['dim_feedforward'],
                                               dropout=0.0,
                                               activation=attn_configs['activation'],
                                               normalize_before=attn_configs['normalize_before']),
                                      self.nlayers) 
        
        self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                         out_dim=d_model, 
                                                                         text_dim=None)

        self.query_feat = nn.Embedding(self.nqueries, d_model)
        self.query_embed = nn.Embedding(self.nqueries, d_model)

        self.query_norm = nn.LayerNorm(d_model)
        self.query_box = MLP(d_model, d_model, 4, 3)
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        self.query_class = nn.Linear(d_model, num_classes)
        if 'refer' in loss_config['losses']:
            self.reason_module = META_ARCH_REGISTRY.get(reason_configs['name'])(reason_configs)
        else:
            self.reason_module = None

    def forward(self,
                frame_queries=None, # b t nq c
                multiscales=None, # b c t h w
                text_inputs=None,):
        nqf = frame_queries.shape[2]
        batch_size, _, nf, *_ = multiscales[list(multiscales.keys())[0]].shape

        src = frame_queries.flatten(1,2).permute(1, 0, 2)   # t_nqf LB c
        # nq B c
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) 
        output = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)

        decoder_outputs = []
        cross_weight_by_layer = []
        for i in range(self.nlayers):
            if self.order == 'cross_self':
                output, cross_weight = self.cross_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # LB nq t_nqf
                output = self.self_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

            elif self.order == 'self_cross':
                output = self.self_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                output, cross_weight = self.cross_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # LB nq t_nqf                           
            output = self.ffn_layers[i](
                output
            )                
            cross_weight_by_layer.append(rearrange(cross_weight, 'b nq (t nqf) -> b nq t nqf',t=nf,nqf=nqf, b=batch_size))
            dec_out = self.query_norm(output) # nq b c
            decoder_outputs.append(dec_out.permute(1,0,2)) 

        mask_embeds = [self.query_mask(dec_o) for dec_o in decoder_outputs] # L b nq c
        pred_cls = [self.query_mask(dec_o) for dec_o in decoder_outputs] # L b nq class+1

        mask_features = multiscales[self.mask_scale] # b c t h w

        mask_features = self.vita_mask_features(mask_features.flatten(0,1)) # l_b_t c h w
        mask_features = rearrange(mask_features, '(L b t) c h w -> L b t c h w',L=L,b=B,t=T)
        pred_masks_by_layer = [torch.einsum('lbnc,lbtchw->lbnthw', mask_e, mask_features) for mask_e in mask_embeds]
    
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
        out = {
            'temporal_queries': torch.stack(decoder_outputs,dim=1), # L D b nq c
            'pred_masks': torch.stack(pred_masks_by_layer, dim=1), # L D b nq t h w
            'pred_logits': torch.stack(pred_cls), # L D b nq class+1
            'frame_queries':rearrange(src, '(t nqf) (L b) c -> L b t nqf c',t=T,nqf=nqf,L=L,b=B), # L b t nqf c
            'cross_attn_weights': torch.stack(cross_weight_by_layer, dim=1), # L D b nq t nqf
        }        
        return out 


class Video_SegmentationLoss_VOS111(nn.Module):
    def __init__(self, loss_config) -> None:
        super().__init__()
        self.matching_costs = loss_config['matching_costs']
        self.losses = loss_config['losses']
        self.foreground_weight = loss_config['foregound_weight']
        self.layer_weights = loss_config['layer_weights']

        self.register_buffer('class_weights', torch.tensor(loss_config['class_weights']).float())

    def compute_loss(self, model_outs_by_layer, targets):
        # list[{'pred_masks': b nq T h w, 'pred_obj': b nq 2,}], 
        # {'masks': list[T' h w], b}
        num_objs = sum([tgt_mask.flatten().any().int().sum().item() for tgt_mask in targets['masks']])
        batch_num_objs = sum(comm.all_gather(num_objs))
        num_objs = torch.clamp(torch.tensor(batch_num_objs).float().to(self.device) / comm.get_world_size(), min=1).item()

        assert len(model_outs_by_layer) == len(self.layer_weights)
        loss_values = {
            'loss_mask': torch.tensor(0.).to(self.device),
            'loss_dice': torch.tensor(0.).to(self.device),
            'loss_class': torch.tensor(0.).to(self.device),
        }
        for layer_weight, layer_out in zip(self.layer_weights, model_outs_by_layer):
            if layer_weight != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        loss_dict = self.masks_loss(layer_out, targets, matching_indices, num_objs)
                    elif loss == 'class':
                        loss_dict = self.class_loss(layer_out, targets, matching_indices, num_objs)
                    else:
                        raise ValueError()
                    for key in loss_dict:
                        loss_values[key] += loss_dict[key] * layer_weight
                    
        return loss_values      

    @torch.no_grad()
    def matching(self, model_outs, targets):
        # list[T' H W], b T
        tgt_masks, has_ann = targets['masks'], targets['has_ann']
        # b nq T H W -> list[nq T' h w], b
        src_masks = [pred_mk[:, taylor] for pred_mk, taylor in zip(model_outs['pred_masks'], has_ann)]
        # b nq c 
        src_class = model_outs['pred_class'].softmax(-1)
        batch_size, nq, *_ = src_class.shape
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks[i]  # nq T' H W
            tgt_mask = tgt_masks[i].to(out_mask).unsqueeze(0) # 1 T' H W
            tgt_ids = [0] * len(tgt_mask) # list[int], 1
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) #  nq 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq 1
            cost_class = - src_class[i][:, tgt_ids] # nq 1

            C = self.matching_costs['mask'] * cost_mask + \
                self.matching_costs['dice'] * cost_dice + \
                self.matching_costs['class'] * cost_class
            
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
    def masks_loss(self, model_outs, targets, matching_indices, num_objs):
        # list[T' H W], b; b T
        tgt_masks, has_ann = targets['masks'], targets['has_ann']
        assert len(has_ann) == len(model_outs['pred_masks'])
        # b nq T h w -> list[nq T' h w], batch
        pred_masks = [pred_mk[:, taylor] for pred_mk, taylor in zip(model_outs['pred_masks'], has_ann)] 

        assert len(pred_masks) == len(matching_indices)
         # list[1 T' h w], b
        src_masks = [t[J] for t, (J, _) in zip(pred_masks, matching_indices)] 
        loss_mask, loss_dice = torch.tensor(0.).to(self.device),  torch.tensor(0.).to(self.device)
        for btc_src_mk, btc_tgt_mk in zip(src_masks, tgt_masks):
            btc_tgt_mk = btc_tgt_mk.to(btc_src_mk)
            loss_mask += ce_mask_loss(btc_src_mk.flatten(1), btc_tgt_mk.flatten().unsqueeze(0), num_boxes=1)
            loss_dice += dice_loss(btc_src_mk.flatten(1), btc_tgt_mk.flatten().unsqueeze(0), num_boxes=1)

        losses = {
            "loss_mask": loss_mask / num_objs,
            "loss_dice": loss_dice / num_objs,
        }
        return losses    
    
    def get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def class_loss(self, model_outs, targets, matching_indices, num_objs):
        # b T, list[T' h w] b
        has_ann, tgt_masks = targets['has_ann'], targets['masks']
        src_logits = model_outs["pred_class"] # b nq c

        # b 1, 都是0
        tgt_labels = torch.zeros(len(tgt_masks)).long().to(self.device).unsqueeze(-1)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_labels, matching_indices)]).long()
        idx = self.get_src_permutation_idx(matching_indices)

        target_classes = torch.full(src_logits.shape[:2], len(self.class_weights) - 1, ).long().to(self.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.class_weights)
        losses = {"loss_class": loss_ce}
        return losses


@META_ARCH_REGISTRY.register()
class FrameQuery_VOS111(Video_SegmentationLoss_VOS111):

    def __init__(self, 
                 configs):
        loss_config = configs['loss']
        super().__init__(loss_config)

        attn_configs = configs['attn']
        inputs_projs = configs['inputs_projs'] # None/dict
        self.num_heads = attn_configs['nheads']
        self.nlayers = configs['nlayers']
        self.nqueries = configs['nqueries']
        d_model = configs['d_model']
        self.order = configs['order']
        self.mask_scale = configs['mask_scale']
        self.cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            activation=attn_configs['activation'],
                                                            normalize_before=attn_configs['normalize_before']),
                                        self.nlayers)
        self.self_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=attn_configs['nheads'],
                                                          dropout=0.0,
                                                          activation=attn_configs['activation'],
                                                          normalize_before=attn_configs['normalize_before']),
                                       self.nlayers)  
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=attn_configs['dim_feedforward'],
                                               dropout=0.0,
                                               activation=attn_configs['activation'],
                                               normalize_before=attn_configs['normalize_before']),
                                      self.nlayers) 
        
        self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, out_dim=d_model, 
                                                                         query_dim=None)

        self.query_feat = nn.Embedding(self.nqueries, d_model)
        self.query_embed = nn.Embedding(self.nqueries, d_model)

        self.query_norm = nn.LayerNorm(d_model)
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        self.query_is_obj = nn.Linear(d_model, 2)
        self.mask_stride = 4
    @property
    def device(self,):
        return self.query_embed.weight.device

    def forward_heads(self, output, mask_features):
        # b c t h w
        # nq lb c
        batch_size, _, nf = mask_features.shape[:3]
        _, LB, _ = output.shape
        L = LB // batch_size
        decoder_output = self.query_norm(output) # n Lb c
        decoder_output = decoder_output.transpose(0, 1)   # Lb n c

        class_logits = self.query_is_obj(decoder_output) # Lb nq c
        class_logits = rearrange(class_logits, '(L B) nq c -> L B nq c', L=L, B=batch_size)
        class_logits = class_logits.unbind(0)

        mask_embeds = self.query_mask(decoder_output)  # Lb n c
        mask_embeds = rearrange(mask_embeds, '(L B) n c ->L B n c', L=L, B=batch_size)
        mask_logits = torch.einsum("lbnc,bcthw->lbnthw", mask_embeds, mask_features)   # l b n t h w

        if not self.training:
            mask_logits = F.interpolate(mask_logits.flatten(0, 2), scale_factor=self.mask_stride, mode='bilinear', align_corners=False)
            mask_logits = rearrange(mask_logits, '(l b n) t h w -> l b n t h w', l=L, b=batch_size,)
        mask_logits = mask_logits.unbind(0)


        return class_logits, mask_logits,  rearrange(decoder_output, '(L B) n c -> L B n c', 
                                                     L=L, B=batch_size).unbind(0)

    def forward(self,
                frame_query, # lb t nqf c
                multiscales, # b c t h w
                ):
        multiscales, frame_query = self.inputs_projs(multiscales, frame_query)
        mask_features = multiscales[self.mask_scale]
        B, _, nf, *_ = mask_features.shape 
        src = frame_query.permute(1,2,0,3).flatten(0,1)   # t_nqf LB c
        LB = src.shape[1]
        L = LB // B
        # nq L*B c
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, LB, 1) 
        output = self.query_feat.weight.unsqueeze(1).repeat(1, LB, 1)

        ret = []
        for i in range(self.nlayers):
            if self.order == 'cross_self_lln':
                output = self.cross_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # nq lb c
                output = self.self_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

            elif self.order == 'self_cross_lln':
                output = self.self_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                output = self.cross_layers[i](
                    output, src,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=query_embed
                ) # nqf lb c
                                
            output = self.ffn_layers[i](
                output
            )          
            # list[b n c], list[b n t h w], list[b n c]
            class_logits, mask_logits, temporal_queries = self.forward_heads(output, mask_features)

            ret.extend([{'pred_class': pred_obj, 'pred_masks': pred_mk, 'temporal_queries': tem_query }
                            for pred_obj, pred_mk, tem_query in zip(class_logits, mask_logits, temporal_queries)
                        ])
        return ret





import torch.nn as nn
from models.layers.decoder_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.layers.anyc_trans import MLP
import torch.nn.functional as F
import torch
import copy
from models.layers.utils import zero_module, _get_clones
from models.layers.position_encoding import build_position_encoding
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting
from models.layers.utils import zero_module
from utils.misc import is_dist_avail_and_initialized
from collections import defaultdict
from detectron2.projects.point_rend.point_features import point_sample
from torch.cuda.amp import autocast
from .mask2former import Image_SetMatchingLoss
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
# 假设: 每个sample的 has_ann 都相同
class Video_SetMatchingLoss(nn.Module):
    def __init__(self, 
                 loss_config,
                 num_classes,) -> None:
        super().__init__()
        self.num_classes = num_classes # n=1 / n=0 / n>1
        self.matching_metrics = loss_config['matching_metrics'] # mask: mask/dice; point_sample_mask: ..
        self.losses = loss_config['losses'] 

        self.aux_layer_weights = loss_config['aux_layer_weights']  # int/list
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = loss_config['background_cls_eos']
        self.register_buffer('empty_weight', empty_weight)
        # self.register_buffer('small_obj_weight', torch.tensor(loss_config['small_obj_weight']).float())
        self._warmup_iters = 2000
        self.register_buffer("_iter", torch.zeros([1]))

    @property
    def device(self,):
        return self.empty_weight.device
    
    def compute_loss(self, 
                     model_outs, 
                     targets,
                     video_aux_dict,
                     **kwargs):
        # list[n t' h w], batch
        if 'masks' in targets:
            num_objs = sum([haosen.flatten(1).any(-1).int().sum().item() for haosen in targets['masks']])
        # list[n t' 4], batch
        elif 'boxes' in targets:
            # n t' 2 -> n t -> n
            num_objs = sum([(haosen[:, :, 2:] > 0).all(-1).any(-1).int().sum().item() for haosen in targets['boxes']])
        else:
            raise ValueError('targets里没有boxes/masks, 需要确定数量')
        
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / comm.get_world_size(), min=1).item()
        if isinstance(self.aux_layer_weights, list):
            assert len(self.aux_layer_weights) == (len(model_outs) - 1)
        else:
            self.aux_layer_weights = [self.aux_layer_weights] * (len(model_outs) - 1)

        layer_weights = self.aux_layer_weights + [1.]

        loss_values = {
            'mask_dice':0., 'mask_ce':0.,
            'box_l1': 0., 'box_giou': 0.,
            'class_ce':0., 'reason_ce': 0.,

            'mask_dice_smobj':0., 'mask_ce_smobj':0.,
            'boxMask_dice':0., 'boxMask_ce':0.,
            'color_similarity': 0.,
            'color_intra': 0.,
            'color_inter': 0.,
        }
        
        if ('mask_ce_dice' in self.matching_metrics) or ('mask_ce_dice' in self.losses):
            # mask interpolate
            tgt_mask_shape = targets['masks'][0].shape[-2:] # list[n t H W], b
            for layer_idx in range(len(model_outs)):
                # b t nq h w
                batch_size, nf = model_outs[layer_idx]['pred_masks'].shape[:2]
                model_outs[layer_idx]['pred_masks'] = rearrange(F.interpolate(model_outs[layer_idx]['pred_masks'].flatten(0, 1),
                                                                size=tgt_mask_shape, mode='bilinear', align_corners=False),
                                                                    '(b t) n h w -> b t n h w',b=batch_size, t=nf)
        for taylor, layer_out in zip(layer_weights, model_outs):
            if taylor != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    loss_extra_param = self.losses[loss]
                    if loss == 'mask_dice_ce' :
                        loss_dict = self.loss_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                           loss_extra_param=loss_extra_param)
                    elif loss == 'boxMask_dice_ce':
                        pass
                    elif loss == 'box_l1_giou':
                        loss_dict = self.loss_box_l1_giou(layer_out, targets, matching_indices, num_objs,
                                                          loss_extra_param=loss_extra_param)
                    elif loss == 'class_ce':
                        loss_dict = self.loss_class_ce(layer_out, targets, matching_indices, num_objs,
                                                       loss_extra_param=loss_extra_param)
                    elif loss == 'point_mask_dice_ce':
                        loss_dict = self.loss_point_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                                 loss_extra_param=loss_extra_param)
                    elif loss == 'color_similairty':
                        loss_dict = self.loss_color_similarity(layer_out, targets, matching_indices, num_objs,
                                                                 loss_extra_param=loss_extra_param, video_aux_dict=video_aux_dict)
                    elif loss == 'refer':
                        if 'refer_score' in layer_out:
                            loss_dict = self.loss_prrl(layer_out, targets, matching_indices, num_objs,
                                                                    loss_extra_param=loss_extra_param)
                        else:
                            loss_dict = {'reason_ce': 0}
                    else:
                        raise ValueError()
                    
                    for key, value in loss_dict.items():
                        loss_values[key] = loss_values[key] + value
    
        return loss_values      

    @torch.no_grad()
    def matching(self, layer_out, targets):
        batch_size = len(targets['masks']) if 'masks' in targets else len(targets['boxes'])
        indices = [] 
        has_ann = targets['has_ann']
        for i in range(batch_size):
            C = 0.

            if 'class_prob' in self.matching_metrics:
                out_cls = layer_out['pred_class'][i].softmax(-1) # nq c
                tgt_cls = targets['classes'][i] # n
                cost_class = - out_cls[:, tgt_cls] # nq n
                C += self.matching_metrics['class_prob']['prob'] * cost_class

            if 'mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i][has_ann[i]].permute(1, 0, 2, 3).contiguous()  # nq t' h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni t' H W
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) 
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
                C += self.matching_metrics['mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['mask_dice_ce']['dice'] * cost_dice

            if 'box_l1_giou' in self.matching_metrics:
                raise ValueError()
                out_box = layer_out['pred_boxes'][i][has_ann[i]].sigmoid() # nq t' 4
                tgt_bbox = targets['boxes'][i] # ni t' 4 
                cost_l1 = 0.
                cost_giou = 0.
                for haosen in range(tgt_bbox.shape[1]):
                    haosen_out_box = out_box[:, haosen]
                    haosen_tgt_box = tgt_bbox[:, haosen]
                    cost_l1 += torch.cdist(haosen_out_box, haosen_tgt_box, p=1) 
                    cost_giou += 1 - box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(haosen_out_box),
                                                                box_ops.box_cxcywh_to_xyxy(haosen_tgt_box))
                
                C += self.matching_metrics['box_l1_giou']['l1'] * cost_l1 + \
                      self.matching_metrics['box_l1_giou']['giou'] + cost_giou
 
            if 'point_mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i][has_ann[i]].permute(1, 0, 2, 3).contiguous() # nq t' h w
                tgt_mask = targets['masks'][i].to(out_mask)# ni t' H W
                nf = out_mask.shape[1]

                out_mask = out_mask.flatten(0, 1)[:, None]
                tgt_mask = tgt_mask.flatten(0, 1)[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.matching_metrics['point_mask_dice_ce']['num_points'],
                                           2, device=self.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1) # nqt s
                tgt_mask = rearrange(tgt_mask, '(nq t) s -> nq t s',t=nf)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1) # nit s
                out_mask = rearrange(out_mask, '(nq t) s -> nq t s',t=nf)
                with autocast(enabled=False):
                    out_mask = out_mask.float().flatten(1) # nq num_points
                    tgt_mask = tgt_mask.float().flatten(1)
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                C += self.matching_metrics['point_mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['point_mask_dice_ce']['dice'] * cost_dice
            
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def loss_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "mask_ce": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
            "mask_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
        }
        return losses   

    def loss_point_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        has_ann = targets['has_ann'] # b t
        src_masks = outputs['pred_masks'].permute(0, 2, 1, 3, 4).contiguous() # b nq t h w
        tgt_masks = targets['masks'] # list[n t' h w]
        # list[nq t' h w] -> n_sigma t' h w
        src_masks = torch.cat([t[J][:, haosen] for t, (J, _), haosen in zip(src_masks, indices, has_ann)],dim=0) 
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        nf = src_masks.shape[1]

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1).unsqueeze(1).contiguous() # nt' 1 h w
        target_masks = tgt_masks.flatten(0, 1).unsqueeze(1).contiguous() 

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                loss_extra_param['num_points'],
                loss_extra_param['oversample_ratio'],
                loss_extra_param['importance_sample_ratio'],
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1) # nt' s

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1) # nt' s

        # point_logits = rearrange(point_logits, '(n t) s -> n (t s)',t=nf)
        # point_labels = rearrange(point_labels, '(n t) s -> n (t s)',t=nf)

        losses = {
            "mask_dice": ce_mask_loss(point_logits, point_labels, num_objs),
            "mask_ce": dice_loss(point_logits, point_labels, num_objs),
        }

        del src_masks
        del target_masks
        return losses        

    def loss_class_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_class"].float() # b nq c

        idx = self._get_src_permutation_idx(indices)
        # list[n], b -> bn
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['classes'], indices)]) 
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"class_ce": loss_ce}
        return losses

    def loss_box_l1_giou(self, outputs, targets, indices, num_objs, loss_extra_param): 
        tgt_boxes = targets['boxes'] # list[n tl 4], b
        has_ann = targets['has_ann'] # b t

        src_boxes = outputs['pred_boxes'].sigmoid().permute(0, 2, 1, 3).contiguous() # b nq t 4

        src_boxes = torch.cat([t[J][:, haosen] for t, (J, _), haosen in zip(src_boxes, indices, has_ann)], dim=0) # n_sum t' 4
        tgt_boxes = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes, indices)], dim=0) # n_sum t' 4
        
        nf = tgt_boxes.shape[1]
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='none').flatten(1) # n_sum t'4

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes.flatten(0, 1)),
                                    box_ops.box_cxcywh_to_xyxy(tgt_boxes.flatten(0, 1)))) # n_sumt'
        
        loss_giou = loss_giou.view(-1, nf).contiguous()
        return {
            'box_l1': loss_l1.sum(-1).sum() / num_objs,
            'box_giou': loss_giou.sum(-1).sum() / num_objs
        }

    def loss_prrl(self, outputs, targets, indices, num_objs, loss_extra_param):
        layer_gscore_output = outputs['refer_score'] # b nq
        referent_idx = targets['referent_objs'] # list[list[int]], batch
        referent_idx = [haosen[0] for haosen in referent_idx]
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  indices): # b
            sel_idx = src_idx.tolist().index(ref_idx)
            match_as_gt_idx = tgt_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output, match_as_gt_indices, reduction='none') # b
        return {'reason_ce': choose_loss.sum() / len(referent_idx)}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

@META_ARCH_REGISTRY.register()
class TemporalEncoderReason(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.video_nqueries = configs['video_nqueries']
        self.nlayers = configs['nlayers']
        self.mask_scale = configs['mask_scale']
        reason_configs = configs['reason_module']
        self.mask_spatial_stride = multiscale_shapes[self.mask_scale].spatial_stride
        num_classes = configs['num_classes']
        from models.layers.anyc_trans import Linear_NormAct
        # self.input_projs = _get_clones(Linear_NormAct(in_features=d_model, out_features=d_model, norm='ln'), 3)
        self.cross_layers = _get_clones(CrossAttentionLayer_ww(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                        self.nlayers)
        self.self_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=attn_configs['nheads'],
                                                          dropout=0.0,
                                                          normalize_before=attn_configs['normalize_before']),
                                       self.nlayers)  
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=attn_configs['dim_feedforward'],
                                               dropout=0.0,
                                               normalize_before=attn_configs['normalize_before']),
                                      self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.temporal_query_poses = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_feats = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_norm = nn.LayerNorm(d_model)
        
        self.head_outputs = configs['head_outputs']
        assert 'mask' in self.head_outputs
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.head_outputs:
            self.query_class = nn.Linear(d_model, num_classes + 1)    

        self.loss_module = Video_SetMatchingLoss(loss_config=configs['loss'], num_classes=num_classes)
        self.reason_module = META_ARCH_REGISTRY.get(reason_configs['name'])(reason_configs)
        
    @property
    def device(self,):
        return self.temporal_query_feats.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c t h w 
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_3d(mem.permute(0, 2, 1,3, 4)).permute(0, 2, 1, 3, 4) for mem in memories]  # b c t h w
        memories = [rearrange(mem, 'b c t h w -> (t h w) b c').contiguous() for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c t h w -> (t h w) b c').contiguous() for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c t h w
        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales=None, # b c t h w
                frame_queries=None, #lb t nqf c
                text_inputs=None, #lb s c
                ):
        
        mask_features = multiscales[self.mask_scale] # b c t h w
        batch_size, _, nf, *_ = mask_features.shape
        LB, _, nqf = frame_queries.shape[:3]
        L = LB // batch_size
        text_inputs_by_layer = text_inputs.unbind(L)
        assert L == 3
        memory = rearrange(frame_queries, '(l b) t nqf c -> l (t nqf) b c',l=L, b=batch_size)
        memory = memory.unbind(0) # list[t_nqf b c]
        
        # nq b c
        temporal_query_poses = self.temporal_query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporal_query_feats = self.temporal_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)

        vid_ret = []
        # b nq class, b nq t h w
        vid_class, vid_mask = \
            self.forward_heads(temporal_query_feats=temporal_query_feats, mask_features=mask_features,) # first sight you re not human
        vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})

        for i in range(self.nlayers):
            level_index = i % len(memory)

            temporal_query_feats, cross_weight = self.cross_layers[i](
                tgt=temporal_query_feats, # nq b c
                memory=memory[level_index],  # t_nqf b c
                memory_mask=None,  # b*h nq thw
                memory_key_padding_mask=None,
                pos=None, 
                query_pos=temporal_query_poses, # nq b c
            )
            temporal_query_feats = self.self_layers[i](
                temporal_query_feats,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=temporal_query_poses,
            )
            temporal_query_feats = self.ffn_layers[i](
                temporal_query_feats 
            )
            # b nq class, b t nq h w
            vid_class, vid_mask = \
                self.forward_heads(temporal_query_feats=temporal_query_feats, mask_features=mask_features) # first sight you re not human
            # reason
            cross_weight = rearrange(cross_weight, 'b nq (t nqf) -> b nq t nqf',t=nf,nqf=nqf, b=batch_size)
            grounding_score = self.reason_module(temporal_queries=temporal_query_feats.permute(1,0,2),  # b nq c
                                                frame_queries=rearrange(memory[level_index], '(t nqf) b c -> b t nqf c',t=nf, nqf=nqf), # b t nqf c
                                                frame_queries_grounding_score=None, 
                                                cross_attn_weights=cross_weight,  # # b nq t nqf
                                                is_3d=True, is_2d=False,
                                                amrs=text_inputs_by_layer[level_index].amr,
                                                amr_token_feats=text_inputs_by_layer[level_index].amr_feats,
                                                amr_token_seg_ids=text_inputs_by_layer[level_index].amr_seg_ids,
                                                node_alignments=None,
                                                text_feats=text_inputs_by_layer[level_index].text_feats,
                                                text_pad_masks=text_inputs_by_layer[level_index].text_pad_masks) # list[vi nq]
            refer_score = torch.stack([haosen[0] for haosen in grounding_score], dim=0) # b nq
            if self.training:
                vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask, 'refer_score': refer_score})
            else:
                refer_score = torch.stack([refer_score.softmax(-1), torch.zeros_like(refer_score)])  # b nq 2
                refer_score = refer_score.unsqueeze(1).repeat(1, nf, 1, 1) # b t nq 2
                vid_ret.append({'pred_class':refer_score, 'pred_masks': vid_mask,})
        return vid_ret

    def forward_heads(self, temporal_query_feats,  mask_features): # nq b c; b c t h w
        batch_size, _, nf, *_ = mask_features.shape

        temporal_query_feats = self.temporal_query_norm(temporal_query_feats) # nq b c
        temporal_query_feats = temporal_query_feats.transpose(0, 1).contiguous() # b nq c

        class_logits = self.query_class(temporal_query_feats) if 'class' in self.head_outputs else None # b n class+1
        mask_embeds = self.query_mask(temporal_query_feats)  # b n c
        mask_logits = torch.einsum("bqc,bcthw->bqthw", mask_embeds, mask_features) 
        batch_size, nq, nf = mask_logits.shape[:3]
        mask_logits = F.interpolate(mask_logits.flatten(0, 1), scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)
        mask_logits = rearrange(mask_logits, '(b n) t h w -> b t n h w',b=batch_size, n=nq)

        if self.training:
            return class_logits, mask_logits
        else:
            # b t n c
            return class_logits.softmax(-1).unsqueeze(1).repeat(1, nf, 1, 1) if class_logits is not None else None, mask_logits
    
    def compute_loss(self, outputs, targets, video_aux_dict=None, **kwargs):
        assert self.training
        return self.loss_module.compute_loss(model_outs=outputs,
                                             targets=targets,
                                             video_aux_dict=video_aux_dict)

