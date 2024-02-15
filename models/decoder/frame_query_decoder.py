from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat, rearrange
from models.layers.decoder_layers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from models.layers.anyc_trans import MLP
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from scipy.optimize import linear_sum_assignment

from models.layers.utils import zero_module, _get_clones
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting

class Image_SegmentationLoss(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.losses = configs['losses'] # masks, boxes, classes, scores,
        self.style = configs['style'] # referent/objects
        self.matching_costs = configs['matching_costs']
        self.aux_layers = configs['aux_layers']
        self.foreground_weight = configs['foregound_weight']

        # referent 可以generalize 成一个object

    def handle_targets(self, targets):
        # 缩放target masks
        if 'masks' in targets:
            target_masks = targets['masks'] # list[ni h w]
            batch_size = len(target_masks)
            for btc_idx in range(batch_size):
                start = int(self.mask_out_stride // 2)
                im_h, im_w = target_masks[btc_idx].shape[-2:]
                target_masks[btc_idx] = target_masks[btc_idx][:, start::self.mask_out_stride, start::self.mask_out_stride] 
                assert target_masks[btc_idx].size(1) * self.mask_out_stride == im_h
                assert target_masks[btc_idx].size(2) * self.mask_out_stride == im_w
            targets['masks'] = target_masks

        # 根据style改变目标
        if self.style == 'referent':
            if 'boxes' in targets:
                # list[1 4]
                targets['boxes'] = [box[[ref]] for box, ref in zip(targets['boxes'], targets['referent_idx'])]
            if 'masks' in targets:
                targets['masks'] = [mask[[ref]] for mask, ref in zip(targets['masks'], targets['referent_idx'])]
        return targets

    def compute_loss(self, model_outs, targets):
        targets = self.handle_targets(targets)
        # list[{'pred_masks', 'pred_boxes':}], {'masks', 'boxes'}

        # list[ni h w]
        num_objs = torch.stack(targets['masks'], dim=0).any(-1).int().sum()
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        assert len(model_outs) == len(self.layer_weights)

        loss_values = {}

        for layer_weight, layer_out in zip(model_outs, self.layer_weights):
            if layer_weight != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        loss_values.update(self.masks_loss(model_outs, targets, matching_indices, num_objs))
                    if loss == 'boxes':
                        loss_values.update(self.boxes_loss(model_outs, targets, matching_indices, num_objs))
                    if loss == 'classes':
                        loss_values.update(self.classes_loss(model_outs, targets, matching_indices, num_objs))
                    if loss == 'scores':
                        loss_values.update(self.scores_loss(model_outs, targets, matching_indices, num_objs))

        return loss_values      
    
    def compute_loss(self, model_outs, targets):
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
    def matching(self, out_mask_logits, out_class_logits, targets):
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

    def classes_loss(self, layer_out_logits, targets, matching_indices, num_objs):
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

    def masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
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

    def scores_loss(self, layer_gscore_output, targets, matching_indices, global_num_refs):
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

    def boxes_loss():
        pass


@META_ARCH_REGISTRY.register()
class FrameQuery_Refer(Image_SegmentationLoss):
    def __init__(self, 
                 configs,
                 reason_module):
        loss_config = configs['loss']
        super().__init__(loss_config)

        attn_configs = configs['attn']
        inputs_projs = configs['inputs_projs'] # None/dict
        self.num_heads = attn_configs['nheads']
        self.nlayers = configs['nlayers']
        self.nqueries = configs['nqueries']
        d_model = configs['d_model']
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
        
        self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, out_dim=d_model, text_dim=None, feat_dim=None, query_dim=None)

        self.query_feat = nn.Embedding(self.nqueries, d_model)
        self.query_embed = nn.Embedding(self.nqueries, d_model)

        self.query_norm = nn.LayerNorm(d_model)
        self.query_box = MLP(d_model, d_model, 4, 3)
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'refer' in loss_config['losses']:
            self.reason_module = reason_module
        else:
            self.reason_module = None

    def forward(self,
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
            elif self.order == 'cross_lln':
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

        # L D b nq c
        # L b t nqf c
        # L D b nq t nqf
        # L D b nq t h w
        # L D b nq class+1
        
        mask_embeds = [self.mask_embed(dec_o) for dec_o in decoder_outputs] # L b nq c
        pred_cls = [self.class_embed(dec_o) for dec_o in decoder_outputs] # L b nq class+1
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
