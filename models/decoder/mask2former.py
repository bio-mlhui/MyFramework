# multi-scale features, b c h w -> module -> obj queries, predictions, b nq c
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

# referent/objects
# referent 可以看成是objects只有一个的情况

# 可以用到refer/vis身上
class Image_SegmentationLoss(nn.Module):
    def __init__(self, loss_config) -> None:
        super().__init__()
        self.style = loss_config['style']
        self.matching_costs = loss_config['matching_costs']
        self.aux_layers = loss_config['aux_layers']
        self.losses = loss_config['losses']
        self.foreground_weight = loss_config['foregound_weight']

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

    @torch.no_grad()
    def matching(self, out_mask_logits, targets):
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

            C = self.matching_costs['mask'] * cost_mask + \
                self.matching_costs['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def masks_loss(self, out_mask_logits, targets, matching_indices, num_boxes):
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

    def scores_loss(self, layer_gscore_output, matching_indices,  targets):
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

    def classes_loss():
        pass

    def boxes_loss():
        pass


# 考虑每个multiscale的维度都不相等
@META_ARCH_REGISTRY.register()
class Masked_Decoder_Multiscale(Image_SegmentationLoss):
    """ 假设: multiscale的维度都是d_model
    不利用text信息
    """
    def __init__(self,
                 configs,
                 ):
        assert configs['loss']['style'] == 'objects'
        super().__init__(configs['loss'])
        attn_configs = configs['attn']
        inputs_projs = configs['inputs_projs'] # None/dict

        d_model = configs['d_model']
        self.nqueries = configs['nqueries']
        self.nlayers = configs['nlayers']
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']
        num_classes = configs['num_classes']

        assert self.mask_scale not in self.memory_scales
        # decoder的输入要把memory一下, 还有res也proj
        # mask的表示: 每帧的query和mask feature进行卷积
        self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, text_dim=None, out_dim=d_model)

        self.query_poses = nn.Embedding(self.nqueries, d_model)
        self.query_feats = nn.Embedding(self.nqueries, d_model)
        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)

        assert self.nlayers % len(self.memory_scales) == 0

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
        self.nheads = attn_configs['nheads']

        self.query_norm = nn.LayerNorm(d_model)
        self.query_box = MLP(d_model, d_model, 4, 3)
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        self.query_class = nn.Linear(d_model, num_classes + 1)

        self.pos_2d = build_position_encoding(position_embedding_name='2d')

    @property
    def device(self,):
        return self.query_feats.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # projection maybe
        # b c h w -> hw b c
        memories = [rearrange(multiscales[scale], 
                              'b c t h w -> (b t) c h w') for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_2d(torch.zeros_like(mem)[:, 0, :, :].bool(), hidden_dim=mem.shape[1]) for mem in memories]
        memories = [rearrange(mem, 'bt c h w -> (h w) bt c') for mem in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]

        mask_features = multiscales[self.mask_scale] # b c t h w
        return memories, memories_poses, mask_features, size_list

    def forward_heads(self, output, mask_features, attn_mask_target_size):
        # b c t h w
        batch_size, _, nf = mask_features.shape[:3]
        decoder_output = self.query_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c

        box_logits = self.query_box(decoder_output) # bt n 4
        box_logits = rearrange(box_logits, '(b t) nq c -> b t nq c', b=batch_size, t=nf)

        mask_embeds = self.query_mask(decoder_output)  # bt n c
        mask_embeds = rearrange(mask_embeds, '(b t) n c -> b t n c', b=batch_size, t=nf)
        mask_logits = torch.einsum("btqc,bcthw->btqhw", mask_embeds, mask_features)  

        class_logits = self.query_class(decoder_output) # bt n class+1
        class_logits = rearrange(class_logits, '(b t) n c -> b t n c',b=batch_size, t=nf)

        attn_mask = mask_logits.detach().clone().flatten(0, 1)  # bt nq h w
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # b n h w -> b 1 n hw -> b head n hw -> b*head n hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        
        return class_logits, box_logits, mask_logits, attn_mask, rearrange(decoder_output, '(b t) n c -> b t n c', 
                                                                           b=batch_size, t=nf)
    
    def forward(self, 
                video_multiscales=None, # b c t h w,
                text_inputs=None):
        video_multiscales, text_inputs = self.inputs_projs(video_multiscales, text_inputs)
        # list[hw bt c], list[hw bt c], b c t h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(video_multiscales)
        batch_size_nf = memories[0].shape[1]
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        # nq bt c
        query_poses = self.query_poses.weight.unsqueeze(1).repeat(1, batch_size_nf, 1)
        query_feats = self.query_feats.weight.unsqueeze(1).repeat(1, batch_size_nf, 1)

        ret = []
        # bt nq c
        class_logits, box_logits, mask_logits, attn_mask, frame_queries = self.forward_heads(query_feats, mask_features, attn_mask_target_size=size_list[0])
        # b t n c, b t n 4, b t n h w, bt*head nq hw
        ret.append({'pred_class':class_logits, 'pred_masks': mask_logits, 'pred_boxes': box_logits, 'frame_queries': frame_queries })
        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # 全masked掉的 全注意, 比如有padding
            query_feats = self.cross_layers[i](
                tgt=query_feats,  # n b c
                memory=memories[level_index], # hw b  c
                memory_mask=attn_mask, # b*head n hw
                memory_key_padding_mask=None,
                pos=memories_poses[level_index],  # hw b  c
                query_pos=query_poses, # n b  c
            )
            query_feats = self.self_layers[i](
                query_feats, # n b c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_poses, # n b c
            )
            query_feats = self.ffn_layers[i](
                query_feats # n b c
            )
            class_logits, box_logits, mask_logits, attn_mask, frame_queries = self.forward_heads(query_feats, mask_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.memory_scales)])
            
            ret.append({'pred_class':class_logits, 'pred_masks': mask_logits, 'pred_boxes': box_logits, 'frame_queries': frame_queries })
        
        keys = list(ret[0].keys())
        collated_ret = {}
        for key in keys:
            collated_ret[key] = [ariana_grande[key] for ariana_grande in ret]
        return collated_ret, text_inputs




class Image_SegmentationLoss_VIS111(nn.Module):
    def __init__(self, loss_config) -> None:
        super().__init__()
        self.matching_costs = loss_config['matching_costs']
        self.losses = loss_config['losses']
        self.foreground_weight = loss_config['foregound_weight']
        self.layer_weights = loss_config['layer_weights']

        self.register_buffer('class_weights', torch.tensor(loss_config['class_weights']).float())

    def compute_loss(self, model_outs_by_layer, targets):
        # list[{'pred_masks', 'pred_boxes', 'pred_obj',}], {'masks', 'boxes'}
        # list[ni h w]
        num_objs = torch.cat(targets['masks'], dim=0).flatten(1).any(-1).int().sum().item()
        batch_num_objs = sum(comm.all_gather(num_objs))
        num_objs = torch.clamp(torch.tensor(batch_num_objs).float().to(self.device) / comm.get_world_size(), min=1).item()

        assert len(model_outs_by_layer) == len(self.layer_weights)
        loss_values = {
            'loss_mask': torch.tensor(0.).to(self.device),
            'loss_dice': torch.tensor(0.).to(self.device),
            'loss_l1': torch.tensor(0.).to(self.device),
            'loss_giou': torch.tensor(0.).to(self.device),
            'loss_class': torch.tensor(0.).to(self.device),
        }
        for layer_weight, layer_out in zip(self.layer_weights, model_outs_by_layer):
            if layer_weight != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        loss_dict = self.masks_loss(layer_out, targets, matching_indices, num_objs)
                    elif loss == 'boxes':
                        loss_dict = self.boxes_loss(layer_out, targets, matching_indices, num_objs)
                    elif loss == 'class':
                        loss_dict = self.class_loss(layer_out, targets, matching_indices, num_objs)
                    else:
                        raise ValueError()
                    for key in loss_dict:
                        loss_values[key] += loss_dict[key] * layer_weight
                    
        return loss_values      

    @torch.no_grad()
    def matching(self, model_outs, targets):
        # b nq H W
        # list[ni H W]
        tgt_masks, has_ann, tgt_boxes = targets['masks'], targets['has_ann'], targets['boxes']
        src_masks = model_outs['pred_masks'].flatten(0,1)[has_ann]  # bt' nq h w
        src_boxes = model_outs['pred_boxes'].flatten(0, 1)[has_ann].sigmoid() # bt' nq 4
        src_class = model_outs['pred_class'].flatten(0, 1)[has_ann].softmax(-1) # bt' nq c
        batch_size, nq, h, w = src_masks.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks[i]  # nq H W
            tgt_mask = tgt_masks[i].to(out_mask) # ni H W
            tgt_ids = [0] * len(tgt_mask) # list[int], ni
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
            out_bbox = src_boxes[i] # nq 4, logis
            tgt_bbox = tgt_boxes[i] # ni 4 
            cost_l1 = torch.cdist(out_bbox, tgt_bbox, p=1) 
            cost_giou = - box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                      box_ops.box_cxcywh_to_xyxy(tgt_bbox))
            
            cost_class = - src_class[i][:, tgt_ids]

            C = self.matching_costs['mask'] * cost_mask + \
                self.matching_costs['dice'] * cost_dice + \
                self.matching_costs['class'] * cost_class + \
                self.matching_costs['l1'] * cost_l1 + \
                self.matching_costs['giou'] * cost_giou
            
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def masks_loss(self, model_outs, targets, matching_indices, num_objs):
        # list[ni H W], bt'; bT
        tgt_masks, has_ann = targets['masks'], targets['has_ann']
        pred_masks = model_outs['pred_masks'].flatten(0, 1)[has_ann] # bT->bt' nq h w
        src_masks = torch.cat([t[J] for t, (J, _) in zip(pred_masks, matching_indices)], dim=0) # n_sum h w
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, matching_indices)], dim=0) # n_sum h w
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "loss_mask": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
            "loss_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
        }
        return losses    
    
    def boxes_loss(self, model_outs, targets, matching_indices, num_objs): 
        # list[ni 4], bt'
        tgt_boxes, has_ann = targets['boxes'], targets['has_ann']
        src_boxes = model_outs['pred_boxes'].flatten(0, 1)[has_ann].sigmoid() # bt' nq 4  

        src_boxes = torch.cat([t[J] for t, (J, _) in zip(src_boxes, matching_indices)], dim=0) # n_sum 4
        tgt_boxes = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes, matching_indices)], dim=0) # n_sum 4
        
        losses = {}
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='none') # n_sum 1
        losses['loss_l1'] = loss_l1.sum() / num_objs

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_objs
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
        # bT, list[ni h w] bt'
        has_ann, tgt_masks = targets['has_ann'], targets['masks']
        src_logits = model_outs["pred_class"].flatten(0,1)[has_ann] # bt' nq c
        # list[ni], bt', 都是类0
        tgt_labels = [torch.tensor([0] * len(bt_mk)).long().to(self.device) for bt_mk in tgt_masks]

        # list[n], bt'
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(tgt_labels, matching_indices)]).long()
        idx = self.get_src_permutation_idx(matching_indices)

        target_classes = torch.full(src_logits.shape[:2], len(self.class_weights) - 1, ).long().to(self.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.class_weights)
        losses = {"loss_class": loss_ce}
        return losses

@META_ARCH_REGISTRY.register()
class Masked_Decoder_Multiscale_VIS111(Image_SegmentationLoss_VIS111):
    """ 假设: multiscale的维度都是d_model
    不利用text信息
    """
    def __init__(self,
                 configs,
                 ):
        super().__init__(configs['loss'])
        attn_configs = configs['attn']
        inputs_projs = configs['inputs_projs'] # None/dict

        d_model = configs['d_model']
        self.nqueries = configs['nqueries']
        self.nlayers = configs['nlayers']
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']

        assert self.mask_scale not in self.memory_scales
        # decoder的输入要把memory一下, 还有res也proj
        # mask的表示: 每帧的query和mask feature进行卷积
        self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                          out_dim=d_model)

        self.query_poses = nn.Embedding(self.nqueries, d_model)
        self.query_feats = nn.Embedding(self.nqueries, d_model)
        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)

        assert self.nlayers % len(self.memory_scales) == 0

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
        self.nheads = attn_configs['nheads']

        self.query_norm = nn.LayerNorm(d_model)
        self.query_box = MLP(d_model, d_model, 4, 3)
        self.query_mask = MLP(d_model, d_model, d_model, 3)

        self.query_is_obj = nn.Linear(d_model, 2)
        self.pos_2d = build_position_encoding(position_embedding_name='2d')
        self.mask_stride = 4


    @property
    def device(self,):
        return self.query_feats.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # projection maybe
        # b c h w -> hw b c
        memories = [rearrange(multiscales[scale], 
                              'b c t h w -> (b t) c h w') for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_2d(torch.zeros_like(mem)[:, 0, :, :].bool(), hidden_dim=mem.shape[1]) for mem in memories]
        memories = [rearrange(mem, 'bt c h w -> (h w) bt c') for mem in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]

        mask_features = multiscales[self.mask_scale] # b c t h w
        return memories, memories_poses, mask_features, size_list

    def forward_heads(self, output, mask_features, attn_mask_target_size):
        # b c t h w
        batch_size, _, nf = mask_features.shape[:3]
        decoder_output = self.query_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c

        class_logits = self.query_is_obj(decoder_output) # bt nq 2
        class_logits = rearrange(class_logits, '(b t) nq c -> b t nq c', b=batch_size, t=nf)
        box_logits = self.query_box(decoder_output) # bt n 4
        box_logits = rearrange(box_logits, '(b t) nq c -> b t nq c', b=batch_size, t=nf)

        mask_embeds = self.query_mask(decoder_output)  # bt n c
        mask_embeds = rearrange(mask_embeds, '(b t) n c -> b t n c', b=batch_size, t=nf)
        mask_logits = torch.einsum("btqc,bcthw->btqhw", mask_embeds, mask_features)  

        attn_mask = mask_logits.detach().clone().flatten(0, 1)  # bt nq h w
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # b n h w -> b 1 n hw -> b head n hw -> b*head n hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        
        if not self.training:
            mask_logits = F.interpolate(mask_logits.flatten(0, 1), scale_factor=self.mask_stride, mode='bilinear', align_corners=True)
            mask_logits = rearrange(mask_logits, '(b t) n h w -> b t n h w',b=batch_size, t=nf)
        
        return class_logits, box_logits, mask_logits, attn_mask, rearrange(decoder_output, '(b t) n c -> b t n c', 
                                                                           b=batch_size, t=nf)
    
    def forward(self, 
                video_multiscales):
        video_multiscales = self.inputs_projs(video_multiscales)
        # list[hw bt c], list[hw bt c], b c t h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(video_multiscales)
        batch_size_nf = memories[0].shape[1]
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        # nq bt c
        query_poses = self.query_poses.weight.unsqueeze(1).repeat(1, batch_size_nf, 1)
        query_feats = self.query_feats.weight.unsqueeze(1).repeat(1, batch_size_nf, 1)

        ret = []
        # bt nq c
        class_logits, box_logits, mask_logits, attn_mask, frame_queries = self.forward_heads(query_feats, mask_features, attn_mask_target_size=size_list[0])
        # b t n c, b t n 4, b t n h w, bt*head nq hw
        ret.append({'pred_class': class_logits, 'pred_masks': mask_logits, 'pred_boxes': box_logits, 'frame_queries': frame_queries })
        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # 全masked掉的 全注意, 比如有padding
            query_feats = self.cross_layers[i](
                tgt=query_feats,  # n b c
                memory=memories[level_index], # hw b  c
                memory_mask=attn_mask, # b*head n hw
                memory_key_padding_mask=None,
                pos=memories_poses[level_index],  # hw b  c
                query_pos=query_poses, # n b  c
            )
            query_feats = self.self_layers[i](
                query_feats, # n b c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_poses, # n b c
            )
            query_feats = self.ffn_layers[i](
                query_feats # n b c
            )
            # bt nq c
            class_logits, box_logits, mask_logits, attn_mask, frame_queries = self.forward_heads(query_feats, mask_features, attn_mask_target_size=size_list[(i + 1) % len(self.memory_scales)])
            # b t n c, b t n 4, b t n h w, bt*head nq hw
            ret.append({'pred_class': class_logits, 'pred_masks': mask_logits, 'pred_boxes': box_logits, 'frame_queries': frame_queries })
        
        return ret