
import util.box_ops as box_ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import get_world_size, is_dist_avail_and_initialized
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def ce_mask_loss(inputs, targets, num_boxes):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: 
            b=n_sigma thw
        targets: b=n_sigma thw
            (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # n_sigma=b thw
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # mean(b mean(thw)), 对于a2d来说，num_boxes=
    return ce_loss.mean(1).sum() / num_boxes


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


######## 换decoder和matching
# n个query, 把一个视频当成多张图片
# 每个query都conditioned on sentence-level information
class Matching_V1(nn.Module):
    """
    每个query预测 referred概率, referent的box, referent的mask
    
    对于图片中没出现referent的情况, 每个query都预测class 0; box/mask不管
    对于图片中出现referent的情况
        进行matching, matched query进行box/mask预测
    """
    def __init__(self,  
                 eos_coef, 
                 losses, 
                 mask_out_stride,
                 matching_costs):
        super().__init__()
        self.num_classes = 2
        self.eos_coef = eos_coef
        self.losses = losses
        class_weight = torch.ones(2)
        class_weight[-1] = self.eos_coef
        self.register_buffer('class_weight', class_weight)
        self.mask_out_stride = mask_out_stride
        self.matching_costs = matching_costs

    def loss_labels(self, outputs, targets, indices, num_boxes):
        # indices: list[[int], [int]], batch_size 
        batch_size, nq, nf, _ = outputs['pred_logits'].shape # b n t 2
        
        src_logits = outputs['pred_logits']  # b n t 2
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool() # list[t]
        
        target_classes = torch.ones([batch_size, nq, nf], device=src_logits.device).long() # b n t
        
        for batch_idx in range(batch_size):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        # b*n*t 2; b*n*t
        loss_ce = F.cross_entropy(src_logits.flatten(0,2), target_classes.flatten(), self.class_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes): 
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool() # list[t] -> n_sigma*t
        src_boxes = outputs['pred_boxes']  # b n t 4
        src_boxes = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0)  # list[t 4] -> n_sigma*t 4
        target_boxes = torch.cat([t['boxes']['referent_idx'] for t in targets], dim=0).to(src_boxes)  # list[t 4] -> n_sigma*t 4
        
        src_boxes = src_boxes[is_consistent]
        target_boxes = target_boxes[is_consistent]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool() # list[t] -> n_sigma*t
        
        src_masks = outputs["pred_masks"]  # b n t h w        
        src_masks = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)  # list[t h w] -> n_sigma*t h w
    
        target_masks = torch.cat([t["masks"]['referent_idx'] for t in targets], dim=0).to(src_masks) # list[t h w] -> n_sigma*t h w
        
        src_masks = src_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        target_masks = target_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(1) * self.mask_out_stride == im_h
        assert target_masks.size(2) * self.mask_out_stride == im_w
  
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def matching(self, outputs, targets):
        src_logits = outputs["pred_logits"].transpose(1, 2) # b t n 2
        src_boxes = outputs["pred_boxes"].transpose(1, 2)   # b t n 4
        src_masks = outputs["pred_masks"].transpose(1, 2)   # b t n h w
        
        bs, nq, nf, h, w = src_masks.shape 

        target_masks = torch.stack([t["masks"]['referent_idx'] for t in targets], dim=0) # list[t h w] -> b t h w
        target_masks = target_masks.to(src_masks)
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w
        
        target_boxes = torch.stack([t['boxes']['referent_idx'] for t in targets], dim=0) # list[t 4] -> b t 4
        is_valid = torch.stack([t['valid']['referent_idx'] for t in targets], dim=0).bool() # list[t] -> b t

        indices = [] 
        for i in range(bs):
            out_prob = src_logits[i].softmax(dim=1) # t n 2
            out_bbox = src_boxes[i]  # t n 4
            out_mask = src_masks[i]  # t n h w

            tgt_bbox = target_boxes[i]     # t 4
            tgt_mask = target_masks[i]     # t h w
            tgt_valid = is_valid[i]    # t
            
            
            tgt_is_referred = (~tgt_valid).long()  # t
            out_prob = out_prob # t n 2
            
            cost_refer = []
            for t in range(nf):
                out_prob_split = out_prob[t]    # n 2
                tgt_is_referred_split = tgt_is_referred[t] # 0/1 
                cost_refer.append(-out_prob_split[:, [tgt_is_referred_split]])  # n 1
            cost_refer = torch.stack(cost_refer, dim=0).mean(0)  # [t n 1] -> [n, 1]

            cost_bbox, cost_giou = [], []
            for t in range(nf):
                out_bbox_split = out_bbox[t]    # n 4
                tgt_bbox_split = tgt_bbox[t].unsqueeze(0)  # 1 4

                cost_bbox_split = torch.cdist(out_bbox_split, tgt_bbox_split, p=1)   # n 1

                cost_giou_split = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox_split),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox_split)) # n 1
                
                cost_bbox.append(cost_bbox_split)
                cost_giou.append(cost_giou_split)
            cost_bbox = torch.stack(cost_bbox, dim=0).mean(0) # n 1
            cost_giou = torch.stack(cost_giou, dim=0).mean(0) # n 1

            
            cost_mask = batch_sigmoid_ce_loss(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0)) # n t h w : 1 t h w -> n 1
            cost_dice = -batch_dice_loss(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))

            # Final cost matrix
            C = self.matching_costs['cost_refer'] * cost_refer +\
                self.matching_costs['cost_bbox'] * cost_bbox + \
                self.matching_costs['cost_giou'] * cost_giou + \
                self.matching_costs['cost_mask'] * cost_mask + \
                self.matching_costs['cost_dice'] * cost_dice 

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        return indices

    def forward(self, outputs, targets, indices, ):
        # list[n t] -> list[t] -> b t
        target_valid = torch.stack([t["valid"]['referent_idx'] for t in targets], dim=0).reshape(-1)
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

# 换成focal loss
class Matching_V0_V1(nn.Module):
    """
    每个query预测 referred概率, referent的box, referent的mask
    
    对于图片中没出现referent的情况, 每个query都预测class 0; box/mask不管
    对于图片中出现referent的情况
        进行matching, matched query进行box/mask预测
    """
    def __init__(self,  
                 eos_coef, 
                 losses, 
                 mask_out_stride,):
        super().__init__()
        self.num_classes = 2
        self.eos_coef = eos_coef
        self.losses = losses
        class_weight = torch.ones(2)
        class_weight[-1] = self.eos_coef
        self.register_buffer('class_weight', class_weight)
        self.mask_out_stride = mask_out_stride

    def loss_labels(self, outputs, targets, indices, num_boxes):
        # indices: list[[int], [int]], batch_size 
        batch_size, nq, nf, _ = outputs['pred_logits'].shape # b n t 2
        
        src_logits = outputs['pred_logits']  # b n t 2
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t]
        
        target_classes = torch.ones([batch_size, nq, nf], device=src_logits.device).long() # b n t
        
        for batch_idx in range(batch_size):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        # b*n*t 2; b*n*t
        loss_ce = F.cross_entropy(src_logits.flatten(0,2), target_classes.flatten(), self.class_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes): 
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t] -> n_sigma*t
        src_boxes = outputs['pred_boxes']  # b n t 4
        src_boxes = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0)  # list[t 4] -> n_sigma*t 4
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0).to(src_boxes)  # list[t 4] -> n_sigma*t 4
        
        src_boxes = src_boxes[is_consistent]
        target_boxes = target_boxes[is_consistent]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t] -> n_sigma*t
        
        src_masks = outputs["pred_masks"]  # b n t h w        
        src_masks = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)  # list[t h w] -> n_sigma*t h w
    
        target_masks = torch.cat([t["masks"] for t in targets], dim=0).to(src_masks) # list[t h w] -> n_sigma*t h w
        
        src_masks = src_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        target_masks = target_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(1) * self.mask_out_stride == im_h
        assert target_masks.size(2) * self.mask_out_stride == im_w
  
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def matching(self, outputs, targets):
        src_logits = outputs["pred_logits"] # b t n 2
        src_boxes = outputs["pred_boxes"]   # b t n 4
        src_masks = outputs["pred_masks"]   # b t n h w
        
        bs, nf, nq, h, w = src_masks.shape 

        # handle mask padding issue # b t h_max w_max
        target_masks, valid = nested_tensor_from_tensor_list_visiblility([t["masks"] for t in targets], 
                                                             size_divisibility=32,
                                                             split=False).decompose()
        target_masks = target_masks.to(src_masks) # [B, T, H, W]

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        indices = [] # list[T([idx]), T([0])]
        for i in range(bs): # 得到这个sample的和referred object最match的query的idx
            out_prob = src_logits[i] # t n 2
            out_bbox = src_boxes[i]            # t n 4
            out_mask = src_masks[i]            # t n h w

            tgt_ids = targets[i]["labels"]     # t
            tgt_bbox = targets[i]["boxes"]     # t 4
            tgt_mask = target_masks[i]         # t h w
            tgt_valid = targets[i]["valid"]    # t

            # class cost
            # 每个query有t个pred, 对这t个Pred去平均得到这个query的分数
            cost_class = []
            for t in range(nf):
                if tgt_valid[t] == 0:
                    continue

                out_prob_split = out_prob[t]    # n 2
                tgt_ids_split = tgt_ids[t].unsqueeze(0)  # [0], 

                out_prob_split = out_prob_split.softmax(dim=-1)
                if self.num_classes == 1:
                    cost_class_split = -out_prob_split[:, [0]] # n 1
                else:
                    cost_class_split = -out_prob_split[:, tgt_ids_split]


                cost_class.append(cost_class_split)
            cost_class = torch.stack(cost_class, dim=0).mean(0)  # [t n 1] -> [n, 1]

            # box cost
            # we average the cost on every frame
            cost_bbox, cost_giou = [], []
            for t in range(nf):
                out_bbox_split = out_bbox[t]    # n 4
                tgt_bbox_split = tgt_bbox[t].unsqueeze(0)  # 1 4

                # Compute the L1 cost between boxes
                cost_bbox_split = torch.cdist(out_bbox_split, tgt_bbox_split, p=1)   # n 1

                # Compute the giou cost betwen boxes
                cost_giou_split = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_split),
                                                box_cxcywh_to_xyxy(tgt_bbox_split)) # n 1
                
                cost_bbox.append(cost_bbox_split)
                cost_giou.append(cost_giou_split)
            cost_bbox = torch.stack(cost_bbox, dim=0).mean(0) # n 1
            cost_giou = torch.stack(cost_giou, dim=0).mean(0) # n 1

            # mask cost
            # Compute the focal loss between masks
            if self.mask_loss_type == 'focal':
                cost_mask = sigmoid_focal_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))
            elif self.mask_loss_type == 'ce':
                cost_mask = sigmoid_ce_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0)) # n t h w : 1 t h w -> n 1
            else:
                raise ValueError()

            # Compute the dice loss betwen masks
            cost_dice = -dice_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + \
                self.cost_mask * cost_mask + self.cost_dice * cost_dice  # [n, 1]

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        # list[tuple], length is batch_size
        return indices

    def forward(self, outputs, ori_targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      list[{'labels':T(t h w)(都是0), 'masks':T(t h w), 'valid': T(t,), 'box': T(t, 4)}], batch
        """
        targets = self.targets_handler(ori_targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    for key in l_dict.keys():
                        losses[key] += l_dict[key]
        return losses

# 训练集中存在不consistent的sample
class Matching_V1(nn.Module):
    """
    每个query预测 referred概率, referent的box, referent的mask
    
    对于图片中没出现referent的情况, 每个query都预测class 0; box/mask不管
    对于图片中出现referent的情况
        进行matching, matched query进行box/mask预测
    """
    def __init__(self,  
                 eos_coef, 
                 losses, 
                 mask_out_stride,):
        super().__init__()
        self.num_classes = 2
        self.eos_coef = eos_coef
        self.losses = losses
        class_weight = torch.ones(2)
        class_weight[-1] = self.eos_coef
        self.register_buffer('class_weight', class_weight)
        self.mask_out_stride = mask_out_stride

    def loss_labels(self, outputs, targets, indices, num_boxes):
        # indices: list[[int], [int]], batch_size 
        batch_size, nq, nf, _ = outputs['pred_logits'].shape # b n t 2
        
        src_logits = outputs['pred_logits']  # b n t 2
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t]
        
        target_classes = torch.ones([batch_size, nq, nf], device=src_logits.device).long() # b n t
        
        for batch_idx in range(batch_size):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        # b*n*t 2; b*n*t
        loss_ce = F.cross_entropy(src_logits.flatten(0,2), target_classes.flatten(), self.class_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes): 
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t] -> n_sigma*t
        src_boxes = outputs['pred_boxes']  # b n t 4
        src_boxes = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0)  # list[t 4] -> n_sigma*t 4
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0).to(src_boxes)  # list[t 4] -> n_sigma*t 4
        
        src_boxes = src_boxes[is_consistent]
        target_boxes = target_boxes[is_consistent]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        is_consistent = torch.cat([t['valid'] for t in targets]).bool() # list[t] -> n_sigma*t
        
        src_masks = outputs["pred_masks"]  # b n t h w        
        src_masks = torch.cat([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)  # list[t h w] -> n_sigma*t h w
    
        target_masks = torch.cat([t["masks"] for t in targets], dim=0).to(src_masks) # list[t h w] -> n_sigma*t h w
        
        src_masks = src_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        target_masks = target_masks[is_consistent].flatten(1) # n_sigma_consistent*t h*w
        
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(1) * self.mask_out_stride == im_h
        assert target_masks.size(2) * self.mask_out_stride == im_w
  
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def matching(self, outputs, targets):
        src_logits = outputs["pred_logits"] # b t n 2
        src_boxes = outputs["pred_boxes"]   # b t n 4
        src_masks = outputs["pred_masks"]   # b t n h w
        
        bs, nf, nq, h, w = src_masks.shape 

        # handle mask padding issue # b t h_max w_max
        target_masks, valid = nested_tensor_from_tensor_list_visiblility([t["masks"] for t in targets], 
                                                             size_divisibility=32,
                                                             split=False).decompose()
        target_masks = target_masks.to(src_masks) # [B, T, H, W]

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        indices = [] # list[T([idx]), T([0])]
        for i in range(bs): # 得到这个sample的和referred object最match的query的idx
            out_prob = src_logits[i] # t n 2
            out_bbox = src_boxes[i]            # t n 4
            out_mask = src_masks[i]            # t n h w

            tgt_ids = targets[i]["labels"]     # t
            tgt_bbox = targets[i]["boxes"]     # t 4
            tgt_mask = target_masks[i]         # t h w
            tgt_valid = targets[i]["valid"]    # t

            # class cost
            # 每个query有t个pred, 对这t个Pred去平均得到这个query的分数
            cost_class = []
            for t in range(nf):
                if tgt_valid[t] == 0:
                    continue

                out_prob_split = out_prob[t]    # n 2
                tgt_ids_split = tgt_ids[t].unsqueeze(0)  # [0], 

                out_prob_split = out_prob_split.softmax(dim=-1)
                if self.num_classes == 1:
                    cost_class_split = -out_prob_split[:, [0]] # n 1
                else:
                    cost_class_split = -out_prob_split[:, tgt_ids_split]


                cost_class.append(cost_class_split)
            cost_class = torch.stack(cost_class, dim=0).mean(0)  # [t n 1] -> [n, 1]

            # box cost
            # we average the cost on every frame
            cost_bbox, cost_giou = [], []
            for t in range(nf):
                out_bbox_split = out_bbox[t]    # n 4
                tgt_bbox_split = tgt_bbox[t].unsqueeze(0)  # 1 4

                # Compute the L1 cost between boxes
                cost_bbox_split = torch.cdist(out_bbox_split, tgt_bbox_split, p=1)   # n 1

                # Compute the giou cost betwen boxes
                cost_giou_split = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_split),
                                                box_cxcywh_to_xyxy(tgt_bbox_split)) # n 1
                
                cost_bbox.append(cost_bbox_split)
                cost_giou.append(cost_giou_split)
            cost_bbox = torch.stack(cost_bbox, dim=0).mean(0) # n 1
            cost_giou = torch.stack(cost_giou, dim=0).mean(0) # n 1

            # mask cost
            # Compute the focal loss between masks
            if self.mask_loss_type == 'focal':
                cost_mask = sigmoid_focal_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))
            elif self.mask_loss_type == 'ce':
                cost_mask = sigmoid_ce_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0)) # n t h w : 1 t h w -> n 1
            else:
                raise ValueError()

            # Compute the dice loss betwen masks
            cost_dice = -dice_coef(out_mask.transpose(0, 1), tgt_mask.unsqueeze(0))

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + \
                self.cost_mask * cost_mask + self.cost_dice * cost_dice  # [n, 1]

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        # list[tuple], length is batch_size
        return indices

    def forward(self, outputs, ori_targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      list[{'labels':T(t h w)(都是0), 'masks':T(t h w), 'valid': T(t,), 'box': T(t, 4)}], batch
        """
        targets = self.targets_handler(ori_targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    for key in l_dict.keys():
                        losses[key] += l_dict[key]
        return losses


#### 
# 完全是单帧
# 每个query都conditioned
class Matching_V0(nn.Module):
    """
    每个query预测 referred概率, referent的box, referent的mask
    
    对于图片中没出现referent的情况, 每个query都预测class 0; box/mask不管
    对于图片中出现referent的情况
        进行matching, matched query进行box/mask预测
    """
    def __init__(self,  
                 eos_coef, 
                 losses, 
                 mask_out_stride,
                 matching_costs):
        super().__init__()
        self.num_classes = 2
        self.eos_coef = eos_coef
        self.losses = losses
        class_weight = torch.ones(2)
        class_weight[-1] = self.eos_coef
        self.register_buffer('class_weight', class_weight)
        self.mask_out_stride = mask_out_stride
        self.matching_costs = matching_costs

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        indices: [[], []], bt
        """
        src_logits = outputs['pred_logits']  # bt n 2
        bt, nq, _ = outputs['pred_logits'].shape # bt n 2
        
        # list[t] -> bt
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool() 
        target_classes = torch.ones([bt, nq], device=src_logits.device).long() # bt n
        
        for batch_idx in range(bt):
            target_classes[batch_idx, indices[batch_idx][0][0]] = (~is_consistent[batch_idx]).long()

        # btn 2, btn
        loss_ce = F.cross_entropy(src_logits.flatten(0,1), target_classes.flatten(), self.class_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes): 
        # list[t] -> bt
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool()
        src_boxes = outputs['pred_boxes']  # bt n 4
        # list[4] -> bt 4
        src_boxes = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_boxes)], dim=0) 
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes']['referent_idx'] for t in targets], dim=0).to(src_boxes)  
        
        src_boxes = src_boxes[is_consistent]
        target_boxes = target_boxes[is_consistent]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        # list[n t 4] -> list[t 4] -> bt 4
        is_consistent = torch.cat([t['valid']['referent_idx'] for t in targets]).bool()
        
        src_masks = outputs["pred_masks"]  # bt n h w  
        # list[h w] -> bt h w
        src_masks = torch.stack([src[J[0]] for (J, _), src in zip(indices, src_masks)], dim=0)  
        # list[n t h w] -> list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"]['referent_idx'] for t in targets], dim=0).to(src_masks) # list[t h w] -> n_sigma*t h w
        
        src_masks = src_masks[is_consistent].flatten(1) # bt hw
        target_masks = target_masks[is_consistent].flatten(1) # bt hw
        
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(1) * self.mask_out_stride == im_h
        assert target_masks.size(2) * self.mask_out_stride == im_w
  
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    @torch.no_grad()
    def matching(self, outputs, targets):
        src_logits = outputs["pred_logits"].transpose(1, 2) # bt n 2
        src_boxes = outputs["pred_boxes"].transpose(1, 2)   # bt n 4
        src_masks = outputs["pred_masks"].transpose(1, 2)   # bt n h w
        bt, nf, h, w = src_masks.shape 

        # list[t h w] -> bt h w
        target_masks = torch.cat([t["masks"]['referent_idx'] for t in targets], dim=0)
        target_masks = target_masks.to(src_masks)
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(1) * self.mask_out_stride == im_h
        assert target_masks.size(2) * self.mask_out_stride == im_w
        # list[t 4] -> bt 4
        target_boxes = torch.cat([t['boxes']['referent_idx'] for t in targets], dim=0) 
        # list[t] -> bt 4
        is_valid = torch.cat([t['valid']['referent_idx'] for t in targets], dim=0).bool()

        indices = [] 
        for i in range(bt):
            out_prob = src_logits[i].softmax(dim=-1) # n 2
            out_bbox = src_boxes[i]  # n 4
            out_mask = src_masks[i]  # n h w

            tgt_bbox = target_boxes[i].unsqueeze(0) # 1 4
            tgt_mask = target_masks[i].unsqueeze(0) # 1 h w
            tgt_valid = is_valid[i]    # True/False
            
            tgt_is_referred = (~tgt_valid).long()  # 1/0
            out_prob = out_prob # n 2
            
            cost_refer = -out_prob[:, [tgt_is_referred]] # n 1

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 1
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # n hw : 1 hw -> n 1
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = self.referent_decoder_matching_costs['cost_refer'] * cost_refer +\
                self.referent_decoder_matching_costs['cost_bbox'] * cost_bbox + \
                self.referent_decoder_matching_costs['cost_giou'] * cost_giou + \
                self.referent_decoder_matching_costs['cost_mask'] * cost_mask + \
                self.referent_decoder_matching_costs['cost_dice'] * cost_dice 

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        return indices

    def forward(self, outputs, targets, indices, ):
        # list[t] -> bt
        target_valid = torch.cat([t["valid"]['referent_idx'] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses





