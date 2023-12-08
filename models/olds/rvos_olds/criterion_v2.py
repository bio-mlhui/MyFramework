import torch
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops
from util.misc import (nested_tensor_from_tensor_list_visiblility,
                        get_world_size, is_dist_avail_and_initialized,)


from einops import rearrange

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


def sigmoid_mask_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Args:
        inputs: n_sigma thw
        targets: n_sigma thw
    """
    prob = inputs.sigmoid() # n_sigma thw 
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss.mean(1).sum() / num_boxes

def class_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Args:
        inputs: b (t nq) 2
        targets: b (t nq) 2, class probability, {0, 1}
    """
    assert False
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # b tn, log(1/pt)
    prob = inputs.softmax(dim=-1)  # b tn 2
    p_t = (prob * targets).sum(-1) # b tn  # pt
    loss = ce_loss * ((1 - p_t) ** gamma) # b tn

    if alpha >= 0:
        alpha_t = alpha * targets[:,:, 0] + (1 - alpha) * targets[:,:, 1]
        loss = alpha_t * loss
        
    return loss.mean(1).sum() / num_boxes


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

class SetCriterion(nn.Module):
    """ This class computes the loss for ReferFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25, mask_loss_type='focal'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef # [1, 0.1]
        self.register_buffer('empty_weight', empty_weight)
        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4
        self.mask_loss_type = mask_loss_type

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        indices: list[([idx], [0])], batch_size
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # b t n 2
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t n k -> b (t n) k')

        # list[T(tn), T(tn)], batch
        valid_indices = []
        valids = [target['valid'] for target in targets] # b个 T(t,)
        for valid, (indice_i, indice_j) in zip(valids, indices): 
            valid_ind = valid.nonzero().flatten()     #这个instance的哪些帧是有效的
            valid_i = valid_ind * nq + indice_i       # 由于是t*n排列, 所以 [_ _ _,  _ _ _,  ] 一共有t个，每个有n个，现在已经知道了这个sample选第indice_i个query
            valid_j = valid_ind + indice_j * nf       # valid_j没有用
            valid_indices.append((valid_i, valid_j))
        # ([0, 1, 2, ..batch],  [tn的index, ...])
        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  # b tn, 都初始化为 空类(1)
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0  # 选到的query的并且为valid的帧的class设置为 0， T(b tn)
        else:
            target_classes[idx] = target_classes_o
        # b tn 3 , 初始化为0， 由于是onehot, 将目标类别设置为1
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) 
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1) # self[i][j][index[i][j][k]] = 1
        # b tn 3.scatter([b tn 1]， 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        if self.mask_loss_type == 'focal': # b tn 2
            loss_ce = class_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        elif self.mask_loss_type == 'ce':
            # b tn 2 -> b 2 tn -> mean(b tn)
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes_onehot.transpose(1, 2), self.empty_weight)
        else:
            raise ValueError()
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)  

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]  
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
            list[([idx], [0])], batch
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]  # b t n h w
        src_masks = src_masks.transpose(1, 2)  # b n t h w

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list_visiblility([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) # b t h w

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] # n_sigma=b t h w;; 因为只有一个instance
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [n_sigma=b, thw]

        target_masks = target_masks.flatten(1) # [n_sigma=b, thw] ;; 因为只有一个instance, 所以不用target idx, n_simga = b 

        if self.mask_loss_type == 'focal': 
            loss_mask = sigmoid_mask_focal_loss(src_masks, target_masks, num_boxes)
        elif self.mask_loss_type == 'ce':
            loss_mask = ce_mask_loss(src_masks, target_masks, num_boxes)
        else:
            raise ValueError()
        
        losses = {
            "loss_mask": loss_mask,
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_attns(self, outputs, targets, indices, num_boxes):
        assert "pred_attns" in outputs
        if outputs['pred_attns'] == None:
            return {
                "loss_attn": 0.
            }
        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_attns"]  # b t n hi wi
        src_masks = src_masks.transpose(1, 2)  # b n t h w

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list_visiblility([t["masks"] for t in targets], 
                                                              size_divisibility=32, split=False).decompose()
        target_masks = target_masks.to(src_masks) # b t h w
        target_masks = F.interpolate(target_masks, size=src_masks.shape[-2:], mode='bilinear')
        target_masks = 2 * target_masks - 1

        src_masks = src_masks[src_idx] # n_sigma=b t h w;; 因为只有一个instance
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [n_sigma=b, thw]

        target_masks = target_masks.flatten(1) # [n_sigma=b, thw] ;; 因为只有一个instance, 所以不用target idx, n_simga = b 
        loss_attn = -1 * torch.clamp(src_masks * target_masks, max=0).mean(dim=1).sum() / num_boxes  
              
        return  {
            "loss_attn": loss_attn,
        }
         


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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'attns': self.loss_attns
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


    def targets_handler(self, targets):
        """
        targets: list[list[dict], b], t_valid
        
        targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 NOTE: Since every frame has one object at most
                 "labels": Tensor of dim [num_frames] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_frames, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_frames, h, w], h,w in origin size 
        """
        nf, bs = len(targets), len(targets[0])
        targets = list(zip(*targets)) # list[list[dict], batch], nf -> list[list[dict], nf], b
        outputs = []
        for batch in targets:
            batch_out = {}
            labels = [time_batch['labels'][time_batch['referred_instance_idx']] for time_batch in batch] # [0, 0, ] nf
            boxes = [time_batch['boxes'][time_batch['referred_instance_idx']] for time_batch in batch] # [T(4), T(4),], nf
            masks = [time_batch['masks'][time_batch['referred_instance_idx']] for time_batch in batch] # [T(h w), T(h,w)], nf
            valid = [time_batch['valid'][time_batch['referred_instance_idx']] for time_batch in batch] # [1, 1,, ], nf
            
            labels = torch.stack(labels, dim=0).long() # T(t ), 都是0
            boxes = torch.stack(boxes, dim=0).float() # T(t 4)
            masks = torch.stack(masks, dim=0).float() # T(t h w)
            valid = torch.stack(valid, dim=0).long() # T(t ), 如果数据增强导致某个帧的instance不见了
            
            batch_out['labels'] = labels
            batch_out['boxes'] = boxes
            batch_out['masks'] = masks
            batch_out['valid'] = valid
            outputs.append(batch_out)
        return outputs

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
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


