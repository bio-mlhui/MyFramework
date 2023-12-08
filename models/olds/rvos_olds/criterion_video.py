import torch
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops
from util.misc import (nested_tensor_from_tensor_list_visiblility,
                        get_world_size, is_dist_avail_and_initialized,)
from detectron2.projects.point_rend.point_features import point_sample

from einops import rearrange
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from util.misc import NestedTensor

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from scipy.optimize import linear_sum_assignment

_matching_entrypoints = {}

def register_matching(fn):
    matching_name = fn.__name__
    _matching_entrypoints[matching_name] = fn

    return fn


def matching_entrypoints(matching_name):
    try:
        return _matching_entrypoints[matching_name]
    except KeyError as e:
        print(f'matching {matching_name} not found')


def nested_tensor_from_masks_list(tensor_list):
    # list[ni t hi wi]
    n_max = max(len(ttt) for ttt in tensor_list)
    H = max([a.shape[2] for a in tensor_list])
    W = max([a.shape[3] for a in tensor_list])
    bs, nf = len(tensor_list), tensor_list[0].shape[1]
    dtype, device = tensor_list[0].dtype, tensor_list[0].device
    tensor = torch.zeros([bs, n_max, nf, H, W], dtype=dtype, device=device)  # b n_max t H W
    mask = torch.ones([bs, n_max, H, W], dtype=torch.bool, device=device) # b n_max H W
    
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]].copy_(img)
        m[:img.shape[0], :img.shape[2], :img.shape[3]] = False
    return NestedTensor(tensor, mask)

def nested_tensor_from_box_list(tensor_list):
    # list[ni t 4] -> b n_max t 4
    n_max = max(len(ttt) for ttt in tensor_list)
    bs, nf = len(tensor_list), tensor_list[0].shape[1]
    dtype, device = tensor_list[0].dtype, tensor_list[0].device
    tensor = torch.zeros([bs, n_max, nf, 4], dtype=dtype, device=device)  # b n_max t H W
    mask = torch.ones([bs, n_max], dtype=torch.bool, device=device) # b n_max
    
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
        m[:img.shape[0]] = False
    return NestedTensor(tensor, mask)

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


class Matching_Video(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
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

# refer, masks 
class Matching_Video_Refer(Matching_Video):
    def __init__(self,
                 eos_coef, # 0.1
                 mask_out_stride, 
                 # important
                 losses,  # refer, masks
                 matching_costs, # {class, mask:, dice:}
                 mask_is_point,
                 num_points,
                 ):
        super().__init__()
        self.need_matching = True
        self.losses = losses
        self.costs = vars(matching_costs)
        
        assert ('refer' in losses) and ('masks' in losses)
        assert 'boxes' not in losses # video matching no boxes
        assert ('refer' in matching_costs) and ('mask' in matching_costs) and ('dice' in matching_costs)
        
        cls_weight = torch.ones(2)
        cls_weight[-1] = eos_coef 
        
        self.register_buffer('cls_weight', cls_weight)
        self.mask_is_point = mask_is_point
        self.num_points=num_points
        
        self.mask_out_stride = mask_out_stride

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'refer': self.loss_refer,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def loss_refer(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']  # b n 2
        assert src_logits.shape[-1] == 2

        # idx = self._get_src_permutation_idx(indices)
        gt_valids = torch.stack([t['valid'] for t in targets])  # list[t] -> b t
        gt_refer_labels = (~(gt_valids.any(dim=-1, keepdim=True))).long() # b 1

        target_classes = torch.ones(src_logits.shape[:2], device=src_logits.device).long() # b n 都是1
        for batch_idx, (src_idx,_) in enumerate(indices):
            target_classes[batch_idx][src_idx] = gt_refer_labels[batch_idx]
        # target_classes[idx] = gt_refer_labels # b n
    
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.cls_weight)
        losses = {'loss_refer': loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs # b t n h w

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"].transpose(1, 2)  # b n t h w

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
        target_masks = target_masks.to(src_masks) # b t h w

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        if self.mask_is_point:
            raise NotImplementedError()
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks[src_idx] # n_sigma=b t h w;; 因为只有一个instance
        # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False) 
        src_masks = src_masks.flatten(1) # [n_sigma=b, thw]

        target_masks = target_masks.flatten(1) # [n_sigma=b, thw] ;; 因为只有一个instance, 所以不用target idx, n_simga = b 
        
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    @torch.no_grad()
    def matching(self, outputs, targets):
        # targets: after refer handler
        src_logits = outputs["pred_logits"] # b n 2
        assert src_logits.shape[-1] == 2
        
        src_masks = outputs["pred_masks"]   # b t n h w
        
        bs, nf, nq, h, w = src_masks.shape 
        target_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
        target_masks = target_masks.to(src_masks) 

        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        indices = [] # list[T([idx]), T([0])]
        for i in range(bs): 
            out_prob = src_logits[i].softmax(dim=-1) # n 2
            out_mask = src_masks[i]  # t n h w

            tgt_valid = targets[i]["valid"]    # t
            tgt_refer_label = (~(tgt_valid.any())).long() # 如果至少一个帧是valid的, 那么gt refer label为0; 否则为1
            tgt_mask = target_masks[i]   # t h w
            cost_class = -out_prob[:, [tgt_refer_label]] # n 1

            if self.mask_is_point:
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
            else:
                cost_mask = batch_sigmoid_ce_loss(out_mask.transpose(0, 1).flatten(1), tgt_mask.unsqueeze(0).flatten(1)) # n thw; 1 thw
                cost_dice = batch_dice_loss(out_mask.transpose(0, 1).flatten(1), tgt_mask.unsqueeze(0).flatten(1)) # n thw; 1 thw

            # Final cost matrix
            C = self.costs['refer'] * cost_class + self.costs['mask'] * cost_mask + \
                self.costs['dice'] * cost_dice  # [n, 1]

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)  
            indices.append((src_ind.long(), tgt_ind.long()))
            
        # list[tuple], length is batch_size
        return indices

    def forward(self, outputs, targets, indices):
        # targets: after refer handler
        assert 'pred_logits' in outputs and ('pred_masks' in outputs) 
        
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
        return losses

@register_matching
def refer_video(matching_configs, ):
    configs = vars(matching_configs)
    assert set(configs['losses']) == set(['refer', 'masks'])
    return Matching_Video_Refer(eos_coef=configs['eos_coef'],
                                mask_out_stride=configs['mask_out_stride'],
                                # important
                                losses=configs['losses'], 
                                matching_costs=configs['matching_costs'],
                                mask_is_point=configs['mask_is_point'],
                                num_points=configs['num_points'])  
                                

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


# targets里的mask是b n t h w
# logits没有经过任何sigmod或者softmax

class Matching_Video_AllInstance(Matching_Video):
    def __init__(self,
                 num_classes,
                 losses,  # labels, masks, boxes, 作为dict的key使用
                 costs, # {ce:, mask:, dice:, giou, bbox}
                 eos_coef, 
                 mask_out_stride,
                 
                 mask_is_point,
                 num_points=None,
                 oversample_ratio=None,
                 importance_sample_ratio=None
                 ):
        super().__init__()
        assert num_classes > 2
        self.num_classes = num_classes + 1
        self.losses = losses
        self.costs = vars(costs)
        
        self.num_points=num_points
        self.mask_is_point = mask_is_point
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        assert ('labels' in losses) and ('masks' in losses)
        assert ('ce' in costs) and ('mask' in costs) and ('dice' in costs)
        assert ('boxes' not in losses)
        
        cls_weight = torch.ones(num_classes + 1)
        cls_weight[-1] = eos_coef 
        
        self.register_buffer('cls_weight', cls_weight)
    
        self.mask_out_stride = mask_out_stride

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_logits,
            'masks': self.loss_masks,
            'token': self.loss_token,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
      
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2] # b n k
        indices = []

        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"] # ni

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids] # nq ni

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.costs['mask'] * cost_mask
                + self.costs['ce'] * cost_class
                + self.costs['dice'] * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]        

    def loss_token(self, outputs, targets, indices, num_boxes,):
        """
        pred_token: b n 50732
        targets: 'token': b n
        """
        assert 'pred_token' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_token'] # b nq c
        src_logits = src_logits[src_idx] # n_sigma c
        
        # n_sigma
        target_token = torch.cat([t["token"][J] for t, (_, J) in zip(targets, indices)], dim=0).long()
        return {'loss_object_token': F.cross_entropy(src_logits, target_token,)}

    def loss_logits(self, outputs, targets, indices, num_boxes,):
        """
        targets: list[{'labels': b n}]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].permute(0, 2, 1)  # b c nq
        assert src_logits.shape[1] == self.num_classes
        bs, _, nq, device = *src_logits.shape, src_logits.device
        idx = self._get_src_permutation_idx(indices)
        # list(ni) -> n_sigma
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])  # list[ni, ] -> n_sigma
        
        # # valid应该已经在准备数据的时候已经准备好了
        # instance_valids = [t['valid'].long() for t in targets] # list[ni t]
        
        # # 不valid的帧让对应真是类别变为空类别
        # for i, t_valid in enumerate(instance_valids):
        #     # ni t -> ni
        #     instance_classes[i][~ (t_valid.any(dim=-1))] = self.num_classes
                
        target_classes = (torch.ones([bs, nq], device=device) * (self.num_classes - 1)).long() # b nq
        target_classes[idx] = target_classes_o # n_sigma = n_sigma
        
        return {
            'loss_object_ce': F.cross_entropy(src_logits, target_classes, weight=self.cls_weight)
        }


    def loss_masks(self, outputs, targets, indices, num_boxes):
        # indices: list[src=[10, 20, 4, 10], tgt=[3,1,0,2]]
        # targets: list[{"masks": n t h w}]
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"]  # b n t h w
        src_masks = src_masks[src_idx] # n_sigma t h w

        target_masks = torch.cat([t["masks"][J] for t, (_, J) in zip(targets, indices)], dim=0).to(src_masks) # list[ni(重新排序) t h w] -> n_sigma t h w

        src_masks = src_masks.flatten(0, 1)[:, None] # n_sigma*t 1 h w
        target_masks = target_masks.flatten(0, 1)[:, None] # n_sigma*t 1 h w
        
        if self.mask_is_point:
            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1) 
            
            mask_loss = ce_mask_loss(point_logits, point_labels, num_boxes)
            mask_dice_loss = dice_loss(point_logits, point_labels, num_boxes)
            
        else:
            # downsample ground truth masks with ratio mask_out_stride
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            
            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w

            mask_loss = ce_mask_loss(src_masks.flatten(1), target_masks.flatten(1), num_boxes)
            mask_dice_loss = dice_loss(src_masks.flatten(1), target_masks.flatten(1), num_boxes)
            
        del src_masks
        del target_masks    
        return  {
                "loss_object_mask": mask_loss,
                "loss_object_dice": mask_dice_loss,
            }
        
    @torch.no_grad()
    def matching(self, outputs, targets):
        """
        Params:
            "pred_logits": b n c
            "pred_masks": b n t h w
        """
        # b n c
        # b n t h w
        bs, num_queries, num_classes = outputs["pred_logits"].shape
        assert num_classes == self.num_classes
        indices = []

        # Iterate through batch size
        for b in range(bs):
            
            out_prob = outputs["pred_logits"][b].softmax(-1)  # nq c
            tgt_ids = targets[b]["labels"] # n

            cost_class = -out_prob[:, tgt_ids] # nq n

            out_mask = outputs["pred_masks"][b]  # nq t h w
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # n t h w

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            if self.mask_is_point:
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    
            else:
                start = int(self.mask_out_stride // 2)
                im_h, im_w = tgt_mask.shape[-2:]
                
                tgt_mask = tgt_mask[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
                assert tgt_mask.size(2) * self.mask_out_stride == im_h
                assert tgt_mask.size(3) * self.mask_out_stride == im_w
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq thw; n thw -> nq n
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq thw; 1 thw -> nq n
            
            # Final cost matrix
            C = (
                self.costs['mask'] * cost_mask
                + self.costs['ce'] * cost_class
                + self.costs['dice'] * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def forward(self, outputs, targets, indices):
        # targets: after refer handler
        assert 'pred_logits' in outputs and ('pred_masks' in outputs) 
        # list[n t (1是valid, 0是invalid)] -> n_sigma t
        target_valid = torch.cat([t["valid"] for t in targets], dim=0).reshape(-1) # n_sgima t
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses


@register_matching
def allinstance_video(configs, num_classes):
    return Matching_Video_AllInstance(
        num_classes=num_classes,
        losses=configs.losses,
        costs=configs.matching_costs,
        eos_coef=configs.eos,
        mask_out_stride=configs.mask_out_stride,
        mask_is_point=configs.mask_is_point,)



class No_Matching_Refer(Matching_Video):
    def __init__(self,
                 mask_out_stride, 
                 # important
                 losses,  # refer, masks
                 ):
        super().__init__()
        assert set(losses) == set(['masks'])
        self.losses = losses
        self.need_matching = False
        
        assert mask_out_stride == 4
        self.mask_out_stride = mask_out_stride

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def loss_masks(self, outputs, targets, num_boxes):
        assert "pred_masks" in outputs
        # b t h w

        src_masks = outputs["pred_masks"]  # b t h w

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = torch.stack([t['masks'] for t in targets], dim=0) # list[t h w] -> b t h w
        target_masks = target_masks.to(src_masks) # b t h w

        # downsample ground truth masks with ratio mask_out_stride
        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        
        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride] 
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_masks = src_masks.flatten(1) # [n_sigma=b, thw]

        target_masks = target_masks.flatten(1) # [n_sigma=b, thw] 
        
        losses = {
            "loss_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def forward(self, outputs, targets):
        """
        outputs:
        'pred_masks': b t h w
        """
        # targets: after refer handler
        assert 'pred_masks' in outputs
        
        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1) # [B, T] -> [B*T]
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_boxes))
        return losses


@register_matching
def no_matching_video(matching_configs,):
    configs = vars(matching_configs)
    assert set(configs['losses']) == set(['masks'])
    return No_Matching_Refer(mask_out_stride=configs['mask_out_stride'],
                                            losses=configs['losses']) 


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


class Matching_Refer_New:
    # 假设
    def loss_masks(self, outputs, targets, indices, num_boxes):
        pass
        # 对于 不consistent的text-video对, 即所有输入帧都没出现referent, 不做mask/box监督, 所有queries都预测class 1
        # 对于 某个时刻出现某个时刻不出现, is_referred_queries预测is_referred
        # 每个queries的temporal queries预测不出现/出现
        # 对于不出现的帧, 每个temporal queries不预测box/mask
        # 预测不出现的帧也做mask/box监督, 不出现的box监督预测[0,0,0,0];
    def loss_labels(self, outputs, targets, indices, num_boxes):
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
        # b tn 2 -> b 2 tn -> mean(b tn)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes_onehot.transpose(1, 2), self.empty_weight)
        losses = {'loss_refer': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # 只有一个box, 所以不需要tgt_index
        # 对于 不consistent的text-video对, 不做box/mask监督, 只做refer监督, 
        # 对于 某个时刻出现某个时刻不出现, 不出现的帧也做mask/box监督, 不出现的box监督预测[0,0,0,0]; 
        src_boxes = outputs['pred_boxes']  # b n t 4
        src_boxes = [src[J[0]] for (J, _), src in zip(indices, src_boxes)]  # list[t 4]
        target_boxes = [t['boxes'] for t in targets]  # list[t 4]
        
        is_valid = [t['valid'] for t in targets] # list[t]
        is_consistent_by_batch = [is_val.any().item() for is_val in is_valid] # list[True/False]
        
    
        loss_bbox = F.l1_loss(src_boxes[is_valid], target_boxes[is_valid], reduction='none') # n_sigma*t 4
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes[is_valid]),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes[is_valid])))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        # 对于不consistent的sample, 
        #     不预测mask
        # 对于有时候出现有时候不出现的sample
        #     被matched的object query的t个perframe query, 出现的帧预测mask, 不出现的帧不预测mask
        #     其他(n-1)*t个perframe query不预测mask
        # 对于整个帧都出现的sample
        #     被matched的object query的t个perframe query 预测mask
        #     其他(n-1)*t个perframe query不预测mask
        src_masks = outputs["pred_masks"]  # b n t h w
        batch_size = src_masks.shape[0]
        
        src_masks = [src[J[0]] for (J, _), src in zip(indices, src_masks)]   # list[t h w]
        is_valid = [t['valid'] for t in targets] # list[t]
        is_consistent_by_batch = [is_val.any().item() for is_val in is_valid] # list[True/False]
        target_masks = [t["masks"] for t in targets] # list[t h w]
        
        src_masks = [src_masks[i] for i in range(batch_size) if is_consistent_by_batch[i] == True]
        target_masks = [target_masks[i] for i in range(batch_size) if is_consistent_by_batch[i] == True]
        is_valid = [is_valid[i] for i in range(batch_size) if is_consistent_by_batch[i] == True]
        
        src_masks = torch.cat(src_masks, dim=0) # n_sigma*t h w
        target_masks = torch.cat(target_masks, dim=0) # n_sigma*t h w
        target_masks = target_masks.to(src_masks) 
        
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