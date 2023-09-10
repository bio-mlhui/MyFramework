"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import nested_tensor_from_tensor_list_visiblility
from einops import rearrange
INF = 100000000

def dice_coef(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1).unsqueeze(1) # [N, 1, THW]
    targets = targets.flatten(1).unsqueeze(0) # [1, M, THW]
    numerator = 2 * (inputs * targets).sum(2)
    denominator = inputs.sum(-1) + targets.sum(-1)

    # NOTE coef doesn't be subtracted to 1 as it is not necessary for computing costs
    coef = (numerator + 1) / (denominator + 1)
    return coef

def sigmoid_focal_coef(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1) # [N, M, THW]
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1) # [N, M, THW]

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    coef = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        coef = alpha_t * coef

    return coef.mean(2) # [N, M]


def sigmoid_ce_coef(inputs, targets):
    """
    targets: 1 t h w
    inputs: n t h w
    """
    N, M = len(inputs), len(targets)
    inputs = inputs.flatten(1).unsqueeze(1).expand(-1, M, -1) # [N, M, THW]
    targets = targets.flatten(1).unsqueeze(0).expand(N, -1, -1) # [N, M, THW]

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")  # N M THW

    return ce_loss.mean(2) # [N, M]

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                       cost_mask: float = 1, cost_dice: float = 1, num_classes: int = 1, mask_loss_type = 'focal'):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_mask: This is the relative weight of the sigmoid focal loss of the mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_classes = num_classes
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 \
            or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.mask_out_stride = 4
        self.mask_loss_type = mask_loss_type

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
        targets = list(zip(*targets)) # list[list[dict], t_valid], b
        outputs = []
        for batch in targets:
            batch_out = {}
            labels = [time_batch['labels'][time_batch['referred_instance_idx']] for time_batch in batch] 
            boxes = [time_batch['boxes'][time_batch['referred_instance_idx']] for time_batch in batch]
            masks = [time_batch['masks'][time_batch['referred_instance_idx']] for time_batch in batch]
            valid = [time_batch['valid'][time_batch['referred_instance_idx']] for time_batch in batch]
            
            labels = torch.stack(labels, dim=0).long() # T(t )
            boxes = torch.stack(boxes, dim=0).float() # T(t 4)
            masks = torch.stack(masks, dim=0).float() # T(t h w)
            valid = torch.stack(valid, dim=0).long() # T(t )
            
            batch_out['labels'] = labels
            batch_out['boxes'] = boxes
            batch_out['masks'] = masks
            batch_out['valid'] = valid
            outputs.append(batch_out)
        return targets
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries_per_frame, num_frames, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries_per_frame, num_frames, 4] with the predicted box coordinates
                 "pred_masks": Tensor of dim [batch_size, num_queries_per_frame, num_frames, h, w], h,w in 4x size
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 NOTE: Since every frame has one object at most
                 "labels": Tensor of dim [num_frames] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_frames, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_frames, h, w], h,w in origin size 
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
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

                if self.mask_loss_type == 'focal':
                    assert False
                    out_prob_split = out_prob_split.sigmoid()
                    # Compute the classification cost.
                    alpha = 0.25
                    gamma = 2.0
                    neg_cost_class = (1 - alpha) * (out_prob_split ** gamma) * (-(1 - out_prob_split + 1e-8).log())
                    pos_cost_class = alpha * ((1 - out_prob_split) ** gamma) * (-(out_prob_split + 1e-8).log())
                    if self.num_classes == 1:  # binary referred
                        cost_class_split = pos_cost_class[:, [0]] - neg_cost_class[:, [0]]
                    else:
                        cost_class_split = pos_cost_class[:, tgt_ids_split] - neg_cost_class[:, tgt_ids_split] 
                        
                elif self.mask_loss_type == 'ce':
                    out_prob_split = out_prob_split.softmax(dim=-1)
                    if self.num_classes == 1:
                        cost_class_split = -out_prob_split[:, [0]] # n 1
                    else:
                        cost_class_split = -out_prob_split[:, tgt_ids_split]
                else:
                    raise ValueError()

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
                


def build_matcher(binary,
                  dataset_profile,
                  set_cost_class,
                  set_cost_bbox,
                  set_cost_giou,
                  set_cost_mask,
                  set_cost_dice,
                  mask_loss_type = 'focal'):
    if binary:
        num_classes = 1
    else:
        if dataset_profile == 'ytvos':
            num_classes = 65 
        elif dataset_profile== 'davis':
            num_classes = 78
        elif dataset_profile == 'a2d' or dataset_profile == 'jhmdb':
            num_classes = 1
        else: 
            num_classes = 91  # for coco
    return HungarianMatcher(cost_class=set_cost_class,
                            cost_bbox=set_cost_bbox,
                            cost_giou=set_cost_giou,
                            cost_mask=set_cost_mask,
                            cost_dice=set_cost_dice,
                            num_classes=num_classes,
                            mask_loss_type=mask_loss_type
                            )


