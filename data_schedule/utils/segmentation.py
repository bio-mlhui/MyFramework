import numpy as np

# def bounding_box_from_mask(mask):
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]
#     return rmin, rmax, cmin, cmax # y1, y2, x1, x2


def small_object_weighting(masks, sop):
    """
    Input:
        - masks:
            T(b 1 h w) {-1, 1}
    Output:
        - weighting:
            T(b 1 h w)
    """    
    if sop == 0.:
        return torch.ones_like(masks).float()
    else:
        # b 1 h w
        booled = (masks.detach() > 0).float()
        not_booled = 1 - booled
        
        # (b,)
        num_forground_pixels = booled.sum(dim=(1,2,3))
        whole_pixels = masks.shape[2] * masks.shape[3]
        num_background_pixels = whole_pixels - num_forground_pixels
        
        # (b 1 h w) / (b 1 1 1)
        weight = booled / (num_forground_pixels[:, None, None, None] **sop)
        weight += not_booled / (num_background_pixels[:, None, None, None] ** sop)
        
        # b 1 h w / b 1 1 1
        weight = whole_pixels * weight / (weight.sum(dim=(1, 2,3))[:, None, None, None])
        
        return weight

def bounding_box_from_mask(mask): # h w, bool
    if not mask.any():
        return torch.zeros([4]).float()
    rows = torch.any(mask, dim=1) # h
    cols = torch.any(mask, dim=0) # w
    row_indexs = torch.where(rows)[0]
    rmin, rmax = row_indexs.min(), row_indexs.max()

    col_indexs = torch.where(cols)[0]
    cmin, cmax = col_indexs.min(), col_indexs.max()
    return torch.tensor([cmin, rmin, cmax, rmax]).float() # x1y1x2y2

# def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the bounding boxes around the provided masks.

#     Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
#     ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

#     Args:
#         masks (Tensor[N, H, W]): masks to transform where N is the number of masks
#             and (H, W) are the spatial dimensions.

#     Returns:
#         Tensor[N, 4]: bounding boxes
#     """
#     if masks.numel() == 0:
#         return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

#     n = masks.shape[0]

#     bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

#     for index, mask in enumerate(masks):
#         y, x = torch.where(mask != 0) # h w

#         bounding_boxes[index, 0] = torch.min(x)
#         bounding_boxes[index, 1] = torch.min(y)
#         bounding_boxes[index, 2] = torch.max(x)
#         bounding_boxes[index, 3] = torch.max(y)

#     return bounding_boxes


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import decode
import numpy as np
import torch
from tqdm import tqdm

def get_AP_PAT_IOU_PerFrame(coco_evaluate_file, coco_predictions):
    coco_gt = COCO(coco_evaluate_file)
    coco_pred = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
    coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
    ap_metrics = coco_eval.stats[:6]
    metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}

    precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
    metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
    metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
    
    return metrics
    
def calculate_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_mask = decode(gt_annot['segmentation'])
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_mask = decode(pred_annot['segmentation'])
        iou, intersection, union = compute_iou(torch.tensor(pred_mask).unsqueeze(0),
                                               torch.tensor(gt_mask).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou

def compute_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-6):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zero if both are 0
    iou = (intersection + EPS) / (union + EPS)  # EPS is used to avoid division by zero
    return iou, intersection, union

# bbox
def calculate_bbox_precision_at_k_and_iou_metrics(coco_gt: COCO, coco_pred: COCO):
    print('evaluating bbox precision@k & iou metrics...')
    counters_by_iou = {iou: 0 for iou in [0.5, 0.6, 0.7, 0.8, 0.9]}
    total_intersection_area = 0
    total_union_area = 0
    ious_list = []
    for instance in tqdm(coco_gt.imgs.keys()):  # each image_id contains exactly one instance
        gt_annot = coco_gt.imgToAnns[instance][0]
        gt_bbox = gt_annot['bbox'] # xywh
        gt_bbox = [
            gt_bbox[0],
            gt_bbox[1],
            gt_bbox[2] + gt_bbox[0],
            gt_bbox[3] + gt_bbox[1],
        ]
        pred_annots = coco_pred.imgToAnns[instance]
        pred_annot = sorted(pred_annots, key=lambda a: a['score'])[-1]  # choose pred with highest score
        pred_bbox = pred_annot['bbox']  # xyxy
        iou, intersection, union = compute_bbox_iou(torch.tensor(pred_bbox).unsqueeze(0),
                                               torch.tensor(gt_bbox).unsqueeze(0))
        iou, intersection, union = iou.item(), intersection.item(), union.item()
        for iou_threshold in counters_by_iou.keys():
            if iou > iou_threshold:
                counters_by_iou[iou_threshold] += 1
        total_intersection_area += intersection
        total_union_area += union
        ious_list.append(iou)
    num_samples = len(ious_list)
    precision_at_k = np.array(list(counters_by_iou.values())) / num_samples
    overall_iou = total_intersection_area / total_union_area
    mean_iou = np.mean(ious_list)
    return precision_at_k, overall_iou, mean_iou

