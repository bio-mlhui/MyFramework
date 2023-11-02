
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