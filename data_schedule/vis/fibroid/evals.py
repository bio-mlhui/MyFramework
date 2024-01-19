from data_schedule.vos.registry import register_vos_metric
import os
from glob import glob
from tqdm import tqdm
import shutil
from functools import partial
from PIL import Image
import numpy as np
import torch
import detectron2.utils.comm as comm
import logging
import pycocotools.mask as mask_util
from pycocotools.mask import decode as decode_rle
from data_schedule.vos.fibroid import metrics


# 把这些prediction mask可视化到output_dir, 并且打包
@register_vos_metric
def fibroid_web(model_preds,
                 output_dir,
                 **kwargs):
    assert comm.is_main_process()
    # output_dir: epc1_iter500_sap8099/eval_dataset1
    if os.path.exists(os.path.join(output_dir, 'web')):
        shutil.rmtree(os.path.join(output_dir, 'web'))
    os.makedirs(os.path.join(output_dir, 'web')) 

    for pred in model_preds:
        video_id = pred['video_id'] # str
        frame_name = pred['frame_name'] # list[str], t'
        masks = pred['masks']# list[rle], nq
        scores = pred['scores'] # nq

        max_idx = torch.tensor(scores).argmax()
        mask = masks[max_idx] # rle
        mask = decode_rle(mask) # h w
        mask = torch.as_tensor(mask, dtype=torch.uint8).contiguous()

        mask = Image.fromarray(255 * mask.int().numpy()).convert('L')

        save_path = os.path.join(output_dir, 'web', video_id)
        os.makedirs(save_path, exist_ok=True)
        mask.save(os.path.join(save_path, f'{frame_name}.png'))
    
    return {}
            

@register_vos_metric
def fibroid_all_medi(model_preds, 
                     dataset_meta,
                     **kwargs):
    assert comm.is_main_process()

    iou_by_test_sample = []
    dice_by_test_sample = []
    preds_by_test_sample = []
    gt_by_test_sample = []
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')

    for pred in model_preds:
        video_id = pred['video_id'] # str
        frame_name = pred['frame_name'] # list[str], t'
        masks = pred['masks']# list[rle], nq
        scores = pred['scores'] # nq

        max_idx = torch.tensor(scores).argmax()
        pred_mask = masks[max_idx] # rle
        pred_mask = decode_rle(pred_mask)
        pred_mask = torch.as_tensor(pred_mask, dtype=torch.uint8).contiguous() # h w

        gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
        gt_mask = gt_mask[0].int() # 0/1

        preds_by_test_sample.append(pred_mask)
        gt_by_test_sample.append(gt_mask)

        tp, fp, fn, tn = metrics.get_stats(pred_mask[None, None, ...], gt_mask[None, None, ...], 
                                           mode='binary')
        iou_score = metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        dice = metrics.dice(tp, fp, fn, tn, reduction='micro')

        iou_by_test_sample.append(iou_score)
        dice_by_test_sample.append(dice)

    mean_iou = torch.tensor(iou_by_test_sample).mean()
    mean_dice = torch.tensor(dice_by_test_sample).mean()

    preds_by_test_sample = torch.stack(preds_by_test_sample, dim=0).unsqueeze(1) # N 1 h w
    gt_by_test_sample = torch.stack(gt_by_test_sample, dim=0).unsqueeze(1) # N 1 h w

    tp, fp, fn, tn = metrics.get_stats(preds_by_test_sample, gt_by_test_sample, 
                                        mode='binary')
    overall_iou = metrics.iou_score(tp, fp, fn, tn, reduction='micro')    
    recall = metrics.recall(tp, fp, fn, tn, reduction='micro-imagewise') 
    precision = metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')

    all_medi = {
        'mean_iou': mean_iou,
        'dice': mean_dice,
        'overall_iou': overall_iou, # J/overallIoU
        'recall': recall,
        'precision': precision,
        'F': 2 * precision * recall / (precision + recall)
    }  
    return all_medi   
    