from data_schedule.vis.evaluator_utils import register_vis_metric
from data_schedule.vis.apis import VIS_Evaluator_OutAPI_EvalFn_API
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
import subprocess
import data_schedule.vis.polyp.metrics as metrics

@register_vis_metric
def polyp_web(model_preds,
              output_dir,
             **kwargs):
    assert comm.is_main_process()
    # output_dir: epc1_iter500_sap8099/eval_dataset1/web
    if os.path.exists(os.path.join(output_dir, 'web')):
        shutil.rmtree(os.path.join(output_dir, 'web'))
    os.makedirs(os.path.join(output_dir, 'web')) 
    VIS_Evaluator_OutAPI_EvalFn_API
    for pred in model_preds:
        video_id = pred['video_id'] # str
        frame_name = pred['frame_name'] # str
        masks = pred['masks']# list[rle], nt
        boxes = pred['boxes'] # nt 4, x1y1x2y2绝对值
        scores = pred['classes'] # nt c

        foreground_scores = scores[:, :-1].sum(-1) # nq

        max_idx = torch.tensor(foreground_scores).argmax()
        mask = masks[max_idx] # rle
        mask = decode_rle(mask) # h w
        mask = torch.as_tensor(mask, dtype=torch.uint8).contiguous()

        mask = Image.fromarray(255 * mask.int().numpy()).convert('L')

        save_path = os.path.join(output_dir, 'web', video_id)
        os.makedirs(save_path, exist_ok=True)
        mask.save(os.path.join(save_path, f'{frame_name}.png'))
    return {}
            

@register_vis_metric
def polyp_vps_evaluator(output_dir,
                  dataset_meta,
                  **kwargs):
    assert comm.is_main_process()
    root = dataset_meta.get('root')
    eval_set_name = dataset_meta.get('eval_set_name')
    pred_path = os.path.join(output_dir, 'web')
    assert os.path.exists(pred_path)
    logging.info(subprocess.call(f'python -u vps_evaluator.py \
                 --gt_root {root}\
                 --pred_root {pred_path} \
                 --data_lst {eval_set_name}  \
                 --txt_name ', 
                 shell=True))
    
    # 提取最后的结果
    return {}


@register_vis_metric
def polyp_all_medi(model_preds, 
                   dataset_meta,
                     **kwargs):
    assert comm.is_main_process()
    iou_by_test_sample = []
    dice_by_test_sample = []
    # fbeta_by_test_sample = []
    gt_by_test_sample = []
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')
    VIS_Evaluator_OutAPI_EvalFn_API
    for pred in tqdm(model_preds):
        video_id = pred['video_id'] # str
        frame_name = pred['frame_name'] # str
        masks = pred['masks'] # list[rle], nq
        for mask_idx in range(len(masks)):
            masks[mask_idx]['counts'] = masks[mask_idx]['counts'].encode('utf-8')
        scores = torch.tensor(pred['classes']) # nt c

        foreground_scores = scores[:, :-1].sum(-1) # nq
        max_idx = foreground_scores.argmax()
        pred_mask = masks[max_idx] # rle
        pred_mask = decode_rle(pred_mask)
        pred_mask = torch.as_tensor(pred_mask, dtype=torch.uint8).contiguous().long() # h w

        gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
        gt_mask = gt_mask[0].int() # 0/1

        # preds_by_test_sample.append(pred_mask)
        gt_by_test_sample.append(gt_mask)

        tp, fp, fn, tn = metrics.get_stats(pred_mask[None, None, ...], gt_mask[None, None, ...], 
                                           mode='binary')
        iou_score = metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        dice = metrics.dice(tp, fp, fn, tn, reduction='micro')
        # recall = metrics.recall(tp, fp, fn, tn, reduction='micro-imagewise') 
        # precision = metrics.precision(tp, fp, fn, tn, reduction='micro-imagewise')
        # fbeta = metrics.fbeta_score(tp, fp, fn, tn, reduction='micro')
        # fbeta_by_test_sample.append(fbeta)
        iou_by_test_sample.append(iou_score)
        dice_by_test_sample.append(dice)

    mean_iou = torch.tensor(iou_by_test_sample).mean()
    mean_dice = torch.tensor(dice_by_test_sample).mean()
    # mena_fbeta = torch.tensor(fbeta_by_test_sample).mean()

    all_medi = {
        'mean_iou': mean_iou,
        'dice': mean_dice,
        # 'recall': recall,
        # 'precision': precision,
        # 'F': 2 * precision * recall / (precision + recall)
    }  
    return all_medi   
    