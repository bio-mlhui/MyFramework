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
def polyp_metric_aggregator(metrics_by_vid_frame, dataset_meta, eval_meta_keys, **kwargs):
    # output: eval_metrics
    # video: frame_name: metric/ vid_metrics

    eval_metrics = {}
    # video, frame_name
    # perframe metrics
    metric_names = metrics_by_vid_frame[list(eval_meta_keys.keys())[0]][eval_meta_keys[list(eval_meta_keys.keys())[0]][0]]
    for taylor_swift in metric_names:
        eval_metrics[taylor_swift] = torch.tensor([metrics_by_vid_frame[video][frame][taylor_swift]  for video in eval_meta_keys.keys() for frame in eval_meta_keys[video]]).mean()
    
    # metrics by each video
    mean_iou_by_each_video = {}
    mean_dice_by_each_video = {}
    for video in eval_meta_keys:
        mean_iou_by_each_video[video] = torch.tensor([metrics_by_vid_frame[video][fname]['iou'] for fname in eval_meta_keys[video]]).mean()
        mean_dice_by_each_video[video] = torch.tensor([metrics_by_vid_frame[video][fname]['dice'] for fname in eval_meta_keys[video]]).mean()
        
    mean_iou_by_each_video = dict(sorted(mean_iou_by_each_video.items(), key=lambda x: x[1]))
    mean_dice_by_each_video = dict(sorted(mean_dice_by_each_video.items(), key=lambda x: x[1]))    
    logging.debug(f'mean_iou_by_each_video: {mean_iou_by_each_video}')
    logging.debug(f'mean_dice_by_each_video: {mean_dice_by_each_video}')
    
    return eval_metrics


