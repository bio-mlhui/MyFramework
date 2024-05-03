import torch  
import os
import shutil
from PIL import Image
from ..evaluator_utils import register_render_metric
import logging

@register_render_metric
def image3d_optimize_metric_aggregator(dataset_meta, eval_meta_keys, metrics_by_view_id, **kwargs):
    # {view_id: {key:value}}
    eval_metrics = {}
    # video, frame_name
    # perframe metrics
    metric_names = metrics_by_view_id[eval_meta_keys[0]]
    for taylor_swift in metric_names:
        eval_metrics[taylor_swift] = torch.tensor([metrics_by_view_id[view][taylor_swift] for view in eval_meta_keys]).mean().cpu()
    
    psnr_each_view = [metrics_by_view_id[haosen]['psnr'] for haosen in metrics_by_view_id.keys()]
    logging.debug(f'psnr_by_each_view: {psnr_each_view}')
    
    return eval_metrics

@register_render_metric
def multivew3d_learn_metric_aggregator(dataset_meta, 
                                       eval_meta_keys,
                                       metrics_by_scene, **kwargs):
    eval_metrics = {}
    # video, frame_name
    # perframe metrics
    metric_names = metrics_by_scene[list(eval_meta_keys.keys())[0]][eval_meta_keys[list(eval_meta_keys.keys())[0]][0]]
    for taylor_swift in metric_names:
        eval_metrics[taylor_swift] = torch.tensor([metrics_by_scene[video][frame][taylor_swift]  \
                                                   for video in eval_meta_keys.keys() for frame in eval_meta_keys[video]]).mean()
    
    # metrics by each video
    mean_iou_by_each_video = {}
    for video in eval_meta_keys:
        mean_iou_by_each_video[video] = torch.tensor([metrics_by_scene[video][fname]['psnr'] for fname in eval_meta_keys[video]]).mean()
        
    mean_iou_by_each_video = dict(sorted(mean_iou_by_each_video.items(), key=lambda x: x[1]))    
    logging.debug(f'psnr_by_each_scene: {mean_iou_by_each_video}')
    
    return eval_metrics




