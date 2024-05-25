import torch  
import os
import shutil
from PIL import Image
from ..evaluator_utils import register_render_metric
import logging


@register_render_metric
def text3d_metric_aggregator(dataset_meta, 
                            eval_meta_keys,
                            metrics_by_scene, 
                            **kwargs):
    # {scene_id: {view_id: {key:value}}}
    eval_metrics = {}
    scene_metrics = {key: metrics_by_scene[key].pop('scene_metrics') for key in metrics_by_scene.keys()}
    # video, frame_name
    # perframe metrics
    if len(metrics_by_scene[list(eval_meta_keys.keys())[0]]) != 0:
        metric_names = metrics_by_scene[list(eval_meta_keys.keys())[0]][eval_meta_keys[list(eval_meta_keys.keys())[0]][0]]
        for taylor_swift in metric_names:
            eval_metrics[taylor_swift] = torch.tensor([metrics_by_scene[video][frame][taylor_swift]  \
                                                    for video in eval_meta_keys.keys() for frame in eval_meta_keys[video]]).mean()
    
    # print specific metrics for each scene
    # # metrics by each video
    # mean_iou_by_each_video = {}
    # for video in eval_meta_keys:
    #     mean_iou_by_each_video[video] = torch.tensor([metrics_by_scene[video][fname]['psnr'] for fname in eval_meta_keys[video]]).mean()
        
    # mean_iou_by_each_video = dict(sorted(mean_iou_by_each_video.items(), key=lambda x: x[1]))    
    # logging.debug(f'psnr_by_each_scene: {mean_iou_by_each_video}')
    
    return eval_metrics



