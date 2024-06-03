import torch  
import os
import shutil
from PIL import Image
import torchvision
from data_schedule.render.scene_utils.image_utils import psnr as compute_psnr
import logging


_render_metric_entrypoints = {}

def register_render_metric(fn):
    render_metric_name = fn.__name__
    if render_metric_name in _render_metric_entrypoints:
        raise ValueError(f'render_metric name {render_metric_name} has been registered')
    _render_metric_entrypoints[render_metric_name] = fn

    return fn

def render_metric_entrypoint(render_metric_name):
    try:
        return _render_metric_entrypoints[render_metric_name]
    except KeyError as e:
        print(f'render_metric Name {render_metric_name} not found')

# view_level
@register_render_metric
def psnr(view_pred, dataset_meta, **kwargs):
    rendering = torch.clamp(view_pred['rendering'], 0.0, 1.0).cuda()  # 0-1, float # 3 h w
    gt_image = torch.clamp(view_pred['view_camera'].original_image.cuda(), 0.0, 1.0)

    psnr = compute_psnr(rendering, gt_image).mean().double()

    return {'psnr': psnr}

@register_render_metric
def web(view_pred, output_dir, 
        append_scene_id=False, **kwargs):
    if append_scene_id:
        save_dir = os.path.join(output_dir, 'web', view_pred['scene_id'])
    else:
        save_dir = os.path.join(output_dir, 'web')
    os.makedirs(save_dir, exist_ok=True) 
    view_id = view_pred['view_id']
    rendering = view_pred['rendering']  # 0-1, float # 3 h w
    torchvision.utils.save_image(rendering, os.path.join(save_dir, f'{view_id:05d}.png'))
    return {}

# scene_level
@register_render_metric
def gs_to_mesh(scene_model, output_dir, append_scene_id=False, **kwargs):
    save_dir = os.path.join(output_dir, 'mesh')
    os.makedirs(save_dir, exist_ok=True) 
    scene_model.save_model(mode='model', output_dir=save_dir)
    scene_model.save_model(mode='geo+tex', output_dir=save_dir)
    return {}

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
