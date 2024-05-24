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
    # get_mesh
    #mesh = scene_model.get_mesh()

    return {}





