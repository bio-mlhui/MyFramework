

import torch  
import os
import shutil
from PIL import Image

_videnoise_metric_entrypoints = {}

def register_videnoise_metric(fn):
    videnoise_metric_name = fn.__name__
    if videnoise_metric_name in _videnoise_metric_entrypoints:
        raise ValueError(f'videnoise_metric name {videnoise_metric_name} has been registered')
    _videnoise_metric_entrypoints[videnoise_metric_name] = fn

    return fn

def videnoise_metric_entrypoint(videnoise_metric_name):
    try:
        return _videnoise_metric_entrypoints[videnoise_metric_name]
    except KeyError as e:
        print(f'videnoise_metric Name {videnoise_metric_name} not found')

@register_videnoise_metric
def web(frame_pred, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    videnoiseed_img = frame_pred['videnoiseed_img'] # t 3 h w, 0-1
    save_path = os.path.join(output_dir, 'web', video_id)
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f'{frame_name}.png')
    if os.path.exists(png_path):
        os.remove(png_path)
    return {}


@register_videnoise_metric
def web_video(video_model, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    model.save_videnoiseed_frames(output_dir)
    # video_id = frame_pred['video_id']
    # frames = f
    # videnoiseed_img = frame_pred['videnoiseed_img'] # t 3 h w, 0-1
    # save_path = os.path.join(output_dir, 'web', video_id)
    # os.makedirs(save_path, exist_ok=True)
    # png_path = os.path.join(save_path, f'{frame_name}.png')
    # if os.path.exists(png_path):
    #     os.remove(png_path)
    return {}



@register_videnoise_metric
def videnoise_aggregator(video_model, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    model.save_videnoiseed_frames(output_dir)
    # video_id = frame_pred['video_id']
    # frames = f
    # videnoiseed_img = frame_pred['videnoiseed_img'] # t 3 h w, 0-1
    # save_path = os.path.join(output_dir, 'web', video_id)
    # os.makedirs(save_path, exist_ok=True)
    # png_path = os.path.join(save_path, f'{frame_name}.png')
    # if os.path.exists(png_path):
    #     os.remove(png_path)
    return {}