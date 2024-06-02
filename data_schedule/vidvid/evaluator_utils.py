

import torch  
import os
import shutil
from PIL import Image
import torch.nn.functional as F
_videnoise_metric_entrypoints = {}

from torchvision.io import write_video
from torchvision.utils import save_image
def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1), force_video=False):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return save_path


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
def web_video(video_model, output_dir, video_id, frames, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web_video', video_id), exist_ok=True) 
    video_model.save_video_frames(frames, video_id, output_dir)
    # input_video = video_model.input_video.float()
    # original_image_size = video_model.original_image_size
    # input_video = F.interpolate(input_video, size=original_image_size, mode='bilinear', align_corners=False).permute(1,0,2,3)

    # # low, high = -1, 1
    # # input_video.clamp_(min=low, max=high)
    # # input_video.sub_(low).div_(max(high - low, 1e-5))

    # # input_video = input_video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
    # assert input_video.shape[1] == len(frames)
    # for haosen, frame in zip(input_video.permute(1,0,2,3)[:1], frames[:1]):
    #     save_image([haosen], os.path.join(output_dir, 'web_video', video_id, f'{frame}.png'), 
    #                normalize=True, value_range=(-1, 1))
        
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
def videnoise_aggregator(**kwargs):
    return {}