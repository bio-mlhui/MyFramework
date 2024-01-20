_vis_metric_entrypoints = {}

def register_vis_metric(fn):
    vis_metric_name = fn.__name__
    if vis_metric_name in _vis_metric_entrypoints:
        raise ValueError(f'vis_metric name {vis_metric_name} has been registered')
    _vis_metric_entrypoints[vis_metric_name] = fn

    return fn

def vis_metric_entrypoint(vis_metric_name):
    try:
        return _vis_metric_entrypoints[vis_metric_name]
    except KeyError as e:
        print(f'vis_metric Name {vis_metric_name} not found')


import torch  
import os
import shutil
from PIL import Image

@register_vis_metric
def mask_dice_iou(frame_pred, dataset_meta, **kwargs):
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')
    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
    gt_mask = gt_mask[0].int() # h w

    inter, union    = (pred_mask*gt_mask).sum(), (pred_mask+gt_mask).sum()
    dice = (2*inter+1)/(union+1)
    iou = (inter+1)/(union-inter+1)

    return {'dice': dice, 'iou': iou}


@register_vis_metric
def web(frame_pred, output_dir, **kwargs):

    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w

    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    mask = Image.fromarray(255 * pred_mask.int().numpy()).convert('L')
    save_path = os.path.join(output_dir, 'web', video_id)

    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f'{frame_name}.png')
    if os.path.exists(png_path):
        os.remove(png_path)
    mask.save(png_path)
    return {}