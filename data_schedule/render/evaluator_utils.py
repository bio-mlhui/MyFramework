import torch  
import os
import shutil
from PIL import Image
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


@register_render_metric
def psnr(view_pred, dataset_meta, **kwargs):
    scene_name = view_pred['scene_name']
    view_point = view_pred['view_point']
    get_gt_view_fn = dataset_meta.get('get_gt_view_fn')
    gt_view = get_gt_view_fn(scene_name=scene_name, view_points=[view_point])[0] # h w
    
    pred_view = view_pred['pred']

    # psnr
    psnr = PSNR(pred_view, gt_view)

    return {'psnr': psnr}

@register_render_metric
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


