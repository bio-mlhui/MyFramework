from typing import Optional, Union
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
from data_schedule.utils.segmentation import get_AP_PAT_IOU_PerFrame
import torch  
import os
import shutil
from PIL import Image
from pycocotools.mask import encode, area
from data_schedule.utils.segmentation import bounding_box_from_mask
import json
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.io import write_video
_rvos_metric_entrypoints = {}

def register_rvos_metric(fn):
    rvos_metric_name = fn.__name__
    if rvos_metric_name in _rvos_metric_entrypoints:
        raise ValueError(f'rvos_metric name {rvos_metric_name} has been registered')
    _rvos_metric_entrypoints[rvos_metric_name] = fn

    return fn

def rvos_metric_entrypoint(rvos_metric_name):
    try:
        return _rvos_metric_entrypoints[rvos_metric_name]
    except KeyError as e:
        print(f'rvos_metric Name {rvos_metric_name} not found')

@register_rvos_metric
def web(frame_pred, output_dir, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w
    exp_id = frame_pred['exp_id']
    refer_text = frame_pred['text']
    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    mask = Image.fromarray(255 * pred_mask.int().numpy()).convert('L')
    save_path = os.path.join(output_dir, 'web', video_id, refer_text)

    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f'{frame_name}.png')
    if os.path.exists(png_path):
        os.remove(png_path)
    mask.save(png_path)
    return {}



# @register_rvos_metric
# def web(model_preds, output_dir, remove_pngs=False, zip=False, **kwargs):
#     os.makedirs(os.path.join(output_dir, 'web')) 
#     for pred in model_preds:
#         video_id = pred['video_id']
#         frame_name = pred['frame_name']
#         masks = pred['masks'] # list[h w], nq
#         scores = pred['scores'] # nq
#         exp_id = pred['exp_id']

#         max_idx = torch.tensor(scores).argmax()
#         mask = masks[max_idx] # h w
#         save_path = os.path.join(output_dir, 'web', video_id, exp_id, frame_name)
#         os.makedirs(save_path, exist_ok=True)
#         mask = Image.fromarray((255 * mask.numpy())).convert('L')
#         mask.save(os.path.join(save_path, f'{frame_name}.png'))

#     model_config, train_ckpt, _ = output_dir.split('/')[-3:]
#     zip_file_name = f'{model_config}_{train_ckpt}'
#     # 把web打包成 model_config_train_ckpt.zip
#     shutil.make_archive(os.path.join(output_dir, zip_file_name), 'zip', root_dir=output_dir, base_dir='web')
#     if remove_pngs:
#         shutil.rmtree(os.path.join(output_dir, 'web'))
    
#     return {}


@register_rvos_metric
def mask_iou_intersect_union(frame_pred, dataset_meta, **kwargs):
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


# mean_iou, overall_iou, 0.5, 0.6
@register_rvos_metric
def a2ds_aggregator(metrics_by_vid_frame, dataset_meta, eval_meta_keys, **kwargs):
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


from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import cv2
def generate_instance_canvas(vid_frames, metadata, H, W, pred_mask):
    """pred_mask: h w, score:float"""
    istce_canvas = Visualizer(img_rgb=vid_frames*255, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    istce = Instances([H, W], 
        pred_masks=pred_mask.unsqueeze(0), # 1 H W
        scores=torch.tensor([1]), # 1,
        pred_classes=torch.tensor([0]) # 1,
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()

@register_rvos_metric
def a2ds_video_aggregator(dataset_meta, eval_meta_keys, output_dir, remove_pngs, zip, **kwargs):
    os.makedirs(os.path.join(output_dir, 'web')) 
    # 每个测试video成 video
    disconnect_vid_text_fn = dataset_meta.get('disconnect_vidText_fn')
    get_frames_fn = dataset_meta.get('get_frames_fn')
    for vidTextId in eval_meta_keys.keys():   # vid_text
        video_id, text_id = disconnect_vid_text_fn(vidTextId)
        frame_paths = os.listdir(os.path.join(output_dir, 'web', video_id, text_id))
        frame_paths = [haosen[:-4] for haosen in frame_paths]

        cap = cv2.VideoCapture(os.path.join(dataset_meta.get('root'), 'Release/clips320H', f'{video_id}.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        all_frames = get_frames_fn(video_id, frame_paths) # list[PIL]
        all_frames = torch.stack([F.to_tensor(haosen) for haosen in all_frames], dim=0) # t h w
        # t h w
        pred_masks = torch.stack([torch.from_numpy(np.array(Image.open(os.path.join(output_dir, 'web',\
                                     video_id, f'{f}.png')))).int() for f in all_frames], dim=0).numpy()
        # detectron2
        detectron2_image = torch.stack([torch.from_numpy(generate_instance_canvas(vid_frames=all_frames[frame_idx], 
                                                     metadata=dataset_meta, 
                                                     H=pred_masks.shape[1], W=pred_masks.shape[2], pred_mask=pred_masks[frame_idx])
                            for frame_idx in range(len(all_frames)))], dim=0) # t h w c, uint8
        
        write_video(os.path.join(output_dir, 'web', video_id, f'{text_id}.mp4'), detectron2_image, fps=fps)

    if remove_pngs:
        shutil.rmtree(os.path.join(output_dir, 'web', video_id, text_id))
    
    return {}



@register_rvos_metric
def yrvos_aggregator(metrics_by_vid_frame, dataset_meta, eval_meta_keys, **kwargs):
    return {}

# def generate_coco_img_id(video_id, frame, exp_id):
#     return f'v_{video_id}_f_{frame}_e_{exp_id}'


# def metric_coco(model_preds, coco_file, output_dir=None):
#     coco_preds = [] 
#     for pred in model_preds:
#         video_id = pred['video_id']
#         frame_name = pred['frame_name']
#         masks = pred['masks'] # list[h w], ni
#         scores = pred['scores'] # ni
#         exp_id = pred['exp_id']

#         # list[list[rle], ni], T_ann
#         rles = [mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order="F"))[0] for mask in masks]

#         # rle, scalar, h w
#         for rle, refer_prob, mask in zip(rles, scores, masks):
#             image_id = generate_coco_img_id(video_id=video_id, frame=frame_name, exp_id=exp_id)
#             coco_preds.append({'image_id': image_id,
#                                 'category_id': 1,
#                                 'segmentation': rle,
#                                 'score': refer_prob.item()})
            
#     coco_perframe_preds = comm.all_gather(coco_perframe_preds)
#     coco_perframe_preds = [p for p_list in coco_perframe_preds for p in p_list]

#     eval_metrics = get_AP_PAT_IOU_PerFrame(coco_file, coco_perframe_preds)
#     return eval_metrics    
   
   



# # 生成一堆videos的coco_gt.json
# def generate_coco_gt_file(get_frames_mask_fn,
#                           video_to_texts, 
#                           video_ids, coco_file_name):
#     # first generate coco_file
#     # 每个数据集提供video_ids, get_frames_mask_fn, video_to_text: {'expression':, 'frames':}
#     if os.path.exists(coco_file_name):
#         logging.debug(f'{coco_file_name} has been generated.')
#         return
#     images_id_set = set()
#     images_dict = []
#     coco_gts = []
#     for vid in video_ids:
#         all_ann_frames = video_to_texts[vid]['frames'] # 应该是 有annotation的所有帧
#         all_ann_frames_mask, _ = get_frames_mask_fn(video_id=vid, frames=all_ann_frames) # t h w
#         assert len(all_ann_frames) == len(all_ann_frames_mask)
#         text_anns = video_to_texts[vid]['expressions'] # {exp_id: {'exp', 'obj_ids': list[int]}}
#         for frame, frame_mask in zip(all_ann_frames, all_ann_frames_mask):
#             # h w, list[int], 0, 1, 2, 3
#             for exp_id, exp_dict in text_anns.items():
#                 exp_objs = exp_dict['obj_ids'] # list[int]
#                 # h w -> N h w (bool) -> h w
#                 frame_exp_ann = torch.stack([frame_mask == obj_id for obj_id in exp_objs], dim=0).any(0)
#                 if not frame_exp_ann.any():
#                     continue
#                 else:
#                     image_id = generate_coco_img_id(video_id=vid, frame_id=frame, exp_id=exp_id)
#                     assert image_id not in images_id_set
#                     images_id_set.add(image_id)
#                     gt_mask = frame_exp_ann.numpy()
#                     images_dict.append({'id': image_id, 'height': gt_mask.shape[0], 'width': gt_mask.shape[1]})
#                     mask_rle = encode(gt_mask)
#                     mask_rle['counts'] = mask_rle['counts'].decode('ascii')
#                     mask_area = float(area(mask_rle))
#                     bbox = bounding_box_from_mask(gt_mask) # x1y1x2y2 form 
#                     assert bbox.ndim == 1 and len(bbox) == 4
#                     bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
#                     coco_gts.append({
#                         'id': len(coco_gts),
#                         'image_id': image_id,
#                         'category_id': 1,  
#                         'segmentation': mask_rle,
#                         'area': mask_area,
#                         'bbox': bbox_xywh,
#                         'iscrowd': 0,
#                     })                    

#     dataset_dict = {
#         'categories': [{'id': 1, 'name': 'refer'}],
#         'images': images_dict,
#         'annotations':  coco_gts,
#     }
#     with open(coco_file_name, 'w') as f:
#         json.dump(dataset_dict, f)