
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

from pycocotools.mask import encode, area
from data_schedule.utils.segmentation import bounding_box_from_mask
import json

# 把这些prediction mask可视化到output_dir, 并且打包
def metric_web(model_preds, output_dir, remove_pngs=False):
    assert comm.is_main_process()
    # output_dir: epc1_iter500_sap8099/eval_dataset1
    if os.path.exists(os.path.join(output_dir, 'web')):
        shutil.rmtree(os.path.join(output_dir, 'web'))

    os.makedirs(os.path.join(output_dir, 'web')) 
    for pred in model_preds:
        video_id = pred['video_id']
        frame_name = pred['frame_name']
        masks = pred['masks'] # list[h w], nq
        scores = pred['scores'] # nq
        exp_id = pred['exp_id']

        max_idx = torch.tensor(scores).argmax()
        mask = masks[max_idx] # h w
        save_path = os.path.join(output_dir, 'web', video_id, frame_name, exp_id)
        os.makedirs(save_path, exist_ok=True)
        mask = Image.fromarray((255 * mask.numpy())).convert('L')
        mask.save(os.path.join(save_path, f'{frame_name}.png'))

    model_config, train_ckpt, _ = output_dir.split('/')[-3:]
    zip_file_name = f'{model_config}_{train_ckpt}'
    # 把web打包成 model_config_train_ckpt.zip
    shutil.make_archive(os.path.join(output_dir, zip_file_name), 'zip', root_dir=output_dir, base_dir='web')
    if remove_pngs:
        shutil.rmtree(os.path.join(output_dir, 'web'))
    
    return {}
                       
def metric_coco(model_preds, coco_file, output_dir=None):
    coco_preds = [] 
    for pred in model_preds:
        video_id = pred['video_id']
        frame_name = pred['frame_name']
        masks = pred['masks'] # list[h w], ni
        scores = pred['scores'] # ni
        exp_id = pred['exp_id']

        # list[list[rle], ni], T_ann
        rles = [mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order="F"))[0] for mask in masks]

        # rle, scalar, h w
        for rle, refer_prob, mask in zip(rles, scores, masks):
            image_id = generate_coco_img_id(video_id=video_id, frame=frame_name, exp_id=exp_id)
            coco_preds.append({'image_id': image_id,
                                'category_id': 1,
                                'segmentation': rle,
                                'score': refer_prob.item()})
            
    coco_perframe_preds = comm.all_gather(coco_perframe_preds)
    coco_perframe_preds = [p for p_list in coco_perframe_preds for p in p_list]

    eval_metrics = get_AP_PAT_IOU_PerFrame(coco_file, coco_perframe_preds)
    return eval_metrics    
                

def generate_coco_img_id(video_id, frame, exp_id):
    return f'v_{video_id}_f_{frame}_e_{exp_id}'

# 生成一堆videos的coco_gt.json
def generate_coco_gt_file(get_frames_mask_fn,
                          video_to_texts, 
                          video_ids, coco_file_name):
    # first generate coco_file
    # 每个数据集提供video_ids, get_frames_mask_fn, video_to_text: {'expression':, 'frames':}
    if os.path.exists(coco_file_name):
        logging.debug(f'{coco_file_name} has been generated.')
        return
    images_id_set = set()
    images_dict = []
    coco_gts = []
    for vid in video_ids:
        all_ann_frames = video_to_texts[vid]['frames'] # 应该是 有annotation的所有帧
        all_ann_frames_mask, _ = get_frames_mask_fn(video_id=vid, frames=all_ann_frames) # t h w
        assert len(all_ann_frames) == len(all_ann_frames_mask)
        text_anns = video_to_texts[vid]['expressions'] # {exp_id: {'exp', 'obj_ids': list[int]}}
        for frame, frame_mask in zip(all_ann_frames, all_ann_frames_mask):
            # h w, list[int], 0, 1, 2, 3
            for exp_id, exp_dict in text_anns.items():
                exp_objs = exp_dict['obj_ids'] # list[int]
                # h w -> N h w (bool) -> h w
                frame_exp_ann = torch.stack([frame_mask == obj_id for obj_id in exp_objs], dim=0).any(0)
                if not frame_exp_ann.any():
                    continue
                else:
                    image_id = generate_coco_img_id(video_id=vid, frame_id=frame, exp_id=exp_id)
                    assert image_id not in images_id_set
                    images_id_set.add(image_id)
                    gt_mask = frame_exp_ann.numpy()
                    images_dict.append({'id': image_id, 'height': gt_mask.shape[0], 'width': gt_mask.shape[1]})
                    mask_rle = encode(gt_mask)
                    mask_rle['counts'] = mask_rle['counts'].decode('ascii')
                    mask_area = float(area(mask_rle))
                    bbox = bounding_box_from_mask(gt_mask) # x1y1x2y2 form 
                    assert bbox.ndim == 1 and len(bbox) == 4
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    coco_gts.append({
                        'id': len(coco_gts),
                        'image_id': image_id,
                        'category_id': 1,  
                        'segmentation': mask_rle,
                        'area': mask_area,
                        'bbox': bbox_xywh,
                        'iscrowd': 0,
                    })                    

    dataset_dict = {
        'categories': [{'id': 1, 'name': 'refer'}],
        'images': images_dict,
        'annotations':  coco_gts,
    }
    with open(coco_file_name, 'w') as f:
        json.dump(dataset_dict, f)