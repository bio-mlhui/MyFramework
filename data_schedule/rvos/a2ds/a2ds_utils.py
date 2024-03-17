import wandb
import plotly.express as px
import logging
import os
import numpy as np
import torch
import json
from joblib import Parallel, delayed
import multiprocessing
import torch.distributed as dist
import detectron2.utils.comm as comm

import pycocotools.mask as mask_util
from pycocotools.mask import encode, area

from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.utils.video_clips import generate_windows_of_video
from glob import glob
from PIL import Image
import torchvision.io.video as video_io
from tqdm import tqdm
import torchvision.transforms.functional as F
import h5py
def check_a2ds_validity(root): 
    pass

def get_class_label(idx):
    """idx是np.float64, 想要获得他的label
    """
    return int(str(int(idx))[0]) - 1
def get_action_label(idx):
    """idx是np.float64, 想要获得他的label
    """
    return int(str(int(idx))[1]) - 1
def normalize_text(text_query):
    # 非法输入
    if text_query == 'The left with yellow t shirt on the left running':
        text_query = 'the man with yellow tshirt on the left running'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    normalized_text_query = " ".join(normalized_text_query.lower().split())
    return normalized_text_query

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg'),) for f in frames]

# t' h w, 0是背景, 1-是obj_id
# has_ann: t
# (video_id, frames) -> t' h w, obj_id; has_ann: t
def get_frames_mask(mask_path, video_id, frames, all_obj_ids):
    has_ann = []
    masks = [] # list[ni h w]
    instance_ids = [] # list[ni]
    for frame_name in frames:
        ann_file = os.path.join(mask_path, video_id, f'{frame_name}.h5')
        if os.path.exists(ann_file) and (True not in has_ann):
            frame_annotation = h5py.File(ann_file)
            # n h w (bool) n, n, n (long)
            frame_mask, appear_instances, class_ids, _, _ = illicit(video_id, int(frame_name), frame_annotation)
            appear_instances = [haosen+1 for haosen in appear_instances]
            assert len(appear_instances) == len(frame_mask)
            masks.append(frame_mask)
            instance_ids.append(appear_instances)
            has_ann.append(True)
        else:
            has_ann.append(False)
    h, w = masks[0].shape[1:]
    obj_masks = [] # list[t' h w], n
    for obj_id in all_obj_ids:
        obj_frame_mks = [] # list[h w]
        for frame_idx in range(len(instance_ids)):
            if obj_id in instance_ids[frame_idx]:
                seg_idx = instance_ids[frame_idx].index(obj_id)
                obj_frame_mks.append(masks[frame_idx][seg_idx])
            else:
                obj_frame_mks.append(torch.zeros([h, w]).bool())
        obj_frame_mks = torch.stack(obj_frame_mks, dim=0) # t' h w
        obj_masks.append(obj_frame_mks)
    masks = torch.stack(obj_masks, dim=0) # n t' h w
    has_ann = torch.tensor(has_ann).bool()
    # n t' h w, t, n
    return masks, has_ann, all_obj_ids
    
def illicit(video_id,  frame_idx, frame_file):
    # n
    appear_instances = [int(ins_id) for ins_id in frame_file['instance']]
    # n h w
    masks = torch.from_numpy(np.array(frame_file['reMask'])).transpose(-1, -2).bool()
    masks = masks.unsqueeze(dim=0) if masks.dim() == 2 else masks 
    
    # 4 n / 4
    boxes = torch.from_numpy(np.array(frame_file['reBBox']))  # x1y1x2y2 form
    boxes = boxes.unsqueeze(dim=-1) if boxes.dim() == 1 else boxes
    boxes = boxes.permute(1, 0) # n 4
    assert len(boxes) == len(masks)
    class_ids = torch.tensor([get_class_label(idx) for idx in frame_file['id'][0]]).long() # 0-6
    action_ids = torch.tensor([get_action_label(idx) for idx in frame_file['id'][0]]).long()      
    # instance不unique的
    if video_id == 'EadxBPmQvtg' and frame_idx == 25:
        assert len(masks) == 11 and len(appear_instances) == 11
        assert appear_instances == [0, 1, 2, 3, 4, 5, 6, 7,8,9,1]
        masks = masks[:-1]         
        boxes = boxes[:-1]            
        class_ids = class_ids[:-1]
        action_ids = action_ids[:-1]
        appear_instances = appear_instances[:-1]
    assert len(torch.tensor(appear_instances).unique()) == len(appear_instances)
  
    # mask多于instance的
    if video_id == '95Nq6fQoP2o' and frame_idx == 32:
        assert len(masks) == 7 and len(appear_instances) == 6
        masks = masks[:6]
        boxes = boxes[:6]
        class_ids = class_ids[:6]
        action_ids = action_ids[:6]
    elif video_id == 'I0MlLHTWCro' and frame_idx == 20:
        assert len(masks) == 4 and len(appear_instances) == 2
        masks = masks[:2] 
        boxes = boxes[:2]  
        class_ids = class_ids[:2]
        action_ids = action_ids[:2]         
    elif video_id == 'IRrbHQjE4LQ' and frame_idx == 16:
        assert len(masks) == 6 and len(appear_instances) == 5
        masks = masks[:5] 
        boxes = boxes[:5] 
        class_ids = class_ids[:5]
        action_ids = action_ids[:5]          
    assert len(masks) == len(appear_instances)
    assert len(boxes) == len(masks)
    assert len(class_ids) == len(class_ids)

    return masks, appear_instances, class_ids, action_ids, boxes
     

def connect_vid_text(video_id, text_id):
    return f'{video_id}{"THISISAUNIQUECONNECT_EDSCDSGREOMI"}{text_id}'

def disconnect_vid_text(vid_Text:str):
    return vid_Text.split("THISISAUNIQUECONNECT_EDSCDSGREOMI")[0], vid_Text.split("THISISAUNIQUECONNECT_EDSCDSGREOMI")[-1]


A2DS_CATEGORIES = {
    'adult': 0, 
    'baby': 1, 
    'ball': 2, 
    'bird': 3, 
    'car': 4, 
    'cat': 5, 
    'dog': 6, 
}



def collate_fn(split, aux_collate_fn, batch):
    samples, text_query, auxiliary, meta_or_target = list(zip(*batch))
    samples = list(samples)
    text_query = list(text_query)
    auxiliary = list(auxiliary)
    meta_or_target = list(meta_or_target)
    batch_data = {
        'samples': samples,
        'text_query': text_query,
        'auxiliary': aux_collate_fn(auxiliary)
    }
    if split == 'test':
        batch_data['meta'] = meta_or_target
    else:
        batch_data['targets'] = meta_or_target
    return batch_data


# # 生成每个video, 每个obj的出现的帧的下标, 
# def get_each_obj_appear_frame_idxs(root, save_path='yrvos_train_appear_frame_idxs_by_video_by_obj_id.json'):
#     save_path = os.path.join(root, save_path)
#     if os.path.exists(save_path):
#         with open(save_path, 'r') as f:
#             appear_frame_idxs_by_video_by_obj_id = json.load(f)
#     else:
#         appear_frame_idxs_by_video_by_obj_id = defaultdict(dict)
#         with open(os.path.join(root, 'meta_expressions', 'train', f'meta_expressions.json'), 'r') as f:
#             video_to_texts = json.load(f)['videos']  
#         # {video_id: expressions: {exp_id: exp, obj_id:str}, frames: []}
            
#         with open(os.path.join(root, 'train', f'meta.json'), 'r') as f:
#             video_to_objs = json.load(f)['videos']
#         # {video_id: objects: {obj_id: {category, frames}} 
            
#         for vid_id in tqdm(video_to_texts.keys()):
#             vid_objs = list(video_to_objs[vid_id]['objects'].keys())
#             vid_objs = [int(taylor) for taylor in vid_objs]
#             all_frames = video_to_texts['frames'] # list[str]
#             vid_frames_masks, _ = get_frames_mask(mask_path=os.path.join(root, 'train/Annotations'),
#                                                video_id=vid_id, 
#                                                frames=all_frames) # t h w, 
#             assert len(vid_frames_masks) == len(all_frames)

#             vid_masks_by_obj = torch.stack([vid_frames_masks == taylor for taylor in vid_objs], dim=0) # N t h w

#             appear_frame_idxs_by_obj = vid_masks_by_obj.flatten(2).any(-1).unbind(0) # list[t], N
#             appear_frame_idxs_by_obj = [torch.where(taylor)[0] for taylor in appear_frame_idxs_by_obj] # list[list[int]], N

#             for taylor, cardib in zip(vid_objs, appear_frame_idxs_by_obj):
#                 appear_frame_idxs_by_video_by_obj_id[vid_id][taylor] = cardib

#         with open(save_path, 'w') as f:
#             json.dump(appear_frame_idxs_by_video_by_obj_id, f)       
    
#     return appear_frame_idxs_by_video_by_obj_id

            

        

