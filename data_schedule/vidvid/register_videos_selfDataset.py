from typing import Optional, Union
import json
import os
from functools import partial
import numpy as np
import torch
import logging
from tqdm import tqdm
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from collections import defaultdict
from data_schedule.vidvid.apis import VIDenoise_Meta
from PIL import Image

def videnoise_selfDataset(step_size, # none / int; 0, 6, 13, 19 ...
                split_dataset_name,
                video_ids,
                video_to_frames,
                root_path):
    # image_to_class_id = {}
    # with open(os.path.join(root_path, 'Classification/classification.txt'), 'r') as f:
    #     for line in f:
    #         image, class_name = line.strip().split()
    #         image_to_class_id[image] = CLASS_TO_ID[class_name]

    logging.debug(f'{split_dataset_name} Generating metas...')   
    metas = []
    for vid_id in tqdm(video_ids):
        all_frames = sorted(video_to_frames[vid_id])
        # 不同前景有不同的类别
        # all_frame_classes = np.array([image_to_class_id[f'{vid_id}/{frame}'] for frame in all_frames])
        # poly_class = np.unique(all_frame_classes).tolist()
        # assert len(poly_class) == 1, '这一个clip的前景Mask必须都是同一个polyp'
        # poly_class = poly_class[0]
        # 只有前景一个类别
        poly_class = 0
        if step_size is None:  
            metas.append({
                'video_id': vid_id,
                'all_frames' : all_frames,
                # {obj_id: {class_label: 0,}},
                'all_objs': { 1: {'class_label': poly_class} },
                'meta_idx': len(metas)
            }) 
        else:
            for frame_idx in range(0, len(all_frames), step_size):
                metas.append({
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'all_objs': { 1: {'class_label': poly_class} },
                    'meta_idx': len(metas)
                })                

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')
    return metas



_root = os.getenv('DATASET_PATH')

root = os.path.join(_root, 'videnoise')

visualize_meta_idxs = defaultdict(list)

video_ids = os.listdir(root)

# videnoise
#   video_1
#       frame_1.jpg
#       frame_2.jpg
#       frame_3.jpg
#   video_2
#   video_3

# 都是注册singledataset, 但是meta需要按照dataset的格式
def get_frames(root, 
               video_id,
               frames):
    return [Image.open(os.path.join(root, video_id, f'{f}.png')).convert('RGB') for f in frames]

def single_meta(video_id, frames, ):
    return [
        {
            'video_id': video_id,
            'frames': frames,
            'meta_idx': 0,
        }
    ]

import torchvision.transforms.functional as F
def to_tensor(get_frame_fn):
    # list[pil] -> 0-1 float, t 3 h w
    frames = get_frame_fn()
    tensor_video = torch.stack([F.to_tensor(frame) for frame in frames], dim=0) # t 3 h w, float, 0-1
    return tensor_video

for video_id in video_ids:
    register_str = '_'.join(video_id.split(' '))
    dataset_name = f'{register_str}_selfDataset'
    frames = sorted([png[:-4] for png in os.listdir(os.path.join(root, video_id)) if png.endswith('.png')])
    register_meta = {
        'mode': 'all',
        'get_frames_fn': partial(get_frames, root=root,),
        'videnoise_optimize':{
            'tensor_frames': partial(to_tensor, get_frame_fn=partial(get_frames,  root=root, video_id=video_id, frames=frames),),
            'input_text': "Several people are walking across the road, and many cars are pulling over the road. Many tall buildings and some trees",
            "input_negative_text": "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions",
            'frame_interval': 3, # 只是存储的时候用到
            'fps': 10,
        },
        'eval_meta_keys':{
            video_id: frames
        }
        
    }
    DatasetCatalog.register(dataset_name, partial(single_meta,
                                                  video_id=video_id,
                                                  frames=frames))  
      
    MetadataCatalog.get(dataset_name).set(**register_meta, 
                                        visualize_meta_idxs=[]) 
   


