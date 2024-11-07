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
from .utils import get_frames, get_frames_mask, SET_NAME_TO_DIR, SET_NAME, SET_NAME_TO_MODE

# 根据有Label的进行clip分割，每个有label帧代表一个clip, clip大小由mapper决定
def cardiac(dataset_name,
                video_ids,
                video_to_frames,
                video_to_anno_frames,): 
    if dataset_name == 'card_labeled_train' or dataset_name == 'card_labeled_test':
        metas = []
        for vid_id in tqdm(video_ids):
            all_frames = sorted(video_to_frames[vid_id]) # list[str]
            anno_frames = video_to_anno_frames[vid_id] # list[str]

            for frame_idx in anno_frames:
                metas.append({
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })                

        logging.debug(f'{dataset_name} Total metas: [{len(metas)}]')
        return metas
    
    elif dataset_name == 'card_unlabeled':
        metas = []
        for vid_id in tqdm(video_ids):
            all_frames = sorted(video_to_frames[vid_id]) # list[str]
            assert len(video_to_anno_frames[vid_id]) == 0
            for frame_idx in all_frames:
                metas.append({
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })                

        logging.debug(f'{dataset_name} Total metas: [{len(metas)}]')
        return metas
    else:
        raise ValueError()
    
_root = os.getenv('DATASET_PATH')
root = os.path.join(_root, 'cardiacUDC/transformed')

visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['card_labeled_train'] = [] 
visualize_meta_idxs['card_labeled_test'] = []
visualize_meta_idxs['card_unlabeled'] = []

polyp_meta = {
    'thing_classes': ['lv', 'rv', 'la', 'ra'],
    'thing_colors': [(255., 0., 0.), (0., 255., 0.), (0, 0, 255), (255, 255, 0)],
    'root': root
}

for name in SET_NAME:
    set_dir = SET_NAME_TO_DIR[name]
    set_dir = os.path.join(root, set_dir)
    video_ids = os.listdir(os.path.join(set_dir, 'Frame'))

    video_to_frames = {
        vid: sorted([png[:-4] for png in os.listdir(os.path.join(set_dir, 'Frame', vid)) if png.endswith('.jpg')])\
            for vid in video_ids
    }
    video_to_anno_frames = {}
    for vid in video_to_frames.keys():
        if os.path.exists(os.path.join(set_dir, 'GT', vid)):
            video_to_anno_frames[vid] = sorted([png[:-4] for png in os.listdir(os.path.join(set_dir, 'GT', vid)) if png.endswith('.npy')])
        else:
            video_to_anno_frames[vid] = []

    mode = SET_NAME_TO_MODE[name]
    if mode == 'train':
        train_meta = copy.deepcopy(polyp_meta)
        train_meta.update({
            'name': name,
            'mode': 'train',
            'get_frames_fn': partial(get_frames, frames_path=os.path.join(set_dir, 'Frame')),
            'get_frames_mask_fn': partial(get_frames_mask, mask_path=os.path.join(set_dir, 'GT'),),
        })
        DatasetCatalog.register(name, partial(cardiac,
                                                video_ids=video_ids, 
                                                dataset_name=name,
                                                video_to_frames=video_to_frames,
                                                video_to_anno_frames=video_to_anno_frames))    
        MetadataCatalog.get(name).set(**train_meta, 
                                            visualize_meta_idxs=visualize_meta_idxs[name])
         
    elif mode == 'evaluate':
        validate_meta = copy.deepcopy(polyp_meta)
        validate_meta.update({
            'name': name,
            'mode': 'evaluate',
            'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, os.path.join(set_dir, 'Frame'))),
            'get_frames_gt_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, os.path.join(set_dir, 'GT')),),
            'eval_meta_keys': video_to_anno_frames
        })
        DatasetCatalog.register(name, partial(cardiac,dataset_name=name,
                                                      video_ids=video_ids,
                                                      video_to_frames=video_to_frames,
                                                      video_to_anno_frames=video_to_anno_frames))
          
        MetadataCatalog.get(name).set(**validate_meta, 
                                      visualize_meta_idxs=visualize_meta_idxs[name])


 

