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
from .fibroid_utils import get_frames, get_frames_mask


# 单帧是vis, 整个视频是vos
def fibroid_train(step_size, # none / int; 0, 6, 13, 19 ...
                  split_dataset_name,
                  video_ids):
    video_to_frames = FIBROID_TRAIN_VIDEO_TO_FRAMES
    logging.debug(f'{split_dataset_name} Generating metas...')   
    metas = []
    for vid_id in tqdm(video_ids):
        all_frames = sorted(video_to_frames[vid_id])
        if step_size is None:  
            # fibroid_train
            # video_id:
            # meta_idx 
            # all_frames: list[str], 这个视频的所有frame的string, 排序好的
            metas.append({
                'video_id': vid_id,
                'all_frames' : all_frames,
                'meta_idx': len(metas)
            }) 
        else:
            # fibroid_train_step[6]
            # 'video_id': 
            # 'frame_idx': 
            # 'all_frames': 
            # 'meta_idx'
            for frame_idx in range(0, len(all_frames), step_size):
                metas.append({
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })                

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')
    return metas


def fibroid_evaluate(eval_video_ids,
                     split_dataset_name,
                     step_size,):
    if (step_size is not None) and (step_size > 1):
        logging.warning('为什么 evaluate的时候step size大于1呢')
        raise ValueError()
    
    video_to_frames = FIBROID_VALIDATE_VIDEO_TO_FRAMES
    metas = []
    for video_id in eval_video_ids:
        all_frames = sorted(video_to_frames[video_id])
        if step_size == None:
            metas.append({
                'video_id': video_id,
                'all_frames': all_frames,
                'meta_idx': len(metas)
            })     
    
        else:   
            for frame_idx in range(0, len(all_frames), step_size):
                metas.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })                                 

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')  
    return metas


# train: video/step; whether for_each_refer_text

# evaluate
# 'video_id': 
# 'frame_idx' : 
# 'all_frames':
# 'meta_idx': 

# fibroid_test_ForEachReferText 
    # offline测试, 整个video都输入
# fibroid_test_step[1]_ForEachReferText 
    # online 测试, 用前5帧/前5帧的大scale sampling的信息去分割第6帧的mask / 或者offline的时候, 用大temporal scale抽5帧, 去预测这一帧的mask
    # 如果online测试的时候用的是连续5帧, 那么尽管fps很大, 但是没有global information, 连续5帧应该换成hybrid frames
    # 很复杂的测试流程, 比如一个test video, 用5中不同的temporal scale去抽样, 得到5中不同的scale的信息结果, 然后汇总起来, 去得到每帧的最优的结果 
# fibroid_test_step[6]_ForEachReferText 很少用

# fibroid_train_ForEachReferText,  可以做hybrid temporal training
# fibroid_train_step[6]_ForEachReferText, 一个视频抽0,6,12,18..., 然后mapper按照这些frame_idx抽clip frames


_root = os.getenv('DATASET_PATH')
root = os.path.join(_root, 'uterus_myoma/Dataset/temp5')
fibroid_meta = {
    'thing_classes': ['rumor', 'not rumor'],
    'thing_colors': [(255., 140., 0.), (0., 255., 0.)],
    'root': root
}

FIBROID_TRAIN_VIDEO_IDS = sorted([taylor_swift for taylor_swift in os.listdir(os.path.join(root, 'train/images')) if\
                   os.path.isdir(os.path.join(root, 'train/images', taylor_swift))])

FIBROID_TRAIN_VIDEO_TO_FRAMES = {
    video_id: sorted([png[:-4] for png in os.listdir(os.path.join(root, 'train/images', video_id)) if png.endswith('.png')])\
        for video_id in FIBROID_TRAIN_VIDEO_IDS
}

FIBROID_VALIDATE_VIDEO_IDS = sorted([justin_bieber for justin_bieber in os.listdir(os.path.join(root, 'test/images')) if\
                   os.path.isdir(os.path.join(root, 'test/images', justin_bieber))])

FIBROID_VALIDATE_VIDEO_TO_FRAMES = {
    video_id: sorted([png[:-4] for png in os.listdir(os.path.join(root, 'test/images', video_id)) if png.endswith('.png')])\
        for video_id in FIBROID_VALIDATE_VIDEO_IDS
}

visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['fibroid_train_step6'] = [] 
visualize_meta_idxs['fibroid_train'] = []  # 不抽样, 50个video, 随机抽一个scale, hybrid temporal scale training
visualize_meta_idxs['fibroid_validate'] = []
visualize_meta_idxs['fibroid_validate_step1'] = []

train_meta = copy.deepcopy(fibroid_meta)
train_meta.update({
    'mode': 'train',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'train/images')),
    'get_frames_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, 'train/labels'),),
})

validate_meta = copy.deepcopy(fibroid_meta)
validate_meta.update({
    'mode': 'evaluate',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'test/images')),
    'get_frames_gt_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, 'test/labels'),),
})
evaluate_step_sizes = [1, None,] 
train_step_sizes = [6, 12, 18, None]

# validate
for step_size in evaluate_step_sizes:
    step_identifer = '' if step_size is None else f'_step[{step_size}]'
    split_name = f'fibroid_validate{step_identifer}'
    DatasetCatalog.register(split_name, partial(fibroid_evaluate,
                                                eval_video_ids=FIBROID_VALIDATE_VIDEO_IDS, 
                                                split_dataset_name=split_name, 
                                                step_size=step_size))    
    MetadataCatalog.get(split_name).set(**validate_meta, step_size=step_size,
                                        visualize_meta_idxs=visualize_meta_idxs[split_name])

# train
for step_size in train_step_sizes:
    step_identifer = '' if step_size is None else f'_step[{step_size}]'
    split_name = f'fibroid_train{step_identifer}'
    DatasetCatalog.register(split_name, partial(fibroid_train,
                                                video_ids=FIBROID_TRAIN_VIDEO_IDS, 
                                                split_dataset_name=split_name,
                                                step_size=step_size,))    
    MetadataCatalog.get(split_name).set(**train_meta, step_size=step_size,
                                            visualize_meta_idxs=visualize_meta_idxs[split_name]) 



 

