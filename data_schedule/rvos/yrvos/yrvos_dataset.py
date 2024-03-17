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
from .yrvos_utils import YRVOS_CATEGORIES, YRVOS_TRAIN_VIDEO_IDS, YRVOS_TEST_VIDEO_IDS,\
                check_yrvos_validity, visualize_youtube_rvos, \
                    get_frames, get_frames_mask, normalize_text, get_each_obj_appear_frame_idxs, connect_vid_text, disconnect_vid_text
from data_schedule.rvos.apis import RVOS_Dataset

# 数据集的mask标注的每个instance必须是从1,2,3,4; 0是没有标注的地方(background)
# 保证 obj_ids是int而不是string, 并且等于mask图片上的int

def yrvos_train(root, 
                for_each_refer_text,
                step_size, # none / int; 0, 6, 13, 19 ...
                split_dataset_name,
                video_ids):
    check_yrvos_validity(root)
    with open(os.path.join(root, 'meta_expressions', 'train', f'meta_expressions.json'), 'r') as f:
        video_to_texts = json.load(f)['videos']  
    # {video_id: expressions: {exp_id: exp, obj_id:str},  frames: []}
    with open(os.path.join(root, 'train', f'meta.json'), 'r') as f:
        video_to_objs = json.load(f)['videos']
    # {video_id: objects: {obj_id: {category, frames}} 

    logging.debug(f'{split_dataset_name} Generating metas...')   
    metas = []
    for vid_id in video_ids:
        all_frames = sorted(video_to_texts[vid_id]['frames'])
        all_exps = video_to_texts[vid_id]['expressions'] # {exp_id: exp, obj_id}
        assert len(set(list(all_exps.keys()))) == len(list(all_exps.keys()))
        all_objs = video_to_objs[vid_id]['objects'] # {obj_id: {category, frames}
        all_obj_ids = list(all_objs.keys())
        assert len(set(all_obj_ids)) == len(all_obj_ids)
        
        all_objs = {int(obj_id): {'class_label': YRVOS_CATEGORIES[all_objs[obj_id]['category']]}
                            for obj_id in all_objs.keys()} # 不要假设连续
        assert 0 not in list(all_objs.keys())
        all_exps = {exp_id: {'exp': all_exps[exp_id]['exp'], 'obj_ids': [int(all_exps[exp_id]['obj_id'])],} # youtube_rvos只有一个物体
                            for exp_id in all_exps.keys()}
        if step_size is None:
            if for_each_refer_text:
                for exp_id in all_exps.keys():
                    metas.append({
                        'video_id': vid_id,
                        'referent_text': all_exps[exp_id]['exp'],
                        'referent_objs': all_exps[exp_id]['obj_ids'],
                        'all_frames' : all_frames,
                        'all_objs': all_objs,
                        'meta_idx': len(metas),
                    })    
            else:
                metas.append({
                    'video_id': vid_id,
                    'all_frames' : all_frames,
                    'all_objs': all_objs,
                    'all_exps':  all_exps,
                    'meta_idx': len(metas)
                }) 
        else:
            if for_each_refer_text:
                RVOS_Dataset
                for exp_id in all_exps.keys():
                    for frame_idx in range(0, len(all_frames), step_size):
                        metas.append({
                            'video_id': vid_id,
                            'referent_text': all_exps[exp_id]['exp'],
                            'referent_objs': all_exps[exp_id]['obj_ids'],
                            'frame_idx': frame_idx,
                            'all_frames': all_frames,
                            'all_objs': all_objs,
                            'meta_idx': len(metas)
                        })
            else:
                RVOS_Dataset
                for frame_idx in range(0, len(all_frames), step_size):
                    metas.append({
                        'video_id': vid_id,
                        'frame_idx': frame_idx,
                        'all_frames': all_frames,
                        'all_exps': all_exps,
                        'all_objs': all_objs,
                        'meta_idx': len(metas)
                    })                

    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')
    return metas


def yrvos_evaluate(root,
                   eval_video_ids,
                   split_dataset_name,
                   step_size,):
    if (step_size is not None) and (step_size > 1):
        logging.warning('为什么 evaluate的时候step size大于1呢')
        raise ValueError()
    check_yrvos_validity(root)
    # if 'validate' in split_dataset_name:
    #     with open(os.path.join(root, 'meta_expressions', 'train', 'meta_expressions.json'), 'r') as f:
    #         video_to_texts = json.load(f)['videos']           
    #     # {video_id: expressions: {exp_id: exp}, 
    #     #            frames: []}
    #     coco_file_name = os.path.join(root, f'{split_dataset_name}_coco_gt.json')
    #     raise NotImplementedError()
    #     generate_coco_gt_file(get_frames_mask_fn=MetadataCatalog.get(split_dataset_name).get('get_frames_mask_fn'),
    #                           video_to_texts=video_to_texts, coco_file_name=coco_file_name,
    #                           video_ids=eval_video_ids)

    with open(os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json'), 'r') as f:
        video_to_texts = json.load(f)['videos'] 
    # {video_id: expressions: {exp_id: exp}, frames: []}  

    metas = []
    for video_id in eval_video_ids:
        all_frames = sorted(video_to_texts[video_id]['frames'])
        all_exps = video_to_texts[video_id]['expressions'] 
        if step_size == None:
            for exp_id in all_exps.keys():
                metas.append({
                    'video_id': video_id,
                    'exp_id': exp_id,
                    'referent_text': all_exps[exp_id]['exp'],
                    'all_frames': all_frames,
                    'meta_idx': len(metas)
                })
    
        else:    
            for exp_id in all_exps.keys():
                for frame_idx in range(0, len(all_frames), step_size):
                    metas.append({
                        'video_id': video_id,
                        'exp_id': exp_id,
                        'referent_text': all_exps[exp_id]['exp'],
                        'frame_idx': frame_idx,
                        'all_frames': all_frames,
                        'meta_idx': len(metas)
                    })  
                                
    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')  
    return metas


# 两个参数:
# foreachrefer/ allexiststext
# step_size
# 所有train的再加上valSplit[300][2024]

# yrvos_test_ForEachRefer 
    # offline测试, 整个video都输入
# yrvos_test_step[1]_ForEachRefer 
    # online 测试, 用前5帧/前5帧的大scale sampling的信息去分割第6帧的mask / 或者offline的时候, 用大temporal scale抽5帧, 去预测这一帧的mask
    # 如果online测试的时候用的是连续5帧, 那么尽管fps很大, 但是没有global information, 连续5帧应该换成hybrid frames
    # 很复杂的测试流程, 比如一个test video, 用5中不同的temporal scale去抽样, 得到5中不同的scale的信息结果, 然后汇总起来, 去得到每帧的最优的结果 
# yrvos_test_step[6]_ForEachRefer 很少用

# yrvos_train_ForEachRefer,  可以做hybrid temporal training
# yrvos_train_step[6]_ForEachRefer, 一个视频抽0,6,12,18..., 然后mapper按照这些frame_idx抽clip frames

# obj_ids需要int, string变换

_root = os.getenv('DATASET_PATH')
root = os.path.join(_root, 'youtube_rvos')
yrvos_meta = {
    'root': root,
    'category_to_ids':YRVOS_CATEGORIES,
    'thing_classes': ['refer', 'not_refer'],
    'thing_colors': [(255., 140., 0.), (0., 255., 0.)],
    'normalize_text_fn': normalize_text,
    'connect_vidText_fn': connect_vid_text,
    'disconnect_vidText_fn': disconnect_vid_text,
}

visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['yrvos_train_AllExistsText'] = []
visualize_meta_idxs['yrvos_train_step[6]_AllExistsText'] = []

visualize_meta_idxs['yrvos_train_ForEachRefer'] = []
visualize_meta_idxs['yrvos_train_step[6]_ForEachRefer'] = []

visualize_meta_idxs['yrvos_test_ForEachRefer'] = [ ]
visualize_meta_idxs['yrvos_test_step[1]_ForEachRefer'] = [ ]

train_meta = copy.deepcopy(yrvos_meta)
train_meta.update({
    'mode': 'train',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'train/JPEGImages')),
    'get_frames_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, 'train/Annotations'),),
    'get_each_obj_appear_frame_idxs': partial(get_each_obj_appear_frame_idxs, root=root)
})

test_meta = copy.deepcopy(yrvos_meta)
test_meta.update({
    'mode': 'evaluate',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'valid/JPEGImages')),
})


# test eval_meta_keys
with open(os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json'), 'r') as f:
    test_video_to_exps = json.load(f)['videos']

eval_meta_keys = {} # vidText: frame
for eval_vid in YRVOS_TEST_VIDEO_IDS:
    for eval_exp_id in  test_video_to_exps[eval_vid]['expressions'].keys():
        eval_meta_keys[connect_vid_text(eval_vid, eval_exp_id)] = sorted(test_video_to_exps[eval_vid]['frames'])
assert sum([len(eval_meta_keys[key]) for key in eval_meta_keys.keys()]) == 22046
for step_size in [1, None,]:
    step_identifer = '' if step_size is None else f'_step[{step_size}]'
    split_name = f'yrvos_test{step_identifer}_ForEachRefer'
    DatasetCatalog.register(split_name, partial(yrvos_evaluate,
                                                eval_video_ids=YRVOS_TEST_VIDEO_IDS, 
                                                split_dataset_name=split_name,
                                                step_size=step_size,
                                                root=root))    
    MetadataCatalog.get(split_name).set(**test_meta, step_size=step_size,
                                        eval_meta_keys = eval_meta_keys,
                                        visualize_meta_idxs=visualize_meta_idxs[split_name])

for step_size in  [1, 6, 12, 18, None]:
    step_identifer = '' if step_size is None else f'_step[{step_size}]'
    for wfer in [True, False]:
        wfer_postfix = '_ForEachRefer' if wfer else '_AllExistsText'
        split_name = f'yrvos_train{step_identifer}{wfer_postfix}'
        DatasetCatalog.register(split_name, partial(yrvos_train,
                                                    for_each_refer_text=wfer,
                                                    video_ids=YRVOS_TRAIN_VIDEO_IDS, 
                                                    step_size=step_size,
                                                    split_dataset_name=split_name,
                                                    root=root,))    
        MetadataCatalog.get(split_name).set(**train_meta, step_size=step_size,
                                            visualize_meta_idxs=visualize_meta_idxs[split_name]) 

# num_validate_videos = [200, 300, 400]
# validate_meta = copy.deepcopy(yrvos_meta)
# validate_meta.update({
#     'mode': 'evaluate',
#     'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'train/JPEGImages')),
# })
# for num_val in num_validate_videos:
#     train_val_split_identifier = f'_valSap[{num_val}][2024]'
#     g = torch.Generator()
#     g.manual_seed(2024)
#     sampled_validate_idxs = torch.randperm(len(YRVOS_TRAIN_VIDEO_IDS), generator=g)[:num_val].numpy()
#     train_idxs = np.setdiff1d(np.arange(len(YRVOS_TRAIN_VIDEO_IDS)), sampled_validate_idxs)
#     validate_video_ids = YRVOS_TRAIN_VIDEO_IDS[sampled_validate_idxs].tolist()
#     train_video_ids = YRVOS_TRAIN_VIDEO_IDS[train_idxs].tolist()

#     for step_size in train_step_sizes:
#         step_identifer = '' if step_size is None else f'_step[{step_size}]'

#         for wfer in whether_for_each_refer:
#             wfer_postfix = '_ForEachRefer' if wfer else ''
#             split_name = f'yrvos_train{train_val_split_identifier}{step_identifer}{wfer_postfix}'
#             DatasetCatalog.register(split_name, partial(yrvos_train,
#                                                         for_each_refer_text=wfer,
#                                                         video_ids=train_video_ids, 
#                                                         step_size=step_size,
#                                                         split_dataset_name=split_name,
#                                                         root=root,))    
#             MetadataCatalog.get(split_name).set(**train_meta, step_size=step_size,
#                                                 visualize_meta_idxs=visualize_meta_idxs[split_name]) 
    
#     for step_size in evaluate_step_sizes:
#         step_identifer = '' if step_size is None else f'_step[{step_size}]'
#         split_name = f'yrvos_validate{train_val_split_identifier}{step_identifer}'
#         DatasetCatalog.register(split_name, partial(yrvos_evaluate, 
#                                                     eval_video_ids=validate_video_ids,
#                                                     split_dataset_name=split_name,
#                                                     step_size=step_size,
#                                                     root=root,))    
#         MetadataCatalog.get(split_name).set(**validate_meta, step_size=step_size,
#                                             visualize_meta_idxs=visualize_meta_idxs[split_name])  

