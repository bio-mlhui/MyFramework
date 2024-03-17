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
import torchvision.io.video as video_io
import torchvision.transforms.functional as F
from data_schedule.rvos.apis import RVOS_Dataset
from .a2ds_utils import A2DS_CATEGORIES, get_frames, get_frames_mask, normalize_text, connect_vid_text, get_action_label, get_class_label, disconnect_vid_text
import pandas

_root = os.getenv('DATASET_PATH')
root = os.path.join(_root, 'a2d_sentences')

# video_to_frames to save loading time
if not os.path.exists(os.path.join(root, 'Release/clips')):
    video_path = os.path.join(root, 'Release/clips320H')
    video_to_frames = {}
    for vid_id in tqdm(os.listdir(video_path)):
        vid_id = vid_id[:-4]
        vframes, _, _ = video_io.read_video(filename=os.path.join(video_path, f'{vid_id}.mp4'), pts_unit='sec', output_format='TCHW')
        vframes = [F.to_pil_image(fram) for fram in vframes]
        os.makedirs(os.path.join(root, 'Release/clips', vid_id))
        for idx, fram in enumerate(vframes):
            fram.save(os.path.join(root, 'Release/clips', vid_id, f'{(idx+1):05d}.jpg'))

if not os.path.exists(os.path.join(root, 'a2ds_yrvos_meta.json')):
    import h5py
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'split']
    with open(os.path.join(root, 'a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
    unused_ids.append('e7Kjy13woXg')
    videos = videos[~videos.video_id.isin(unused_ids)]
    train_videos = videos[videos.split== 0]
    train_video_ids = train_videos['video_id'].tolist()  

    validate_videos = videos[videos.split==1]
    validate_video_ids = validate_videos['video_id'].tolist()  

    with open(os.path.join(root, f'a2d_annotation.txt'), 'r') as f:
        text_annotations = pandas.read_csv(f)
    text_annotations = text_annotations[text_annotations.instance_id != '1 (copy)']

    video_to_text = defaultdict(dict) # vid: {exp_1: {}, }
    video_to_objs = defaultdict(dict) # vid: {obj_id: {'category'}}

    for index, row in text_annotations.iterrows():
        instance_id = str(int(row['instance_id']) + 1)
        text = row['query']
        video_id = row['video_id']
        video_to_text[video_id][f'{index}'] = {'exp': text, 'obj_id': instance_id}


    for video_id in tqdm(video_to_text.keys()):
        if video_id == 'e7Kjy13woXg':
            continue
        # action_ids = torch.tensor([get_action_label(idx) for idx in frame_file['id'][0]]).long()   
        all_objs =  [video_to_text[video_id][key]['obj_id'] for key in video_to_text[video_id]]
        instance_id_to_class_id = {key: None for key in all_objs}

        for haosen in os.listdir(os.path.join(root, 'a2d_annotation_with_instances', video_id)):
            frame_file = h5py.File(os.path.join(root, 'a2d_annotation_with_instances', video_id, haosen))
            class_ids = [get_class_label(idx) for idx in frame_file['id'][0]] # 0 0 0 2 3
            appear_instances = [str(int(ins_id)+1) for ins_id in frame_file['instance']] # 2 1 4 7 3
            assert set(appear_instances).issubset(set(list(instance_id_to_class_id.keys())))
            class_ids = class_ids[:len(appear_instances)]
           #  assert len(class_ids) == len(appear_instances)
            for hh1, hh2 in zip(appear_instances, class_ids):
                if instance_id_to_class_id[hh1] is None:
                    instance_id_to_class_id[hh1] = hh2
                else:
                    if hh2 != instance_id_to_class_id[hh1]:
                        print('false')
        for key in instance_id_to_class_id.keys():
            assert instance_id_to_class_id[key] is not None

        all_objs = {key: {'class_label': instance_id_to_class_id[key]} for key in all_objs}
        video_to_objs[video_id] = all_objs
    
    with open(os.path.join(root, 'a2ds_yrvos_meta.json'), 'w') as f:
        json.dump({'A2DS_TRAIN_VIDEO_IDS': train_video_ids,
                   'YRVOS_TEST_VIDEO_IDS': validate_video_ids,
                   'a2ds_video_to_text': video_to_text,
                   'a2ds_video_to_objs': video_to_objs}, f)    
    # train set的samples  
    # validate set的samples 
   
 
else:
    with open(os.path.join(root, 'a2ds_yrvos_meta.json'), 'r') as f:
        a2ds_yrvos_meta = json.load(f)
    A2DS_TRAIN_VIDEO_IDS = a2ds_yrvos_meta['A2DS_TRAIN_VIDEO_IDS']
    A2DS_TEST_VIDEO_IDS = a2ds_yrvos_meta['YRVOS_TEST_VIDEO_IDS']
    A2DS_VIDEO_TO_TEXT = a2ds_yrvos_meta['a2ds_video_to_text']
    A2DS_VIDEO_TO_OBJS = a2ds_yrvos_meta['a2ds_video_to_objs']



# 数据集的mask标注的每个instance必须是从1,2,3,4; 0是没有标注的地方(background)
# 保证 obj_ids是int而不是string, 并且等于mask图片上的int

def a2ds_train(root, 
                for_each_refer_text,
                step_size, # none / int; 0, 6, 13, 19 ...
                split_dataset_name,
                video_ids):

    logging.debug(f'{split_dataset_name} Generating metas...')   
    metas = []
    for vid_id in video_ids:
        all_frames = sorted(os.listdir(os.path.join(root, 'Release/clips', vid_id)))
        all_frames = [haosen[:-4] for haosen in all_frames]

        have_ann_frames = sorted(os.listdir(os.path.join(root, 'a2d_annotation_with_instances', vid_id)))
        have_ann_frames = [int(haosen[:-3])-1 for haosen in have_ann_frames]  

        all_exps = A2DS_VIDEO_TO_TEXT[vid_id] # {exp_id: exp, obj_id}
        assert len(set(list(all_exps.keys()))) == len(list(all_exps.keys()))
        all_objs = A2DS_VIDEO_TO_OBJS[vid_id] # {obj_id: {'class_label'}
        all_objs = {int(key): value  for key, value in all_objs.items()} # 不要假设连续
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
                    for frame_idx in have_ann_frames:
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


def a2ds_evaluate(root,
                   eval_video_ids,
                   split_dataset_name,
                   step_size,):
    if (step_size is not None) and (step_size > 1):
        logging.warning('为什么 evaluate的时候step size大于1呢')
        raise ValueError()
    metas = []
    for video_id in eval_video_ids:
        all_frames = sorted(os.listdir(os.path.join(root, 'Release/clips', video_id)))
        all_frames = [haosen[:-4] for haosen in all_frames]
        all_exps = A2DS_VIDEO_TO_TEXT[video_id]
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
                have_ann_frames = sorted(os.listdir(os.path.join(root, 'a2d_annotation_with_instances', video_id)))
                have_ann_frames = [int(haosen[:-3]) for haosen in have_ann_frames]
                for frame_idx in have_ann_frames:
                    metas.append({
                        'video_id': video_id,
                        'exp_id': exp_id,
                        'referent_text': all_exps[exp_id]['exp'],
                        'frame_idx': frame_idx-1,
                        'all_frames': all_frames,
                        'meta_idx': len(metas)
                    })  
                                
    logging.debug(f'{split_dataset_name} Total metas: [{len(metas)}]')  
    return metas



a2ds_meta = {
    'root': root,
    'thing_classes': ['refer', 'not_refer'],
    'thing_colors': [(255., 140., 0.), (0., 255., 0.)],
    'normalize_text_fn': normalize_text,
    'connect_vidText_fn': connect_vid_text,
    'disconnect_vidText_fn': disconnect_vid_text,
}

visualize_meta_idxs = defaultdict(list)

train_meta = copy.deepcopy(a2ds_meta)
train_meta.update({
    'mode': 'train',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'Release/clips')),
    'get_frames_mask_fn': partial(get_frames_mask, mask_path=os.path.join(root, 'a2d_annotation_with_instances'),),
    # 'get_each_obj_appear_frame_idxs': partial(get_each_obj_appear_frame_idxs, root=root)
})

test_meta = copy.deepcopy(a2ds_meta)
test_meta.update({
    'mode': 'evaluate',
    'get_frames_fn': partial(get_frames, frames_path=os.path.join(root, 'Release/clips')),
})


eval_meta_keys = {} # vidText: frame
for eval_vid in A2DS_TEST_VIDEO_IDS:
    for eval_exp_id in  A2DS_VIDEO_TO_TEXT[eval_vid].keys():
        all_frames = sorted(os.listdir(os.path.join(root, 'a2d_annotation_with_instances', eval_vid)))
        eval_meta_keys[connect_vid_text(eval_vid, eval_exp_id)] = [haosen[:-3] for haosen in all_frames]

for step_size in [1, None,]:
    step_identifer = '' if step_size is None else f'_step[{step_size}]'
    split_name = f'a2ds_test{step_identifer}_ForEachRefer'
    DatasetCatalog.register(split_name, partial(a2ds_evaluate,
                                                eval_video_ids=A2DS_TEST_VIDEO_IDS, 
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
        split_name = f'a2ds_train{step_identifer}{wfer_postfix}'
        DatasetCatalog.register(split_name, partial(a2ds_train,
                                                    for_each_refer_text=wfer,
                                                    video_ids=A2DS_TRAIN_VIDEO_IDS, 
                                                    step_size=step_size,
                                                    split_dataset_name=split_name,
                                                    root=root,))    
        MetadataCatalog.get(split_name).set(**train_meta, step_size=step_size,
                                            visualize_meta_idxs=visualize_meta_idxs[split_name]) 

