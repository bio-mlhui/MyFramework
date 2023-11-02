import json
import os
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import shutil
import h5py
import pandas
from functools import partial
from typing import Union
import copy
from PIL import Image
import numpy as np
from einops import rearrange
import torch
import torchvision.transforms.functional as F

from data_schedule.registry import register_data_schedule
from .augmentation_videos import video_aug_entrypoints

from util.misc import is_dist_avail_and_initialized, all_gather, to_device
import torch.distributed as dist


import pycocotools.mask as mask_util
from pycocotools.mask import encode, area
from detectron2.data import DatasetCatalog

import wandb
import plotly.express as px

from util.box_ops import box_xyxy_to_cxcywh
from torch.utils.data import DataLoader, Dataset
import torchvision.io.video as video_io
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, bounding_box_from_mask, generate_windows_of_video
from .metric_utils import get_AP_PAT_IOU_PerFrame
from .utils import CollatorWithAux, DatasetWithAux



__all__ = ['a2ds_schedule']


# group-level functions
def get_class_label(idx):
        """idx是np.float64, 想要获得他的label
        """
        return int(str(int(idx))[0]) - 1

def get_action_label(idx):
        """idx是np.float64, 想要获得他的label
        """
        return int(str(int(idx))[1]) - 1
    
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
     

def read_text_annotations_groupby_video(root):
    with open(os.path.join(root, f'a2d_annotation.txt'), 'r') as f:
        text_annotations = pandas.read_csv(f)
    text_annotations = text_annotations[text_annotations.instance_id != '1 (copy)']
    text_annotations['exp_id'] = np.arange(len(text_annotations))
    text_annotations = text_annotations.astype({"instance_id":int})
    return text_annotations.groupby(by='video_id')     


def visualize_dataset_information(root):
    pass


def generate_pFpE_validate_coco_file(root):
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'split']
    with open(os.path.join(root, 'a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
    unused_ids.append('e7Kjy13woXg')
    videos = videos[~videos.video_id.isin(unused_ids)]
    validate_videos = videos[videos.split==1]
    
    text_annotations_by_video = read_text_annotations_groupby_video(root)
    
    images_id_set = set()
    perframe_coco_evaluation = []
    images_dict = []
    for video_id in tqdm(validate_videos['video_id'].tolist()):
        annotated_frames = sorted(glob((os.path.join(root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
        annotated_frames = [int(f.split('/')[-1].split('.')[0]) for f in annotated_frames] # start from 1
        text_annotations = text_annotations_by_video.get_group(video_id).to_dict('records')
        
        for frame in annotated_frames:
            f = h5py.File(os.path.join(root, f'a2d_annotation_with_instances/{video_id}', f'{frame:05d}.h5'))
            masks, appear_objs, *_, boxes = illicit(video_id, frame, f)
            f.close()
            
            for exp_dict in text_annotations:
                refenent_obj_id = exp_dict['instance_id']
                exp_id = exp_dict['exp_id']
                if refenent_obj_id in appear_objs:
                    referent_idx = appear_objs.index(refenent_obj_id)
                    image_id = f'v_{video_id}_f_{frame}_e_{exp_id}'
                    assert image_id not in images_id_set
                    images_id_set.add(image_id)
                    
                    gt_mask = masks.numpy()[referent_idx]
                    images_dict.append({'id': image_id, 'height': gt_mask.shape[0], 'width': gt_mask.shape[1]})
                    
                    mask_rle = encode(gt_mask)
                    mask_rle['counts'] = mask_rle['counts'].decode('ascii')
                    mask_area = float(area(mask_rle))
                    
                    bbox = boxes.numpy()[referent_idx] # x1y1x2y2 form 
                    assert bbox.ndim == 1 and len(bbox) == 4
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    perframe_coco_evaluation.append({
                        'id': len(perframe_coco_evaluation),
                        'image_id': image_id,
                        'category_id': 1,  
                        'segmentation': mask_rle,
                        'area': mask_area,
                        'bbox': bbox_xywh,
                        'iscrowd': 0,
                    })
                    
    print(f'there are a total {len(perframe_coco_evaluation)} (image, exp_id)')
    assert len(perframe_coco_evaluation) == 3800
    dataset_dict = {
        'categories': [{'id': 1, 'name': 'dummy_class'}],
        'images': images_dict,
        'annotations':  perframe_coco_evaluation,
    }
    with open(os.path.join(root, f'pFpE_validate_coco.json'), 'w') as f:
        json.dump(dataset_dict, f)

# 和世界联系
# 只要模型输出每个annotated frame的预测每个一个就行
# 和generate params没关系
def validate(loader, model, device, is_distributed, is_main_process, output_dir, validate_metrics, root):
    eval_metrics = {}
    if 'pFpE_mAP_Pat_IOU' in validate_metrics:
        pFpE_coco_file = os.path.join(root, 'pFpE_validate_coco.json')
        if not os.path.exists(pFpE_coco_file):
            generate_pFpE_validate_coco_file(root)
            
        coco_perframe_preds = [] # 3800 * 1 * n
        for idx, batch_dict in tqdm(enumerate(loader)):
            samples = to_device(batch_dict['samples'], device) # list[t 3 h w]
            targets = to_device(batch_dict['targets'], device)
            text_queries = batch_dict['text_query']
            auxiliary = to_device(batch_dict['auxiliary'], device)
            # [nq t' h w], [n t'/n], batch;
            model_preds = model.sample(samples, text_queries, auxiliary, targets,)
            model_mask_preds = model_preds['query_pred_masks'] # [n t' h w], batch
            nf_by_batch = [mm.shape[1] for mm in model_mask_preds]
            query_mask_preds = [] # [n h w] bt'
            for bch_idx in range(len(model_mask_preds)):
                bch_mask_pred = model_mask_preds[bch_idx].split(1, dim=1) # list[n 1 h w], t'
                bch_mask_pred = [bm.squeeze(1) for bm in bch_mask_pred] # list[n h w], t'
                query_mask_preds.extend(bch_mask_pred)
            
            # [nq t'], batch
            model_prob_preds = model_preds['query_pred_is_referred_prob'] 
            if model_prob_preds[0].dim() == 1:
                for idx in range(len(model_prob_preds)):
                    model_prob_preds[idx] = model_prob_preds[idx].unsqueeze(-1).repeat(1, nf_by_batch[idx])
                    
            # [nq t'], b -> [nq], bt'
            query_refer_prob_preds = []
            for bch_idx in range(len(model_prob_preds)):
                bch_prob_preds = model_prob_preds[bch_idx].split(1, dim=1) # nq 1, t'
                bch_prob_preds = [bpp.squeeze(1) for bpp in bch_prob_preds] # [nq], t'
                query_refer_prob_preds.extend(bch_prob_preds)
 
            query_rle_masks = [[mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in frame_mask_pred.cpu()] for frame_mask_pred in query_mask_preds]
            
            # [n, n h w, list[rle]n ] bt_has_ann_sum
            model_outputs = [{'scores': s, 'masks': m, 'rle_masks': rle}
                            for s, m, rle in zip(query_refer_prob_preds, query_mask_preds, query_rle_masks)]
            image_ids = [] # bt
            for t in targets:
                image_ids.extend(t['image_ids'])
            for p, image_id in zip(model_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    coco_perframe_preds.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})
            
        if is_distributed:
            coco_perframe_preds = all_gather(coco_perframe_preds)
            coco_perframe_preds = [p for p_list in coco_perframe_preds for p in p_list]

        if is_main_process:
            eval_metrics.update(get_AP_PAT_IOU_PerFrame(pFpE_coco_file, coco_perframe_preds))
        else:
            eval_metrics = {}
            
        if is_distributed:
            dist.barrier()
    else:
        raise ValueError()
    return eval_metrics

# utils
def generate_train_validate_samples(root, generate_params):
    # trainset:
    # 保证每个sample都是consistent的
    # validateset:
    # 保证覆盖所有的annotated frames
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'split']
    with open(os.path.join(root, 'a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
    unused_ids.append('e7Kjy13woXg')
    videos = videos[~videos.video_id.isin(unused_ids)]
    text_annotations_groupby_video = read_text_annotations_groupby_video(root)
    
    
    # train set的samples  
    train_videos = videos[videos.split== 0]
    train_video_ids = train_videos['video_id'].tolist()    
    params_by_vid = [(root, video_id, 'train', text_annotations_groupby_video.get_group(video_id).to_dict('records'), generate_params) for video_id in train_video_ids]
    n_jobs = min(multiprocessing.cpu_count(), 12)
    train_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
    train_samples = [s for l in train_samples for s in l] # 3016个视频, 5357个 train sample
    if generate_params['name'] == '61m61m':
        assert len(train_samples) == 15741

    # validate set的samples 
    validate_videos = videos[videos.split==1]
    validate_video_ids = validate_videos['video_id'].tolist()     
    params_by_vid = [(root, video_id, 'validate', text_annotations_groupby_video.get_group(video_id).to_dict('records'), generate_params) for video_id in validate_video_ids]
    n_jobs = min(multiprocessing.cpu_count(), 12)
    validate_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
    validate_samples = [s for l in validate_samples for s in l]  # 1295个 test sample
    
    # 15741, 3800; 5357, 1295
    with open(os.path.join(root, f'{generate_params["name"]}_TrainValidate_samples.json'), 'w') as f:
        json.dump({'train': train_samples, 'validate':validate_samples}, f)

def generate_samples_of_one_video(root, video_id, split, text_annotations, generate_params):       
    vframes, _, _ = video_io.read_video(filename=os.path.join(root, 'Release/clips320H', f'{video_id}.mp4'), pts_unit='sec', output_format='TCHW')
    
    all_frames = (np.arange(len(vframes)) + 1).tolist()
    annotated_frames = sorted(glob((os.path.join(root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
    annotated_frames = [int(f.split('/')[-1].split('.')[0]) for f in annotated_frames] # start from 1
    appear_objs_by_annotated_frame = {}
    for frame in annotated_frames:
        f = h5py.File(os.path.join(root, f'a2d_annotation_with_instances/{video_id}', f'{frame:05d}.h5'))
        _, appear_objs, *_ = illicit(video_id, frame, f)
        f.close()
        appear_objs_by_annotated_frame[frame] = appear_objs
        
    if split == 'train': 
        # 保证每个(window, text)肯定是consistent的并且能够用对应的帧做监督
        train_window_size = generate_params['train_window_size']
        train_window_step = generate_params['train_window_step']
        filter_name = generate_params['train_filter'] # middle/all
        samples = [] 
        
        if filter_name == 'middle':
            # pad first and last to make sure the same to the public evaluation
            if (annotated_frames[0] - 1) < (train_window_size // 2):
                all_frames = [all_frames[0]] * (train_window_size//2 - (annotated_frames[0] - 1)) + all_frames
            if (len(all_frames) - annotated_frames[-1]) < ((train_window_size - 1) // 2):
                all_frames = all_frames + [all_frames[-1]] * ((train_window_size - 1) // 2 - (len(all_frames) - annotated_frames[-1])) 
            
        # 抽取所有可能的windows
        sampled_windows = generate_windows_of_video(all_frames, window_size=train_window_size, window_step=train_window_step, 
                                                    force_all_used=True)
        # 过滤掉没有annotated帧的window, a2ds特有的情况
        a2ds_sampled_windows = []
        for sample_window in sampled_windows:
            if len(set(sample_window) & set(annotated_frames)) == 0:
                pass
            a2ds_sampled_windows.append(sample_window)
        sampled_windows = a2ds_sampled_windows
        
        filtered_windows = []
        # 只要中间帧是annotated frame的window
        if filter_name == 'middle':
            for sample_window in sampled_windows:
                if sample_window[train_window_size//2] in annotated_frames:
                    filtered_windows.append(sample_window)
        elif filter_name == 'all':
            filtered_windows = sampled_windows
        assert len(filtered_windows) > 0
        
        
        for window in filtered_windows:
            window_appear_instances = []
            for frame in window:
                if frame in annotated_frames:
                    window_appear_instances.extend(appear_objs_by_annotated_frame[frame])
            # 肯定出现过的物体, 不一定是全部
            window_appear_instances = set(window_appear_instances) 
            for exp_dict in text_annotations:
                referent_obj_id = exp_dict['instance_id']
                if referent_obj_id in window_appear_instances:
                    samples.append({
                        'video_id':video_id,
                        'window':window, 
                        'exp_id': exp_dict['exp_id'], 
                    })
        return samples
    
    elif split == 'validate': 
        # 不一定需要是consistent的
        # 覆盖所有的annotated frames, 并且每个annotated frames只出现在其中一个window里
        validate_window_size = generate_params['validate_window_size']
        validate_window_step = generate_params['validate_window_step']
        filter_name = generate_params['validate_filter'] # middle/all
        samples = [] 
        
        if filter_name == 'middle':
            # pad first and last to make sure the same to the public evaluation
            if (annotated_frames[0] - 1) < (validate_window_size // 2):
                all_frames = [all_frames[0]] * (validate_window_size//2 - (annotated_frames[0] - 1)) + all_frames
            if (len(all_frames) - annotated_frames[-1]) < ((validate_window_size - 1) // 2):
                all_frames = all_frames + [all_frames[-1]] * ((validate_window_size - 1) // 2 - (len(all_frames) - annotated_frames[-1])) 
            
            
        # 抽取所有可能的windows
        sampled_windows = generate_windows_of_video(all_frames, window_size=validate_window_size, window_step=validate_window_step, force_all_used=True)
        # 过滤掉没有annotated帧的window, a2ds特有的情况
        a2ds_sampled_windows = []
        for sample_window in sampled_windows:
            if len(set(sample_window) & set(annotated_frames)) == 0:
                pass
            a2ds_sampled_windows.append(sample_window)
        sampled_windows = a2ds_sampled_windows
        
        filtered_windows = []
        # 只要中间帧是annotated frame的window
        if filter_name == 'middle':
            for sample_window in sampled_windows:
                if sample_window[validate_window_size//2] in annotated_frames:
                    filtered_windows.append(sample_window)
        elif filter_name == 'all':
            filtered_windows = sampled_windows
        assert len(filtered_windows) > 0
        
        # 保证每个annotated frame只出现在一个window里
        for frame in annotated_frames:
            appear = [1 if frame in window else 0 for window in filtered_windows]
            assert sum(appear) == 1

        for window in filtered_windows:
            window_appear_instances = []
            for frame in window:
                if frame in annotated_frames:
                    window_appear_instances.extend(appear_objs_by_annotated_frame[frame])
            window_appear_instances = set(window_appear_instances) 
            # 保证每个sample是consistent的, (充分条件)
            for exp_dict in text_annotations:
                referent_obj_id = exp_dict['instance_id']
                if referent_obj_id in window_appear_instances:
                    samples.append({
                        'video_id':video_id,
                        'window':window, 
                        'exp_id': exp_dict['exp_id'], 
                    })
        return samples

def generate_text_aux_file(root, version):
    pass

       
@register_data_schedule
def a2ds_schedule(configs, is_distributed, process_id, num_processes):
    root = configs['data_dir']
    root = os.path.join(root, 'a2d_sentences')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    validate_batch_size= configs['validate_batch_size']
    
    # 训练样本的生成, 测试样本的生成
    generate_trainvalidate_params: dict = configs['generate_trainvalidate_params']
    # 训练时额外的数据 
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    video_aux_version: int = configs['video_aux_version']
    # 训练数据增强, 测试数据增强
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    # 训练时的SGD的loading
    training_seed: int  = configs['training_seed']
    train_batch_size= configs['train_batch_size']
    # 测试时候的metrics
    validate_metrics = configs['validate_metrics']

    
    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize', 'hflip_ResizeSmaller', "resizeSmaller"]
        assert train_augmentation['name'] in amr_legi_augs
        assert validate_augmentation['name'] in amr_legi_augs
    
    # dataset part 
    trainvalidate_samples_file = os.path.join(root, f'{generate_trainvalidate_params["name"]}_TrainValidate_samples.json')
    if not os.path.exists(trainvalidate_samples_file):
        if process_id == 0:
            generate_train_validate_samples(root, generate_trainvalidate_params)
        if is_distributed:
            dist.barrier()
    with open(trainvalidate_samples_file, 'r') as f:
        samples = json.load(f)
        train_samples = samples['train']
        validate_samples = samples['validate']
    
    if text_aux_version != 0:
        text_aux_file = os.path.join(root, f'text_to_aux.json')
        # you need to generate outside the main module
        with open(text_aux_file, 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if video_aux_version != 0:
        video_aux_file = os.path.join(root, f'video_aux_v{video_aux_version}.json')
        if not os.path.exists(video_aux_file):
            generate_text_aux_file(root, video_aux_file)
        with open(video_aux_file, 'r') as f:
            video_aux_by_auxid = json.load(f)
    else:
        video_aux_by_auxid = None        
        
    text_annotations_by_videoid = read_text_annotations_groupby_video(root)
    
    create_train_aug = video_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)
    train_dataset = A2DS_Dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='train',
                                samples = train_samples,
                                augmentation=train_aug,
                                text_annotations_by_videoid=text_annotations_by_videoid,
                                
                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                video_aux_version=video_aux_version,
                                video_aux_by_auxid=video_aux_by_auxid) 
    
    create_validate_aug = video_aug_entrypoints(validate_augmentation['name'])
    validate_aug = create_validate_aug(validate_augmentation)                     
    validate_dataset = A2DS_Dataset(root=root,
                                    pt_tokenizer_dir=pt_tokenizer_dir,
                                split='validate',
                                samples = validate_samples,
                                augmentation=validate_aug,
                                text_annotations_by_videoid=text_annotations_by_videoid,
                                
                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                video_aux_version=video_aux_version,
                                video_aux_by_auxid=video_aux_by_auxid) 

    # dataloader part
    sampler_train = TrainRandomSampler_ByEpoch_Distributed(train_dataset,
                                    num_replicas=num_processes,
                                    rank=process_id,
                                    seed=training_seed,)
    train_loader = DataLoader(train_dataset,
                            batch_size=train_batch_size,
                            sampler=sampler_train,
                            collate_fn=train_dataset.collator, 
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)
            
    sampler_validate = Evaluate_Sampler_Distributed(validate_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    validate_loader = DataLoader(validate_dataset, 
                                batch_size=validate_batch_size, 
                                sampler=sampler_validate,
                                collate_fn=validate_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
       
    def my_dataset_function():
        return [{}]
    from detectron2.data import DatasetCatalog
    DatasetCatalog.register('a2ds', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('a2ds').thing_classes = ['r', 'nr']
    MetadataCatalog.get('a2ds').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

        
    return train_loader, sampler_train,\
            validate_loader, partial(validate, validate_metrics=validate_metrics, root=root),\
                None, None

def a2ds_normalize_text(text_query):
    # 非法输入
    if text_query == 'The left with yellow t shirt on the left running':
        text_query = 'the man with yellow tshirt on the left running'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

class A2DS_Dataset(DatasetWithAux):

    # h: 320, 
    # train: w_min: 320, w_max:768, 15741
    # validate:  w_min: 390, w_max:640, 3800
    def __init__(self, 
                 root,
                 split,
                 samples,
                 augmentation, 
                 text_annotations_by_videoid,
                 
                 text_aux_version,
                 text_aux_by_auxid,
                 video_aux_version,
                 video_aux_by_auxid,
                 pt_tokenizer_dir,
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                         text_aux_by_auxid=text_aux_by_auxid,
                         video_aux_version=video_aux_version,
                         video_aux_by_auxid=video_aux_by_auxid,
                         pt_tokenizer_dir=pt_tokenizer_dir,
                         )
        self.root = root
        self.object_classes = ['adult','baby','ball','bird', 'car', 'cat', 'dog']
        self.samples = samples
        self.text_annotations_by_videoid = text_annotations_by_videoid

        self.augmentation = augmentation
        
        self.split = split
        collator_kwargs = {}
        if text_aux_version == 1 or text_aux_version == 2 or text_aux_version == 3 or text_aux_version == 4:
            collator_kwargs['tokenizer'] = self.tokenizer
        self.collator = Collator(split=split,
                                 text_aux_version=text_aux_version,
                                 video_aux_version=video_aux_version,
                                 **collator_kwargs)
                
    def __getitem__(self, item_idx):
        if self.split == 'test':
            video_id, window_frames, exp_id = self.samples[item_idx]['video_id'], self.samples[item_idx]['window'], self.samples[item_idx]['exp_id']
            window_frames = sorted(window_frames)
                            
            exp_dict = self.text_annotations_by_videoid.get_group(video_id).set_index('exp_id').to_dict('index')[exp_id]
            text_query = exp_dict['query']
            text_query = a2ds_normalize_text(text_query)
            
            # videos
            vframes, _, _ = video_io.read_video(filename=os.path.join(self.root, 'Release/clips320H', f'{video_id}.mp4'), pts_unit='sec', output_format='TCHW')
            H, W = vframes.shape[-2:]
            vframes = vframes[[w-1 for w in window_frames]]
            vframes = [F.to_pil_image(frame) for frame in vframes] # list[PIL Image]
            has_ann = torch.zeros([len(vframes)]).bool()
            annotated_frames = sorted(glob((os.path.join(self.root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
            annotated_frames = [int(f.split('/')[-1].split('.')[0]) for f in annotated_frames] # start from 1
            image_ids = []
            for idx, frame in enumerate(window_frames):
                if frame in annotated_frames:
                    has_ann[idx] = True
                    image_ids.append(f'v_{video_id}_f_{frame}_e_{exp_id}')
            meta = {
                'orig_size': torch.tensor([len(vframes), H, W]), # t h w
                'size': torch.tensor([len(vframes), H, W]), # t h w
                'image_ids': image_ids,
                'has_ann': has_ann,
            }
            vframes, text_query, meta = self.augmentation(vframes, [text_query], meta)
            return vframes, text_query[0], self.get_aux(item_idx, exist_queries=None,
                                                        video_auxid=video_id, text_auxid=text_query[0]), meta
        
        elif self.split == 'train' or self.split == 'validate':
            # start from 1, sorted
            video_id, window_frames, exp_id = self.samples[item_idx]['video_id'], self.samples[item_idx]['window'], self.samples[item_idx]['exp_id']
            window_frames = sorted(window_frames)
            exp_dict = self.text_annotations_by_videoid.get_group(video_id).set_index('exp_id').to_dict('index')[exp_id]
            text_query = exp_dict['query']
            text_query = a2ds_normalize_text(text_query)
            referent_obj_id = exp_dict['instance_id']
            
            vframes, _, _ = video_io.read_video(filename=os.path.join(self.root, 'Release/clips320H', f'{video_id}.mp4'), pts_unit='sec', output_format='TCHW')
            H, W = vframes.shape[-2:]
            vframes = vframes[[w-1 for w in window_frames]]
            vframes = [F.to_pil_image(frame) for frame in vframes] # list[PIL Image]
            has_ann = torch.zeros([len(vframes)]).bool()
            annotated_frames = sorted(glob((os.path.join(self.root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
            annotated_frames = [int(f.split('/')[-1].split('.')[0]) for f in annotated_frames] # start from 1

            masks_by_ann_frame = [] # list[ni hi wi], nf -> n_max hi wi
            classid_by_ann_frame = [] # list[ni], nf
            appear_instances_by_ann_frame = [] # list[ni]
            num_annframe_in_window = 0
            image_ids = []
            for idx, frame in enumerate(window_frames):
                if frame in annotated_frames:
                    frame_annotation = h5py.File(os.path.join(self.root, 'a2d_annotation_with_instances', video_id, f'{frame:05d}.h5'))
                    # n h w (bool) n, n, n (long)
                    masks, appear_instances, class_ids, _, _ = illicit(video_id, frame, frame_annotation)
                    frame_annotation.close()
                    masks_by_ann_frame.append(masks)
                    classid_by_ann_frame.append(class_ids)
                    appear_instances_by_ann_frame.append(torch.tensor(appear_instances))
                    num_annframe_in_window += 1
                    has_ann[idx] = True
                    image_ids.append(f'v_{video_id}_f_{frame}_e_{exp_id}')
            assert num_annframe_in_window > 0
            window_appear_objs = torch.cat(appear_instances_by_ann_frame).unique().tolist()
            annotated_exps_by_object = [] # list[list[str],] n
            video_text_annotations = self.text_annotations_by_videoid.get_group(video_id).to_dict('records')
            for obj_id in window_appear_objs:
                annotated_exps_by_object.append([a2ds_normalize_text(t['query']) for t in video_text_annotations if t['instance_id'] == obj_id])
        
            obj_idx_map = {obj:idx for idx, obj in enumerate(window_appear_objs)}
            
            masks = torch.zeros([len(window_appear_objs), num_annframe_in_window, H, W], dtype=torch.bool) # n t h w
            class_ids = torch.empty([len(window_appear_objs)], dtype=torch.int64) # n
            
            for idx, (mask, appear_instances, classid) in enumerate(zip(masks_by_ann_frame, appear_instances_by_ann_frame, classid_by_ann_frame)):
                # n h w
                idx_map = [obj_idx_map[ins.item()] for ins in appear_instances]
                masks[idx_map, idx] = mask
                class_ids[idx_map] = classid

            boxes = []
            valids = []
            for object_mask in masks.flatten(0, 1).long():
                if (object_mask > 0).any():
                    y1, y2, x1, x2 = bounding_box_from_mask(object_mask.numpy())
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valids.append(1)
                    boxes.append(box)
                else:
                    box = torch.tensor([0,0,0,0]).to(torch.float)
                    valids.append(0)
                    boxes.append(box)
            boxes = torch.stack(boxes, dim=0) # nt 4
            boxes = rearrange(boxes, '(n t) c -> n t c', n=masks.shape[0], t=masks.shape[1])
            valids = torch.tensor(valids).long() # nt                               
            valids = rearrange(valids, '(n t) -> n t', n=masks.shape[0], t=masks.shape[1])
            
            referent_idx = window_appear_objs.index(referent_obj_id)  
            
            appear_texts = [text_query]
            for foo in annotated_exps_by_object:
                appear_texts.extend(foo)
                
            
                    
            targets = {
                'has_ann': has_ann,
                'masks': masks, # n t' h w (bool)
                'class_labels': class_ids, # n
                'boxes': boxes, # n t' 4, xyxy, float
                'valid': valids, # n t', 0/1

                'referent_idx': referent_idx, 
                'image_ids': image_ids,
                'orig_size': torch.tensor([len(vframes), H, W]), # T h w
                'size': torch.tensor([len(vframes),  H, W]), # T h w
            }
            
            vframes, appear_texts, targets = self.augmentation(vframes, appear_texts, targets)
            text_query = appear_texts[0]
            appear_texts = appear_texts[1:]
            
            cnt = 0
            for idx, foo in enumerate(annotated_exps_by_object):
                annotated_exps_by_object[idx] = appear_texts[cnt:(cnt+len(foo))]
                cnt += len(foo)
            assert cnt == len(appear_texts)
            
            return vframes, text_query, self.get_aux(item_idx, queries_by_objid=annotated_exps_by_object,
                                                     video_auxid=None, text_auxid=text_query), targets

    def get_aux(self, item_idx, queries_by_objid,
                video_auxid,
                text_auxid):
        aux = {}
        aux['sample_idx'] = item_idx
        aux['exist_queries'] = queries_by_objid
        aux.update(self.get_text_aux(text_auxid, queries_by_objid))
        aux.update(self.get_video_aux(video_auxid))
        return aux
    
    def __len__(self):
        return len(self.samples)


class Collator(CollatorWithAux):
    def __init__(self, split, 
                 text_aux_version,
                 video_aux_version,
                 **kwargs
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                       video_aux_version=video_aux_version,
                       **kwargs)
        self.split = split
    def __call__(self, batch):
    
        samples, text_query, auxiliary, meta_or_target = list(zip(*batch))
        samples = list(samples)
        text_query = list(text_query)
        auxiliary = list(auxiliary)
        meta_or_target = list(meta_or_target)
        
        batch_data = {
            'samples': samples,
            'text_query': list(text_query),
            'auxiliary': self.batching_aux(auxiliary)
        }
        if self.split == 'test':
            batch_data['meta'] = meta_or_target
        elif self.split == 'train' or self.split == 'validate':
            batch_data['targets'] = meta_or_target
    
        return batch_data





  

        

