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

from PIL import Image
import numpy as np
from einops import rearrange
import torch
import torchvision.transforms.functional as F

from datasets.registry import register_data_schedule
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
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, bounding_box_from_mask, nested_tensor_from_videos_list_with_stride,\
    generate_windows_of_video
from datasets.rvos.metric_utils import get_AP_PAT_IOU_PerFrame
# text aux 和train/test无关
from .utils import Dataset_WithTextAux


__all__ = ['a2ds_perWinPerExp']


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
    
    # instance不unique的
    if video_id == 'EadxBPmQvtg' and frame_idx == 25:
        assert len(masks) == 11 and len(appear_instances) == 11
        assert appear_instances == [0, 1, 2, 3, 4, 5, 6, 7,8,9,1]
        masks = masks[:-1]         
        boxes = boxes[:-1]            
        appear_instances = appear_instances[:-1]
    assert len(torch.tensor(appear_instances).unique()) == len(appear_instances)
    
    # mask多于instance的
    if video_id == '95Nq6fQoP2o' and frame_idx == 32:
        assert len(masks) == 7 and len(appear_instances) == 6
        masks = masks[:6]
    elif video_id == 'I0MlLHTWCro' and frame_idx == 20:
        assert len(masks) == 4 and len(appear_instances) == 2
        masks = masks[:2]            
    elif video_id == 'IRrbHQjE4LQ' and frame_idx == 16:
        assert len(masks) == 6 and len(appear_instances) == 5
        masks = masks[:5]           
    assert len(masks) == len(appear_instances)
    
    class_ids = torch.tensor([get_class_label(idx) for idx in frame_file['id'][0]]).long() # 0-6
    action_ids = torch.tensor([get_action_label(idx) for idx in frame_file['id'][0]]).long()
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

def generate_coco_perframe_evaluate_file(root):
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'split']
    with open(os.path.join(root, 'a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
    unused_ids.append('e7Kjy13woXg')
    videos = videos[~videos.video_id.isin(unused_ids)]
    test_videos = videos[videos.split==1]
    
    text_annotations_by_video = read_text_annotations_groupby_video(root)
    
    images_id_set = set()
    perframe_coco_evaluation = []
    images_dict = []
    for video_id in tqdm(test_videos['video_id'].tolist()):
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
    with open(os.path.join(root, f'pWpE_pAFpE_coco_evaluate.json'), 'w') as f:
        json.dump(dataset_dict, f)

# 和世界联系
# 只要模型输出每个annotated frame的预测每个一个就行
# 和generate params没关系
def test(loader, model, device, is_distributed, is_main_process, output_dir, perFrame_eval_coco_file):
    coco_perframe_preds = [] # 3800 * 1 * n
    for batch_dict in tqdm(loader):
        samples = batch_dict['samples'].to(device)
        targets = to_device(batch_dict['targets'], device)
        text_queries = batch_dict['text_query']
        auxiliary = batch_dict['auxiliary']
        
        frame_has_ann = [t['inputvideo_whichframe_hasann'] for t in targets]
        # [n t h w], [n t/n], batch; 不是T
        # ['query_mask_preds', 'query_refer_prob_preds']
        model_preds_by_batch = model.sample(samples, text_queries, auxiliary, targets)

        # [n t h w] -> n bt h w -> bt n h w -> [n h w] bt
        query_mask_preds = torch.cat(model_preds_by_batch['query_pred_masks'], dim=1).permute(1, 0, 2, 3).split(1, dim=0)
        
        query_refer_prob_preds = model_preds_by_batch['query_pred_is_referred_prob']
        # [n t / n] -> [n t]
        if query_refer_prob_preds[0].dim() == 1:
            for idx in range(len(query_refer_prob_preds)):
                query_refer_prob_preds[idx] = query_refer_prob_preds[idx].unsqueeze(-1).repeat(1, frame_has_ann[idx].int().sum())
        # [n t] -> n bt -> bt n -> [n] bt
        query_refer_prob_preds = torch.cat(query_refer_prob_preds, dim=1).permute(1, 0).split(1, dim=0) 
        
        query_rle_masks = [[mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                            for mask in frame_mask_pred.cpu()] for frame_mask_pred in query_mask_preds]
        
        # [n, n h w, list[rle]n ] bt_has_ann_sum
        model_outputs = [{'scores': s, 'masks': m, 'rle_masks': rle}
                        for s, m, rle in zip(query_refer_prob_preds, query_mask_preds, query_rle_masks)]

        # 每个test sample 
        # 有t个被annotate的帧, 返回 t*n个预测
        image_ids = []
        for t in targets:
            image_ids.extend(t['perframe_eval']['image_ids'])
        for p, image_id in zip(model_outputs, image_ids):
            for s, m in zip(p['scores'], p['rle_masks']):
                # int, h w
                coco_perframe_preds.append({'image_id': image_id,
                                    'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                    'segmentation': m,
                                    'score': s.item()})
        
    if is_distributed:
        coco_perframe_preds = all_gather(coco_perframe_preds)
        coco_perframe_preds = [p for p_list in coco_perframe_preds for p in p_list]

    if is_main_process:
        eval_metrics = get_AP_PAT_IOU_PerFrame(perFrame_eval_coco_file, coco_perframe_preds)
    else:
        eval_metrics = None
        
    if is_distributed:
        dist.barrier()
    return eval_metrics


def generate_train_test_samples(root, generate_params):
    # trainset:
    # 保证每个sample都是consistent的
    # testset:
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
    train_samples = [s for l in train_samples for s in l] 
    assert len(train_samples) == 15741

    # test set的samples 
    test_videos = videos[videos.split==1]
    test_video_ids = test_videos['video_id'].tolist()     
    params_by_vid = [(root, video_id, 'test', text_annotations_groupby_video.get_group(video_id).to_dict('records'), generate_params) for video_id in test_video_ids]
    n_jobs = min(multiprocessing.cpu_count(), 12)
    test_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
    test_samples = [s for l in test_samples for s in l] 
    
    # 15741, 3800
    with open(os.path.join(root, f'pWpE_{generate_params["name"]}_TrainTest_samples.json'), 'w') as f:
        json.dump({'train': train_samples, 'test':test_samples}, f)

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
    
    elif split == 'test': 
        # 不一定需要是consistent的
        # 覆盖所有的annotated frames, 并且每个annotated frames只出现在其中一个window里
        test_window_size = generate_params['test_window_size']
        test_window_step = generate_params['test_window_step']
        filter_name = generate_params['test_filter'] # middle/all
        samples = [] 
        
        if filter_name == 'middle':
            # pad first and last to make sure the same to the public evaluation
            if (annotated_frames[0] - 1) < (test_window_size // 2):
                all_frames = [all_frames[0]] * (test_window_size//2 - (annotated_frames[0] - 1)) + all_frames
            if (len(all_frames) - annotated_frames[-1]) < ((test_window_size - 1) // 2):
                all_frames = all_frames + [all_frames[-1]] * ((test_window_size - 1) // 2 - (len(all_frames) - annotated_frames[-1])) 
            
            
        # 抽取所有可能的windows
        sampled_windows = generate_windows_of_video(all_frames, window_size=test_window_size, window_step=test_window_step, force_all_used=True)
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
                if sample_window[test_window_size//2] in annotated_frames:
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
    generate_traintest_params: dict = configs['generate_traintest_params']
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    train_augmentation: dict = configs['train_augmentation']
    test_augmentation: dict = configs['test_augmentation']
    training_seed: int  = configs['training_seed']
    root = configs['root']
    num_workers= configs['num_workers']
    model_max_stride= configs['model_max_stride']
    trainset_seed= configs['trainset_seed']
    train_batch_size= configs['train_batch_size']
    test_batch_size= configs['test_batch_size']
    
    if amr_are_used:
        # 因为amr的生成是非常耗时间的, 最多换一个right/left
        assert train_augmentation['name'] in ['fixsize', 'justnormalize', 'resize']
        assert test_augmentation['name'] in ['justnormalize', 'fixsize', 'resize']
    
    # dataset part
    generate_train_test_name = generate_traintest_params["name"]
    traintest_samples_file = os.path.join(root, f'{generate_train_test_name}_TrainTest_samples.json')
    if not os.path.exists(traintest_samples_file):
        if process_id == 0:
            generate_train_test_samples(root, generate_traintest_params)
        if is_distributed:
            dist.barrier()
    with open(traintest_samples_file, 'r') as f:
        samples = json.load(f)
        train_samples = samples['train']
        test_samples = samples['test']
    
    text_aux_file = os.path.join(root, f'text_aux_v{text_aux_version}.json')
    if not os.path.exists(text_aux_file):
        generate_text_aux_file(root, text_aux_version)
    with open(text_aux_file, 'r') as f:
        text_aux_by_expid = json.load(f)
    text_annotations_by_videoid = read_text_annotations_groupby_video(root)
    
    create_train_aug = video_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_aug)
    train_dataset = A2DS_TrainSet(root=configs['root'],
                                model_max_stride=configs['model_max_stride'],
                                samples = train_samples,
                                augmentation=train_aug,
                                text_aux_by_expid=text_aux_by_expid,
                                text_aux_version=text_aux_version,
                                text_annotations_by_videoid=text_annotations_by_videoid) 
    
    create_test_aug = video_aug_entrypoints(test_augmentation['name'])
    test_aug = create_test_aug(test_aug)                     
    test_dataset = A2DS_TestSet(root=configs['root'],
                                model_max_stride=configs['model_max_stride'],
                                samples = test_samples,
                                augmentation=test_aug,
                                text_aux_by_expid=text_aux_by_expid,
                                text_aux_version=text_aux_version,
                                text_annotations_by_videoid=text_annotations_by_videoid) 

    # dataloader part
    sampler_train = TrainRandomSampler_ByEpoch_Distributed(train_dataset,
                                    num_replicas=num_processes,
                                    rank=process_id,
                                    seed=configs['trainset_seed'],)
    train_loader = DataLoader(train_dataset,
                            batch_size=configs['batch_size'],
                            sampler=sampler_train,
                            collate_fn=train_dataset.collator, 
                            num_workers=configs['num_workers'],
                            pin_memory=True,
                            persistent_workers=True)
            
    sampler_test = Evaluate_Sampler_Distributed(test_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    test_loader = DataLoader(test_dataset, 
                                batch_size=configs['eval_batch_size'], 
                                sampler=sampler_test,
                                collate_fn=test_dataset.collator,
                                num_workers=configs['num_workers'],
                                pin_memory=True,
                                persistent_workers=True)
       
    def my_dataset_function():
        return [{}]
    from detectron2.data import DatasetCatalog
    DatasetCatalog.register('a2ds_PerWindow_PerExp', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('a2ds_PerWindow_PerExp').thing_classes = ['r', 'nr']
    MetadataCatalog.get('a2ds_PerWindow_PerExp').thing_colors = [(255., 140., 0.), (0., 255., 0.)]


    return train_loader, sampler_train,\
            None, None, \
            test_loader, partial(test, perFrame_eval_coco_file=os.path.join(root, f'perFrame_coco_evaluate.json'))



class A2DS_Dataset(Dataset_WithTextAux):

    # h: 320, 
    # train: w_min: 320, w_max:768, 15741
    # test:  w_min: 390, w_max:640, 3800
    def __init__(self, 
                 root,
                 model_max_stride,
                 split,
                 samples,
                 augmentation, 
                 text_aux_by_expid,
                 text_aux_version,
                 text_annotations_by_videoid) -> None:
        super().__init__()
        self.root = root
        self.text_annotations_groupby_video = read_text_annotations_groupby_video(root)
        self.object_classes = ['adult','baby','ball','bird', 'car', 'cat', 'dog']
        
        self.samples = samples[split]

        self.text_annotations_by_videoid = text_annotations_by_videoid
        self.text_aux_by_expid = text_aux_by_expid
                          
        self.augmentation = augmentation
        self.text_aux_version = text_aux_version

        self.split = split
        if self.text_aux_version == 0:            
            class V0Collator:
                def __init__(self, max_stride, split) -> None:
                    self.split = split
                    self.max_stride = max_stride
                def __call__(self, batch):
                    if self.split == 'train':
                        samples, text_query,  aux,   = list(zip(*batch))
                        # list[t c h w] -> t b c H W
                        samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
                        targets = list(targets)
                        return {
                            'samples': samples, # NT(t b 3 h w)
                            'text_query': list(text_query), # b[str]
                            'auxiliary': {},
                            'targets': targets,
                        }
                    else:
                        samples, text_query,  _, metas  = list(zip(*batch))
                        # list[t c h w] -> t b c H W
                        samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
                        metas = list(metas)
                        return {
                            'samples': samples, # NT(t b 3 h w)
                            'text_query': list(text_query), # b[str]
                            'auxiliary': {},
                            'metas': metas, 
                        }
            self.collator = V0Collator(model_max_stride, split=self.split)
            
        elif self.text_aux_version == 1:
            file_path = os.path.join(root, f"a2ds_{'train' if train else 'test'}_nxGraph.json")
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            class V1Collator:
                def __init__(self, pad_token_id) -> None:
                    self.pad_token_id = pad_token_id
                def __call__(self, batch):
                    # list[T(t 3 hi wi)], list[ [None,..,dict,..None] ], list[str], list[dict]
                    # list[T(2 Ei)], list[T(V_i+E_i)],
                    # list[list[int]] 
                    # list[list[int]]，
                    # list[list[int]]
                    samples, targets, text_query, auxiliary  = list(zip(*batch))
                    samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=32)
                    *_, H, W = samples.tensors.shape
                    # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
                    targets = list(zip(*targets))

                    amrs = [s_dic['amrs'] for s_dic in auxiliary]
                    seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
                    token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
                    token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]
                    return {
                            'samples': samples, # NT(t b 3 h w)
                            'targets': targets, # list[[None...None], [dict...dict]], t
                            'text_query': text_query, # list[str], b,
                            'auxiliary':{
                                'exist_queries': [s_dic['exist_queries'] for s_dic in auxiliary], # list[list[str], N], 
                                'amrs': amrs, 
                                'seg_ids': text_pad_token_ids(list(seg_ids), 0)[0], # b (V+E)max
                                'token_splits': token_splits, # list[list[int]]
                                'token_ids': text_pad_token_ids(token_ids, self.pad_token_id)[0],  # b max
                            }
                        }  
            self.collator = V1Collator(self.tokenizer.pad_token_id)
            
        elif self.text_aux_version == 2:
            file_path = os.path.join(root, f"a2ds_{'train' if train else 'test'}_nxGraph.json")
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            from datasets.propbank_frames import PropBankFrames
            self.pbframes = PropBankFrames('/home/xhh/workspace/rvos_encoder/datasets/propbank-frames/frames')
            self.all_predicates = list(self.pbframes.rolesets_meaning.keys())
            class V2Collator:
                def __call__(self, batch):
                    # list[T(t 3 hi wi)], list[ [None,..,dict,..None] ], list[str], list[dict]
                    # list[T(2 Ei)], list[T(V_i+E_i)],
                    # list[list[int]] 
                    # list[list[int]]，
                    # list[list[int]]
                    samples, targets, text_query, auxiliary  = list(zip(*batch))
                    samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=32)
                    *_, H, W = samples.tensors.shape
                    # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
                    targets = list(zip(*targets))

                    graphs = [s_dic['graphs'] for s_dic in auxiliary]
                    seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
                    token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
                    tokens_ids =  [s_dic['tokens_ids'] for s_dic in auxiliary]
                    return{
                            'samples': samples, # NT(t b 3 h w)
                            'targets': targets, # list[[None...None], [dict...dict]], t
                            'text_query': text_query, # list[str], b,
                            'auxiliary':{
                                'exist_queries': [s_dic['exist_queries'] for s_dic in auxiliary], # list[list[str], N], b
                                'first_noun': [s_dic['first_noun'] for s_dic in auxiliary],
                                'graphs': Batch.from_data_list(list(graphs)), 
                                'seg_ids': pad_token_ids(list(seg_ids), 0)[0], # b (V+E)max
                                'token_splits': list(token_splits), # list[list[int]]
                                'tokens_ids': pad_token_ids(list(tokens_ids), self.pad_token_id)[0],  # b max
                            }
                        }  
            self.collator = V2Collator()
        elif self.text_aux_version == 3:
            file_path = os.path.join(root, f"a2ds_{'train' if train else 'test'}_nxGraph.json")
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            self.prefix = ""
            self.max_src_length = 256
            class V3Collator:
                def __init__(self, tokenizer) -> None:
                    self.tokenizer = tokenizer
                    self.label_pad_token_id = -100
                    
                def pad_model_inputs(self, features):
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.label_pad_token_id,
                        key="labels",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="joint_ids",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="seg_ids",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="srcEtgt_ids",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="srcEtgt_segids",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="Esrctgt_ids",
                    )
                    padding_func(
                        features,
                        padding_side=self.tokenizer.padding_side,
                        pad_token_id=self.tokenizer.pad_token_id,
                        key="Esrctgt_segids",
                    )

                def __call__(self, batch):
                    # list[T(t 3 hi wi)], list[ [None,..,dict,..None] ], list[str], list[dict]
                    # list[T(2 Ei)], list[T(V_i+E_i)],
                    # list[list[int]] 
                    # list[list[int]]，
                    # list[list[int]]
                    samples, targets, text_query, auxiliary  = list(zip(*batch))
                    samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=32)
                    *_, H, W = samples.tensors.shape
                    # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
                    targets = list(zip(*targets))
      
                    model_inputs = [aux['model_inputs'] for aux in auxiliary] # list[dict{'input_ids': [list[int]], ..}]
                    self.pad_model_inputs(model_inputs)
                    model_inputs = self.tokenizer.pad(model_inputs,
                                                    padding=True,
                                                    max_length=None,
                                                    pad_to_multiple_of=None,
                                                    return_tensors='pt')
                    return {
                        'samples': samples, # NT(t b 3 h w)
                        'targets': targets, # list[[None...None], [dict...dict]], t
                        'text_query': text_query, # list[str], b,
                        'auxiliary':{
                            'exist_queries': [s_dic['exist_queries'] for s_dic in auxiliary], # list[list[str], N], b
                            'model_inputs': model_inputs, 
                        }}
            self.collator = V3Collator(tokenizer=self.tokenizer)
        else:
            raise ValueError()
    
        
    def __getitem__(self, item_idx):
        if self.split == 'test':
            video_id, window_frames, exp_id = self.samples[item_idx]['video_id'], self.samples[item_idx]['window'], self.samples[item_idx]['exp_id']
            window_frames = sorted(window_frames)
                            
            exp_dict = self.text_annotations_groupby_video.get_group(video_id).set_index('exp_id').to_dict('index')[exp_id]
            text_query = exp_dict['query']
            text_query = " ".join(text_query.lower().split())
            
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
            return vframes, text_query[0], self.get_text_aux(item_idx), meta 
        
        elif self.split == 'train':
            # start from 1, sorted
            video_id, window_frames, exp_id = self.samples[item_idx]['video_id'], self.samples[item_idx]['window'], self.samples[item_idx]['exp_id']
            window_frames = sorted(window_frames)
            exp_dict = self.text_annotations_groupby_video.get_group(video_id).set_index('exp_id').to_dict('index')[exp_id]
            text_query = exp_dict['query']
            text_query = " ".join(text_query.lower().split())
            referent_obj_id = exp_dict['instance_id']
            
            vframes, _, _ = video_io.read_video(filename=os.path.join(self.root, 'Release/clips320H', f'{video_id}.mp4'), pts_unit='sec', output_format='TCHW')
            H, W = vframes.shape[-2:]
            vframes = vframes[[w-1 for w in window_frames]]
            vframes = [F.to_pil_image(frame) for frame in vframes] # list[PIL Image]
            has_ann = torch.zeros([len(vframes)]).bool()
            annotated_frames = sorted(glob((os.path.join(self.root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
            annotated_frames = [int(f.split('/')[-1].split('.')[0]) for f in annotated_frames] # start from 1

            masks_by_ann_frame = []
            classid_by_ann_frame = []
            appear_instances_by_ann_frame = []
            num_annframe_in_window = 0
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
            assert num_annframe_in_window > 0
            window_appear_objs = torch.cat(appear_instances_by_ann_frame).unique().tolist()
            annotated_exps_by_object = [] # list[list[str],] n
            video_text_annotations = self.text_annotations_groupby_video.get_group(video_id).to_dict('recorsd')
            for obj_id in window_appear_objs:
                annotated_exps_by_object.append([" ".join(t['query'].lower().split()) for t in video_text_annotations if t['instance_id'] == obj_id])
        
            obj_idx_map = {obj:idx for idx, obj in enumerate(window_appear_objs)}
            
            masks = torch.zeros([len(window_appear_objs), num_annframe_in_window, H, W], dtype=torch.bool) # n t h w
            class_ids = torch.empty([len(window_appear_objs)], dtype=torch.int64) # n
            
            for idx, (mask, appear_instances, classid) in enumerate(zip(masks_by_ann_frame, appear_instances_by_ann_frame, classid_by_ann_frame)):
                # n h w
                idx_map = [obj_idx_map[ins] for ins in appear_instances]
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
                
                'orig_size': torch.tensor([len(vframes), H, W]), # T h w
                'size': torch.tensor([len(vframes),  H, W]), # T h w
            }
            
            vframes, appear_texts, targets = self.augmentations(vframes, appear_texts, targets)
            text_query = appear_texts[0]
            appear_texts = appear_texts[1:]
            
            cnt = 0
            for idx, foo in enumerate(annotated_exps_by_object):
                annotated_exps_by_object[idx] = appear_texts[cnt:(cnt+len(foo))]
                cnt += len(foo)
            assert cnt == len(appear_texts)
            
            return vframes, text_query, self.get_text_aux(item_idx), targets

    def __len__(self):
        return len(self.samples)
