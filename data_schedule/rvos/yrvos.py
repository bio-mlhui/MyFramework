import json
import os
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import shutil
from functools import partial
from typing import Optional, Union

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
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, generate_windows_of_video,\
    bounding_box_from_mask,DatasetWithAux, CollatorWithAux
from data_schedule.rvos.metric_utils import get_AP_PAT_IOU_PerFrame

__all__ = ['youtube_schedule', 'yrvos_v300s1999_schedule']

def show_dataset_information_and_validate(root):
    
    information = {'num_frame_range': None,
                   'num_videos': [],
                   'num_frame_by_video': [],
                   'video_id_by_video': [],
                   'split_by_video': [],
                   'w_by_video': [], 
                   'h_by_video': [], 
                   'referent_classes': [], # extend
                   'num_objects_by_video': [], # append
                   'color_by_video': []
                   } 
    split_colors = ['#F85F08', '#0858F8']
    splits = ['train', 'valid']
    for split, color in zip(splits, split_colors):
        if split == 'train':
            with open(os.path.join(root, 'meta_expressions', split, 'meta_expressions.json'), 'r') as f:
                video_annotations = json.load(f)['videos']            
            with open(os.path.join(root, split, 'meta.json'), 'r') as f:
                video_objects_annotations = json.load(f)['videos']  # 类别信息是有用的, 还包含object第一次出现之后的帧
                
            video_ids = [key for key in video_annotations.keys()] 
            assert len(set(video_ids)) == len(video_ids)
            for video_id in video_ids:
                assert video_id in video_objects_annotations
            
            for video_id in tqdm(video_ids):
                # 数据集中的每个video都是原video的一个clip, 所以并不一定都是从0开始
                # 每个video都是相隔5帧的多个图片
                video_annotated_frames = sorted(video_annotations[video_id]['frames'])
                frame_dirs = [os.path.join(root, 'train/JPEGImages', video_id, f'{f}.jpg') for f in video_annotated_frames]
                mask_dirs = [os.path.join(root, 'train/Annotations', video_id, f'{f}.png') for f in video_annotated_frames]
                assert torch.tensor([os.path.exists(f) for f in frame_dirs]).all()
                assert torch.tensor([os.path.exists(f) for f in mask_dirs]).all()
                # 该video标注的expressions 
                video_expressions = video_annotations[video_id]['expressions']
                # 该video中出现的所有objects
                video_appear_objects = video_objects_annotations[video_id]['objects']
                assert len(set(video_annotated_frames)) == len(video_annotated_frames)
                assert len(set(list(video_expressions.keys()))) == len(list(video_expressions.keys()))
                assert len(set(list(video_appear_objects.keys()))) == len(list(video_appear_objects.keys()))

                # 2. video的resolution信息
                one_of_frame = torch.tensor(np.array(Image.open(frame_dirs[0])))
                # 3. referent出现的帧数占整个video的比例
                # 4. refeerent的大小分布
                # 5. referent类别的分布
                referent_obj_ids = [exp_dict['obj_id'] for exp, exp_dict in video_expressions.items()] # 1, 1, 2, 2

                information['num_frame_by_video'].append(len(video_annotated_frames))
                information['w_by_video'].append(one_of_frame.shape[1])
                information['h_by_video'].append(one_of_frame.shape[0])
                information['video_id_by_video'].append(video_id)
                information['split_by_video'].append(split)
                information['num_objects_by_video'].append(len(list(video_appear_objects.keys())))
                information['referent_classes'].extend([video_appear_objects[obj_id]['category'] for obj_id in referent_obj_ids])
                information['color_by_video'].append(color)
            information['num_videos'].append(len(video_ids))
            assert len(video_ids) == 3471
        elif split == 'valid':
            with open(os.path.join(root, 'meta_expressions', split, 'meta_expressions.json'), 'r') as f:
                video_annotations = json.load(f)['videos']                            
            video_ids = [key for key in video_annotations.keys()]  # 507
            assert len(set(video_ids)) == len(video_ids)
            with open(os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json'), 'r') as f:
                test_video_annotations = json.load(f)['videos'] 
            test_video_ids = [key for key in test_video_annotations] # 305
            assert len(set(test_video_ids)) == len(test_video_ids)  

            # test \in valid
            assert len(set(test_video_ids) - set(video_ids)) == 0
            for video_id in tqdm(video_ids):
                # 数据集中的每个video都是原video的一个clip, 所以并不一定都是从0开始
                # 每个video都是相隔5帧的多个图片
                video_annotated_frames = sorted(video_annotations[video_id]['frames'])
                # if video_id in test_video_ids:
                #     frame_dirs = [os.path.join(root, 'test/JPEGImages', video_id, f'{f}.jpg') for f in video_annotated_frames]
                # else:
                frame_dirs = [os.path.join(root, 'valid/JPEGImages', video_id, f'{f}.png') for f in video_annotated_frames]
                assert torch.tensor([os.path.exists(f) for f in frame_dirs]).all()
                # 该video标注的expressions 
                video_expressions = video_annotations[video_id]['expressions']
                assert len(set(video_annotated_frames)) == len(video_annotated_frames)
                assert len(set(list(video_expressions.keys()))) == len(list(video_expressions.keys()))
                one_of_frame = torch.tensor(np.array(Image.open(frame_dirs[0])))
                information['num_frame_by_video'].append(len(video_annotated_frames))
                information['w_by_video'].append(one_of_frame.shape[1])
                information['h_by_video'].append(one_of_frame.shape[0])
                information['video_id_by_video'].append(video_id)
                information['split_by_video'].append(split)
                information['color_by_video'].append(color)
            information['num_videos'].append(len(video_ids))
            assert len(video_ids) == 507
    min_nframe = np.array(information['num_frame_by_video']).min()
    max_nframe = np.array(information['num_frame_by_video']).max() 
    information['num_frame_range'] = np.arange(min_nframe, max_nframe+1)  
    wandb.log(data={
        'video resolution':  wandb.Plotly(px.scatter({'x':information['w_by_video'], 'y':information['h_by_video'],
                                                      's': information['split_by_video'],
                                                      'c':information['color_by_video']}, 
                                                     x='x', y='y', symbol='s', color='c')),
        'number of frames': wandb.Plotly(px.bar({'x':information['num_frame_range'], 'y': information['num_frame_by_video'],
                                                  'c':information['color_by_video']}, 
                                                   x='x', y='y', color='c', barmode='group')),
    })

# static method
def test(loader, model, device, is_distributed, is_main_process, output_dir):
    save_dir = os.path.join(output_dir, 'Annotations')
    # 如果存在了annotation 目录，让主进程删除重新Make
    if is_main_process:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    if is_distributed:
        dist.barrier()

    for batch_dict in tqdm(loader):
        samples = to_device(batch_dict['samples'], device)
        text_query = batch_dict['text_query']
        meta_data = to_device(batch_dict['meta'], device)
        auxiliary = to_device(batch_dict['auxiliary'], device)
        assert len(meta_data) == 1
        batch_video_ids = [meta_data[i]['video_id'] for i in range(len(meta_data))]
        batch_frames = [meta_data[i]['all_frames'] for i in range(len(meta_data))]
        batch_exp_id = [meta_data[i]['exp_id'] for i in range(len(meta_data))]
        
        # list[n t' h w], list[n t']
        # ['query_pred_masks', 'query_pred_is_referred_prob',]]
        preds = model.sample(samples=samples, text_queries=text_query, auxiliary=auxiliary, targets=meta_data)
        
        assert len(preds['query_pred_masks']) == len(batch_video_ids)
        for pred_masks, pred_refer_prob, video_id, all_frames, exp_id in zip(preds['query_pred_masks'], 
                                                                            preds['query_pred_is_referred_prob'],
                                                                            batch_video_ids, batch_frames, batch_exp_id):
            # n t' -> t'
            sort_idx = pred_refer_prob.argmax(dim=0) # 每一帧的最牛的query
            # t n h w, t -> list[h w], t -> t h w
            pred_masks = torch.stack([frame_pred[idx] for frame_pred, idx in zip(pred_masks.permute(1, 0, 2, 3), sort_idx)], dim=0)

            dir_path = os.path.join(save_dir, video_id, exp_id)
            os.makedirs(dir_path,exist_ok=True)
            for frame_mask, frame in zip(pred_masks, all_frames):
                frame_mask = frame_mask.to('cpu').float()
                frame_mask = Image.fromarray((255 * frame_mask.numpy())).convert('L')

                # assert not os.path.exists(os.path.join(dir_path, f'{frame}.png'))

                frame_mask.save(os.path.join(dir_path, f'{frame}.png'))
                
    if is_distributed:
        dist.barrier()
    if is_main_process:
        print('creating a zip file with the predictions...')
        zip_file_path = os.path.join(output_dir, f'submission')
        shutil.make_archive(zip_file_path, 'zip', root_dir=output_dir, base_dir='Annotations')
        print('a zip file was successfully created.')
        shutil.rmtree(save_dir)  # remove the uncompressed annotations for memory efficiency
    if is_distributed:
        dist.barrier() 
    return {}

# semi-static method  
def validate(loader, model, device, is_distributed, is_main_process, output_dir, pFpE_coco_file):
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
        eval_metrics = get_AP_PAT_IOU_PerFrame(coco_file, coco_perframe_preds)
    else:
        eval_metrics = None
        
    if is_distributed:
        dist.barrier()
    return eval_metrics


# utils
def generate_train_validate_test_samples(root, generate_params, validate_sample_number=300, validate_sample_seed=1999):
    
    with open(os.path.join(root, 'meta_expressions', 'train', 'meta_expressions.json'), 'r') as f:
        train_video_annotations = json.load(f)['videos']           
    all_train_video_ids = np.array([key for key in train_video_annotations.keys()]) # 3471
    
    if validate_sample_number != 0:
        # 分开一点成validate set
        np.random.seed(validate_sample_seed)
        g = torch.Generator()
        g.manual_seed(validate_sample_seed)
        sampled_validate_idxs = torch.randperm(len(all_train_video_ids), generator=g)[:validate_sample_number]
        train_idxs = np.setdiff1d(np.arange(len(all_train_video_ids)), sampled_validate_idxs.numpy())
        train_video_ids = all_train_video_ids[train_idxs]
        validate_video_ids = all_train_video_ids[sampled_validate_idxs]
    else:
        train_video_ids = all_train_video_ids
        validate_video_ids = []
    
    # 生成validate set的coco annotation file
    if validate_sample_number != 0:
        images_id_set = set()
        perframeperExp_coco_evaluation = []
        images_dict = []
        for video_id in validate_video_ids:
            all_frames = train_video_annotations[video_id]["frames"]
            for frame in all_frames:
                # h w
                frame_annotation = torch.from_numpy(np.array(Image.open(os.path.join(root, 'train/Annotations', video_id, f'{frame}.jpg'))))
                for exp_id, exp_dict in train_video_annotations[video_id]["expressions"].items():
                    referent_obj_id = int(exp_dict['obj_id'])
                    referent_frame_mask = (frame_annotation == referent_obj_id) # h w
                    if referent_frame_mask.any():
                        image_id = f'v_{video_id}_f_{frame}_e_{exp_id}'
                        assert image_id not in images_id_set
                        images_id_set.add(image_id)
                        gt_mask = referent_frame_mask.numpy()
                        images_dict.append({'id': image_id, 'height': gt_mask.shape[0], 'width': gt_mask.shape[1]})
                        mask_rle = encode(gt_mask)
                        mask_rle['counts'] = mask_rle['counts'].decode('ascii')
                        mask_area = float(area(mask_rle))
                
                        bbox = bounding_box_from_mask(gt_mask) # x1y1x2y2 form 
                        assert bbox.ndim == 1 and len(bbox) == 4
                        bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        perframeperExp_coco_evaluation.append({
                            'id': len(perframeperExp_coco_evaluation),
                            'image_id': image_id,
                            'category_id': 1,  
                            'segmentation': mask_rle,
                            'area': mask_area,
                            'bbox': bbox_xywh,
                            'iscrowd': 0,
                        })
        dataset_dict = {
            'categories': [{'id': 1, 'name': 'dummy_class'}],
            'images': images_dict,
            'annotations':  perframeperExp_coco_evaluation,
        }
        with open(os.path.join(root, f'pWpE_{validate_sample_number}-{validate_sample_seed}_ValidatePerFrameCoCo.json'), 'w') as f:
            json.dump(dataset_dict, f)
    else:
        print('不用生成validate set的coco文件, 因为没有validate set')
    with open(os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json'), 'r') as f:
        test_video_annotations = json.load(f)['videos']    
    with open(os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json'), 'r') as f:
        test2_video_annotations = json.load(f)['videos'] 
    assert len(test_video_annotations.keys()) == 507
    assert len(test2_video_annotations.keys()) == 305
    assert len(set(list(test2_video_annotations.keys())) - set(list(test_video_annotations.keys()))) == 0
    test_video_ids = list(set(list(test_video_annotations.keys())) - set(list(test2_video_annotations.keys())))
    # Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    # two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    # currently only be done on the competition 'validation' subset using the competition's server, as
    # annotations were publicly released only for the 'train' subset of the competition.
    # train/test的meta file中object frames是
    # all the frame indices after its first occurrence, for every object in every video.
    # 并且 Please note that small objects (smaller than 100 pixels in 256p resolution) are ignored in meta file
    # 会有60000多个
    assert len(test_video_ids) == 202
    # train set的samples      
    params_by_vid = [(root, train_video_annotations[video_id], video_id, 'train', generate_params) for video_id in train_video_ids]
    n_jobs = min(multiprocessing.cpu_count(), 12)
    train_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
    train_samples = [s for l in train_samples for s in l]   
    
    print(f'there are {len(train_samples)} training samples')

    # test set的samples
    params_by_vid = [(root, test_video_annotations[video_id], video_id, 'test', generate_params) for video_id in test_video_ids]
    n_jobs = min(multiprocessing.cpu_count(), 12)
    test_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
    test_samples = [s for l in test_samples for s in l]   
    print(f'there are {len(test_samples)} test samples')   

    if validate_sample_number != 0:
        # validate set的samples
        params_by_vid = [(root, train_video_annotations[video_id], video_id, 'validate', generate_params) for video_id in validate_video_ids]
        n_jobs = min(multiprocessing.cpu_count(), 12)
        validate_samples = Parallel(n_jobs)(delayed(generate_samples_of_one_video)(*p) for p in tqdm(params_by_vid))
        validate_samples = [s for l in validate_samples for s in l]   
        print(f'there are {len(validate_samples)} validate samples')
    else:
        validate_samples = []

    with open(os.path.join(root, f'{generate_params["name"]}_TrainValidateTest.json'), 'w') as f:
        json.dump({'train': train_samples, 'validate': validate_samples, 'test': test_samples}, f)
    
def generate_samples_of_one_video(root, video_annotations, video_id, split, generate_params):
    samples = []
    video_expressions = video_annotations['expressions'] # exp_id:{exp, obj_id} / exp_id:{exp}
    all_frames = sorted(video_annotations['frames'])
    if split == 'train':
        train_window_size = generate_params['train_window_size']
        train_window_step = generate_params['train_window_step']
        sampled_windows = generate_windows_of_video(all_frames, window_size=train_window_size,window_step=train_window_step,
                                                    force_all_used=True)
        # pad window size
        last_window_len = len(sampled_windows[-1])
        if last_window_len < train_window_size:
            if len(all_frames) > train_window_size:
                sampled_windows[-1] = all_frames[-train_window_size:]
            else:
                delta = train_window_size - last_window_len
                sampled_windows[-1] = sampled_windows[-1] + [all_frames[-1]] * delta
        for wd in sampled_windows:
            assert len(wd) == train_window_size   
                            
        for window in sampled_windows:
            # 这个window中出现的所有objects
            window_annotation = [os.path.join(root, 'train/Annotations', video_id, f'{idx}.png') for idx in window]
            mask_annotations = [torch.tensor(np.array(Image.open(p))) for p in window_annotation]  # list[h w, uint8], window_size
            mask_annotations = torch.stack(mask_annotations, dim=0) # t h w
            appear_objects = mask_annotations.unique().tolist()
            # appear_objects = set().union(*[m.unique().tolist() for m in mask_annotations])
            for exp_id, exp_dict in video_expressions.items():
                referent_id = exp_dict['obj_id']
                if int(referent_id) not in appear_objects:  #这个window中没出现exp refer 的object
                    continue
                samples.append({
                    'video_id': video_id,
                    'window': window,
                    'exp_id': exp_id,
                })
        return samples
    elif split == 'test':
        test_window_size = generate_params['test_window_size']
        test_window_step = generate_params['test_window_step']
        sampled_windows = generate_windows_of_video(all_frames, window_size=test_window_size,window_step=test_window_step,
                                                    force_not_interleave=True, force_all_used=True)
        # 最后一个window的大小可能小于test window size
        for window in sampled_windows:
            for exp_id, exp_dict in video_expressions.items():
                samples.append({
                    'video_id': video_id,
                    'window': window,
                    'exp_id': exp_id,
                })
        return samples
        
    elif split == 'validate':
        validate_window_size = generate_params['validate_window_size']
        validate_window_step = generate_params['validate_window_step']
        sampled_windows = generate_windows_of_video(all_frames, window_size=validate_window_size, window_step=validate_window_step,
                                                        force_not_interleave=True, force_all_used=True)
        for window in sampled_windows:
            # 这个window中出现的所有objects
            window_annotation = [os.path.join(root, 'train/Annotations', video_id, f'{idx}.png') for idx in window]
            mask_annotations = [torch.tensor(np.array(Image.open(p))) for p in window_annotation]  # list[h w, uint8], window_size
            mask_annotations = torch.stack(mask_annotations, dim=0) # t h w
            appear_objects = mask_annotations.unique().tolist()
            # appear_objects = set().union(*[m.unique().tolist() for m in mask_annotations])
            for exp_id, exp_dict in video_expressions.items():
                referent_id = exp_dict['obj_id']
                if int(referent_id) not in appear_objects:  #这个window中没出现exp refer 的object
                    continue
                samples.append({
                    'video_id': video_id,
                    'window': window,
                    'exp_id': exp_id,
                })
        return samples

@register_data_schedule
def yrvos_v300s1999_schedule(configs, is_distributed, process_id, num_processes):
    yrvos_schedule_id='v300s1999'
    generate_trainvalidatetest_params: dict = configs['generate_trainvalidatetest_params']
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    test_augmentation: dict = configs['test_augmentation']
    training_seed: int  = configs['training_seed']
    root = configs['root']
    num_workers= configs['num_workers']
    train_batch_size= configs['train_batch_size']
    validate_batch_size = configs['validate_batch_size']
    test_batch_size= configs['test_batch_size']
    
    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize']
        assert train_augmentation['name'] in amr_legi_augs
        assert test_augmentation['name'] in amr_legi_augs
        assert validate_augmentation['name'] in amr_legi_augs
    
    root = configs['root']
    traintest_samples_file = os.path.join(root, f'v300s1999_{generate_trainvalidatetest_params["name"]}_TrainValidateTest_samples.json')
    if not os.path.exists(traintest_samples_file):
        if process_id == 0:
            generate_train_validate_test_samples(root, generate_trainvalidatetest_params, 
                                                 validate_sample_number=300, validate_sample_seed=1999)
        if is_distributed:
            dist.barrier()
    with open(traintest_samples_file, 'r') as f:
        samples = json.load(f)
        train_samples = samples['train']
        test_samples = samples['test']
        validate_samples = samples['validate']

    with open(os.path.join(root, 'meta_expression', 'train', 'meta_expressions.json'), 'r') as f:
        trainvalidate_textann_by_videoid = json.load(f)["videos"]
        
    with open(os.path.join(root, 'meta_expression', 'valid', 'meta_expressions.json'), 'r') as f:
        test_textann_by_videoid = json.load(f)["videos"]
                
    with open(os.path.join(root, 'train', 'meta.json'), 'r') as f:
        trainvalidate_objann_by_videoid = json.load(f)["videos"]
        

    create_train_aug = video_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)       
    train_dataset = YRVOS_Dataset(root=root, 
                                       split='train',
                                       samples=train_samples,
                                       augmentation=train_aug,
                                       text_annotations_by_videoid=trainvalidate_textann_by_videoid,
                                       text_aux_version = text_aux_version,
                                       obj_annotations_groupby_videoid=trainvalidate_objann_by_videoid,)
    
    validate_aug = video_aug_entrypoints(validate_augmentation['name'])
    validate_aug = create_train_aug(validate_augmentation) 
    validate_dataset = YRVOS_Dataset(root=root, 
                                       split='validate',
                                       samples=validate_samples,
                                       augmentation=validate_aug,
                                       text_annotations_by_videoid=trainvalidate_textann_by_videoid,
                                       text_aux_version = text_aux_version,
                                       obj_annotations_groupby_videoid=trainvalidate_objann_by_videoid)
    
    test_aug = video_aug_entrypoints(test_augmentation['name'])
    test_aug = create_train_aug(test_augmentation) 
    test_dataset = YRVOS_Dataset(root=root, 
                                       split='test',
                                       samples=test_samples,
                                       augmentation=test_aug,
                                       text_annotations_by_videoid=test_textann_by_videoid,
                                       text_aux_version = text_aux_version,
                                       obj_annotations_groupby_videoid=None)

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
                                collate_fn=test_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
            
    sampler_test = Evaluate_Sampler_Distributed(test_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    test_loader = DataLoader(test_dataset, 
                                batch_size=test_batch_size, 
                                sampler=sampler_test,
                                collate_fn=test_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
    def my_dataset_function():
        return [{}]
    DatasetCatalog.register('youtube_rvos', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('youtube_rvos').thing_classes = ['r', 'nr']
    MetadataCatalog.get('youtube_rvos').thing_colors = [(255., 140., 0.), (0., 255., 0.)]
    
    pFpE_validate_coco_file = os.path.join(root, f'{yrvos_schedule_id}_pFpE_validate_coco.json')
    return train_loader, sampler_train, \
            validate_loader, partial(validate, pFpE_coco_file=pFpE_validate_coco_file),\
            test_loader, test

import logging
@register_data_schedule
def youtube_schedule(configs, is_distributed, process_id, num_processes):
    all_categories = {'truck': 1, 'leopard': 2, 'knife': 3, 'sedan': 4, 'parrot': 5, 'snail': 6, 'snowboard': 7, 'sign': 8, 'rabbit': 9, 'bus': 10, 'crocodile': 11, 'penguin': 12, 'skateboard': 13, 'eagle': 14, 'dolphin': 15, 'lion': 16, 'others': 17, 'surfboard': 18, 'earless_seal': 19, 'cow': 20, 'hat': 21, 'lizard': 22, 'duck': 23, 'dog': 24, 'hedgehog': 25, 'zebra': 26, 'bird': 27, 'turtle': 28, 'hand': 29, 'elephant': 30, 'motorbike': 31, 'bike': 32, 'bear': 33, 'tiger': 34, 'ape': 35, 'fish': 36, 'deer': 37, 'frisbee': 38, 'snake': 39, 'horse': 40, 'bucket': 41, 'train': 42, 'owl': 43, 'parachute': 44, 'monkey': 45, 'airplane': 46, 'person': 47, 'paddle': 48, 'plant': 49, 'mouse': 50, 'camel': 51, 'shark': 52, 'raccoon': 53, 'squirrel': 54, 'giraffe': 55, 'boat': 56, 'sheep': 57, 'giant_panda': 58, 'whale': 59, 'tennis_racket': 60, 'toilet': 61, 'umbrella': 62, 'frog': 63, 'cat': 64, 'fox': 65}
    category_name_to_id = {}
    cnt = 0
    for key in all_categories.keys():
        category_name_to_id[key] = cnt
        cnt += 1

    root = configs['data_dir']
    root = os.path.join(root, 'youtube_rvos')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    test_batch_size= configs['test_batch_size']
    assert test_batch_size == 1
    # 如果每个sample的视频有大量的交集，那么模型可以通过记住一个sample的结果来分割另一个sample
    generate_traintest_params: dict = configs['generate_traintest_params']
    # 训练时额外的数据 
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    video_aux_version: int = configs['video_aux_version']
    # 训练数据增强, 测试数据增强
    train_augmentation: dict = configs['train_augmentation']
    test_augmentation: dict = configs['test_augmentation']
    # 训练时的SGD的loading
    training_seed: int  = configs['training_seed']
    train_batch_size= configs['train_batch_size']
    
    assert len(glob(os.path.join(root, f'train/JPEGImages', '*'))) == 3471 
    assert len(glob(os.path.join(root, f'valid/JPEGImages', '*'))) == 202 
    assert len(glob(os.path.join(root, f'test/JPEGImages', '*'))) == 305
    assert len(glob(os.path.join(root, f'train/Annotations', '*'))) == 3471

    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize', 'hflip_ResizeSmaller', "resizeSmaller"]
        assert train_augmentation['name'] in amr_legi_augs
        assert test_augmentation['name'] in amr_legi_augs
    
    traintest_samples_file = os.path.join(root, f'{generate_traintest_params["name"]}_TrainValidateTest.json')
    if not os.path.exists(traintest_samples_file):
        if process_id == 0:
            generate_train_validate_test_samples(root, generate_traintest_params, validate_sample_number=0, validate_sample_seed=None)
        if is_distributed:
            dist.barrier()
    with open(traintest_samples_file, 'r') as f:
        samples = json.load(f)
        train_samples = samples['train']
        test_samples = samples['test']
    logging.info(f'there are {len(train_samples)} training samples')
    print(f'there are {len(train_samples)} training samples')
    logging.info(f'there are {len(test_samples)} test samples')
    print(f'there are {len(test_samples)} test samples')
    if text_aux_version != 0:
        with open(os.path.join(root, 'meta_expressions', f'text_to_aux.json'), 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if video_aux_version != 0:
        with open(os.path.join(root, f'video_to_aux.json'), 'r') as f:
            video_aux_by_auxid = json.load(f)
    else:
        video_aux_by_auxid = None          

    with open(os.path.join(root, 'train', f'meta.json'), 'r') as f:
        train_video_to_objs = json.load(f)['videos']
    with open(os.path.join(root, 'meta_expressions', 'train', f'meta_expressions.json'), 'r') as f:
        train_video_to_texts = json.load(f)['videos']   

    create_train_aug = video_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)       
    train_dataset = YRVOS_Dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='train',
                                samples = train_samples,
                                augmentation=train_aug,
                                video_to_texts=train_video_to_texts,
                                video_to_objects=train_video_to_objs,
                                catname_to_id=category_name_to_id,

                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                video_aux_version=video_aux_version,
                                video_aux_by_auxid=video_aux_by_auxid)

    with open(os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json'), 'r') as f:
        test_video_annotations = json.load(f)['videos']    
    with open(os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json'), 'r') as f:
        test2_video_annotations = json.load(f)['videos'] 
    assert len(test_video_annotations.keys()) == 507
    assert len(test2_video_annotations.keys()) == 305
    assert len(set(list(test2_video_annotations.keys())) - set(list(test_video_annotations.keys()))) == 0
    test_video_ids = list(set(list(test_video_annotations.keys())) - set(list(test2_video_annotations.keys())))
    test_video_to_texts = {key:test_video_annotations[key] for key in test_video_ids}
    test_aug = video_aug_entrypoints(test_augmentation['name'])
    test_aug = create_train_aug(test_augmentation) 
    test_dataset = YRVOS_Dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='test',
                                samples = test_samples,
                                augmentation=test_aug,
                                video_to_texts=test_video_to_texts,
                                video_to_objects=None,
                                catname_to_id=category_name_to_id,
                                
                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                video_aux_version=video_aux_version,
                                video_aux_by_auxid=video_aux_by_auxid) 

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

    sampler_test = Evaluate_Sampler_Distributed(test_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    test_loader = DataLoader(test_dataset, 
                                batch_size=test_batch_size, 
                                sampler=sampler_test,
                                collate_fn=test_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
    def my_dataset_function():
        return [{}]
    DatasetCatalog.register('youtube_rvos', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('youtube_rvos').thing_classes = ['r', 'nr']
    MetadataCatalog.get('youtube_rvos').thing_colors = [(255., 140., 0.), (0., 255., 0.)]
    
    return train_loader, sampler_train, None, None, test_loader, partial(test),

def yrvos_normalize_text(text_query):
    if text_query == 'cannot describe too little':
        text_query = 'an airplane not moving'
    elif text_query == 'a red clothe':
        text_query = 'a red cloth'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1


# 训练集是perWindow_perExp, 即(clip, text)
# test set是(video, text) 对

class YRVOS_Dataset(DatasetWithAux):      
    def __init__(self, 
                 root,
                 split,
                 samples,
                 augmentation,
                 video_to_texts,
                 video_to_objects,
                catname_to_id,
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
        self.split = split
        self.catname_to_id = catname_to_id
        self.samples = samples
        self.video_to_texts = video_to_texts
        self.video_to_objects = video_to_objects
        self.augmentation = augmentation 
        if self.split == 'test':
            self.video_root = os.path.join(self.root, f'valid/JPEGImages')
        elif self.split in ['train', 'validate']:
            self.video_root = os.path.join(self.root, f'train/JPEGImages')
            self.mask_ann_root = os.path.join(self.root, f'train/Annotations')
        else:
            raise ValueError()
        collator_kwargs = {}
        if text_aux_version == 1 or text_aux_version == 2 or text_aux_version == 3 or text_aux_version == 4:
            collator_kwargs['tokenizer'] = self.tokenizer

        self.collator = Collator(split=split,
                                 text_aux_version=text_aux_version,
                                 video_aux_version=video_aux_version,
                                 **collator_kwargs)

    def __getitem__(self, sample_idx):
        if self.split == 'train' or self.split == 'validate':
            video_id, window_frames, exp_id = self.samples[sample_idx]['video_id'], \
                                            self.samples[sample_idx]['window'], self.samples[sample_idx]['exp_id']
            all_exps_dict = self.video_to_texts[video_id]['expressions'] # exp_id : {exp, obj_id}
            all_objs_dict = self.video_to_objects[video_id]['objects'] # obj_id : {category:name, frames,}
            text_query = all_exps_dict[exp_id]['exp']
            text_query = yrvos_normalize_text(text_query)

            vframes = [Image.open(os.path.join(self.video_root, video_id, f'{f}.jpg'),) for f in window_frames]
            width, height = vframes[0].size
            # t h w
            all_objects_masks = torch.stack([torch.from_numpy(np.array(Image.open(os.path.join(self.mask_ann_root, video_id, f'{f}.png')))) for f in window_frames], dim=0) #
            appear_obj_ids = set(all_objects_masks.unique().tolist())
            if 0 in appear_obj_ids:
                appear_obj_ids = list(appear_obj_ids - set([0]))
            appear_obj_ids = sorted(appear_obj_ids) # 1，2，3, 5
            annotated_exps_by_object = []
            masks_by_object = [] 
            obj_classes_by_object = []
            for obj_id in appear_obj_ids:
                masks_by_object.append(all_objects_masks == obj_id) # t h w, uint8
                obj_classes_by_object.append(self.catname_to_id[all_objs_dict[str(obj_id)]["category"]])
                obj_exps = [value['exp'] for key, value in all_exps_dict.items() if int(value['obj_id']) == obj_id]
                if len(obj_exps) == 0:
                    print('there are some objects that in the video has no expressions')
                annotated_exps_by_object.append(obj_exps)
            masks = torch.stack(masks_by_object, dim=0) # n t h w, bool
            class_labels = torch.tensor(obj_classes_by_object).long() # n
            referent_idx = appear_obj_ids.index(int(all_exps_dict[exp_id]['obj_id'])) # 在1, 2, 3, 5中的下标    
            targets = {
                'has_ann': torch.ones(len(vframes)).bool(), # t
                'masks': masks, # n t h w (bool) 
                'class_labels': class_labels, # n
                'referent_idx': referent_idx,
                'orig_size': torch.tensor([len(vframes), height, width]), # T h w
                'size': torch.tensor([len(vframes),  height, width]), # T h w
            }
            
            flatten_texts = [text_query]
            for att in annotated_exps_by_object:
                flatten_texts.extend(att)
            vframes, flatten_texts, targets = self.augmentation(vframes, flatten_texts, targets)
            text_query = flatten_texts[0]
            cnt = 1
            for idx in range(len(annotated_exps_by_object)):
                num_texts = len(annotated_exps_by_object[idx])
                annotated_exps_by_object[idx] = flatten_texts[cnt:(cnt+num_texts)]
                cnt += num_texts
            assert (cnt - 1) == sum([len(ttt) for ttt in annotated_exps_by_object]) 
                
            return vframes, text_query,\
                  self.get_aux(sample_idx, annotated_exps_by_object, video_auxid=None, text_auxid=text_query), targets
            
        elif self.split == 'test': 
            video_id, window_frames, exp_id = self.samples[sample_idx]['video_id'], \
                                            self.samples[sample_idx]['window'], self.samples[sample_idx]['exp_id']          
            all_exps_dict = self.video_to_texts[video_id]["expressions"] # exp_id: exp
            text_query = all_exps_dict[exp_id]['exp']
            text_query = yrvos_normalize_text(text_query)
            vframes = [Image.open(os.path.join(self.video_root, video_id, f'{f}.jpg')) for f in window_frames]
            width, height = vframes[0].size
            meta_data = {
                'size': torch.tensor([len(vframes),  height, width]),
                'orig_size': torch.tensor([len(vframes),  height, width]),
                'has_ann': torch.ones(len(vframes)).bool(), # T
                'video_id': video_id,
                'all_frames': window_frames,
                'exp_id': exp_id,
            }
            vframes, text_query, meta_data = self.augmentation(vframes, [text_query], meta_data)  # T(t 3 h w), list[none]
            text_query = text_query[0]
            return vframes, text_query, \
                self.get_aux(sample_idx, None, video_auxid=None, text_auxid=text_query), meta_data

    def get_aux(self, item_idx, exist_queries,
                video_auxid,
                text_auxid):
        aux = {}
        aux['sample_idx'] = item_idx
        aux['exist_queries'] = exist_queries
        aux.update(self.get_text_aux(text_auxid, exist_queries))
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
        else:
            batch_data['targets'] = meta_or_target
    
        return batch_data