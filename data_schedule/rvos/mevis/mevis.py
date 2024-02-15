"""
Ref-YoutubeVOS data loader
"""
import os
from PIL import Image
import json
import numpy as np
import random
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

from utils.misc import is_dist_avail_and_initialized, all_gather, to_device
import torch.distributed as dist


import pycocotools.mask as mask_util
from pycocotools.mask import encode, area
from detectron2.data import DatasetCatalog
import logging
import wandb
import plotly.express as px

from util.box_ops import box_xyxy_to_cxcywh
from torch.utils.data import DataLoader, Dataset
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, generate_windows_of_video,\
    bounding_box_from_mask,DatasetWithAux, CollatorWithAux
from data_schedule.rvos.metric_utils import get_AP_PAT_IOU_PerFrame

__all__ = ['mevis_schedule']


def generate_metas(root, num_frames):
    meta_by_split = {}
    for split in ['train', 'valid', 'valid_u']:
        with open(os.path.join(root, split, 'meta_expressions.json'), 'r') as f:
            video_to_texts = json.load(f)['videos']
        videos = list(video_to_texts.keys())
        print('number of video in the datasets:{}'.format(len(videos)))
        metas = []
        for vid in videos:
            vid_data = video_to_texts[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if split == 'train' or split == 'valid_u':
                if vid_len < 2:
                    continue
                all_objs, all_obj_anns = [], []
                for exp_id, exp_dict in vid_data['expressions'].items():           
                    obj_ids =  [int(x) for x in exp_dict['obj_id']]
                    ann_ids = [str(x) for x in exp_dict['anno_id']] 
                    assert len(obj_ids) == len(ann_ids)
                    for exp_o, exp_a in zip(obj_ids, ann_ids):
                        if exp_o not in all_objs:
                            all_objs.append(exp_o)
                            all_obj_anns.append(exp_a)
                
                for exp_id, exp_dict in vid_data['expressions'].items():
                    for frame_id in range(0, vid_len, num_frames): # 锚点
                        meta = {}
                        meta['video'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['video_objects'] = all_objs
                        meta['video_object_anns'] = all_obj_anns
                        referent_idxs = [all_objs.index(oi) for oi in exp_dict['obj_id']]
                        meta['frames'] = vid_frames
                        meta['exp_id'] = exp_id
                        meta['length'] = vid_len
                        meta['frame_id'] = frame_id
                        meta['referent_idxs'] = referent_idxs
                        metas.append(meta)
            else:
                for exp_id, exp_dict in vid_data['expressions'].items():
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['frames'] = vid_frames
                    meta['exp_id'] = exp_id
                    meta['category'] = 0
                    meta['length'] = vid_len
                    metas.append(meta)
        meta_by_split[split] = metas

    with open(os.path.join(root, f'split_metas_{num_frames}.json'), 'w') as f:
        json.dump(meta_by_split, f)
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
            sort_idx = pred_refer_prob.sigmoid(dim=0) # 大于0.5的合并
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
        # shutil.rmtree(save_dir)  # remove the uncompressed annotations for memory efficiency
    if is_distributed:
        dist.barrier() 
    return {}
   

@register_data_schedule
def mevis_schedule(configs, is_distributed, process_id, num_processes):
    
    root = configs['data_dir']
    root = os.path.join(root, 'mevis')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    test_batch_size= configs['test_batch_size']
    local_range = configs['local_range']
    assert test_batch_size == 1
    add_validate_to_train = configs['add_validate_to_train']
    train_window_size = configs['train_window_size']
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
    
    assert len(glob(os.path.join(root, f'train/JPEGImages', '*'))) == 1662  
    assert len(glob(os.path.join(root, f'valid_u/JPEGImages', '*'))) == 50  # train+valid作为训练
    assert len(glob(os.path.join(root, f'valid/JPEGImages', '*'))) == 140  # 作为test

    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize', 'hflip_ResizeSmaller', "resizeSmaller"]
        assert train_augmentation['name'] in amr_legi_augs
        assert test_augmentation['name'] in amr_legi_augs
    
    if not os.path.exists(os.path.join(root, f'split_metas_{train_window_size}.json')):
        if process_id == 0:
            generate_metas(root, train_window_size)
        if is_distributed:
            dist.barrier()

    with open(os.path.join(root, f'split_metas_{train_window_size}.json'), 'r') as f:
        metas = json.load(f)
        train_metas = metas['train']
        validate_metas = metas['valid_u']
        test_metas = metas['valid']    

    if text_aux_version != 0:
        with open(os.path.join(root, f'text_to_aux.json'), 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if video_aux_version != 0:
        with open(os.path.join(root, f'video_to_aux.json'), 'r') as f:
            video_aux_by_auxid = json.load(f)
    else:
        video_aux_by_auxid = None          

    with open(os.path.join(root, 'train', f'mask_dict.json'), 'r') as f:
        train_video_to_anns = json.load(f)
    with open(os.path.join(root, 'train', f'meta_expressions.json'), 'r') as f:
        train_video_to_texts = json.load(f)['videos']  
    with open(os.path.join(root, 'valid_u', f'mask_dict.json'), 'r') as f:
        valid_video_to_anns = json.load(f)   
    with open(os.path.join(root, 'valid_u', f'meta_expressions.json'), 'r') as f:
        valid_video_to_texts = json.load(f)['videos'] 


    create_train_aug = video_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)       
    train_dataset = Mevis_dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='train',
                                 local_range=local_range,
                                metas = train_metas,
                                train_window_size=train_window_size,
                                augmentation=train_aug,
                                video_to_texts=train_video_to_texts,
                                video_to_anns=train_video_to_anns,

                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                video_aux_version=video_aux_version,
                                video_aux_by_auxid=video_aux_by_auxid)

    if add_validate_to_train:     
        validate_dataset = Mevis_dataset(root=root,
                                    pt_tokenizer_dir=pt_tokenizer_dir,
                                    split='validate',
                                    metas = validate_metas,
                                    local_range=local_range,
                                    train_window_size=train_window_size,
                                    augmentation=train_aug,
                                    video_to_texts=valid_video_to_anns,
                                    video_to_anns=valid_video_to_texts,

                                    text_aux_by_auxid=text_aux_by_auxid,
                                    text_aux_version=text_aux_version,
                                    video_aux_version=video_aux_version,
                                    video_aux_by_auxid=video_aux_by_auxid)   
        from torch.utils.data import ConcatDataset
        collator = train_dataset.collator
        train_dataset = ConcatDataset([train_dataset, validate_dataset])
        train_dataset.collator = collator

    with open(os.path.join(root,'valid', 'meta_expressions.json'), 'r') as f:
        test_video_to_texts = json.load(f)['videos']  

    test_aug = video_aug_entrypoints(test_augmentation['name'])
    test_aug = create_train_aug(test_augmentation) 
    test_dataset = Mevis_dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='test',
                                 train_window_size=None,
                                metas = test_metas,
                                local_range=None,
                                augmentation=test_aug,
                                video_to_texts=test_video_to_texts,
                                video_to_anns=None,

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

def mevis_normalize_text(text_query):
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

from detectron2.structures import Boxes, BoxMode, PolygonMasks


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax # y1, y2, x1, x2

import copy
from detectron2.data import detection_utils as utils
from pycocotools import mask as maskUtils 
class Mevis_dataset(DatasetWithAux):
    def __init__(self, 
                root,
                pt_tokenizer_dir,
                train_window_size,
                split,
                metas,
                local_range,
                augmentation,
                video_to_texts,
                video_to_anns,
                
                text_aux_by_auxid,
                text_aux_version,
                video_aux_version,
                video_aux_by_auxid):

        super().__init__(text_aux_version=text_aux_version,
                         text_aux_by_auxid=text_aux_by_auxid,
                         video_aux_version=video_aux_version,
                         video_aux_by_auxid=video_aux_by_auxid,
                         pt_tokenizer_dir=pt_tokenizer_dir,
                         )
        self.local_range = local_range
        self.root = root
        self.num_frames = train_window_size
        self.metas = metas
        self.text_annotations_by_videoid = video_to_texts
        self.video_to_anns = video_to_anns

        self.augmentation = augmentation
        if split == 'train':
            self.img_root = os.path.join(self.root, 'train', 'JPEGImages')
        elif split == 'validate':
            self.img_root = os.path.join(self.root, 'valid_u', 'JPEGImages')
        elif split == 'test':
            self.img_root = os.path.join(self.root, 'valid', 'JPEGImages')
        self.split = split
        collator_kwargs = {}
        if text_aux_version == 1 or text_aux_version == 2 or text_aux_version == 3 or text_aux_version == 4:
            collator_kwargs['tokenizer'] = self.tokenizer
        self.collator = Collator(split=split,
                                 text_aux_version=text_aux_version,
                                 video_aux_version=video_aux_version,
                                 **collator_kwargs)
        
    def __len__(self):
        return len(self.metas)
    
    def get_annotation_clip_masks(self, ann_id, window_indexs, video_length):
        obj_ann = self.video_to_anns[ann_id] # list[], v_dieo_length
        assert video_length == len(obj_ann)
        obj_win_ann = [obj_ann[si] for si in window_indexs] #list[rle/None]
        for srwa in obj_win_ann:
            if srwa is not None:
                H, W = srwa['size']
                break
        masks = []
        for ann in obj_win_ann:
            if ann is None:
                masks.append(torch.zeros([H, W]).bool()) # 我后面是用valid来不学习你的
            else:
                masks.append(torch.from_numpy(np.array(maskUtils.decode(ann), dtype=np.uint8)).bool())
        return torch.stack(masks, dim=0)

    def __getitem__(self, idx):
        # if self.split == 'train':
        #     instance_check = False
        #     while not instance_check:
        if self.split == 'train' or self.split == 'validate':
            meta = self.metas[idx]  # dict

            video, text_query, all_objs, all_obj_anns, frames, frame_id, referent_idxs = \
                        meta['video'], meta['exp'], meta['video_objects'], meta['video_object_anns'], \
                    meta['frames'], meta['frame_id'], meta['referent_idxs']
            # clean up the caption
            text_query = mevis_normalize_text(text_query)
            vid_len = len(frames)
            assert len(set(all_objs)) == len(all_objs) and (len(set(all_obj_anns)) == len(all_obj_anns))
            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.num_frames != 1:
                # local sample
                sample_id_before = random.randint(1, self.local_range) # 可以控制3
                sample_id_after = random.randint(1, self.local_range)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()
            # load masks of all objects at this window
            object_masks = [self.get_annotation_clip_masks(ann, sample_indx, vid_len) for ann in all_obj_anns]
            object_masks = torch.stack(object_masks, dim=0) # ni t h w
            referent_masks = object_masks[referent_idxs] # n_r t h w
            if not referent_masks.any():
                idx = random.randint(0, self.__len__() - 1)
                return self.__getitem__(idx)

            # 去掉没有出现的物体
            all_appear = object_masks.flatten(1).any(-1) # ni

            appear_objects = torch.tensor(all_objs)[all_appear]
            appear_refs = [all_objs[ref_idx] for ref_idx in referent_idxs if all_appear[ref_idx] == True]
            referent_idxs = [appear_objects.tolist().index(aref) for aref in appear_refs]
            appear_obj_masks = object_masks[all_appear] # ni t h w

            boxes = []
            for object_mask in appear_obj_masks.flatten(0, 1).long():
                if (object_mask > 0).any():
                    y1, y2, x1, x2 = bounding_box_from_mask(object_mask.numpy())
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    boxes.append(box)
                else:
                    box = torch.tensor([0,0,0,0]).to(torch.float)
                    boxes.append(box)
            boxes = torch.stack(boxes, dim=0) # nt 4
            boxes = rearrange(boxes, '(n t) c -> n t c', n=appear_obj_masks.shape[0], t=appear_obj_masks.shape[1])

            vframes = [Image.open(os.path.join(self.img_root,video, frames[sample_indx[j]] + '.jpg')).convert('RGB')\
                        for j in range(self.num_frames)]
            width, height = vframes[0].size
            targets = { 
                'has_ann': torch.ones(len(vframes)).bool(),
                'masks': appear_obj_masks, # n t' h w (bool)
                'boxes': boxes, # n t' 4, xyxy, float
                'referent_idx': referent_idxs, 
                'orig_size': torch.tensor([len(vframes), height, width]), # T h w
                'size': torch.tensor([len(vframes),  height, width]), # T h w
            } # plt.imshow(appear_obj_masks[referent_idxs[0]][1])
            # import matplotlib.pyplot as plt
            # plt.imshow(vframes[0])
            # plt.savefig('./test.png')
            # plt.imshow(appear_obj_masks[referent_idxs[0]][0])
            # plt.savefig('./test.png')
            vframes, text_query, targets = self.augmentation(vframes, [text_query], targets) 
            text_query = text_query[0]

            return vframes, text_query,\
                    self.get_aux(idx, None, video_auxid=None, text_auxid=text_query), targets
        elif self.split == 'test':
            meta = self.metas[idx]  # dict
            video, text_query, frames, exp_id = \
                        meta['video'], meta['exp'], meta['frames'], meta['exp_id']
            # clean up the caption
            text_query = mevis_normalize_text(text_query)
            vid_len = len(frames)
        
            vframes = [Image.open(os.path.join(self.img_root,video, frame + '.jpg')).convert('RGB')\
                        for frame in range(frames)]
            width, height = vframes[0].size
            targets = { 
                'size': torch.tensor([len(vframes),  height, width]),
                'orig_size': torch.tensor([len(vframes),  height, width]),
                'has_ann': torch.ones(len(vframes)).bool(), # T
                'video_id': video,
                'all_frames': frames,
                'exp_id': exp_id,
            }
            vframes, text_query, targets = self.augmentation(vframes, [text_query], targets) 
            text_query = text_query[0]

            return vframes, text_query,\
                    self.get_aux(idx, None, video_auxid=None, text_auxid=text_query), targets
        
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
        return len(self.metas)

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
