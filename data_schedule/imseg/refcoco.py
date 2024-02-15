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

from utils.misc import is_dist_avail_and_initialized, all_gather, to_device
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

from .imgseg_aug_eval import imgseg_evalaug_entrypoints
from .imgseg_aug_train import imgseg_trainaug_entrypoints
from pycocotools.mask import frPyObjects as frPoly
from pycocotools.mask import decode as decode_rle


__all__ = ['refcocog_imgseg_schedule']

def visualize_dataset_information(root):
    pass

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = frPoly(polygons, height, width)
        mask = decode_rle(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def get_img_ids(samples):
    split_images = samples['images']
    split_images = [sim['image_id'] for sim in split_images]
    split_images = np.array(split_images, dtype=np.int64)
    return np.unique(split_images)

@register_data_schedule
def refcocog_imgseg_schedule(configs, is_distributed, process_id, num_processes):
    root = configs['data_dir']
    root = os.path.join(root, 'refer')
    coco_annotation_file = os.path.join(root, 'trainval2014_ann/annotations/instances_train2014.json')
    num_workers= configs['num_workers']
    validate_batch_size= configs['validate_batch_size']
    assert validate_batch_size == 1
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    training_seed: int  = configs['training_seed']
    train_batch_size = configs['train_batch_size']
    validate_metrics = configs['validate_metrics']
    
    # dataset part 
    with open(os.path.join(root, 'refer/refcocog', f'instances_refcocog_train.json'), 'r') as f:
        train_samples = json.load(f)
    categories = train_samples['categories']
    assert len(categories) == 80
    all_category_ids = [foo['id'] for foo in categories]
    catToLabel = {cid:idx for idx, cid in enumerate(all_category_ids)}
    train_samples = get_img_ids(train_samples)

    with open(os.path.join(root, 'refer/refercocog',f'instances_refcocog_val.json'), 'r') as f:
        val1_samples = json.load(f)
    val1_samples = get_img_ids(val1_samples)
    with open(os.path.join(root, 'refer/refercocog',f'instances_refcocog_test.json'), 'r') as f:
        val2_samples = json.load(f)    
    val2_samples = get_img_ids(val2_samples)
    val_length, test_length = len(val1_samples), len(val2_samples)

    val_samples = val1_samples + val2_samples
    split_val_param = {
        'refcocog_val': val_length,
        'refcocog_test': test_length
    }        
    
    create_train_aug = imgseg_trainaug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)
    assert isinstance(train_aug, list)
    train_dataset = REFCOCO(root=root,
                            split='train',
                            catToLabel=catToLabel,
                            coco_annotation_file=coco_annotation_file,
                            samples = train_samples,
                            augmentation=train_aug,) 
    
    create_validate_aug = imgseg_evalaug_entrypoints(validate_augmentation['name'])
    validate_aug = create_validate_aug(validate_augmentation)                     
    validate_dataset = REFCOCO(root=root,
                            split='validate',
                            catToLabel=catToLabel,
                            coco_annotation_file=coco_annotation_file,
                            samples = val_samples,
                            augmentation=validate_aug,) 

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
    DatasetCatalog.register('refcocog', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('refcocog').thing_classes = ['r', 'nr']
    MetadataCatalog.get('refcocog').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

    return train_loader, sampler_train,\
            validate_loader, partial(validate, validate_metrics=validate_metrics, root=root, split_val_param=split_val_param),\
                None, None
from detectron2.data import transforms as T
class REFCOCO(Dataset):
    def __init__(self, 
                 root,
                 catToLabel,
                 coco_annotation_file,
                 split,
                 samples,
                 augmentation,
                 ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        from pycocotools.coco import COCO
        self.coco = COCO(coco_annotation_file)
        self.samples = samples
        self.augmentation = augmentation
        self.catToLabel = catToLabel
        self.collator = Collator(split=split)

    def decode_annotations(self, annotations, H, W):
        # list[dict] -> dict['masks': ni h w, 'boxes': ni 4, 'valid': ni]  
        obj_boxes = []
        # n h w
        obj_masks = convert_coco_poly_to_mask([ann['segmentation'] for ann in annotations], height=H, width=W)
        obj_valids = obj_masks.flatten(1).any(-1) # n
        assert obj_valids.all()
        class_labels = torch.tensor([self.catToLabel[ann['category_id']]  for ann in annotations])
        for mask in obj_masks:
            y1, y2, x1, x2 = bounding_box_from_mask(mask.numpy())
            box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
            obj_boxes.append(box)
        return {
            'masks': obj_masks,
            'boxes': torch.stack(obj_boxes, dim=0),
            'valids': obj_valids,
            'class_labels': class_labels
        }
    
    def __getitem__(self, item_idx):
        # 每个图像成为一个样本
        if self.split == 'train':
            image_id = self.samples[item_idx]
            img_file_name = self.coco.imgs[image_id]['file_name']
            H, W = self.coco.imgs[image_id]['height'], self.coco.imgs[image_id]['width']
            image = F.to_pil_image(os.path.join(self.root, 'refer', f'train2014', f'train2014', img_file_name))
            assert image.size[0] == W and image.size[1] == H
            image, transforms = T.apply_transform_gens(self.augmentation, image)

            all_obj_anns = self.coco.imgToAnns[image_id]
            all_obj_anns = self.decode_annotations(all_obj_anns, H=H, W=W)
            targets = {
                'masks': all_obj_anns['masks'], # n h w (bool)
                'class_labels': all_obj_anns['class_labels'], # n
                'boxes': all_obj_anns['boxes'], # n 4, xyxy, float
                'valid': all_obj_anns['valids'], # n
                'image_ids': image_id,
                'orig_size': torch.tensor([H, W]),
                'size': torch.tensor([H, W]),
            }
            
            
            return self.augmentation(image, targets)
        
        elif self.split == 'validate':
            image_id = self.samples[item_idx]
            img_file_name = self.coco.imgs[image_id]['file_name']
            H, W = self.coco.imgs[image_id]['height'], self.coco.imgs[image_id]['width']
            image = F.to_pil_image(os.path.join(self.root, 'refer', f'train2014', f'train2014', img_file_name))
            assert image.size[0] == W and image.size[1] == H

            all_obj_anns = self.coco.imgToAnns[image_id]
            all_obj_anns = self.decode_annotations(all_obj_anns, H=H, W=W)
            targets = {
                'masks': all_obj_anns['masks'], # n h w (bool)
                'class_labels': all_obj_anns['class_labels'], # n
                'boxes': all_obj_anns['boxes'], # n 4, xyxy, float
                'valid': all_obj_anns['valids'], # n
                'image_ids': image_id,
                'orig_size': torch.tensor([H, W]),
                'size': torch.tensor([H, W]), # h w
            }
            return self.augmentation(image, targets)
        
        else:
            raise ValueError()
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
    
        if self.split == 'validate':
            asembles, meta = list(zip(*batch))
            # list[list[vid, text, meta, callback, aux], num_asembles;], batch
            batch_data = {
                'asembles': list(asembles),
                'meta': list(meta),
            }
        
        elif self.split == 'train':
            samples, targets = list(zip(*batch))

            batch_data = {
                'samples': samples,
                'targets': list(targets)
            }
    
        return batch_data


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
            asembles = to_device(batch_dict['asembles'], device) # list[(video, text, callback)]
            metas = to_device(batch_dict['meta'], device)
            auxiliary = to_device(batch_dict['auxiliary'], device)
            
            # [nq t' h w], [n t'/n], batch;
            model_preds = model.sample(asembles, auxiliary)
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
            for t in metas:
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







  

        

