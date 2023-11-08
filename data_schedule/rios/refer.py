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
from .augmentation_rios import image_aug_entrypoints

from util.misc import is_dist_avail_and_initialized, all_gather, to_device
import torch.distributed as dist


import pycocotools.mask as mask_util
from pycocotools.mask import encode, area
from detectron2.data import DatasetCatalog

import wandb
import plotly.express as px

from util.box_ops import box_xyxy_to_cxcywh
from torch.utils.data import DataLoader, Dataset
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, bounding_box_from_mask, generate_windows_of_video
from .metric_utils import get_AP_PAT_IOU_PerFrame
from .utils import CollatorWithAux, DatasetWithAux

import logging

__all__ = ['refcocog_schedule']
from pycocotools.mask import frPyObjects as frPoly
from pycocotools.mask import decode as decode_rle

def generate_coco_eval_file(tgt_file_name, samples):
    raise ValueError()
    samples_iamges = samples['images']
    samples_anns = samples['annotations']
    assert len(samples_iamges) == len(samples_anns)
    images_id_set = set()
    coco_evaluation = []
    images_dict = []
    for img, ann in zip(samples_iamges, samples_anns):
        test_image_id = ann['image_id']
        assert test_image_id not in images_id_set
        images_id_set.add(test_image_id)
        image_height, image_width = img['height'], img['width']
        images_dict.append({'id': test_image_id, 'height': image_height, 'width': image_width})
        coco_evaluation.append({
            'id': len(coco_evaluation),
            'image_id': ann['image_id'],
            'category_id': 1,  
            'segmentation': ann['segmentation'],
            'area': ann['area'],
            'iscrowd': ann['iscrowd'],
        })
    logging.info(f'there are a total {len(coco_evaluation)} (image, exp_id)')   
    print(f'there are a total {len(coco_evaluation)} (image, exp_id)')
    dataset_dict = {
        'categories': [{'id': 1, 'name': 'dummy_class'}],
        'images': images_dict,
        'annotations':  coco_evaluation,
    }
    with open(tgt_file_name, 'w') as f:
        json.dump(dataset_dict, f)


def visualize_dataset_information(root):
    pass

# 和世界联系
# 只要模型输出每个annotated frame的预测每个一个就行
# 和generate params没关系
def validate(loader, model, device, is_distributed, is_main_process, output_dir, validate_metrics, root):
    eval_metrics = {}
    for set_name, set_foo in loader.items():
        logging.info(f'evaluating {set_name}')
        print(f'evaluating {set_name}')
        loader = set_foo['loader']
        coco_eval_file = set_foo['coco_eval_file']
        assert os.path.exists(coco_eval_file)
        coco_preds = []
        for idx, batch_dict in tqdm(enumerate(loader)):
            samples = to_device(batch_dict['samples'], device) # list[t 3 h w]
            targets = to_device(batch_dict['targets'], device)
            text_queries = batch_dict['text_query']
            auxiliary = to_device(batch_dict['auxiliary'], device)
            # [nq h w], [nq], batch;
            model_preds = model.sample(samples, text_queries, auxiliary, targets,)
            model_mask_preds = model_preds['query_pred_masks'] # [nq h w], batch
            model_prob_preds = model_preds['query_pred_is_referred_prob'] # [nq]
            query_rle_masks = [[mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in sample_mask_pred.cpu()] for sample_mask_pred in model_mask_preds]
            
            # [n, n h w, list[rle]], batch
            model_outputs = [{'scores': s, 'masks': m, 'rle_masks': rle}
                            for s, m, rle in zip(model_prob_preds, model_mask_preds, query_rle_masks)]
            image_ids = [t['image_id'] for t in targets]
            for p, image_id in zip(model_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    coco_preds.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})   
        if is_distributed:
            coco_preds = all_gather(coco_preds)
            coco_preds = [p for p_list in coco_preds for p in p_list]

        if is_main_process:
            ret_metrics = get_AP_PAT_IOU_PerFrame(coco_eval_file, coco_preds, not_compute_pat=('not compute_pat' in validate_metrics))
            add_set_name_metrics = {f'{set_name}_{key}':val for key, val in ret_metrics.items()}
            eval_metrics.update(add_set_name_metrics)
        else:
            eval_metrics = {}
                
        if is_distributed:
            dist.barrier()
    return eval_metrics

def generate_text_aux_file(root, version):
    pass

def refcoco_normalize_text(text_query ,set_name, sent_id):
    if set_name == 'refcoco+':
        if str(sent_id) == '126910':
            assert text_query == 'sis'
            text_query = 'sister'
    if set_name == 'refcocog':
        if str(sent_id) == '29232':
            assert text_query == '{}'
            text_query = 'man wearing black pants'
        if str(sent_id) == '268':
            assert text_query == '{}'
            text_query = 'woman surfing above the water'
    # if text_query == 'main i white shirt'

    # 非法输入
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

def convert_poly_to_mask_for_single_obj_torchBool(poly_seg, height, width):
    rles = frPoly(poly_seg, height, width)
    mask = decode_rle(rles) # uint8, h, w, areai
    if len(mask.shape) < 3:
        mask = mask[..., None]
    mask = torch.as_tensor(mask, dtype=torch.uint8)
    mask = mask.any(dim=2).bool() # 比如一个poly有2个分开的区域, 将两个mask合并
    return mask

# 25799张图像
# 49822个referent: 42226/2573/5023
# 95020个句子, 80512/4896/9602
@register_data_schedule
def refcocog_schedule(configs, is_distributed, process_id, num_processes):
    root = configs['data_dir']
    imgs_dir = os.path.join(root, 'refer/train2014/train2014')
    # cocotrain14_ann_file = os.path.join(root, 'refer/trainval2014_ann/annotations/instances_train2014.json')
    # for img_id in self.imgToAnns.keys():
    #     assert len(self.imgToAnns[img_id]) == len(coco.imgToAnns[int(img_id)])
    root = os.path.join(root, 'refer/refcocog')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    validate_batch_size= configs['validate_batch_size']
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    image_aux_version: int = configs['image_aux_version']
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    training_seed: int  = configs['training_seed']
    train_batch_size= configs['train_batch_size']
    validate_metrics = configs['validate_metrics']

    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize', 'hflip_ResizeSmaller', "resizeSmaller"]
        assert train_augmentation['name'] in amr_legi_augs
        assert validate_augmentation['name'] in amr_legi_augs
   
    splits = ['train', 'test', 'val']
    split_to_samples = {}
    cat_to_ids = None
    for split in splits:
        with open(os.path.join(root, f'instances_refcocog_{split}.json'), 'r') as f:
            split_samples = json.load(f)
        split_num_samples = len(split_samples['images']) # 80512,
        assert split_num_samples == len(split_samples['annotations'])
        if process_id == 0:
            logging.info(f'Number of {split} samples: {split_num_samples}')
            print(f'Number of {split} samples: {split_num_samples}')
        if cat_to_ids is None:
            categories = split_samples['categories']
            all_category_ids = [foo['id'] for foo in categories]
            assert len(all_category_ids) == 80
            cat_to_ids = {cid:idx for idx, cid in enumerate(all_category_ids)}
        split_to_samples[split] = split_samples
        # if split == 'test' or split == 'val':
        #     tgt_file_name = os.path.join(root, f'{split}_coco_eval.json')
        #     if not os.path.exists(tgt_file_name):
        #         generate_coco_eval_file(tgt_file_name, split_samples)

    with open(os.path.join(root, f'global_mappings.json'), 'r') as f:
        globals = json.load(f) 
    imgToAnns = globals['imgToAnns']
    imgToRefs = globals['imgToRefs']
    if text_aux_version != 0:
        text_aux_file = os.path.join(root, f'text_to_aux.json')
        # you need to generate outside the main module
        with open(text_aux_file, 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if image_aux_version != 0:
        pass
    else:
        image_aux_by_auxid = None        
    
    create_train_aug = image_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)
    train_dataset = REFCOCO(imgs_dir=imgs_dir,
                            pt_tokenizer_dir=pt_tokenizer_dir,
                            split='train',
                            set_name='refcocog',
                            imgToRefs=imgToRefs,
                            catToId=cat_to_ids,
                            imgToAnns=imgToAnns,
                            samples = split_to_samples['train'],
                            augmentation=train_aug,
                            text_aux_by_auxid=text_aux_by_auxid,
                            text_aux_version=text_aux_version,
                            image_aux_version=image_aux_version,
                            image_aux_by_auxid=image_aux_by_auxid) 
    
    create_validate_aug = image_aug_entrypoints(validate_augmentation['name'])
    validate_aug = create_validate_aug(validate_augmentation)                     
    validate_val_dataset = REFCOCO(imgs_dir=imgs_dir,
                            pt_tokenizer_dir=pt_tokenizer_dir,
                            split='validate',
                            set_name='refcocog',
                            catToId=cat_to_ids,
                            imgToRefs=imgToRefs,
                            imgToAnns=imgToAnns,
                            samples = split_to_samples['val'],
                            augmentation=validate_aug,
                            text_aux_by_auxid=text_aux_by_auxid,
                            text_aux_version=text_aux_version,
                            image_aux_version=image_aux_version,
                            image_aux_by_auxid=image_aux_by_auxid) 

    validate_test_dataset = REFCOCO(imgs_dir=imgs_dir,
                            pt_tokenizer_dir=pt_tokenizer_dir,
                            split='validate',
                            set_name='refcocog',
                            catToId=cat_to_ids,
                            imgToRefs=imgToRefs,
                            imgToAnns=imgToAnns,
                            samples = split_to_samples['test'],
                            augmentation=validate_aug,
                            text_aux_by_auxid=text_aux_by_auxid,
                            text_aux_version=text_aux_version,
                            image_aux_version=image_aux_version,
                            image_aux_by_auxid=image_aux_by_auxid) 
    
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
            
    sampler_validate_val = Evaluate_Sampler_Distributed(validate_val_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    validate_val_loader = DataLoader(validate_val_dataset, 
                                batch_size=validate_batch_size, 
                                sampler=sampler_validate_val,
                                collate_fn=validate_val_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
    sampler_validate_test = Evaluate_Sampler_Distributed(validate_test_dataset, 
                                    num_replicas=num_processes, 
                                    rank=process_id,)
    validate_test_loader = DataLoader(validate_test_dataset, 
                                batch_size=validate_batch_size, 
                                sampler=sampler_validate_test,
                                collate_fn=validate_test_dataset.collator,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)
       
    def my_dataset_function():
        return [{}]
    from detectron2.data import DatasetCatalog
    DatasetCatalog.register('refcocog_rios', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('refcocog_rios').thing_classes = ['r', 'nr']
    MetadataCatalog.get('refcocog_rios').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

        
    return train_loader, sampler_train,\
            {'refcocog_val': {"loader": validate_val_loader, "coco_eval_file":os.path.join(root, 'instances_refcocog_val.json')},
             "refcocog_test": {"loader": validate_test_loader, "coco_eval_file":os.path.join(root, 'instances_refcocog_test.json')}}, \
                partial(validate, validate_metrics=validate_metrics, root=root),\
                None, None

@register_data_schedule
def refcoco_all_schedule(configs, is_distributed, process_id, num_processes):
    root = configs['data_dir']
    imgs_dir = os.path.join(root, 'refer/train2014/train2014')
    root = os.path.join(root, 'refer')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    validate_batch_size= configs['validate_batch_size']
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    image_aux_version: int = configs['image_aux_version']
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    training_seed: int  = configs['training_seed']
    train_batch_size= configs['train_batch_size']
    validate_metrics = configs['validate_metrics']

    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize', 'hflip_ResizeSmaller', "resizeSmaller"]
        assert train_augmentation['name'] in amr_legi_augs
        assert validate_augmentation['name'] in amr_legi_augs
    
    # 合并所有图像, 没有validate和test
    splits = ['train', 'test', 'val']
    split_to_samples = {}
    cat_to_ids = None
    for split in splits:
        with open(os.path.join(root, f'instances_refcocog_{split}.json'), 'r') as f:
            split_samples = json.load(f)
        split_num_samples = len(split_samples['images']) # 80512,
        assert split_num_samples == len(split_samples['annotations'])
        if process_id == 0:
            logging.info(f'Number of {split} samples: {split_num_samples}')
            print(f'Number of {split} samples: {split_num_samples}')
        if cat_to_ids is None:
            categories = split_samples['categories']
            all_category_ids = [foo['id'] for foo in categories]
            assert len(all_category_ids) == 80
            cat_to_ids = {cid:idx for idx, cid in enumerate(all_category_ids)}
        split_to_samples[split] = split_samples

    with open(os.path.join(root, f'global_mappings.json'), 'r') as f:
        globals = json.load(f) 
    imgToAnns = globals['imgToAnns']
    imgToRefs = globals['imgToRefs']
    if text_aux_version != 0:
        text_aux_file = os.path.join(root, f'text_to_aux.json')
        # you need to generate outside the main module
        with open(text_aux_file, 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if image_aux_version != 0:
        pass
    else:
        image_aux_by_auxid = None        
    
    create_train_aug = image_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)
    train_dataset = REFCOCO(imgs_dir=imgs_dir,
                            pt_tokenizer_dir=pt_tokenizer_dir,
                            split='train',
                            set_name='refcocog',
                            imgToRefs=imgToRefs,
                            catToId=cat_to_ids,
                            imgToAnns=imgToAnns,
                            samples = split_to_samples['train'],
                            augmentation=train_aug,
                            text_aux_by_auxid=text_aux_by_auxid,
                            text_aux_version=text_aux_version,
                            image_aux_version=image_aux_version,
                            image_aux_by_auxid=image_aux_by_auxid) 
     
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
            
    def my_dataset_function():
        return [{}]
    from detectron2.data import DatasetCatalog
    DatasetCatalog.register('refcocog_rios', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('refcocog_rios').thing_classes = ['r', 'nr']
    MetadataCatalog.get('refcocog_rios').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

        
    return train_loader, sampler_train,\
            None, None, None, None


class REFCOCO(DatasetWithAux):
    def __init__(self, 
                 imgs_dir,
                 split,
                 set_name,
                 samples,
                 augmentation, 
                 imgToRefs,
                 imgToAnns,
                 catToId,
                 text_aux_version,
                 text_aux_by_auxid,
                 image_aux_version,
                 image_aux_by_auxid,
                 pt_tokenizer_dir,
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                         text_aux_by_auxid=text_aux_by_auxid,
                         video_aux_version=image_aux_version,
                         video_aux_by_auxid=image_aux_by_auxid,
                         pt_tokenizer_dir=pt_tokenizer_dir,
                         )
        self.set_name = set_name
        self.imgs_dir = imgs_dir
        self.samples = samples['images'] # {file_name, h, w, original_id, id, caption, dataset_name}
        self.samples_anns = samples['annotations']
        assert len(self.samples) == len(self.samples_anns)
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.category_id_map = catToId
        self.augmentation = augmentation

        self.split = split
        collator_kwargs = {}
        if text_aux_version == 1 or text_aux_version == 2 or text_aux_version == 3 or text_aux_version == 4:
            collator_kwargs['tokenizer'] = self.tokenizer
        self.collator = Collator(split=split,
                                 data_ins=self,
                                 text_aux_version=text_aux_version,
                                 image_aux_version=image_aux_version,
                                 **collator_kwargs)
        
    def decode_annotations(self, annotations, H, W):
        # list[dict] -> dict['masks': ni h w, bool, 'boxes': ni 4, float, 'valid': ni, bool]  
        obj_boxes = []
        # n h w, bool
        obj_masks = [convert_poly_to_mask_for_single_obj_torchBool(ann['segmentation'], height=H, width=W) for ann in annotations]
        assert len(obj_masks) > 0
        obj_masks = torch.stack(obj_masks, dim=0)
        obj_valids = obj_masks.flatten(1).any(-1) # n
        class_labels = torch.tensor([self.category_id_map[ann['category_id']]  for ann in annotations])
        for vli, mask in zip(obj_valids, obj_masks):
            if vli:
                y1, y2, x1, x2 = bounding_box_from_mask(mask.numpy())
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                obj_boxes.append(box)
            else:
                box = torch.tensor([0,0,0,0]).to(torch.float)
        return {
            'masks': obj_masks,
            'boxes': torch.stack(obj_boxes, dim=0),
            'valids': obj_valids,
            'class_labels': class_labels
        }

    def decode_text(self, all_texts, all_ann_ids):
        # {sent_ids: [0, 1, 2], 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id'}, ann_id -> list[str]
        ann_to_texts = {}
        for text in all_texts:
            sentences = text['sentences'] # list[{tokens, raw, sent_id, sent}]
            assert len(sentences) > 0 # 保证这个图像至少有一个有refcoco text annotation
            sentences = [refcoco_normalize_text(sent['sent'], self.set_name, sent['sent_id']) for sent in sentences]
            ann_id = text['ann_id']
            if ann_id in ann_to_texts:
                ann_to_texts[ann_id] += sentences
            else:
                ann_to_texts[ann_id] = sentences
        assert len(set(list(ann_to_texts.keys())) - set(list(all_ann_ids))) == 0  
        # !TODO: assert len(set(list(all_ann_ids)) - set(list(ann_to_texts.keys()))) == 0 # 有的annotation没有text
        # TODO: refcoco并没有为coco图像的每个obj ann提供text ann
        ret = []
        refcoco_have_at_least_one_text_ann_for_this_image = False
        for ann_id in all_ann_ids:
            if ann_id in ann_to_texts:
                refcoco_have_at_least_one_text_ann_for_this_image = True
                ret.append(ann_to_texts[ann_id])
            else:
                ret.append([''])
        assert refcoco_have_at_least_one_text_ann_for_this_image
        return ret
    
    def __getitem__(self, item_idx):
        # 每个图像的每个(obj, text)成为一个测试样本
        if self.split == 'train' or self.split == 'validate':
            chosen_sample = self.samples[item_idx] 
            chosen_sample_ann = self.samples_anns[item_idx]
            file_name, H, W, original_image_id, test_img_id, text_query, sent_id = chosen_sample['file_name'], chosen_sample['height'],chosen_sample['width'],\
                                                            chosen_sample['original_id'], chosen_sample['id'],chosen_sample['caption'], chosen_sample['sent_id']
            ann_id = chosen_sample_ann['original_id']
            text_query = refcoco_normalize_text(text_query, set_name=self.set_name, sent_id=sent_id)  
            if text_query == 'man i a white shirt':
                print('this')
                pass
            image = Image.open(os.path.join(self.imgs_dir, file_name)).convert("RGB") 
            assert image.size[0] == W and image.size[1] == H
            all_obj_anns = self.imgToAnns[str(original_image_id)] # 因为Json导致key成了string
            all_obj_ann_ids = [aoa['id'] for aoa in all_obj_anns]
            referent_idx = all_obj_ann_ids.index(ann_id)  # 每个obj能区分的就是annotation id
            all_obj_anns = self.decode_annotations(all_obj_anns, H=H, W=W)
            all_refs = self.imgToRefs[str(original_image_id)]
            appear_texts = self.decode_text(all_refs, all_obj_ann_ids) # list[[list[str]]], 按照all_obj_ann_ids排列每个物体的sentences
            targets = {
                'masks': all_obj_anns['masks'], # n h w (bool)
                'class_labels': all_obj_anns['class_labels'], # n
                'boxes': all_obj_anns['boxes'], # n 4, xyxy, float
                'valid': all_obj_anns['valids'], # n
                'referent_idx': referent_idx, 
                'image_id': test_img_id,
                'orig_size': torch.tensor([H, W]),
                'size': torch.tensor([H, W]), # h w
            }
            flatten_texts = [text_query]
            for atxt in appear_texts:
                flatten_texts.extend(copy.deepcopy(atxt))
            
            image, flatten_texts, targets = self.augmentation(image, flatten_texts, targets)
            text_query = flatten_texts[0]
            cnt = 1
            for idx, foo in enumerate(appear_texts):
                appear_texts[idx] = flatten_texts[cnt:(cnt+len(foo))]
                cnt += len(foo)
            assert cnt == len(flatten_texts)

            return image, text_query, self.get_aux(item_idx, queries_by_objid=appear_texts,
                                                     image_auxid=None, text_auxid=text_query), targets

    def get_aux(self, item_idx, queries_by_objid,
                image_auxid,
                text_auxid):
        aux = {}
        aux['sample_idx'] = item_idx
        aux['exist_queries'] = queries_by_objid
        aux.update(self.get_text_aux(text_auxid, queries_by_objid))
        return aux
    
    def __len__(self):
        return len(self.samples)

class Collator(CollatorWithAux):
    def __init__(self, split, 
                 data_ins,
                 text_aux_version,
                 image_aux_version,
                 **kwargs
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                       video_aux_version=image_aux_version,
                       **kwargs)
        self.split = split
        self.data_ins = data_ins

    def __call__(self, batch):
        samples, text_query, auxiliary, meta_or_target = list(zip(*batch))
        samples = list(samples)
        text_query = list(text_query)
        auxiliary = list(auxiliary)
        meta_or_target = list(meta_or_target)

        batch_size = len(samples)
        
        batch_data = {
            'samples': samples,
            'text_query': text_query,
            'auxiliary': self.batching_aux(auxiliary)
        }

        # if self.split == 'train':
        #     if 'amrs' in batch_data['auxiliary']:
        #         amrs = batch_data['auxiliary']['amrs']
        #         num_edges = [am.num_edges for am in amrs]
        #         while sum(num_edges) == 0:
        #             new_sample_idxs = torch.randperm(200)[:batch_size]
        #             # change this batch by calling data_ins
        #             new_batch = [self.data_ins.__getitem__(idx) for idx in new_sample_idxs]
        #             samples, text_query, auxiliary, meta_or_target = list(zip(*new_batch))
        #             samples = list(samples)
        #             text_query = list(text_query)
        #             auxiliary = list(auxiliary)
        #             meta_or_target = list(meta_or_target)
        #             batch_data = {
        #                 'samples': samples,
        #                 'text_query': text_query,
        #                 'auxiliary': self.batching_aux(auxiliary)
        #             }
        #             amrs = batch_data['auxiliary']['amrs']
        #             num_edges = [am.num_edges for am in amrs]

        if self.split == 'test':
            batch_data['meta'] = meta_or_target
        elif self.split == 'train' or self.split == 'validate':
            batch_data['targets'] = meta_or_target


        return batch_data


        

