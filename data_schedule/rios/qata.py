import json
import os
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
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
from .augmentation_images import image_aug_entrypoints

from utils.misc import is_dist_avail_and_initialized, all_gather, to_device
import torch.distributed as dist


import pycocotools.mask as mask_util
from pycocotools.mask import encode, area
from detectron2.data import DatasetCatalog
from util.box_ops import box_xyxy_to_cxcywh
from torch.utils.data import DataLoader, Dataset
from .utils import Evaluate_Sampler_Distributed, TrainRandomSampler_ByEpoch_Distributed, bounding_box_from_mask
from .utils import CollatorWithAux, DatasetWithAux

__all__ = ['qata_schedule']

def read_text_annotations_groupby_image(root):
    # train/val
    train_text_by_id = pandas.read_excel(os.path.join(root, 'Train_text_for_Covid19.xlsx'), header=0)
    train_texts = train_text_by_id['Description'].tolist()
    train_img_ids = train_text_by_id['Image'].tolist()
    assert len(train_img_ids) == 7145
    assert len(train_texts) == 7145

    # test
    test_text_by_id = pandas.read_excel(os.path.join(root, 'Test_text_for_Covid19.xlsx'), header=0)
    test_texts = test_text_by_id['Description'].tolist()
    test_img_ids = test_text_by_id['Image'].tolist()
    assert len(test_img_ids) == 2113
    assert len(test_texts) == 2113

    image_ids = train_img_ids + test_img_ids
    texts = train_texts + test_texts
    return {id: text for id, text in zip(image_ids, texts)}    

def read_img_dirs_groupby_image(root):
    # {'train': {id:{'dir':, 'gt_dir':}}}
    # root dir
    trainval_root = os.path.join(root, 'QaTa-COV19/QaTa-COV19-v2/Train Set')
    # train/val
    train_img_ids = pandas.read_excel(os.path.join(root, 'Train_ID.xlsx'), header=0)['Image'].tolist()
    assert len(train_img_ids) == 5716
    for id in train_img_ids:
        assert id.startswith('mask_')
    val_img_ids = pandas.read_excel(os.path.join(root, 'Val_ID.xlsx'), header=0)['Image'].tolist()
    assert len(val_img_ids) == 1429
    for id in val_img_ids:
        assert id.startswith('mask_')
    train_img_ids += val_img_ids

    train_meta = {id:{'dir': os.path.join(trainval_root, 'Images', f'{id[5:]}.png'),
                      'gt_dir': os.path.join(trainval_root, 'Ground-truths', f'{id}.png')} for id in train_img_ids}
    
    # test
    test_root = os.path.join(root, 'QaTa-COV19/QaTa-COV19-v2/Test Set')
    test_text_by_id = pandas.read_excel(os.path.join(root, 'Test_text_for_Covid19.xlsx'), header=0)
    test_img_ids = test_text_by_id['Image'].tolist()
    assert len(test_img_ids) == 2113
    test_meta = {id:{'dir':os.path.join(test_root, 'Images', f'{id}.png'),
                     'gt_dir':os.path.join(test_root, 'Ground-truths', f'mask_{id}.png')} for id in test_img_ids}
    return {
        'train': train_meta,
        'validate': test_meta
    }

def visualize_dataset_information(root):
    pass


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


def generate_text_aux_file():
    pass

@register_data_schedule
def qata_schedule(configs, is_distributed, process_id, num_processes):
    root = configs['data_dir']
    root = os.path.join(root, 'refer_medical/qata')
    pt_tokenizer_dir=configs['pt_tokenizer_dir']
    num_workers= configs['num_workers']
    validate_batch_size= configs['validate_batch_size']
    
    # 训练时额外的数据 
    amr_are_used: bool = configs['amr_are_used']
    text_aux_version: int = configs['text_aux_version']
    image_aux_version: int = configs['image_aux_version']
    # 训练数据增强, 测试数据增强
    train_augmentation: dict = configs['train_augmentation']
    validate_augmentation: dict = configs['validate_augmentation']
    # 训练时的SGD的loading
    training_seed: int  = configs['training_seed']
    train_batch_size= configs['train_batch_size']
    # 测试时候的metrics
    validate_metrics = configs['validate_metrics']

    if amr_are_used:
        amr_legi_augs = ['fixsize', 'justnormalize', 'resize', 'hflip_fixsize']
        assert train_augmentation['name'] in amr_legi_augs
        assert validate_augmentation['name'] in amr_legi_augs
    
    if text_aux_version != 0:
        text_aux_file = os.path.join(root, f'text_to_aux.json')
        # you need to generate outside the main module
        with open(text_aux_file, 'r') as f:
            text_aux_by_auxid = json.load(f)
    else:
        text_aux_by_auxid = None

    if image_aux_version != 0:
        image_aux_file = os.path.join(root, f'image_aux_v{image_aux_version}.json')
        if not os.path.exists(image_aux_file):
            generate_text_aux_file(root, image_aux_file)
        with open(image_aux_file, 'r') as f:
            image_aux_by_auxid = json.load(f)
    else:
        image_aux_by_auxid = None        
        
    text_annotations_by_imageid = read_text_annotations_groupby_image(root)
    img_dir_by_image_id = read_img_dirs_groupby_image(root)
    
    create_train_aug = image_aug_entrypoints(train_augmentation['name'])
    train_aug = create_train_aug(train_augmentation)
    train_dataset = QATA_Dataset(root=root,
                                 pt_tokenizer_dir=pt_tokenizer_dir,
                                 split='train',
                                samples = img_dir_by_image_id['train'],
                                augmentation=train_aug,
                                text_annotations_by_imageid=text_annotations_by_imageid,
                                
                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                image_aux_version=image_aux_version,
                                image_aux_by_auxid=image_aux_by_auxid) 
    
    create_validate_aug = image_aug_entrypoints(validate_augmentation['name'])
    validate_aug = create_validate_aug(validate_augmentation)                     
    validate_dataset = QATA_Dataset(root=root,
                                    pt_tokenizer_dir=pt_tokenizer_dir,
                                split='validate',
                                samples = img_dir_by_image_id['validate'],
                                augmentation=validate_aug,
                                text_annotations_by_imageid=text_annotations_by_imageid,
                                
                                text_aux_by_auxid=text_aux_by_auxid,
                                text_aux_version=text_aux_version,
                                image_aux_version=image_aux_version,
                                image_aux_by_auxid=image_aux_by_auxid) 

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
    DatasetCatalog.register('pata', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('pata').thing_classes = ['r', 'nr']
    MetadataCatalog.get('pata').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

        
    return train_loader, sampler_train,\
            validate_loader, partial(validate, validate_metrics=validate_metrics, root=root),\
                None, None

def qata_normalize_text(text_query):
    # 非法输入
    if text_query == 'The left with yellow t shirt on the left running':
        text_query = 'the man with yellow tshirt on the left running'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

class QATA_Dataset(DatasetWithAux):

    # h: 320, 
    # train: w_min: 320, w_max:768, 15741
    # validate:  w_min: 390, w_max:640, 3800
    def __init__(self, 
                 root,
                 split,
                 samples,
                 augmentation, 
                 text_annotations_by_imageid,
                 
                 text_aux_version,
                 text_aux_by_auxid,
                 image_aux_version,
                 image_aux_by_auxid,
                 pt_tokenizer_dir,
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                         text_aux_by_auxid=text_aux_by_auxid,
                         image_aux_version=image_aux_version,
                         image_aux_by_auxid=image_aux_by_auxid,
                         pt_tokenizer_dir=pt_tokenizer_dir,
                         )
        self.root = root
        self.image_dir_by_id = samples
        self.samples = list(self.image_dir_by_id.keys())
        self.text_annotations_by_imageid = text_annotations_by_imageid
        self.augmentation = augmentation
        
        self.split = split
        collator_kwargs = {}
        if text_aux_version == 1 or text_aux_version == 2 or text_aux_version == 3 or text_aux_version == 4:
            collator_kwargs['tokenizer'] = self.tokenizer
        self.collator = Collator(split=split,
                                 text_aux_version=text_aux_version,
                                 image_aux_version=image_aux_version,
                                 **collator_kwargs)
                
    def __getitem__(self, item_idx):

        if self.split == 'train' or self.split == 'validate':
            image_id = self.samples[item_idx]
            img_dir, gt_dir = self.image_dir_by_id[image_id]['dir'], self.image_dir_by_id[image_id]['gt_dir']
            
            image = Image.open(img_dir)
            pt_dir = Image.open(gt_dir)
            H, W = image.shape[-2:]

            y1, y2, x1, x2 = bounding_box_from_mask(object_mask.numpy())
            box = torch.tensor([x1, y1, x2, y2]).to(torch.float)

            text = self.text_annotations_by_imageid[image_id]
            
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
                                                     image_auxid=None, text_auxid=text_query), targets

    def get_aux(self, item_idx, queries_by_objid,
                image_auxid,
                text_auxid):
        aux = {}
        aux['sample_idx'] = item_idx
        aux['exist_queries'] = queries_by_objid
        aux.update(self.get_text_aux(text_auxid, queries_by_objid))
        aux.update(self.get_image_aux(image_auxid))
        return aux
    
    def __len__(self):
        return len(self.samples)


class Collator(CollatorWithAux):
    def __init__(self, split, 
                 text_aux_version,
                 image_aux_version,
                 **kwargs
                 ) -> None:
        super().__init__(text_aux_version=text_aux_version,
                       image_aux_version=image_aux_version,
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





  

        

