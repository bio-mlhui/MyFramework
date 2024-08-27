
import json
import os
from typing import List
import copy
from functools import partial
import random
import numpy as np
import torch
import logging
from einops import rearrange
from detectron2.data import MetadataCatalog, DatasetCatalog
from data_schedule.registry import Mapper as Ori_Mapper
from data_schedule.registry import MAPPER_REGISTRY
from PIL import Image
from torchvision import transforms as T
from .augmentations import UN_IMG_SEG_EVAL_AUG_REGISTRY, UN_IMG_SEG_TRAIN_AUG_REGISTRY



@MAPPER_REGISTRY.register()
class UN_IMG_SEG_EvalMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name, # potsdam_train
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.get_mask_fn = dataset_meta.get('get_mask_fn')
        self.num_classes = dataset_meta.get('num_classes')
        
        # define your test augmentations
        aug_config = mapper_config['augmentation']
        self.eval_aug = UN_IMG_SEG_EVAL_AUG_REGISTRY.get(aug_config['name'])(aug_config)
    
    def _call(self, data_dict):
        image_id = data_dict['image_id']

        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # unin8, 255
        ret = {
            'image_id': image_id,
            'image': image,
            'mask': mask
        }
        aug_ret = self.eval_aug(ret)        
        return aug_ret


@MAPPER_REGISTRY.register()
class UN_IMG_SEG_TrainMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name, # potsdam3_train
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.num_classes = dataset_meta.get('num_classes')
        
        self.get_mask_fn = dataset_meta.get('get_mask_fn')
        aug_config = mapper_config['augmentation']
        self.train_aug = UN_IMG_SEG_TRAIN_AUG_REGISTRY.get(aug_config['name'])(aug_config)
        
    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        
        ret = {
            'image_id': image_id,
            'image': image,
        }
        aug_ret = self.train_aug(ret)
        
        return aug_ret


@MAPPER_REGISTRY.register()
class UN_IMG_SEG_TrainMapper_withMask(Ori_Mapper):
    def __init__(self,
                 dataset_name, # potsdam3_train
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.num_classes = dataset_meta.get('num_classes')
        
        self.get_mask_fn = dataset_meta.get('get_mask_fn')
        aug_config = mapper_config['augmentation']
        self.train_aug = UN_IMG_SEG_TRAIN_AUG_REGISTRY.get(aug_config['name'])(aug_config)
        
    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w, 
        
        ret = {
            'image_id': image_id,
            'image': image,
            'mask': mask
        }
        aug_ret = self.train_aug(ret)
        
        return aug_ret


import time
@MAPPER_REGISTRY.register()
class STEGO_TrainMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name, # potsdam3_train
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        from detectron2.data import DatasetFromList
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.num_classes = dataset_meta.get('num_classes')
        self.get_mask_fn = dataset_meta.get('get_mask_fn')
        resolution = mapper_config['res']

        nearest_neighbor_file = mapper_config['nearest_neighbor_file']
        if not os.path.exists(nearest_neighbor_file):
            raise ValueError()
        self.nns = torch.load(nearest_neighbor_file)

        features_path = mapper_config['features_path']
        if not os.path.exists(features_path):
            raise ValueError()
        self.features_path = features_path 

        self.transform = T.Compose([
            T.Resize((resolution,resolution)), 
            T.ToTensor(),
        ])
        self.num_neighbors = mapper_config['num_neighbors']
        
    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        image = self.transform(image) # 3 h w 
        
        nn_image_id = self.nns[image_id][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
        nn_image = self.get_image_fn(image_id=nn_image_id)
        nn_image = self.transform(nn_image)
        # image_feat = np.fromfile(os.path.join(self.features_path, f'{image_id}.bin'))
        # image_feat = np.fromfile(os.path.join(self.features_path, f'{image_id}.bin')).reshape([1536, 64, 64]) # c h w
        # pos_image_feat = np.fromfile(os.path.join(self.features_path, f'{nn_image_id}.bin')).reshape([1536, 64, 64]) # c h w
        # image_feat = torch.load(os.path.join(self.features_path, f'{image_id}.bin'))
        # pos_image_feat = torch.load(os.path.join(self.features_path, f'{nn_image_id}.bin'))
        # image_feat= torch.from_numpy(image_feat)
        # pos_image_feat= torch.from_numpy(pos_image_feat)

        ret = {
            'image_id': image_id,
            'image': image,
            'image_pos': nn_image,
            # 'image_feat': image_feat,
            # 'image_pos_feat': pos_image_feat,
        }
        return ret

@MAPPER_REGISTRY.register()
class Common_TrainMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name, # potsdam3_train
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.num_classes = dataset_meta.get('num_classes')
        self.get_mask_fn = dataset_meta.get('get_mask_fn')
        resolution = mapper_config['res']
        
        self.transform = T.Compose([
            T.Resize((resolution,resolution)), 
            T.ToTensor(),
        ])
        
    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        image = self.transform(image) # 3 h w 
        ret = {
            'image_id': image_id,
            'image': image,
        }
        return ret



# @MAPPER_REGISTRY.register()
# class Superpixel_TrainMapper(Ori_Mapper):
#     def __init__(self,
#                  configs,
#                  dataset_name,
#                  mode,
#                  meta_idx_shift,
#                  ): 
#         assert mode == 'evaluate'
#         dataset_meta = MetadataCatalog.get(dataset_name)
#         assert dataset_meta.get('step_size') == None
#         mapper_config = configs['data'][mode][dataset_name]['mapper']
#         super().__init__(meta_idx_shift, dataset_meta)
#         self.augmentation = VIS_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
#         self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')  
#         self.get_frames_fn = dataset_meta.get('get_frames_fn')
        
#     def _call(self, data_dict):
#         VIS_Dataset
#         video_id, all_frames = data_dict['video_id'], data_dict['all_frames']
#         video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
#         aug_ret = {
#             'video': video_frames,
#             'callback_fns': []
#         }
#         VIS_Aug_CallbackAPI
#         aug_ret = self.augmentation(aug_ret)
#         video = aug_ret.pop('video')
#         callback_fns = aug_ret.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
#         VIS_EvalAPI_clipped_video_request_ann
#         return {
#             'video_dict': {'video': video},
#             'meta': {
#                 'video_id': video_id,
#                 'frames': all_frames,
#                 'request_ann': torch.ones(len(all_frames)).bool(),
#                 'callback_fns': callback_fns               
#             }
#         }



