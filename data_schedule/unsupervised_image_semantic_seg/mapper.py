
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
import torch.nn.functional as F


# nearest resize短边image, 然后centercrop到固定大小; 同样的mask; HP有normalize
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
@MAPPER_REGISTRY.register()
class HP_TrainMapper_FiveCrop(Ori_Mapper):
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

        self.resolution = resolution
        self.transform = T.Compose([
            T.Resize(resolution, Image.NEAREST),
            T.CenterCrop(resolution),
            T.ToTensor(),
            normalize
        ])
        self.target_transform = T.Compose([
            T.Resize(resolution, Image.NEAREST),
            T.CenterCrop(resolution),
        ])
        self.geometric_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=resolution, scale=(0.8, 1.0))
        ])
        self.photometric_transforms = T.Compose([
            T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            T.RandomGrayscale(.2),
            T.RandomApply([T.GaussianBlur((5, 5))]),
            # from dataset.photometric_aug import RandomLightingNoise
            # RandomLightingNoise()
        ])
        self.path = os.path.join('/home/xuhuihui/workspace/UNMED/data/Datasets/cocostuff', "cropped", "cocostuff27_five_crop_0.5", "label", "train",)

    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image: Image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w, -1/0-27, long
        mask = mask[None, None] # h w, -1, 0-27

        image = self.transform(image) # 3 h w 
        mask = self.target_transform(mask)[0, 0]

        img_aug = self.photometric_transforms(image)
        
        ret = {
            'image_id': image_id,
            'image': image,
            'mask': mask,
            'img_aug': img_aug,
        }
        return ret

# nearest resize短边image, crop到固定大小, 同样的 mask, 有normalize
@MAPPER_REGISTRY.register()
class HP_EvalMapper(Ori_Mapper):
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
        resolution = mapper_config['res']
        self.transform = T.Compose([
            T.Resize(resolution, Image.NEAREST), 
            T.CenterCrop(resolution),
            T.ToTensor(),
            normalize,
        ])
        self.target_transform = T.Compose([
            T.Resize(resolution, Image.NEAREST),
            T.CenterCrop(resolution),
        ])
        pass
    
    def bilinear_resize_mask(self, mask, shape):
        # h w, uint8, 0-255, label, bilinear
        H, W = mask.shape
        unique_labels = mask.unique()
        lab_to_mask = []
        for lab in unique_labels:
            binary_mask = (mask == lab).float()
            binary_mask = F.interpolate(binary_mask[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0]
            lab_to_mask.append(binary_mask)
        lab_to_mask = torch.stack(lab_to_mask, dim=-1) # h w num_class
        new_mask = lab_to_mask.max(dim=-1)[1] # h w, indices
        new_label = unique_labels[new_mask.flatten()].reshape(shape)
        return new_label
    
    def _call(self, data_dict):
        image_id = data_dict['image_id']

        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w unin8, 255

        image = self.transform(image)
        mask = mask[None, None]
        new_mask = self.target_transform(mask)
        new_mask = new_mask[0, 0]
        return {
           'image': image,
           'image_id': image_id,
           'mask': new_mask
       }



# bilinear resize到固定大小，mask也是bilinear
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
        resolution = mapper_config['res']
        self.transform = T.Compose([
            T.Resize((resolution,resolution), interpolation=Image.BILINEAR), 
            T.ToTensor()
        ])
    
    def bilinear_resize_mask(self, mask, shape):
        # h w, uint8, 0-255, label, bilinear
        H, W = mask.shape
        unique_labels = mask.unique()
        lab_to_mask = []
        for lab in unique_labels:
            binary_mask = (mask == lab).float()
            binary_mask = F.interpolate(binary_mask[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0]
            lab_to_mask.append(binary_mask)
        lab_to_mask = torch.stack(lab_to_mask, dim=-1) # h w num_class
        new_mask = lab_to_mask.max(dim=-1)[1] # h w, indices
        new_label = unique_labels[new_mask.flatten()].reshape(shape)
        return new_label
    
    def _call(self, data_dict):
        image_id = data_dict['image_id']

        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # -1/0-26, long

        image = self.transform(image)
        new_mask = self.bilinear_resize_mask(mask, shape=image.shape[-2:])
        # original_class = mask.unique().tolist()
        # after_class = new_mask.unique().tolist()
        # assert len(set(original_class) - set(after_class)) == 0
        # assert len(set(after_class) - set(original_class)) == 0
        return {
           'image': image,
           'image_id': image_id,
           'mask': new_mask
       }


# bilinear resize到固定大小 没有mask
@MAPPER_REGISTRY.register()
class Common_TrainMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name,
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




@MAPPER_REGISTRY.register()
class STEGO_TrainMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name, 
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

        nearest_neighbor_file = mapper_config['nearest_neighbor_file']
        if not os.path.exists(nearest_neighbor_file):
            raise ValueError()
        self.nns = torch.load(nearest_neighbor_file)

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

        ret = {
            'image_id': image_id,
            'image': image,
            'image_pos': nn_image,
            # 'image_feat': image_feat,
            # 'image_pos_feat': pos_image_feat,
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



