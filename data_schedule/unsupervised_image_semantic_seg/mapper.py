
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
from detectron2.data import MetadataCatalog
from data_schedule.registry import Mapper as Ori_Mapper
from data_schedule.registry import MAPPER_REGISTRY
from PIL import Image
from torchvision import transforms as T

torch_aug_generator = torch.Generator()
torch_aug_generator.manual_seed(1999)

@MAPPER_REGISTRY.register()
class Superpixel_TrainMapper(Ori_Mapper):
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.augmentation = VIS_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')  
        self.get_frames_fn = dataset_meta.get('get_frames_fn')
        
    def _call(self, data_dict):
        VIS_Dataset
        video_id, all_frames = data_dict['video_id'], data_dict['all_frames']
        video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
        aug_ret = {
            'video': video_frames,
            'callback_fns': []
        }
        VIS_Aug_CallbackAPI
        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')
        callback_fns = aug_ret.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        VIS_EvalAPI_clipped_video_request_ann
        return {
            'video_dict': {'video': video},
            'meta': {
                'video_id': video_id,
                'frames': all_frames,
                'request_ann': torch.ones(len(all_frames)).bool(),
                'callback_fns': callback_fns               
            }
        }



class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

@MAPPER_REGISTRY.register()
class Unsupervised_EvalMapper(Ori_Mapper):
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
        res = aug_config['res']
        self.image_transform = T.Compose([
            T.Resize(res, Image.NEAREST), 
            T.CenterCrop(res), 
            T.ToTensor(), 
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize(res, Image.NEAREST), 
            T.CenterCrop(res), 
        ])
        
    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # int
        image = self.image_transform(image)
        mask = self.mask_transform(mask[None, ...])
        ret = {
            'image_id': image_id,
            'image': image,   
            'mask': mask.squeeze(0), ## unsupervised, but compuate loss w.r.t the detached logits for probling
            'foreground_mask': None, # TODO: 当前方法中用了gt label >0 得到了foreground mask, 为了公平，我们也用
        }
        
        return ret


@MAPPER_REGISTRY.register()
class Unsupervised_TrainMapper(Ori_Mapper):
    """
    每个sample是一张图像(对应图像的model_feature), 对图像进行增强; 没有Mask做监督
    
    每个Meta是一张图片
    每个数据集分成固定的(训练, 测试)两个split
    训练和测试不需要读取Mask
    """
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
        # define your train augmentations
        aug_config = mapper_config['augmentation']
        res, crop_type = aug_config['res'], aug_config['loader_crop_type']
        
        self.image_transform = T.Compose([
            T.Resize(res, Image.NEAREST), 
            T.CenterCrop(res) if crop_type == 'center' else T.Lambda(lambda x: x), 
            T.RandomCrop(res) if crop_type == 'random' else T.Lambda(lambda x: x),
            T.ToTensor(), 
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.target_transform = T.Compose([  # hw, int32, -1,0,1,2,3
            T.Resize(res, Image.NEAREST), 
            T.CenterCrop(res) if crop_type == 'center' else T.Lambda(lambda x: x), 
            T.RandomCrop(res) if crop_type == 'random' else T.Lambda(lambda x: x),
        ])        
        self.aug_geometric_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=aug_config["res"], scale=(0.8, 1.0))
        ])
        self.aug_photometric_transform = T.Compose([
            T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            T.RandomGrayscale(.2),
            T.RandomApply([T.GaussianBlur((5, 5))])
        ])
        # torchv_aug generator
        # TODO: 可以把每个Image的不同增强当作不同的Image, 扩充训练集, 这样就比较快乐       
        # pt_model_names = mapper_config['pt_model_names'] # list[str]
        # """
        # 如果是提前抽好特征的话，必须固定住抽特征时候用的增强，
        # 但是会限制Linear层只学一个数据增强下的N个样本
        # """
        # aug_id = f''
        # pt_features = {
        #     pt_model_key: torch.load(f'{dataset_name}_{aug_config['crop_type']}_\
        #         {aug_config['crop_res']}_{pt_model_key}_features.pth', map_location='cpu') for pt_model_key in pt_model_names # N hi wi c
        # }
        # pt_model_names = mapper_config['pt_model_names'] # list[str]
    
    def _call(self, data_dict):
        orig_torch_rng_state = torch.get_rng_state()
        # torchvision transform
         
        # albumentataion transform
        
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w, int32, label, -1 for background
        torch.manual_seed(torch.randint(low=0, high=2147483647, size=[1], generator=torch_aug_generator)[0].data)
        cnt_state = torch.get_rng_state()
        image = self.image_transform(image)
        torch.set_rng_state(cnt_state)
        mask = self.target_transform(mask[None, ...])
        
        coord_entries = torch.meshgrid([torch.linspace(-1, 1, image.shape[1]),
                                        torch.linspace(-1, 1, image.shape[2])], indexing="ij")
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0) # 2 h w, -1, 1定义的坐标

        ret = {
            'image_id': image_id,
            'image': image,   
            'pho_aug_image': self.aug_photometric_transform(image),
            'geo_aug_coord': self.aug_geometric_transform(coord).permute(1, 2, 0),
            'mask': mask.squeeze(0), ## unsupervised, but compuate loss w.r.t the detached logits for probling
            'foreground_mask': None, # TODO: 当前方法中用了gt label >0 得到了foreground mask, 为了公平，我们也用; HP里面没有用到foreground mask
        }
        
        torch.set_rng_state(orig_torch_rng_state)
        return ret









