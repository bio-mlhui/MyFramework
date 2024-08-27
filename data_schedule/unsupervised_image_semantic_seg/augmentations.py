# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from PIL import Image
from detectron2.utils.registry import Registry
UN_IMG_SEG_EVAL_AUG_REGISTRY = Registry('UN_IMG_SEG_EVAL_AUG')
UN_IMG_SEG_TRAIN_AUG_REGISTRY = Registry('UN_IMG_SEG_TRAIN_AUG')
import torchvision.transforms as T
import torch
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
@UN_IMG_SEG_TRAIN_AUG_REGISTRY.register()
class Alignseg_COCOStuff27_TrainAug:
    def __init__(self, configs):
        from .aug_utils import TrainTransforms
        self.train_transforms = TrainTransforms(size_crops=configs["size_crops"],
                                            nmb_crops=configs["nmb_crops"],
                                            min_intersection=configs["min_intersection_crops"],
                                            min_scale_crops=configs["min_scale_crops"],
                                            max_scale_crops=configs["max_scale_crops"],
                                            augment_image=configs["augment_image"])
    
    def __call__(self, ret):
        image = ret['image']
        image = self.train_transforms(image) # images: list[3 h w], 3;  
        ret['image'] = torch.stack(image[0], dim=0) # 3 3 h w
        return ret

@UN_IMG_SEG_TRAIN_AUG_REGISTRY.register()
class KMeans_TrainAug:
    def __init__(self, configs):
        resolution = configs['res']
        self.transform = T.Compose([
            T.Resize((resolution,resolution)), 
            T.ToTensor()
        ])
    
    def __call__(self, ret):
        image, mask = ret['image'], ret['mask']
        image = self.transform(image) # 3 h w 
        
        original_class = mask.unique().tolist()
        mask = F.interpolate(mask[None, None], size=image.shape[-2:], mode='nearest')[0, 0]
        after_class = mask.unique().tolist()
        assert len(set(original_class) - set(after_class)) == 0
        assert len(set(after_class) - set(original_class)) == 0
        ret['image'] = image
        ret['mask'] = mask
        return ret

@UN_IMG_SEG_EVAL_AUG_REGISTRY.register()
class COCOStuff27_EvalAug:
    def __init__(self, configs):
        resolution = configs['res']
        self.transform = T.Compose([
            T.Resize((resolution,resolution)), 
            T.ToTensor()
        ])

    def __call__(self, ret):
        image, mask = ret['image'], ret['mask']
        image = self.transform(image)
        original_class = mask.unique().tolist()
        mask = F.interpolate(mask[None, None], size=image.shape[-2:], mode='nearest')[0, 0]
        after_class = mask.unique().tolist()
        assert len(set(original_class) - set(after_class)) == 0
        assert len(set(after_class) - set(original_class)) == 0
        
        ret['image'] = image
        ret['mask'] = mask
        return ret
    
