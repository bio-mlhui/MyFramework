
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


@MAPPER_REGISTRY.register()
class UNI_TrainMapper(Ori_Mapper):
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

        self.resolution = resolution
        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
        ])
        # self.geometric_transforms = T.Compose([
        #     T.RandomHorizontalFlip(),
        #     T.RandomResizedCrop(size=resolution, scale=(0.8, 1.0))
        # ])
        # self.photometric_transforms = T.Compose([
        #     T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        #     T.RandomGrayscale(.2),
        #     T.RandomApply([T.GaussianBlur((5, 5))]),
        #     # from dataset.photometric_aug import RandomLightingNoise
        #     # RandomLightingNoise()
        # ])

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
        image: Image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w, -1/0-27, long
        image = self.transform(image) # 3 h w 
        # img_aug = self.photometric_transforms(image)

        new_mask = self.bilinear_resize_mask(mask, shape=image.shape[-2:])

        ret = {
            'image_id': image_id,
            'image': image,
            'mask': new_mask,
            # 'img_aug': img_aug,
        }
        return ret

@MAPPER_REGISTRY.register()
class UNI_EvalMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'evaluate'
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
            T.Resize((resolution, resolution)),
            T.ToTensor(),
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
        image: Image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # h w, -1/0-27, long
        image = self.transform(image) # 3 h w 
        new_mask = self.bilinear_resize_mask(mask, shape=image.shape[-2:])

        ret = {
            'image_id': image_id,
            'image': image,
            'mask': new_mask,
        }
        return ret




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

class Uni_UNIMGSEM_AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                # list[3 3 h w] -> b 3 3 h w
                'img': torch.stack([item['image'] for item in batch_dict], dim=0),
                'label': torch.stack([item['mask'] for item in batch_dict], dim=0),
                'img_aug': torch.stack([item['img_aug'] for item in batch_dict], dim=0),
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict],
                'image_ids':[item['image_id'] for item in batch_dict],
            }
        elif mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                },
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()

@MAPPER_REGISTRY.register()
class AggSampleMapper(Ori_Mapper):
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        global_crops_scale = mapper_config['global_crops_scale']
        local_crops_scale = mapper_config['local_crops_scale']
        local_crops_number = mapper_config['local_crops_number']

        from torchvision import transforms
        import models.UN_IMG_SEM.AggSample.utils as utils
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.global_geo_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.local_geo_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(1.0),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(p=0.5),
        ])

        self.to_tensor = transforms.ToTensor()

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def x1x2y1y2(self, coord):
        # 2 H W, (y, x)
        ymin, x1 = coord[:, 0, 0].tolist()
        ymax, x2 = coord[:, -1, -1].tolist()
        assert ymin < ymax
        if x1 > x2:
            wmin, wmax, flip = x2, x1, True
        else:
            wmin, wmax, flip = x1, x2, False

        return (wmin, wmax, ymin, ymax)

    def xywh(self, coord):
        ymin, x1 = coord[:, 0, 0].tolist()
        ymax, x2 = coord[:, -1, -1].tolist()
        assert ymin < ymax
        if x1 > x2:
            wmin, wmax, flip = x2, x1, True
        else:
            wmin, wmax, flip = x1, x2, False

        x_c, y_c = (wmin+wmax)/2, (ymin+ymax)/2
        w, h = wmax-wmin, ymax -ymin
        return x_c,y_c,w,h, flip


    def get_common_relative_coordinates(self, crop1, crop2):
        # 坐标值: -1,1, 大小: 0.0-2.0
        x_c1, y_c1, W1, H1, flip1 = self.xywh(crop1)
        x_c2, y_c2, W2, H2, flip2 = self.xywh(crop2)

        x_min1, x_max1, y_min1, y_max1 = self.x1x2y1y2(crop1)
        x_min2, x_max2, y_min2, y_max2 = self.x1x2y1y2(crop2)

        # # Calculate the boundaries of each crop
        # x_min1, x_max1 = x_c1 - W1 / 2, x_c1 + W1 / 2
        # y_min1, y_max1 = y_c1 - H1 / 2, y_c1 + H1 / 2
        
        # x_min2, x_max2 = x_c2 - W2 / 2, x_c2 + W2 / 2
        # y_min2, y_max2 = y_c2 - H2 / 2, y_c2 + H2 / 2
        assert x_min1 == x_c1 - W1 / 2
        assert x_max1 == x_c1 + W1 / 2
        assert y_min1 == y_c1 - H1 / 2
        assert y_max2 == y_c2 + H2 / 2
        
        # Calculate the boundaries of the intersection
        x_min_inter = max(x_min1, x_min2)
        x_max_inter = min(x_max1, x_max2)
        y_min_inter = max(y_min1, y_min2)
        y_max_inter = min(y_max1, y_max2)
        
        # Check if there is an intersection
        assert (x_max_inter > x_min_inter) and (y_max_inter > y_min_inter)
        
        # Calculate relative coordinates of the intersection w.r.t Crop 1
        x_rel1_min = 2 * (x_min_inter - x_c1) / W1
        x_rel1_max = 2 * (x_max_inter - x_c1) / W1
        y_rel1_min = 2 * (y_min_inter - y_c1) / H1
        y_rel1_max = 2 * (y_max_inter - y_c1) / H1
        
        # inter的大小按照crop1得到还是crop2得到?
        if flip1:
            rel_coords1 = (-x_rel1_min, -x_rel1_max, y_rel1_min, y_rel1_max)
        else:
            rel_coords1 = (x_rel1_min, x_rel1_max, y_rel1_min, y_rel1_max)
        
        # Calculate relative coordinates of the intersection w.r.t Crop 2
        x_rel2_min = 2 * (x_min_inter - x_c2) / W2
        x_rel2_max = 2 * (x_max_inter - x_c2) / W2
        y_rel2_min = 2 * (y_min_inter - y_c2) / H2
        y_rel2_max = 2 * (y_max_inter - y_c2) / H2

        # inter的大小按照crop1得到还是crop2得到?
        if flip2:
            rel_coords2 = (-x_rel2_min, -x_rel2_max, y_rel2_min, y_rel2_max)
        else:
            rel_coords2 = (x_rel2_min, x_rel2_max, y_rel2_min, y_rel2_max)

        return rel_coords1, rel_coords2

    def get_global1(self, image, coord):
        seed = np.random.randint(2147483647)  # 第二个global和第一个global最好有交集
        self._set_seed(seed)
        g1_coord = self.global_geo_transform(coord)

        self._set_seed(seed)
        crop = self.global_geo_transform(image)
        crop = self.global_transfo1(crop)
        return crop, g1_coord       

    def intersect_box(self, coord1, coord2):
        x_min1, x_max1, y_min1, y_max1 = self.x1x2y1y2(coord1)
        x_min2, x_max2, y_min2, y_max2 = self.x1x2y1y2(coord2)
        x_min_inter = max(x_min1, x_min2)
        x_max_inter = min(x_max1, x_max2)
        y_min_inter = max(y_min1, y_min2)
        y_max_inter = min(y_max1, y_max2)
        
        # Check if there is an intersection
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return False
        else:
            return True    


    def get_globa2(self, image, coord, g1_coord):
        can_pass = False
        num_attmpts = 0
        while ((not can_pass) and (num_attmpts <=10)):
            num_attmpts += 1
            seed = np.random.randint(2147483647)  # 第二个global和第一个global最好有交集
            self._set_seed(seed)
            g2_coord = self.global_geo_transform(coord)
            can_pass = self.intersect_box(g2_coord, g1_coord)

        g1_to_g1g2, g2_to_g1g2 = None, None
        if can_pass:
            g1_to_g1g2, g2_to_g1g2 = self.get_common_relative_coordinates(g1_coord, g2_coord)

        self._set_seed(seed)
        crop = self.global_geo_transform(image)
        crop = self.global_transfo2(crop)

        return crop, g2_coord, (g1_to_g1g2, g2_to_g1g2) 

    def get_local(self, image, coord, g1_coord, g2_coord):
        can_pass = False
        num_attmpts = 0
        while ((not can_pass) and (num_attmpts <=15)):
            num_attmpts += 1
            seed = np.random.randint(2147483647)  # 第二个global和第一个global最好有交集
            self._set_seed(seed)
            lx_coord = self.local_geo_transform(coord) # 2 H w

            lx_g1_common = self.intersect_box(lx_coord, g1_coord)
            lx_g2_common = self.intersect_box(lx_coord, g2_coord)
            can_pass =  lx_g1_common and lx_g2_common

        rel1, rel2, rel3, rel4 = None, None, None, None
        if lx_g1_common:
            (rel1, rel2) = self.get_common_relative_coordinates(g1_coord, lx_coord)
        if lx_g2_common:
            (rel3, rel4) = self.get_common_relative_coordinates(g2_coord, lx_coord)

        self._set_seed(seed)
        crop = self.local_geo_transform(image)
        crop = self.local_transfo(crop)

        return crop, lx_coord, (rel1, rel2), (rel3, rel4)

    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        W, H = image.size
        coord = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)]) 
        coord = torch.stack(coord, dim=0) # (y, x) H W
        
        crops = []
        common_samples = {}

        global1_image, global1_coord = self.get_global1(image, coord)
        global2_image, global2_coord, (g1_to_g1g2, g2_to_g1g2) = self.get_globa2(image, coord, g1_coord=global1_coord) 

        crops.extend([self.to_tensor(global1_image), self.to_tensor(global2_image)])
        common_samples.update({
            'g0_to_g0g1':g1_to_g1g2, # (xmin, xmax, ymin, ymax), -1/1
            'g1_to_g0g1': g2_to_g1g2,
        })
        for local_idx in range(self.local_crops_number):
            local_crop, local_coord, (g1_to_lxg1, lx_to_lxg1), (g2_to_lxg2, lx_to_lxg2) = self.get_local(image, coord, 
                                                                                                         global1_coord, global2_coord)
            crops.append(self.to_tensor(local_crop))
            common_samples.update({
                f'g0_to_l{local_idx}g0': g1_to_lxg1,
                f'g1_to_l{local_idx}g1': g2_to_lxg2,
                f'l{local_idx}_to_l{local_idx}g0': lx_to_lxg1,
                f'l{local_idx}_to_l{local_idx}g1': lx_to_lxg2,
            })
            
        return {
            'crops': crops, # list[3 h w] # 2 + local
            'common_samples': common_samples, #  dict
            'image_id': image_id
        }

class AggSampleAUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            images = [item['crops'] for item in batch_dict] 
            # list[list[3 h w], crop] batch -> list[b 3 hi wi], crop
            images = [torch.stack(foo, dim=0) for foo in list(zip(*images))]
            common_samples = [item['common_samples'] for item in batch_dict]
            return {
                'images': images,
                'common_samples': common_samples,
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict],
                'image_ids':[item['image_id'] for item in batch_dict],
            }
        elif mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                },
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()


# bilinear resize到固定大小，mask也是bilinear
@MAPPER_REGISTRY.register()
class CutLer_Mapper(Ori_Mapper):
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

        self.to_tensor = T.ToTensor()

    def _call(self, data_dict):
        image_id = data_dict['image_id']

        image = self.get_image_fn(image_id=image_id)  # PIL Image
        mask = self.get_mask_fn(image_id=image_id) # long int 
        unique_labels = mask.unique().tolist()
        instance_masks = []
        assert len(unique_labels) >= 2
        for uniq_cls in unique_labels:
            if uniq_cls == -1:
                continue
            else:
                instance_masks.append(mask == uniq_cls)
        instance_masks = torch.stack(instance_masks, dim=0) # N h w

        image = self.to_tensor(image)
        
        return {
           'image': image,
           'image_id': image_id,
           'instance_mask': instance_masks
       }

@MAPPER_REGISTRY.register()
class Online_CutLer_EvalCluster_Mapper(Ori_Mapper):
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_image_fn = dataset_meta.get('get_image_fn')
        self.get_instance_mask_fn = dataset_meta.get('get_instance_mask_fn')
        self.transform = T.Compose([
            T.Resize((mapper_config['res'],mapper_config['res']), interpolation=Image.BILINEAR), 
            T.ToTensor()
        ])

    def _call(self, data_dict):
        image_id = data_dict['image_id']
        image = self.get_image_fn(image_id=image_id)  # PIL Image
        (orig_W, orig_H) = image.size
        # semantic_mask = self.get_semantic_mask_fn(image_id=image_id) # h w, -1是背景, 0-cls-1
        instance_masks = self.get_instance_mask_fn(image_id=image_id, orig_height=image.size[1], orig_width=image.size[0]) # ni h w, bool
        if instance_masks is None:
            return None
        image = self.transform(image)
        instance_masks = F.interpolate(instance_masks[None, ...].float(), size=image.shape[-2:], align_corners=False, mode='bilinear')[0] > 0.5
        # from data_schedule.unsupervised_image_semantic_seg.evaluator_alignseg import visualize_cutler
        # whole_image = visualize_cutler(image, gt=instance_masks)
        # Image.fromarray(whole_image.numpy()).save('./test.png')

        return {
           'image': image,
           'image_id': image_id,
           'instance_mask': instance_masks,
           'orig_HW': (orig_H, orig_W),
       }
    
class Online_Cutler_EvalCluster_AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                    'orig_HWs': [item['orig_HW'] for item in batch_dict], # TODO: 原大小->224 -> 28 -> 原大小 有点问题
                },
                'instance_masks': [item['instance_mask'] for item in batch_dict],
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()


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
        image = self.transform(image) # 3 h w 
        
        nn_image_id = self.nns[image_id][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
        nn_image = self.get_image_fn(image_id=nn_image_id)
        nn_image = self.transform(nn_image)

        mask = self.get_mask_fn(image_id=image_id) # h w, -1/0-27, long
        new_mask = self.bilinear_resize_mask(mask, shape=image.shape[-2:])
        ret = {
            'image_id': image_id,
            'image': image,
            'image_pos': nn_image,
            'mask': new_mask,
            # 'image_feat': image_feat,
            # 'image_pos_feat': pos_image_feat,
        }
        return ret



@MAPPER_REGISTRY.register()
class HP_TrainMapper(Ori_Mapper):
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
        self.geometric_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=resolution, scale=(0.8, 1.0))
        ])
        self.photometric_transforms = T.Compose([
            T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            T.RandomGrayscale(.2),
            T.RandomApply([T.GaussianBlur((5, 5))]),
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
        image = self.transform(image) # 3 h w 
        img_aug = self.photometric_transforms(image)

        mask = self.get_mask_fn(image_id=image_id) # h w, -1/0-27, long
        new_mask = self.bilinear_resize_mask(mask, shape=image.shape[-2:])
        ret = {
            'image_id': image_id,
            'image': image,
            'mask': new_mask,
            'img_aug': img_aug,
        }
        return ret



# nearest resize短边image, 然后centercrop到固定大小; 同样的mask; HP有normalize
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
@MAPPER_REGISTRY.register()
class HP_TrainMapper_FiveCrop_original(Ori_Mapper):
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
class HP_EvalMapper_original(Ori_Mapper):
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



