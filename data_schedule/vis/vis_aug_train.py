# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from copy import deepcopy as dcopy
from data_schedule.vis.apis import VIS_Aug_CallbackAPI
import albumentations as A
import cv2
import numpy as np
from data_schedule.utils.segmentation import bounding_box_from_mask
from .vis_aug_utils import VIS_TRAIN_AUG_REGISTRY, get_size_with_aspect_ratio, get_tgt_size, \
    pil_torch_to_numpy, numpy_to_pil_torch

import copy

# video: list[pil]
# masks: n t' h w
# has_ann: t

class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ret):
        if random.random() < self.p:
            video = ret['video']
            w, h = video[0].size
            flipped_video = [F.hflip(frame) for frame in video]
            ret['video'] = flipped_video

            if "masks" in ret:
                VIS_Aug_CallbackAPI
                ret['masks'] = ret['masks'].flip(-1)

            if 'boxes' in ret:
                VIS_Aug_CallbackAPI
                boxes = ret["boxes"] # n t' 4, x1y1x2y2
                valid_box_idxs = boxes.any(-1, keepdim=True) # n t 1
                # n t' (-x2+w y1 -x1+w y2)
                boxes = boxes[:, :, [2, 1, 0, 3]] * (torch.tensor([-1, 1, -1, 1])[None, None, :]) + torch.tensor([w, 0, w, 0])[None, None, :] 
                boxes = torch.where(valid_box_idxs.repeat(1, 1, 4), boxes, torch.zeros_like(boxes))
                ret['boxes'] = boxes
        return ret

class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ret):
        if random.random() < self.p:
            video = ret['video']
            w, h = video[0].size
            flipped_video = [F.vflip(frame) for frame in video]
            ret['video'] = flipped_video

            if "masks" in ret:
                VIS_Aug_CallbackAPI
                ret['masks'] = ret['masks'].flip(-2)

            if 'boxes' in ret:
                VIS_Aug_CallbackAPI
                boxes = ret["boxes"] # n t' 4, x1y1x2y2
                valid_box_idxs = boxes.any(-1, keepdim=True) # n t 1
                # n t' (x1 h-y2 x2 h-y1)
                boxes = boxes[:, :, [0, 3, 2, 1]] * (torch.tensor([1, -1, 1, -1])[None, None, :]) + torch.tensor([0, h, 0, h])[None, None, :] 
                boxes = torch.where(valid_box_idxs.repeat(1, 1, 4), boxes, torch.zeros_like(boxes))
                ret['boxes'] = boxes
         
        return ret

class RandomResize:
    def __init__(self, sizes, max_size=None):
        """
        Input:  
            - sizes: 
                list of (w_final, h_final): 
                    你就是想resize到这些大小, 此时max_size不起作用
                list of (size of shorter side):
                    保持ratio进行resize
                    给出较短边的目标长度,
                        如果max_size没有给出, 则就是将较短边resize到这个长度, 较长边resize到对应大小
                        如果max_size给出, 则是将短边resize到该长度, 如果此时较长边超过了max_size,则按照 较长边放大大max_size 进行resize
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, ret):
        video = ret['video']
        orig_size = video[0].size # w h
        tgt_size = get_tgt_size(video[0].size, random.choice(self.sizes), self.max_size) # h w

        resized_video = [F.resize(frame, tgt_size) for frame in video]
        ratio_width, ratio_height = tuple(float(s) / float(s_orig) for s, s_orig in zip(tgt_size[::-1], orig_size))
        ret['video'] = resized_video

        if "masks" in ret:
            VIS_Aug_CallbackAPI
            masks = ret['masks']
            masks = torch.nn.functional.interpolate(masks.float(), tgt_size, mode='nearest').bool()
            ret['masks'] = masks

        if "boxes" in ret:
            VIS_Aug_CallbackAPI
            boxes = ret["boxes"] # n t' x1y1x2y2
            scaled_boxes = boxes * (torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])[None, None, :])
            ret["boxes"] = scaled_boxes
        
        return ret
    
class RandomRotate90:
    def __init__(self) -> None:
        self.album_aug = A.ReplayCompose(
            [A.RandomRotate90(0.5)]
        )
    
    def __call__(self, ret):
        video = ret['video'] 
        masks = ret['masks'] 
        has_ann = ret['has_ann']
        # list[PIL], n t' h w -> 
        # list[h w 3, 255rgb], t
        # list[list[h w, 01uint8]] t
        video, masks = pil_torch_to_numpy(video=video, masks=masks, has_ann=has_ann)
        replay = self.album_aug(image=video[0], mask=[masks[0][0]])['replay']
        auged_video = []
        auged_mask = []
        for vid, mk in zip(video, masks):
            ret = self.album_aug.replay(replay, image=vid, mask=mk)
            auged_video.append(ret['image'])
            auged_mask.append(ret['mask'])
        
        auged_video, auged_mask = numpy_to_pil_torch(video=auged_video, auged_mask=auged_mask, has_ann=has_ann)

        ret['video'] = auged_video
        ret['mask'] = auged_mask

        return ret

class ComputeBox:
    def __call__(self, ret):
        W, H = ret['video'][0].size
        N, T = ret['masks'].shape[:2] # n t' h w
        boxes = torch.stack([bounding_box_from_mask(mask) for mask in copy.deepcopy(ret['masks']).flatten(0, 1)], dim=0) # Nt' 4
        boxes = rearrange(boxes, '(N T) c -> N T c', N=N, T=T)
        boxes[:, :, 0::2].clamp_(min=0, max=W)
        boxes[:, :, 1::2].clamp_(min=0, max=H)

        ret['boxes'] = boxes

        return ret

class VideoToTensor:
    def __call__(self, ret):
        video = ret['video']
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0) # t 3 h w, float, 0-1
        ret['video'] = tensor_video
        return ret

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ret):
        for t in self.transforms:
            ret = t(ret)
        return ret

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@VIS_TRAIN_AUG_REGISTRY.register()
class WeakPolyP_TrainAug:
    def __init__(self, configs) -> None:
        self.transform = A.ReplayCompose([
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        self.tensor_video = VideoToTensor()
        self.add_box = ComputeBox()

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        video = ret['video'] 
        masks = ret['masks']  # n t' h w
        has_ann = ret['has_ann'] # t
        # list[PIL] -> list[h w 3, 0-1float], t
        # n t' h w -> list[list[h w, 01uint8], 没有annotation的帧box是空] t
        video, masks = pil_torch_to_numpy(video=video, masks=masks, has_ann=has_ann)

        replay = self.transform(image=video[0], masks=[masks[0][0]])['replay']
        auged_video = []
        auged_mask = []
        for vid, mk in zip(video, masks):
            auged_each_frame = self.transform.replay(replay, image=vid, masks=mk)
            auged_video.append(auged_each_frame['image'])
            auged_mask.append(auged_each_frame['masks']) # list[h w, 01uint8]
        
        auged_video, auged_mask = numpy_to_pil_torch(video=auged_video, masks=auged_mask, has_ann=has_ann)

        ret['video'] = auged_video
        ret['masks'] = auged_mask
        
        ret = self.add_box(ret)
        ret = self.tensor_video(ret)

        return ret


@VIS_TRAIN_AUG_REGISTRY.register()
class Fibroid_TrainAug:
    def __init__(self, configs) -> None:
        self.hflip = RandomHFlip(0.5)
        self.resize = RandomResize(sizes=[400, 500, 512, 640], max_size=1333)
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.hflip(ret)
        ret = self.resize(ret)
        ret = self.tensor_video(ret)
        return ret


@VIS_TRAIN_AUG_REGISTRY.register()
class Hflip_RandomResize:
    def __init__(self, configs):
        sizes = configs['sizes'] # list[(w_final, h_final)] / list[短边的目标长度, 保持ratio]
        max_size = configs['max_size']
        flip_prob = configs['flip_prob']
        assert flip_prob > 0
        self.aug =  Compose([
            RandomHFlip(flip_prob),
            RandomResize(sizes, max_size),
            ComputeBox(),
            VideoToTensor(),
        ])

