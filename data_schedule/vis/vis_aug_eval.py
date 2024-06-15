# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from copy import deepcopy as dcopy
from data_schedule.vis.apis import VIS_Aug_CallbackAPI
from .vis_aug_utils import get_size_with_aspect_ratio, get_tgt_size, pil_torch_to_numpy, numpy_to_pil_torch

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from .vis_aug_utils import VIS_EVAL_AUG_REGISTRY

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ret):
        if random.random() < self.p:
            video = ret['video']
            w, h = video[0].size
            flipped_video = [F.hflip(frame) for frame in video]
            ret['video'] = flipped_video

            if 'pred_boxes' in ret:
                VIS_Aug_CallbackAPI
                # list[nt 4], t
                boxes = ret["pred_boxes"] 
                boxes = [bx[:, [2, 1, 0, 3]] * (torch.tensor([-1, 1, -1, 1])[None, :]) + torch.tensor([w, 0, w, 0])[None, :] 
                         for bx in boxes]
                ret['pred_boxes'] = boxes

            if "pred_masks" in ret:
                VIS_Aug_CallbackAPI
                # lis[nt h w], t
                ret['pred_masks'] = [mk.flip(-1) for mk in ret['pred_masks']] 

            if 'callback_fns' in ret:
                VIS_Aug_CallbackAPI
                ret['callback_fns'].append(RandomHorizontalFlip(1.))
         
        return ret

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ret):
        if random.random() < self.p:
            video = ret['video']
            w, h = video[0].size
            flipped_video = [F.vflip(frame) for frame in video]
            ret['video'] = flipped_video

            if 'pred_boxes' in ret:
                VIS_Aug_CallbackAPI
                # list[nt 4], t
                boxes = ret["pred_boxes"] 
                boxes = [bx[:, [0, 3, 2, 1]] * (torch.tensor([1, -1, 1, -1])[None, :]) + torch.tensor([0, h, 0, h])[None, :] 
                         for bx in boxes]
                ret['pred_boxes'] = boxes

            if "pred_masks" in ret:
                VIS_Aug_CallbackAPI
                # lis[nt h w], t
                ret['pred_masks'] = [mk.flip(-2) for mk in ret['pred_masks']] 

            if 'callback_fns' in ret:
                VIS_Aug_CallbackAPI
                ret['callback_fns'].append(RandomVerticalFlip(1.))
         
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

        if 'callback_fns' in ret:
            VIS_Aug_CallbackAPI
            ret['callback_fns'].append(RandomResize(sizes=[orig_size], max_size=None))

        if "pred_masks" in ret:
            assert (len(self.sizes) == 1) and (self.max_size == None)
            VIS_Aug_CallbackAPI
            pred_masks = ret['pred_masks'] # list[nt h w], t
            pred_masks = [torch.nn.functional.interpolate(mk.unsqueeze(0).float(), tgt_size, mode='nearest')[0].bool()
                          for mk in pred_masks]
            ret['pred_masks'] = pred_masks # list[nt h w], t

        if "pred_boxes" in ret:
            VIS_Aug_CallbackAPI
            pred_boxes = ret["pred_boxes"] # list[nt 4], t
            scaled_boxes = [bx * (torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])[None, :])
                            for bx in pred_boxes]
            ret["pred_boxes"] = scaled_boxes

        return ret
    
class VideoToPIL:
    def __call__(self, ret):
        video = ret['video'] # t 3 h w ->
        assert video.dtype == torch.float and (video.max() <= 1) and (video.min() >=0)  
        pil_video = [F.to_pil_image(frame, mode='RGB') for frame in video] # 3 h w, float, 0-1
        ret['video'] = pil_video
        assert 'callback_fns' not in ret
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

class VideoToTensor:
    def __call__(self, ret):
        video = ret['video']
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0) # t 3 h w, float, 0-1
        ret['video'] = tensor_video

        if 'callback_fns' in ret:
            VIS_Aug_CallbackAPI
            ret['callback_fns'].append(VideoToPIL())

        return ret
        

@VIS_EVAL_AUG_REGISTRY.register()
class WeakPolyP_EvalAug:
    def __init__(self, configs) -> None:
        size = configs.pop('size', 352)
        self.resize = RandomResize(
            sizes=[[size, size]],
        )
        self.tensor_video = VideoToTensor() # 先进行tensor

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret

@VIS_EVAL_AUG_REGISTRY.register()
class TimeSMamba_EvalAug:
    def __init__(self, configs) -> None:
        self.resize = RandomResize(
            sizes=[[384, 384]],
        )
        self.tensor_video = VideoToTensor() # 先进行tensor

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret

@VIS_EVAL_AUG_REGISTRY.register()
class Visha_EvalAug:
    def __init__(self, configs) -> None:
        sizes = configs['sizes'] if 'max_size' in configs else [[512,512]]
        max_size = configs['max_size'] if 'max_size' in configs else None
        self.resize = RandomResize(
            sizes=sizes,
            max_size=max_size
        )
        self.tensor_video = VideoToTensor() # 先进行tensor

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret





@VIS_EVAL_AUG_REGISTRY.register()
class Fibroid_EvalAug:
    def __init__(self, configs) -> None:
        self.resize = RandomResize(sizes=[512], max_size=1333)
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret