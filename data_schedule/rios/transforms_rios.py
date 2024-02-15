# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from numpy import random as rand
import copy
from utils.misc import interpolate
import numpy as np
from PIL import Image
import cv2
from einops import rearrange, reduce, repeat
from util.box_ops import box_xyxy_to_cxcywh


def hflip(image, texts, target):
    """
    水平翻转每帧图像, 并且将对应所有object的text query进行翻转
    Input: 
        - images: list of Pillow images
            list[pillow iamges of the same shape]
        - targets:
            dict
    """
    flipped_image = F.hflip(image)

    w, h = image.size
    
    # 比如有的句子就有@
    old_texts = copy.deepcopy(texts)
    texts = []
    for q in old_texts:
        if ('left' in q) or ('right' in q):
            texts.append(q.replace('left', '@').replace('right', 'left').replace('@', 'right'))
        else:
            texts.append(q)

    if 'boxes' in target:
        # n (x1 y1 x2 y2)
        boxes = target["boxes"]
        # n (-x2+w y1 -x1+w y2)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        
    if "masks" in target:
        # n h w
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, texts, target


def resize(image, texts, target, size, max_size=None):

    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """
        Input:
            - image_size: 图片的原先大小
            - size: 较短边的目标长度
            - max_size: 如果 放大较短边 导致 较长边 大于max_size
        """
        # 保持ratio不变，
        # 让较短边resize到size，如果较长边大于了max_size, 则依照较长边到max_size进行resize
        # 返回最终的大小(h_target, w_target)
        w, h = image_size
        # 确定较短边的最终长度, 防止较长边大于max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        # if size is like [w, h], then just scale the images to that(会改变长短比)
        if isinstance(size, (list, tuple)):
            return size[::-1]
        # else if size is a number (短边的目标长度), then we need to determine the final size（固定长短比)
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    
    h, w = size
    
    target["size"] = torch.tensor([h, w])
    if "boxes" in target:
        boxes = target["boxes"] 
        # n (x1*rw y1*rh x2*rw y2*rh)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "masks" in target:
        # n h w
        target['masks'] = (interpolate(target['masks'].float().unsqueeze(0), size, mode="nearest") > 0.5)[0]

    return rescaled_image, texts, target

class ToTensor(object):
    def __call__(self, image, texts, target):
        tensor_image = F.to_tensor(image)
        return tensor_image, texts, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, texts,  target):
        if random.random() < self.p:
            return hflip(image, texts,  target)
        return image, texts, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, texts,  target):
        """
        Input:  
            - video:
                list[pillow image]
            - target:   
                list[dict or none]
        """
        size = random.choice(self.sizes)
        return resize(image, texts, target, size, self.max_size)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, texts, target):
        for t in self.transforms:
            image, texts, target = t(image, texts, target)
        return image, texts, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string