# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import torch
import torchvision.transforms.functional as F

from utils.misc import interpolate
from einops import rearrange

_video_refer_aug_entrypoints = {}

def register_video_refer_aug(fn):
    aug_name = fn.__name__
    _video_refer_aug_entrypoints[aug_name] = fn

    return fn

def video_refer_aug_entrypoints(aug_name):
    try:
        return _video_refer_aug_entrypoints[aug_name]
    except KeyError as e:
        print(f'VideoText Eval Augmentation {aug_name} not found')

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

def get_tgt_size(image_size, size, max_size=None):
    # if size is like [w, h], then just scale the images to that(会改变长短比)
    if isinstance(size, (list, tuple)):
        return size[::-1]
    # else if size is a number (短边的目标长度), then we need to determine the final size（固定长短比)
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)


# class Check(object):
#     def __init__(self,):
#         pass
#     def __call__(self, video, texts, target):
#         if "boxes" in target or "masks" in target:
#             if "masks" in target:
#                 # n t h w -> n t hw
#                 keep = target['masks'].flatten(2).any(-1) # n t
#             else:
#                 # n t 4 -> n t 2 2
#                 boxes = rearrange(target['boxes'], 'n t (s xy) -> n t s xy', s=2, xy=2)
#                 # n t 2 > n t 2 -> n t
#                 keep = torch.all(boxes[:, :, 1, :] > boxes[:, :, 0, :], dim=-1) 
#             target['valid'] = keep.bool()
#         return  video, texts, target


def hflip(video, texts, target):
    """
    水平翻转每帧图像, 并且将对应所有object的text query进行翻转
    Input: 
        - images: list of Pillow images
            list[pillow iamges of the same shape]
        - targets:
            dict
    """
    flipped_video = [F.hflip(frame) for frame in video]

    w, h = video[0].size
    
    texts =[q.replace('left', '@').replace('right', 'left').replace('@', 'right') for q in texts]
    
    if 'boxes' in target:
        # n t (x1 y1 x2 y2)
        boxes = target["boxes"]
        # n t (-x2+w y1 -x1+w y2)
        boxes = boxes[:, :, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        
    if "masks" in target:
        # n t h w
        target['masks'] = target['masks'].flip(-1)

    return flipped_video, texts, target


def resize(video, texts, target, size):
    rescaled_video = [F.resize(frame, size) for frame in video]
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_video[0].size, video[0].size))
    ratio_width, ratio_height = ratios
    
    if "boxes" in target:
        boxes = target["boxes"] 
        # n t (x1*rw y1*rh x2*rw y2*rh)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "masks" in target:
        # n t h w
        target['masks'] = interpolate(target['masks'].float(), size, mode="nearest") > 0.5

    return rescaled_video, texts, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video, texts, target):
        """
        Input:
            - video:
                list[pillow imags of the smae shape], nf
        """
        if random.random() < self.p:
            return hflip(video, texts, target)
        return video, texts, target


class RandomResize(object):
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

    def __call__(self, video, texts,  target):
        """
        Input:  
            - video:
                list[pillow image]
            - target:   
                list[dict or none]
        """
        size = random.choice(self.sizes)
        tgt_size = get_tgt_size(video[0].size, size, self.max_size)
        return resize(video, texts, target, tgt_size)


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, video, texts, target):
        if random.random() < self.p:
            return self.transforms1(video, texts, target)
        return self.transforms2(video, texts, target)


class ToTensor(object):
    def __call__(self, video, texts, target):
        # list[pil] -> t 3 h w
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0)
        return tensor_video, texts, target

# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, video, texts, target):
#         assert type(video) is torch.Tensor
#         normalized_images = F.normalize(video, mean=self.mean, std=self.std)
#         return normalized_images, texts, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video, texts, target):
        for t in self.transforms:
            video, texts, target = t(video, texts, target)
        return video, texts, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    

# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
@register_video_refer_aug
def no_aug(configs):
    return Compose([
        ToTensor(),
    ])

@register_video_refer_aug
def fixsize(configs):
    fixsize = configs['fixsize'] # [[360, 584]]
    assert len(fixsize) == 1
    assert isinstance(fixsize[0], (list, tuple))
    assert len(fixsize[0]) == 2 # w, h
    return Compose([
        RandomResize(fixsize),
        ToTensor(),
    ])

@register_video_refer_aug
def hflip_fixsize(configs):
    fixsize = configs['fixsize']
    assert len(fixsize) == 1
    assert isinstance(fixsize[0], (list, tuple))
    assert len(fixsize[0]) == 2 # w, h
    return Compose([
        RandomHorizontalFlip(0.5),
        RandomResize(fixsize),
        ToTensor(),
    ])

@register_video_refer_aug
def hflip_randomResize(configs):
    sizes = configs['sizes'] # list[(w_final, h_final)] / list[短边的目标长度, 保持ratio]
    max_size = configs['max_size']
    return Compose([
        RandomHorizontalFlip(0.5),
        RandomResize(sizes, max_size),
        ToTensor(),
    ])

@register_video_refer_aug
def randomResize(configs):
    sizes = configs['sizes'] # list[(w_final, h_final)] / list[短边的目标长度, 保持ratio]
    max_size = configs['max_size']
    return Compose([
        RandomResize(sizes, max_size),
        ToTensor()
    ])