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

from util.misc import interpolate
import numpy as np
from PIL import Image
import cv2
from einops import rearrange, reduce, repeat
from util.box_ops import box_xyxy_to_cxcywh
class Check(object):
    def __init__(self,):
        pass
    def __call__(self, video, texts, target):
        """
        Input:
            - video:
                list[pillow images of the same shape]
            - texts
                list[str]
            - target:
                dict
        """
        if "boxes" in target or "masks" in target:
            if "masks" in target:
                # n t h w -> n t hw
                keep = target['masks'].flatten(2).any(-1) # n t
            else:
                # n t 4 -> n t 2 2
                cropped_boxes = rearrange(target['boxes'], 'n t (s xy) -> n t s xy', s=2, xy=2)
                # n t 2 > n t 2 -> n t
                keep = torch.all(cropped_boxes[:, :, 1, :] > cropped_boxes[:, :, 0, :], dim=-1) 
            
            target['valid'] = keep.long()
        return  video, texts, target

# TODO: 如果一个crop把右边的物体放到了左边怎么半?
# TODO: 如果一个crop把左边的2个man crop没了怎么办?
# 固定窗口进行crop
def crop(video, texts, target, region):
    """
    Input:
        - video:
            list[pillow iamges of the same shape]
        - texts:
            list[str]
        - region: 
            (i,j,h,w)
    """
    cropped_video = [F.crop(frame, *region) for frame in video]

    i, j, h, w = region

    target["size"] = torch.tensor([len(video), h, w])
    
    if "boxes" in target:
        # n t 4
        boxes = target["boxes"]
        num_instance, nf, _ = boxes.shape
        boxes = boxes.flatten(0, 1) # nt 4
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        # nt 2 2
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        # area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = rearrange(cropped_boxes.reshape(-1, 4), '(n t) c -> n t c', n=num_instance, t=nf)

    if "masks" in target:
        # n t h w
        target['masks'] = target['masks'][:, :, i:i + h, j:j + w]

    return cropped_video, texts, target


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


def resize(video, texts, target, size, max_size=None):

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

    size = get_size(video[0].size, size, max_size)
    rescaled_video = [F.resize(frame, size) for frame in video]
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_video[0].size, rescaled_video[0].size))
    ratio_width, ratio_height = ratios
    
    h, w = size
    
    target["size"] = torch.tensor([len(video), h, w])
    if "boxes" in target:
        boxes = target["boxes"] 
        # n t (x1*rw y1*rh x2*rw y2*rh)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "masks" in target:
        # n t h w
        target['masks'] = interpolate(
            target['masks'].float(), size, mode="nearest") > 0.5

    return rescaled_video, texts, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, video, texts,  target):
        region = T.RandomCrop.get_params(video[0], self.size) # (i, j, h, w)
        return crop(video, texts, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        # w = [min_size, min(max_size,图片的width)]
        # h = [min_size, min(max_size, 图片的height)
        self.min_size = min_size 
        self.max_size = max_size

    def __call__(self, video, texts,  target):
        """
        Input:
            video: 
                list[pillow images of the same shape]
        """
        
        w = random.randint(self.min_size, min(video[0].width, self.max_size)) 
        h = random.randint(self.min_size, min(video[0].height, self.max_size)) 
        region = T.RandomCrop.get_params(video[0], [h, w]) # (i, j, h, w)
        return crop(video, texts,  target, region)


class CenterCrop(object):
    def __init__(self, size):
        # h w
        self.size = size
        # i, j的计算是以中心点向两边 h/2 w/2
        # i = H/2 - h/2

    def __call__(self, video, texts, target):
        image_width, image_height = video[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(video, texts,  target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video, texts,  target):
        """
        Input:
            - video:
                list[pillow imags of the smae shape], nf
        """
        if random.random() < self.p:
            return hflip(video, texts,  target)
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
        return resize(video, texts, target, size, self.max_size)




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
        tensor_video = [F.to_tensor(frame) for frame in video]
        return tensor_video, texts, target

class Normalize(object):
    """
    normalize tensor images and boxes
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video, texts, target):
        """
        Input:
            images: list[ T(n 3 h w) ]
            target: list[none of dict]
        """
        assert type(video[0]) is torch.Tensor
        normalized_images = [F.normalize(frame, mean=self.mean, std=self.std) for frame in video]
        H, W = normalized_images[0].shape[-2:]
        if 'boxes' in target:
            boxes = target["boxes"] # n t (x1 y1 x2 y2)
            boxes = box_xyxy_to_cxcywh(boxes) # n t (cx cy w h)
            boxes = boxes / torch.tensor([W, H, W, H], dtype=torch.float32)
            target['boxes'] = boxes
        return normalized_images, texts, target


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


# def pad(clip, target, padding):
#     # padding: [w_pad, h_pad], bottom-right
#     padded_image = []
#     for image in clip:
#         padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
    
#     # should we do something wrt the original size?
#     target["size"] = torch.tensor(padded_image[0].size[::-1])
#     if "masks" in target:
#         # n t h w
#         target['masks'] = F.pad(target['masks'].long(), (0, 0, padding[0], padding[1])) > 0.5
#     return padded_image, target
# class RandomPad(object):
#     def __init__(self, max_pad):
#         self.max_pad = max_pad

#     def __call__(self, video, target):
#         pad_x = random.randint(0, self.max_pad)
#         pad_y = random.randint(0, self.max_pad)
#         return pad(video, target, (pad_x, pad_y))


# class RandomErasing(object):

#     def __init__(self, scale, ratio, value=None):
#         self.scale = scale
#         self.ratio = ratio
#         self.value = value

#     def __call__(self, video, target):
#         i,j,h,w,v = T.RandomErasing.get_params(video[0],scale=self.scale,value=self.value )
#         erased_imgs = [T.RandomErasing(img,)]

# 不能改变颜色
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    
    def __call__(self,clip,target):
        imgs = []
        
        target = None
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
            
        return imgs, targets

class RandomHue(object): #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target
    
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, target):
        
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, target

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target
    
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target
    
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, target
    

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image