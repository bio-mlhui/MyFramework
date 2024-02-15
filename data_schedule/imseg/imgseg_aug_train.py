# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import torch
import torchvision.transforms.functional as F

from utils.misc import interpolate
from einops import rearrange

_imgseg_trainaug_entrypoints = {}

def register_imgseg_trainaug(fn):
    aug_name = fn.__name__
    _imgseg_trainaug_entrypoints[aug_name] = fn

    return fn

def imgseg_trainaug_entrypoints(aug_name):
    try:
        return _imgseg_trainaug_entrypoints[aug_name]
    except KeyError as e:
        print(f'imgseg Eval Augmentation {aug_name} not found')

def resize(image, target, tgt_size):
    scaled_image = F.resize(image, tgt_size)
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(scaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    h, w = tgt_size
    target["size"] = torch.tensor([h, w])
    if "boxes" in target:
        boxes = target["boxes"] 
        # n (x1*rw y1*rh x2*rw y2*rh)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "masks" in target:
        # n h w
        target['masks'] = (interpolate(target['masks'].unsqueeze(1).float(), size=tgt_size, mode="nearest") > 0.5)[:, 0]

    return scaled_image, target

import numpy as np
import torchvision.transforms as T
class ResizeScale:
    """
    把目标size(h, w) 随机缩放, 当成新的目标大小, 然后计算h, w要缩放的倍数, 选择最小的进行保持ratio缩放
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
    ):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image, targets):
        scale = np.random.uniform(self.min_scale, self.max_scale)
        input_size = image.shape[:2]

        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        image, targets = resize(image, output_size)
        return image, targets 

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        region = T.RandomCrop.get_params(video[0], self.size) # (i, j, h, w)
        return crop(video, texts, target, region)

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


class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(
        self,
        crop_size: Tuple[int],
        pad: bool = True,
        pad_value: float = 128.0,
        seg_pad_value: int = 255,
    ):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        super().__init__()
        self._init(locals())

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0,
            0,
            pad_size[1],
            pad_size[0],
            original_size[1],
            original_size[0],
            self.pad_value,
            self.seg_pad_value,
        )

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.data.transforms import (
    RandomCrop,
    RandomFlip,
    ResizeScale,
    FixedSizeCrop,
    ResizeShortestEdge,
)

@register_imgseg_trainaug
def resize_crop_ssd_hflip(configs):
    assert isinstance(configs['min_size_train'], list)
    return [
        ResizeShortestEdge(configs['min_size_train'], max_size=configs['max_size_train'], sample_style=configs['sample_stype']),
        RandomCrop(configs['crop_type'], configs['crop_size']),
        ColorAugSSDTransform(configs['format']),
        RandomFlip(),]

@register_imgseg_trainaug
def hflip_lsj_crop(configs):
    image_size = configs['image_size']
    return Compose([
        RandomFlip(),
        ResizeScale(min_scale=configs['min_scale'],
                    max_scale=configs['max_scale'],
                    target_height=image_size,
                    target_width=image_size),
        FixedSizeCrop(crop_size=(image_size, image_size))]
    )