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
        size = configs.pop('size', 352)
        self.transform = A.ReplayCompose([
            A.Resize(size, size),
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
class TimeSMamba_TrainAug:
    def __init__(self, configs) -> None:
        self.transform = A.ReplayCompose([
            A.Resize(384, 384),
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
class Visha_TrainAug:
    def __init__(self, configs) -> None:
        self.transform = A.ReplayCompose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
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
class Visha_TrainAug_v2:
    def __init__(self, configs):
        self.hflip = RandomHFlip(0.5)
        self.resize = RandomResize(sizes=configs['sizes'], max_size=configs['max_size'])
        self.compute_box = ComputeBox()
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        ret = self.hflip(ret)
        ret = self.resize(ret)
        ret = self.compute_box(ret)
        ret = self.tensor_video(ret)
        return ret


@VIS_TRAIN_AUG_REGISTRY.register()
class WeakPolyP_TrainAug_RotateImageToClip:
    def __init__(self, configs) -> None:
        self.ImageToSeqAugmenter = ImageToSeqAugmenter(perspective=True, affine=True, motion_blur=True,
                                    rotation_range=(-20, 20), perspective_magnitude=0.08,
                                    hue_saturation_range=(-5, 5), brightness_range=(-40, 40),
                                    motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                    translate_range=(-0.1, 0.1))
        self.num_frames = configs['num_frames']
        
        self.transform = A.ReplayCompose([
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        self.tensor_video = VideoToTensor()
        self.add_box = ComputeBox()

    def apply_random_sequence_shuffle(self, images, instance_masks):
        perm = list(range(self.num_frames))
        random.shuffle(perm)
        images = [images[i] for i in perm]
        instance_masks = [instance_masks[i] for i in perm]
        return images, instance_masks
    
    def __call__(self, ret):
        VIS_Aug_CallbackAPI
        video = ret['video']  # list[pil], t
        masks = ret['masks']  # n t' h w
        has_ann = ret['has_ann'] # t

        # list[PIL] -> list[h w 3, uint8], t
        # n t' h w -> list[list[h w], n, uint8], t
        seq_images, seq_instance_masks = pil_torch_to_numpy(video=video, masks=masks, has_ann=has_ann, float_image=False)
        assert len(seq_images) == 1 and len(seq_instance_masks) == 1
        static_img, static_mask = seq_images[0], seq_instance_masks[0]
        for t in range(self.num_frames - 1):
            im_trafo, instance_masks_trafo = self.ImageToSeqAugmenter(static_img, static_mask) # h w 3, uint8; list[h w], n, uint8
            seq_images.append(np.uint8(im_trafo))
            seq_instance_masks.append(instance_masks_trafo)
        # list[h w 3], t ;  # list[list[h w, 01uint8]] t 
        seq_images, seq_instance_masks = self.apply_random_sequence_shuffle(seq_images, seq_instance_masks)         
        has_ann = torch.ones(self.num_frames).bool() # T
        seq_images = [np.float32(haosen) / 255.0 for haosen in seq_images] # list[h w 3, 0-1float], t
        replay = self.transform(image=seq_images[0], masks=[seq_instance_masks[0][0]])['replay']
        auged_video = []
        auged_mask = []
        for vid, mk in zip(seq_images, seq_instance_masks):
            auged_each_frame = self.transform.replay(replay, image=vid, masks=mk)
            auged_video.append(auged_each_frame['image'])
            auged_mask.append(auged_each_frame['masks']) # list[h w, 01uint8]
        
        auged_video, auged_mask = numpy_to_pil_torch(video=auged_video, masks=auged_mask, has_ann=has_ann) # n t h w
        # [haosen.save(f'./test{idx}.png') for idx, haosen in enumerate(auged_video)]
        # import matplotlib.pyplot as plt
        # [plt.imsave( f'./mask{idx}.png', auged_mask[0][idx].float().numpy()) for idx in range(len(auged_mask[0]))]
        ret['video'] = auged_video
        ret['masks'] = auged_mask
        ret['has_ann'] = has_ann
        
        ret = self.add_box(ret)
        ret = self.tensor_video(ret)

        return ret
    
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug
from datetime import datetime
class ImageToSeqAugmenter(object):
    def __init__(self, perspective=True, affine=True, motion_blur=True,
                 brightness_range=(-50, 50), hue_saturation_range=(-15, 15), perspective_magnitude=0.12,
                 scale_range=1.0, translate_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, rotation_range=(-20, 20),
                 motion_blur_kernel_sizes=(7, 9), motion_blur_prob=0.5, seed=2024):

        self.basic_augmenter = iaa.SomeOf((1, None), [
                iaa.Add(brightness_range),
                iaa.AddToHueAndSaturation(hue_saturation_range)
            ]
        )

        transforms = []
        if perspective:
            transforms.append(iaa.PerspectiveTransform(perspective_magnitude))
        if affine:
            transforms.append(iaa.Affine(scale=scale_range,
                                         translate_percent=translate_range,
                                         rotate=rotation_range,
                                         order=1,  # cv2.INTER_LINEAR
                                         backend='auto'))
        transforms = iaa.Sequential(transforms)
        transforms = [transforms]

        if motion_blur:
            blur = iaa.Sometimes(motion_blur_prob, iaa.OneOf(
                [
                    iaa.MotionBlur(ksize)
                    for ksize in motion_blur_kernel_sizes
                ]
            ))
            transforms.append(blur)

        self.frame_shift_augmenter = iaa.Sequential(transforms)
        self.seed = seed
    @staticmethod
    def condense_masks(instance_masks):
        condensed_mask = np.zeros_like(instance_masks[0], dtype=np.int8)
        for instance_id, mask in enumerate(instance_masks, 1):
            condensed_mask = np.where(mask, instance_id, condensed_mask)

        return condensed_mask

    @staticmethod
    def expand_masks(condensed_mask, num_instances):
        return [(condensed_mask == instance_id).astype(np.uint8) for instance_id in range(1, num_instances + 1)]

    def __call__(self, image, masks=None, boxes=None): # n h w
        det_augmenter = self.frame_shift_augmenter.to_deterministic()
        if masks is not None:
            masks_np, is_binary_mask = [], []
            boxs_np = []

            for mask in masks:
                
                if isinstance(mask, np.ndarray):
                    masks_np.append(mask.astype(np.bool_))
                    is_binary_mask.append(False)
                else:
                    raise ValueError("Invalid mask type: {}".format(type(mask)))

            num_instances = len(masks_np)
            masks_np = SegmentationMapsOnImage(self.condense_masks(masks_np), shape=image.shape[:2])
            # boxs_np = BoundingBoxesOnImage(boxs_np, shape=image.shape[:2])
            seed = int(datetime.now().strftime('%M%S%f')[-8:])
            imgaug.seed(seed)
            aug_image, aug_masks = det_augmenter(image=self.basic_augmenter(image=image) , segmentation_maps=masks_np)
            imgaug.seed(seed)
            invalid_pts_mask = det_augmenter(image=np.ones(image.shape[:2] + (1,), np.uint8)).squeeze(2)
            aug_masks = self.expand_masks(aug_masks.get_arr(), num_instances)
            # aug_boxes = aug_boxes.remove_out_of_image().clip_out_of_image()
            aug_masks = [mask for mask, is_bm in zip(aug_masks, is_binary_mask)]
            return aug_image, aug_masks #, aug_boxes.to_xyxy_array()

        else:
            masks = [SegmentationMapsOnImage(np.ones(image.shape[:2], np.bool), shape=image.shape[:2])]
            aug_image, invalid_pts_mask = det_augmenter(image=image, segmentation_maps=masks)
            return aug_image, invalid_pts_mask.get_arr() == 0

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



class RandomReverseVideo:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ret):
        if random.random() < self.p:
            ret['video'] = ret['video'][::-1]

            if "masks" in ret:
                VIS_Aug_CallbackAPI # n t h w
                ret['masks'] = ret['masks'].flip(1).contiguous()
            
            if 'has_ann' in ret:
                ret['has_ann'] = ret['has_ann'].flip(0).contiguous()

            if 'boxes' in ret:
                VIS_Aug_CallbackAPI
                ret["boxes"] = ret['boxes'].flip(1).contiguous()
        return ret

@VIS_TRAIN_AUG_REGISTRY.register()
class Flanet_TrainAug:
    def __init__(self, configs) -> None:
        self.transform = A.ReplayCompose([
            A.Resize(352, 352),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
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
class WeakPolyP_TrainAug_resize:
    def __init__(self, configs) -> None:
        self.resize = RandomResize(sizes=[256, 288, 320, 352, 384, 416, 448], max_size=480)
        self.reverse_video = RandomReverseVideo()
        self.transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
        ])
        self.tensor_video = VideoToTensor()
        self.add_box = ComputeBox()

    def __call__(self, ret):
        ret = self.reverse_video(ret)
        ret = self.resize(ret)
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

