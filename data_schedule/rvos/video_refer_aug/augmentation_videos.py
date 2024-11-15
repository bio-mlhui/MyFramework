
from typing import Any
import data_schedule.rvos.video_refer_aug.transforms_videos as T
import torch

_video_refer_aug_entrypoints = {}

def register_aug(fn):
    aug_name = fn.__name__
    _video_refer_aug_entrypoints[aug_name] = fn

    return fn


def video_refer_aug_entrypoints(aug_name):
    try:
        return _video_refer_aug_entrypoints[aug_name]
    except KeyError as e:
        print(f'Video Augmentation {aug_name} not found')

class VideoText_Transforms:
    def __call__(self, source_frames, texts, targets):
        
        # for a2d, the longer edge is always width.
        # if the width is longer than train_max_size
        # then this will resize the frames and masks, not changing raio, along the width to train_max_size
        # e.g for a (320, 720) image, this will goes to (320*576/720, 576) when train_max_size is 576
        
        source_frames, texts, targets = self.transform(source_frames, texts, targets)
        source_frames = torch.stack(source_frames, dim=0)  # nf c h w
        return source_frames, texts, targets

class FixSize(VideoText_Transforms):
    def __init__(self, 
                 fixsize,):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomResize([fixsize]),
            normalize,
            
        ])
      

class Hflip_FixSize(VideoText_Transforms):
    def __init__(self, 
                 fixsize,
                 ):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([fixsize]),
            normalize,
            
        ])

class Hflip_ReSizeSmaller(VideoText_Transforms):
    def __init__(self, 
                 sizes,
                 max_size,
                 ):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(sizes=sizes,
                           max_size=max_size),
            normalize,
            
        ])


class ReSizeSmaller(VideoText_Transforms):
    def __init__(self, 
                 sizes,
                 max_size,):
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomResize(sizes=sizes,
                           max_size=max_size),
            normalize,
        ])
                       
class Hflip_ResizeAndCrop(VideoText_Transforms):
    def __init__(self,  
                 train_max_size=640, 
                 scales=[288, 320, 352, 392, 416],
                 crop_range=[384, 576],
                 ):
        # sizes: 
        # list of (w_final, h_final): 
        #     你就是想resize到这些大小, 此时max_size不起作用
        # list of (size of shorter side):
        #     保持ratio进行resize
        #     给出较短边的目标长度,
        #         如果max_size没有给出, 则就是将较短边resize到这个长度, 较长边resize到对应大小
        #         如果max_size给出, 则是将短边resize到该长度, 如果此时较长边超过了max_size,则按照 较长边放大到max_size 进行resize
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(sizes=scales, max_size=train_max_size),
            T.RandomSizeCrop(crop_range[0], crop_range[1]),
            normalize
        ])
 
class JustNormalize(VideoText_Transforms):
    def __init__(self):
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])                

@register_aug
def justnormalize(configs):
    return JustNormalize()

@register_aug
def fixsize(configs):
    return FixSize(fixsize=configs['fixsize'])

@register_aug
def hflip_fixsize(configs):
    train_transform = Hflip_FixSize(fixsize=configs['fixsize'])
    return train_transform

@register_aug
def hflip_ResizeSmaller(configs):
    return Hflip_ReSizeSmaller(sizes=configs['sizes'],
                                max_size=configs['max_size'])

@register_aug
def resizeSmaller(configs):
    return ReSizeSmaller(sizes=configs['sizes'],
                         max_size=configs['max_size'])

@register_aug
def hflip_resize_and_crop(configs):
    train_transform = Hflip_ResizeAndCrop(train_max_size=configs['train_max_size'], train=True,
                                          scales=configs['scales'],
                                          crop_range=configs['crop_range'])
    return train_transform