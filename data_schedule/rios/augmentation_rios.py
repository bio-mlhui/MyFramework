
from typing import Any
import data_schedule.rios.transforms_rios as T
import torch

_image_aug_entrypoints = {}

def register_aug(fn):
    aug_name = fn.__name__
    _image_aug_entrypoints[aug_name] = fn

    return fn


def image_aug_entrypoints(aug_name):
    try:
        return _image_aug_entrypoints[aug_name]
    except KeyError as e:
        print(f'image Augmentation {aug_name} not found')

class ImgText_Transform:
    def __call__(self, image, texts, targets):
        
        # for a2d, the longer edge is always width.
        # if the width is longer than train_max_size
        # then this will resize the frames and masks, not changing raio, along the width to train_max_size
        # e.g for a (320, 720) image, this will goes to (320*576/720, 576) when train_max_size is 576
        
        image, texts, targets = self.transform(image, texts, targets) # 3 h w
        return image, texts, targets
    
class FixSize(ImgText_Transform):
    def __init__(self, 
                 fixsize,):
        self.transform = T.Compose([
            T.RandomResize([fixsize]),
            T.ToTensor()
        ])
      

class Hflip_FixSize(ImgText_Transform):
    def __init__(self, 
                 fixsize,
                 ):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([fixsize]), 
            T.ToTensor()           
        ])


class No(ImgText_Transform):
    def __init__(self):
        self.transform = T.Compose([
            T.ToTensor()           
        ])

@register_aug
def fixsize(configs):
    return FixSize(fixsize=configs['fixsize'])

@register_aug
def hflip_fixsize(configs):
    return Hflip_FixSize(fixsize=configs['fixsize'])

@register_aug
def no(configs):
    return No()



