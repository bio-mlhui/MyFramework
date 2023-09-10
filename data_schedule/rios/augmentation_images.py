
from typing import Any
import datasets.rios.transforms_images as T
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

class Image_Transforms:
    def __call__(self, images, targets):
        
        # for a2d, the longer edge is always width.
        # if the width is longer than train_max_size
        # then this will resize the frames and masks, not changing raio, along the width to train_max_size
        # e.g for a (320, 720) image, this will goes to (320*576/720, 576) when train_max_size is 576
        
        images, targets = self.transform(images, targets)
        return images, targets
    
class COCO_HFIP_ResizeCrop_Train(Image_Transforms):
    def __init__(self, 
                 scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
                 final_scales = [296, 328, 360, 392, 416, 448, 480, 512] ,
                 max_size=800,
                 respect_boxes=False,
                 flip=0.5) -> None:
        
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomHorizontalFlip(flip), 
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose(
                    [
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600, respect_boxes=respect_boxes),
                        T.RandomResize(final_scales, max_size=640),
                    ]
                ),
            ),
            normalize,
        ])

class COCO_Resize_Eval(Image_Transforms):
    def __init__(self, resize) -> None:
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

class COCO_FixSize_Eval(Image_Transforms):
    def __init__(self, fixsize) -> None:
        normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transform = T.Compose([
            T.RandomResize(fixsize),
            normalize,
        ])      
          

@register_aug
def COCO_HFLIP_ResizeCrop(configs):
    configs = vars(configs)
    train_aug = COCO_HFIP_ResizeCrop_Train(configs['scales'],
                                            configs['final_scales'],
                                            configs['max_size'],
                                            configs['flip'])
    
    eval_aug = COCO_Resize_Eval(configs['eval_resize'])
    
    return train_aug, eval_aug



