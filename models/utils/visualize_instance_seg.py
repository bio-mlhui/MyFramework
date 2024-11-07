import os



from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.visualizer import ColorMode
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import networkx as nx

import os
import numpy as np
import torch
from PIL import Image

class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color

from detectron2.data import MetadataCatalog

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f,)).convert('RGB') for f in frames]

# t' h w, int, obj_ids ;  has_ann t
def get_frames_mask(mask_path, video_id, frames):
    masks = [Image.open(os.path.join(mask_path, video_id, f)).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()


def generate_instance_canvas_uou(image, H, W, mask, num_classes, dataset_name):
    """pred_mask: h w, score:float"""
    metadata = MetadataCatalog.get(dataset_name)
    istce_canvas = MyVisualizer(img_rgb=image, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    
    pred_masks = []
    scores = []
    pred_classes = []
    unique_classes = mask.unique().tolist()
    unique_classes = list(set(unique_classes) - set([255]))
    for clss in unique_classes:
        pred_masks.append((mask==clss).unsqueeze(0))
        scores.append(1)
        pred_classes.append(clss)
    pred_masks = torch.cat(pred_masks, dim=0) 
    scores = torch.tensor(scores)
    pred_classes = torch.tensor(pred_classes)
        
    istce = Instances([H, W], 
        pred_masks=pred_masks, # nq H W
        scores=scores, # nq
        pred_classes=pred_classes # nq
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()


rbg_colors = [
    (156, 31, 23),
    (58, 90, 221),
    (223, 123, 119),
    (46, 140, 40),
    (201, 221, 213),
    (222, 32, 106),
    (145, 70, 230),
    (131, 225, 124),
    (29, 88, 111),
    (157, 113, 153),
    (31, 196, 212),
    (52, 32, 38),
    (156, 238, 33),
    (145, 135, 47),
    (102, 50, 128),
    (210, 145, 215),
    (218, 215, 141),
    (145, 30, 84),
    (226, 40, 207),
    (212, 195, 48),
    (84, 144, 146),
    (51, 29, 193),
    (68, 213, 30),
    (212, 98, 34),
    (162, 23, 188),
    (112, 202, 216),
    (44, 214, 110)
]

color_gt = MetadataCatalog.get('color_gt').set(thing_classes = [str(idx) for idx in range(27)],
                                               thing_colors = rbg_colors)
import time
def visualize_cluster(image, gt, pred, num_classes, save_dir):
    # 3 h w, h w
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    random_name = f'color_pred_{time.time()}'
    MetadataCatalog.get(random_name).set(thing_classes = [str(idx) for idx in range(num_classes)], thing_colors = rbg_colors[:num_classes])
    
    # detectron2
    gt_image = torch.from_numpy(generate_instance_canvas_uou(image=image,  H=H, W=W, mask=gt, num_classes=27, dataset_name='color_gt',))  
    pred_image = torch.from_numpy(generate_instance_canvas_uou(image=image,  H=H, W=W, mask=pred, num_classes=num_classes, dataset_name=random_name,))  
    
    whole_image = torch.cat([torch.from_numpy(image), gt_image, pred_image], dim=1)
    
    Image.fromarray(whole_image.numpy()).save(save_dir)
    

def generate_instance_canvas(image, H, W, mask,dataset_name):
    """pred_mask: h w, score:float"""
    metadata = MetadataCatalog.get(dataset_name)
    istce_canvas = MyVisualizer(img_rgb=image, metadata=metadata, instance_mode=ColorMode.SEGMENTATION)
    
    pred_masks = mask # n h w
    scores = torch.ones(pred_masks.shape[0])
    pred_classes = torch.arange(pred_masks.shape[0]).int()

    istce = Instances([H, W], 
        pred_masks=pred_masks, # nq H W
        scores=scores, # nq
        pred_classes=pred_classes # nq
    )
    istce_canvas.draw_instance_predictions(istce)
    istce_canvas = istce_canvas.get_output()
    return istce_canvas.get_image()

def cls_ag_visualize_pred(image, pred, save_dir):
    """
    image: 3 h w, 0-1
    pred: n h w, bool
    """
    H, W = image.shape[-2:]
    num_classes = pred.shape[0]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    random_name = f'color_pred_{time.time()}'
    MetadataCatalog.get(random_name).set(thing_classes = [str(idx) for idx in range(num_classes)], thing_colors = rbg_colors[:num_classes])
    
    pred_image = torch.from_numpy(generate_instance_canvas(image=image,  H=H, W=W, mask=pred, dataset_name=random_name,))  
    
    whole_image = torch.cat([torch.from_numpy(image), pred_image], dim=1)
    
    Image.fromarray(whole_image.numpy()).save(save_dir)
         
