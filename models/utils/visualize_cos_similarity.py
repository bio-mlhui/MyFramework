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

color_gt = MetadataCatalog.get('foo').set(stuff_classes = [str(idx) for idx in range(27)],
                                               stuff_colors = rbg_colors)
import time
import cv2 as cv2
def visualize_cos_similarity(image, 
                             sampled_points,
                             similarities,):
    """
    image: 0-1, 3 h w, float
    sampled_points: N 2, 0-H-1, H, W
    similarities: N h w, [-1,1], float, -1=blue, 1=red
    save_dir: ../image_id.png
    """
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    
    point_images = []
    superimposed_imgs = []
    for point, sim in zip(sampled_points, similarities):
        # 2, h w,
        istce_canvas = MyVisualizer(img_rgb=image, metadata=None, instance_mode=ColorMode.SEGMENTATION)
        istce_canvas.draw_circle(circle_coord=point.tolist()[::-1], color=(1.0, 0, 0), radius=10)
        istce_canvas = istce_canvas.get_output()
        point_image =  torch.from_numpy(istce_canvas.get_image())  # h w 3

        heatmap = ((sim + 1) / 2).clamp(0, 1).numpy()
        heatmap = np.uint8(255 * heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        superimposed_img = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)
        superimposed_img = torch.from_numpy(np.asarray(superimposed_img)) # h w 3
        
        point_images.append(point_image) # 
        superimposed_imgs.append(superimposed_img)
 
        
    whole_image = torch.cat([torch.cat(point_images, dim=1), torch.cat(superimposed_imgs, dim=1)], dim=0)
    return whole_image
    # Image.fromarray(whole_image.numpy()).save(save_dir)
    
       
