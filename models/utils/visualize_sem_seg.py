import os



from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

from detectron2.utils.visualizer import ColorMode
import torch
import numpy as np
import os

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

def generate_semseg_canvas_uou(image, H, W, mask, num_classes=None, dataset_name=None):
    """pred_mask: h w, score:float"""
    metadata = MetadataCatalog.get(dataset_name)
    istce_canvas = MyVisualizer(img_rgb=image, 
                                metadata=metadata, 
                                instance_mode=ColorMode.SEGMENTATION)
    istce_canvas.draw_sem_seg(mask, alpha=0.9)
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
MetadataCatalog.get('color_gt_27').set(stuff_classes = [str(idx) for idx in range(27)],
                                        stuff_colors = rbg_colors)
import time
def visualize_cluster(image, gt, pred, 
                      num_image_classes, 
                      num_gt_classes,):
    """
    image: 0-1, 3 h w, float
    gt: 0-num_gt_class-1, -1代表背景, h w, long
    pred: 0-num_image_class-1, h w, int/long
    save_dgtir: ../image_id.png
    """
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    random_name = f'color_pred_{time.time()}'
    MetadataCatalog.get(random_name).set(stuff_classes = [str(idx) for idx in range(num_image_classes)],
                                         stuff_colors = rbg_colors[:num_image_classes])
    if num_gt_classes == 27:
        gt_metalog_name = 'color_gt_27'
    else:
        raise NotImplementedError()
    
    gt[gt==-1] = (num_image_classes + 2000) # detectron2: for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
    # detectron2
    gt_image = torch.from_numpy(generate_semseg_canvas_uou(image=image, 
                                                           H=H, W=W, mask=gt, num_classes=num_gt_classes, dataset_name=gt_metalog_name,))  
    pred_image = torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=pred, 
                                                             num_classes=num_image_classes, dataset_name=random_name,))  
    
    whole_image = torch.cat([torch.from_numpy(image), gt_image, pred_image], dim=1)
    return whole_image
    # Image.fromarray(whole_image.numpy()).save(save_dir)
    
       
