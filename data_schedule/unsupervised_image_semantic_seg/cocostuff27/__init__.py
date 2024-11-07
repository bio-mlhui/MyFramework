# cocostuff27_train
# cocostuff27-IIC_eval


from typing import Optional, Union
import json
import os
from functools import partial
import numpy as np
import torch
import logging
from tqdm import tqdm
import copy
from detectron2.data import DatasetCatalog, MetadataCatalog
from collections import defaultdict
import os
import random
from os.path import join
from copy import deepcopy as dcopy
import numpy as np
import torch.multiprocessing
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import contextlib
import io
import pycocotools.mask as mask_util
import numpy as np

cat_id_map = {91: 22, 92: 22, 93: 16, 94: 3, 95: 3, 96: 16, 97: 10, 98: 21, 99: 17, 100: 6, 101: 4, 102: 4, 103: 22, 104: 22, 105: 18, 106: 10, 107: 10, 108: 22, 109: 10, 110: 11, 111: 10, 112: 21, 113: 6, 114: 6, 115: 6, 116: 6, 117: 6, 118: 16, 119: 25, 120: 8, 121: 8, 122: 10, 123: 16, 124: 11, 125: 11, 126: 19, 127: 3, 128: 16, 129: 10, 130: 22, 131: 17, 132: 10, 133: 16, 134: 19, 135: 11, 136: 22, 137: 21, 138: 17, 139: 11, 140: 22, 141: 16, 142: 17, 143: 11, 144: 11, 145: 21, 146: 11, 147: 25, 148: 11, 149: 19, 150: 3, 151: 22, 152: 8, 153: 11, 154: 25, 155: 10, 156: 18, 157: 3, 158: 11, 159: 19, 160: 10, 161: 19, 162: 16, 163: 21, 164: 10, 165: 3, 166: 22, 167: 22, 168: 16, 169: 8, 170: 24, 171: 24, 172: 24, 173: 24, 174: 24, 175: 24, 176: 24, 177: 25, 178: 25, 179: 26, 180: 26, 181: 19,
              255: -1, 
              0: 15, 1: 23, 2: 23, 3: 23, 4: 23, 5: 23, 6: 23, 7: 23, 8: 23, 9: 14, 10: 14, 12: 14, 13: 14, 14: 14, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 26: 0, 27: 0, 30: 0, 31: 0, 32: 0, 33: 20, 34: 20, 35: 20, 36: 20, 37: 20, 38: 20, 39: 20, 40: 20, 41: 20, 42: 20, 43: 13, 45: 13, 46: 13, 47: 13, 48: 13, 49: 13, 50: 13, 51: 7, 52: 7, 53: 7, 54: 7, 55: 7, 56: 7, 57: 7, 58: 7, 59: 7, 60: 7, 61: 9, 62: 9, 63: 9, 64: 9, 66: 9, 69: 9, 71: 5, 72: 5, 73: 5, 74: 5, 75: 5, 76: 5, 77: 2, 78: 2, 79: 2, 80: 2, 81: 2, 83: 12, 84: 12, 85: 12, 86: 12, 87: 12, 88: 12, 89: 12}

def get_image(path, image_id):
    image = Image.open(os.path.join(path, f'{image_id}.jpg')).convert('RGB')
    return image

def get_image_mask(path, image_id):
    mask = torch.from_numpy(np.array(Image.open(os.path.join(path, f'{image_id}.png')))).long()
    for cat_id in torch.unique(mask):
        mask[mask == cat_id] = cat_id_map[int(cat_id.item())]
    return mask

from detectron2.utils.visualizer import GenericMask

def get_instance_mask(dataset_name, image_id, orig_height, orig_width):
    cocoapi = MetadataCatalog.get(dataset_name).get('coco_instance_api')
    annotations = cocoapi.imgToAnns[int(image_id)] # list[dict]
    if len(annotations) == 0:
        return None
    instance_masks = []
    for anno in annotations:
        segm = anno.get("segmentation", None)
        if segm:  # either list[list[float]] or dict(RLE)
            if isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    segm = mask_util.frPyObjects(segm, *segm["size"])
                    # segm = GenericMask(segm)
                    # if isinstance(segm, dict):
                    #     # RLEs
                    #     assert "counts" in segm and "size" in segm
                    #     if isinstance(segm["counts"], list):  # uncompressed RLEs
                    #         h, w = segm["size"]
                    #         assert h == orig_height and w == orig_width
                    #         m = mask_util.frPyObjects(m, h, w)
                    #     mask = mask_util.decode(segm)[:, :]
                else:
                    raise ValueError()
            else:
                # filter out invalid polygons (< 3 points)
                segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                if len(segm) == 0:
                    continue  # ignore this instance
                # rle = mask_util.frPyObjects(segm, orig_height, orig_width)
                # rle = mask_util.merge(rle)
                # mask = mask_util.decode(rle)[:, :] # H W, uint8
            mask = GenericMask(segm, orig_height, orig_width).mask
            mask = torch.from_numpy(mask).bool()
            instance_masks.append(mask)
    if len(instance_masks) == 0:
        instance_masks = None
    else:
        instance_masks = torch.stack(instance_masks, dim=0)
    return instance_masks


root = os.path.join(os.environ['DATASET_PATH'], 'cocostuff')
visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['cocostuff27_train]'] = [] 
visualize_meta_idxs['cocostuff27_iic_eval'] = [] 
visualize_meta_idxs['cocostuff27_iic_train'] = [] 
# visualize_meta_idxs['cocostuff27_iic_train_meta_fivecrop'] = [] 
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


cocostuff27_meta = {
    'thing_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    'thing_colors': rbg_colors,
    'class_labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    'num_classes': 27,
}

tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'evaluate', 'name': 'cocostuff27_iic_eval',})
tep_meta.update({
    'get_image_fn': partial(get_image, path=os.path.join(root, 'images/val2017',)),
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'annotations/stuffthingmaps_trainval2017/val2017/',),), 
    'get_instance_mask_fn': partial(get_instance_mask, dataset_name='cocostuff27_iic_eval'),
})
def cocostuff27_iic_eval_meta():
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(os.path.join(root, 'annotations/annotations/instances_val2017.json'))
    MetadataCatalog.get('cocostuff27_iic_eval').set(coco_instance_api=coco_api)
    file_list = []
    with open(os.path.join(root, "curated", "val2017", "Coco164kFull_Stuff_Coarse_7.txt"), "r") as f:
        file_list = [fn.rstrip() for fn in f.readlines()]
    return [{'image_id': image_id, 'meta_idx': idx} for idx, image_id in enumerate(file_list)]
DatasetCatalog.register('cocostuff27_iic_eval', cocostuff27_iic_eval_meta)    
MetadataCatalog.get('cocostuff27_iic_eval').set(**tep_meta, 
                                                json_file=os.path.join(root, 'annotations/annotations/instances_val2017.json'),
                                               visualize_meta_idxs=visualize_meta_idxs['cocostuff27_iic_eval_meta']) 


# 97702
tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'train', 'name': 'cocostuff27_iic_train',})
tep_meta.update({
     'get_image_fn': partial(get_image, path=os.path.join(root, 'images/train2017',)),
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'annotations/train2017/',),), 
    'get_instance_mask_fn': partial(get_instance_mask, dataset_name='cocostuff27_iic_train'),
})
def cocostuff27_iic_train_meta():
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(os.path.join(root, 'annotations/annotations/instances_train2017.json'))
    MetadataCatalog.get('cocostuff27_iic_train').set(coco_instance_api=coco_api)
    file_list = []
    with open(os.path.join(root, "curated", "train2017", "Coco164kFull_Stuff_Coarse.txt"), "r") as f:
        file_list = [fn.rstrip() for fn in f.readlines()]
    return [{'image_id': image_id, 'meta_idx': idx} for idx, image_id in enumerate(file_list)]
DatasetCatalog.register('cocostuff27_iic_train', cocostuff27_iic_train_meta)    
MetadataCatalog.get('cocostuff27_iic_train').set(**tep_meta, 
                                                  visualize_meta_idxs=visualize_meta_idxs['cocostuff27_iic_train']) 



tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'train', 'name': 'cocostuff27_train',})
tep_meta.update({
     'get_image_fn': partial(get_image, path=os.path.join(root, 'images/train2017',)),
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'annotations/train2017/',),), 
    'get_instance_mask_fn': partial(get_instance_mask, dataset_name='cocostuff27_train'),
})
def cocostuff27_train_meta():
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(os.path.join(root, 'annotations/annotations/instances_train2017.json'))
    MetadataCatalog.get('cocostuff27_train').set(coco_instance_api=coco_api)
    file_list = os.listdir(os.path.join(root, "images/train2017"))
    return [{'image_id': os.path.splitext(image_id)[0], 'meta_idx': idx} for idx, image_id in enumerate(file_list)]
DatasetCatalog.register('cocostuff27_train', cocostuff27_train_meta)    
MetadataCatalog.get('cocostuff27_train').set(**tep_meta, 
                                            visualize_meta_idxs=visualize_meta_idxs['cocostuff27_train']) 


tep_meta = {'mode': 'evaluate', 
            'name': 'cocoval2017_instance_cls_agnostic',
            'get_image_fn': partial(get_image, path=os.path.join(root, 'images/val2017',)),
            'get_instance_mask_fn': partial(get_instance_mask, dataset_name='cocoval2017_instance_cls_agnostic'),}

def cocoval2017_instance_cls_agnostic():
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(os.path.join(root, 'annotations/annotations/coco_cls_agnostic_instances_val2017.json'))
    MetadataCatalog.get('cocoval2017_instance_cls_agnostic').set(coco_instance_api=coco_api)
    file_list = os.listdir(os.path.join(root, "images/val2017"))
    return [{'image_id': os.path.splitext(image_id)[0], 'meta_idx': idx} for idx, image_id in enumerate(file_list)]
DatasetCatalog.register('cocoval2017_instance_cls_agnostic', cocoval2017_instance_cls_agnostic)    
MetadataCatalog.get('cocoval2017_instance_cls_agnostic').set(**tep_meta, 
                                                json_file=os.path.join(root, 'annotations/annotations/instances_val2017.json'),
                                               visualize_meta_idxs=visualize_meta_idxs['cocoval2017_instance_cls_agnostic_meta']) 

# def get_image_mask_fivecrop(path, image_id):
#     mask = torch.from_numpy(np.array(Image.open(os.path.join(path, f'{image_id}.png')))).long()
#     mask = mask - 1
#     return mask

# def cocostuff27_iic_train_meta_fivecrop():
#     file_list = sorted(os.listdir(os.path.join(root, "cropped", "cocostuff27_five_crop_0.5", "img", "train")), key=lambda x: int(os.path.splitext(x)[0]))
#     return [{'image_id': os.path.splitext(image_id)[0], 'meta_idx': idx} for idx, image_id in enumerate(file_list)]

# # 118287 <-> 97702
# tep_meta = dcopy(cocostuff27_meta)
# tep_meta.update({'mode': 'train', 'name': 'cocostuff27_iic_train_meta_fivecrop',})
# tep_meta.update({
#     'get_image_fn': partial(get_image, path=os.path.join(root, "cropped", "cocostuff27_five_crop_0.5", "img", "train")),
#     'get_mask_fn': partial(get_image_mask_fivecrop, path=os.path.join(root, "cropped", "cocostuff27_five_crop_0.5", "label", "train",), )
# })
# DatasetCatalog.register('cocostuff27_iic_train_meta_fivecrop', cocostuff27_iic_train_meta_fivecrop)    
# MetadataCatalog.get('cocostuff27_iic_train_meta_fivecrop').set(**tep_meta, 
#                                                   visualize_meta_idxs=visualize_meta_idxs['cocostuff27_iic_train_meta_fivecrop']) 