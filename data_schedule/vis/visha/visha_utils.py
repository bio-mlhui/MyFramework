import wandb
import plotly.express as px
import logging
import os
import numpy as np
import torch
import json
from joblib import Parallel, delayed
import multiprocessing
import torch.distributed as dist
import detectron2.utils.comm as comm

import pycocotools.mask as mask_util
from pycocotools.mask import encode, area

from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.utils.video_clips import generate_windows_of_video
from glob import glob
from PIL import Image

SET_NAME = ['train', 'test']

SET_NAME_TO_DIR = {
    'train': 'train',
    'test': 'test',}

SET_NAME_TO_NUM_VIDEOS = {
    'train': 50,
    'test': 70,    
}


SET_NAME_TO_PREFIX = {
    'train': 'visha_train',
    'test': 'visha_test',
}

SET_NAME_TO_MODE = {
    'train': 'train',
    'test': 'evaluate'       
}

SET_NAME_TO_GT_TYPE = {
    'train': 'GT',
    'test': 'GT', 
}

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')).convert('RGB') for f in frames]

# t' h w, 0是背景, 1-是obj_id  ;  has_ann: t
def get_frames_mask(mask_path, video_id, frames):
    # masks = [cv2.imread(os.path.join(mask_path, video_id, f'{f}.jpg')) for f in frames]
    if os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.png')):
        masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    elif os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.jpg')):
        masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.jpg')).convert('L') for f in frames]
    else:
        raise ValueError()
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    # assert set(masks.unique().tolist()) == set([0, 255]), f'{masks.unique().tolist()}'
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()


