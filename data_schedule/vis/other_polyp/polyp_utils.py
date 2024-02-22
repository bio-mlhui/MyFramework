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

SET_NAME = ['Poly_Train', 
         'Poly_Hard_Seen', 
         'Poly_Hard_Unseen', 
         'Poly_Easy_Seen', 
         'Poly_Easy_Unseen',]

SET_NAME_TO_DIR = {
    'Poly_Train': 'TrainDataset',
    'Poly_Hard_Seen': 'TestHardDataset/Seen',
    'Poly_Hard_Unseen': 'TestHardDataset/Unseen',
    'Poly_Easy_Seen': 'TestEasyDataset/Seen',
    'Poly_Easy_Unseen': 'TestEasyDataset/Unseen',
}

SET_NAME_TO_NUM_VIDEOS = {
    'Poly_Train': 112,
    'Poly_Hard_Seen': 17,
    'Poly_Hard_Unseen': 37,
    'Poly_Easy_Seen': 33,
    'Poly_Easy_Unseen': 86        
}

SET_NAME_TO_MODE = {
    'Poly_Train': 'train',
    'Poly_Hard_Seen': 'evaluate',
    'Poly_Hard_Unseen': 'evaluate',
    'Poly_Easy_Seen': 'evaluate',
    'Poly_Easy_Unseen': 'evaluate'      
}

SET_NAME_TO_PREFIX = {
    'Poly_Train': 'polyp_train',
    'Poly_Hard_Seen': 'polyp_hard_seen_validate',
    'Poly_Hard_Unseen': 'polyp_hard_unseen_validate',
    'Poly_Easy_Seen': 'polyp_easy_seen_validate',
    'Poly_Easy_Unseen': 'polyp_easy_unseen_validate' 
}

CLASS_TO_ID = {
    'high_grade_adenoma':0, 
    'hyperplastic_polyp':1, 
    'invasive_cancer':2,
    'low_grade_adenoma':3, 
    'sessile_serrated_lesion':4,
    'traditional_serrated_adenoma':5
}

import cv2

def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')).convert('RGB') for f in frames]

# t' h w, 0是背景, 1-是obj_id  ;  has_ann: t
def get_frames_mask(mask_path, video_id, frames):
    # masks = [cv2.imread(os.path.join(mask_path, video_id, f'{f}.jpg')) for f in frames]
    masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    # assert set(masks.unique().tolist()) == set([0, 255]), f'{masks.unique().tolist()}'
    masks = (masks > 0).int()
    return masks, torch.ones(len(frames)).bool()


