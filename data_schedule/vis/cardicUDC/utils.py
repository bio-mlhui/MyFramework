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

SET_NAME = [
         'card_labeled_train',  # 全是有监督的训练, 每个clip中都有一帧有监督
         'card_labeled_test', # 
         'card_unlabeled', # 包含所有clip, 有监督或者没有
         ]

SET_NAME_TO_DIR = {
    'card_labeled_train': 'labeled_train',
    'card_labeled_test': 'labeled_test',
    'card_unlabeled': 'unlabeled',
}

SET_NAME_TO_MODE = {
    'card_labeled_train': 'train',
    'card_labeled_test': 'evaluate',
    'card_unlabeled': 'train',
}

SET_NAME_TO_PREFIX = {
    'card_labeled_train': 'card_labeled_train',
    'card_labeled_test': 'card_labeled_test',
    'card_unlabeled': 'card_unlabeled',
}


def get_frames(frames_path, video_id, frames):
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')) for f in frames]

def get_frames_mask(mask_path, video_id, mid_frame):
    if os.path.exists(os.path.join(mask_path, video_id, f'{mid_frame}.npy')):
        masks = torch.from_numpy(np.load(os.path.join(mask_path, video_id, f'{mid_frame}.npy'))) # h w
        masks = torch.stack([masks == cls_id for cls_id in range(1, 5)], dim=0) # K h w
        return masks
    else:
        return None


