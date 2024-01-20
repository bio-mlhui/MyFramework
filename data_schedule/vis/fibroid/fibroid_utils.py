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

def get_frames(frames_path,
               video_id, 
               frames):
    # 给出video_id, list of 帧strs, 返回这几帧的 list[Image]
    # frames_path根据train/evaluate的不同给出
    return [Image.open(os.path.join(frames_path, video_id, f'{f}.png'),) for f in frames]

# t' h w, int, obj_ids ;  has_ann t
def get_frames_mask(mask_path, video_id, frames):
    masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    assert set(masks.unique().tolist()) == set([0, 255])
    masks = masks > 0
    return masks, torch.ones(len(frames)).bool()


