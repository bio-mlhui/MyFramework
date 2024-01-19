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

# t' h w, 0是背景, 1-是obj_id
# has_ann: t
# (video_id, frames) -> t' h w, obj_id; has_ann: t
# frames_mask 就是0, 1
# 获得的是video-level 的mask
# 单帧的mask 可以通过cv.connectedComponents得到分开的两个
def get_frames_mask(mask_path, video_id, frames):
    # 给出video_id, list of 帧strs, 返回这几帧的 mask: tensor(t' h w, int,)
    # 返回这几帧里哪些帧是有annotation的
    masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
    masks = [np.array(mk) for mk in masks]
    masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
    assert set(masks.unique().tolist()) == set([0, 255])
    masks = masks > 0
    return masks, torch.ones(len(frames)).bool()


