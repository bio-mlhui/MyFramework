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
from PIL import Image

def single_meta():
    return [{
        'dataset_id': 'text_generate_video',
        'video_id': 'text_generate_video_0',
        'meta_idx': 0
    }]

DatasetCatalog.register('text_generate_video', single_meta) 
MetadataCatalog.get('text_generate_video').set(
    mode='all',
    visualize_meta_idxs=[],
    eval_meta_keys={'text_generate_video_0':0}
)  




