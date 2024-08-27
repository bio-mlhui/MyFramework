
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

import numpy as np
import torch.multiprocessing
import torchvision.transforms as T
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

import numpy as np


fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                  1: 1, 5: 1,  # buildings and clutter
                  2: 2, 3: 2,  # vegetation and trees
                  255: -1}

def get_image(path, image_id):
    img = loadmat(os.path.join(path, image_id + ".mat"))["img"]
    return to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])

def get_image_mask(path, image_id):
    label = loadmat(join(path, image_id + ".mat"))["gt"] # fine label
    label = torch.from_numpy(label) # fine label, h, w, int32, 0,1,2,3,4,5, 255
    new_label_map = torch.zeros_like(label).long()
    for fine, coarse in fine_to_coarse.items():
        new_label_map[label == fine] = coarse
    return new_label_map
    
_root = os.getenv('DATASET_PATH')
root = os.path.join(_root, 'POTSDAM/potsdam')

visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['potsdam3_train]'] = [] 
visualize_meta_idxs['potsdam3_val'] = [] 
potsdam3_meta = {
    'thing_classes': ['roads and cars', 'buildings and clutter', 'trees and vegetation'],
    'thing_colors': [(108, 109, 112), (152, 173, 237), (82, 237, 9)],
    'class_labels': ['roads and cars', 'buildings and clutter', 'trees and vegetation'],
    'num_classes': 3,
    'get_image_fn': partial(get_image, path=os.path.join(root, 'imgs',)),
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'gt',),),
}

split_files = {
    "potsdam3_train": ["labelled_train.txt"],
    "potsdam3_val": ["labelled_test.txt"],
}

def return_meta(path, split_name):
    files = []
    for split_file in split_files[split_name]:
        with open(join(path, split_file), "r") as f:
            files.extend(fn.rstrip() for fn in f.readlines())
    files = [{'image_id': image_id, 'meta_idx': idx} for idx, image_id in enumerate(files)]
    return files    

for name,mode in zip(['potsdam3_val', 'potsdam3_train',], ['evaluate', 'train']):
    dataset_meta = copy.deepcopy(potsdam3_meta)
    dataset_meta.update({'mode': mode, 'name': name,})
    DatasetCatalog.register(name, partial(return_meta,path=root,split_name=name))    
    MetadataCatalog.get(name).set(**dataset_meta, 
                                  visualize_meta_idxs=visualize_meta_idxs[name]) 
