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
from . import cutler
import numpy as np


def get_image(path, image_id):
    image = Image.open(os.path.join(path, f'{image_id}.jpg')).convert('RGB')
    return image

def get_image_mask(path, image_id):
    mask = torch.from_numpy(np.array(Image.open(os.path.join(path, f'{image_id}.jpg')))).long()[0]
    return mask

root = '/home/xuhuihui/workspace/rvos_encoder/data_schedule/unsupervised_image_semantic_seg/sample_dataset/sample_data'
visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['sample_dataset]'] = []

cocostuff27_meta = {
    'thing_classes': [0],
    'thing_colors': [(156, 31, 23)],
    'class_labels': [0],
    'num_classes': 1,
}

tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'all', 'name': 'sample_dataset',})
tep_meta.update({
    'get_image_fn': partial(get_image, path=os.path.join(root, 'images',)), 
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'images',),),        
})
def sample_dataset_meta():
    file_list = os.listdir(os.path.join(root, "images"))
    return [{'image_id': os.path.splitext(image_id)[0], 'meta_idx': idx} for idx, image_id in enumerate(file_list)]
DatasetCatalog.register('sample_dataset', sample_dataset_meta)    
MetadataCatalog.get('sample_dataset').set(**tep_meta, 
                                               visualize_meta_idxs=visualize_meta_idxs['sample_dataset']) 


