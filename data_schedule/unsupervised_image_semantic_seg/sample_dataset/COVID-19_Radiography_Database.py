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

import numpy as np
import glob
join = os.path.join


def get_image(path, image_id):
    image = Image.open(os.path.join(path,  f'{image_id}.png')).convert('RGB')
    return image

def get_image_mask(path, image_id):
    # binary mask, 0-255
    mask = Image.open(os.path.join(path, f'{image_id}.png')).convert('L') # h w
    mask = (torch.from_numpy(np.array(mask)) > 128).long() - 1
    return mask

DATASET_PATH = os.environ.get('DATASET_PATH')

visualize_meta_idxs = defaultdict(list)
cocostuff27_meta = {
    'thing_classes': [0],
    'thing_colors': [(156, 31, 23)],
    'class_labels': [0],
    'num_classes': 1,
}
def cutler_meta(root):
    all_files = os.listdir(os.path.join(root, 'images'))
    return [{'image_id': image_id[:-4], 'meta_idx': idx} for idx, image_id in enumerate(all_files)]
REGISTER_NAMES = []
for subdir in ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']:
    DATASET_NAME = f'COVID-19_Radiography_Database_{subdir}'
    REGISTER_NAMES.append(DATASET_NAME)
    root = os.path.join(DATASET_PATH, 'med_unsupervised/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset', subdir)

    tep_meta = dcopy(cocostuff27_meta)
    tep_meta.update({'mode': 'all', 'name': DATASET_NAME,})
    tep_meta.update({
        'get_image_fn': partial(get_image, path=os.path.join(root, 'images',)), 
        'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'masks',),),        
    })
    DatasetCatalog.register(DATASET_NAME, partial(cutler_meta, root=root))
    MetadataCatalog.get(DATASET_NAME).set(**tep_meta, 
                                        visualize_meta_idxs=visualize_meta_idxs[DATASET_NAME]) 


if __name__ == '__main__':
    import logging
    from functools import partial
    from torch.utils.data import DataLoader, ConcatDataset
    from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset, MetadataCatalog

    def visualize_dataset(data_dict, dataset_name):
        image_id = data_dict['image_id']
        get_image_fn = MetadataCatalog.get(dataset_name).get('get_image_fn')
        get_mask_fn = MetadataCatalog.get(dataset_name).get('get_mask_fn')
        image = get_image_fn(image_id=image_id)
        mask = get_mask_fn(image_id=image_id) # -1/0,1,2, H, W
        unique_labels = mask.unique()
        binary_masks = [image] # n h w
        for lab in unique_labels:
            if lab == -1:
                continue
            else:
                b_mask = ((lab == mask).float() * 255).to(torch.uint8).numpy() # h w
                b_mask = Image.fromarray(b_mask, mode='L')
                binary_masks.append(b_mask)

        W, H = image.size

        new_im = Image.new('RGB', (W*len(binary_masks), H))

        x_offset = 0
        for im in binary_masks:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save('./test.png')
        return {'None': None}
    
    for dataset_name in REGISTER_NAMES:
        dataset_dicts = DatasetFromList(DatasetCatalog.get(dataset_name), copy=True, serialize=True)
        dataset = MapDataset(dataset_dicts, partial(visualize_dataset, dataset_name=dataset_name))
        for idx in range(10):
            item = dataset.__getitem__(idx)

