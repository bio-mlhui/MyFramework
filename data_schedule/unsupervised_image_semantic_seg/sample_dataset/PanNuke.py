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

REGISTER_NAMES = ['PanNuke']

def get_image(path, image_id):
    image = np.load(os.path.join(path, f'{image_id}.npy')) / 255.0 # h w 3, 0-1
    return image

# 不进行测试
# def get_image_mask(path, image_id):
#     # binary mask, 0-255
#     mask = np.load(os.path.join(path, f'{image_id}.npy')) # h w
#     return mask

DATASET_PATH = os.environ.get('DATASET_PATH')
DATASET_NAME = 'PanNuke'
root = os.path.join(DATASET_PATH, 'med_unsupervised/PanNuke')
visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs[DATASET_NAME] = []
cocostuff27_meta = {
    'thing_classes': [0],
    'thing_colors': [(156, 31, 23)],
    'class_labels': [0],
    'num_classes': 1,
}

tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'all', 'name': DATASET_NAME,})
tep_meta.update({
    'get_image_fn': partial(get_image, path=os.path.join(root, 'imgs',)), 
    # 'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'masks',),),        
})
def cutler_meta():
    all_files = os.listdir(os.path.join(root, 'imgs'))
    return [{'image_id': image_id[:-4], 'meta_idx': idx} for idx, image_id in enumerate(all_files)]
DatasetCatalog.register(DATASET_NAME, cutler_meta)    
MetadataCatalog.get(DATASET_NAME).set(**tep_meta, 
                                             visualize_meta_idxs=visualize_meta_idxs[DATASET_NAME]) 




if __name__ == '__main__':
    import logging
    from functools import partial
    from torch.utils.data import DataLoader, ConcatDataset
    from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset, MetadataCatalog
    # # first crop each dataset and save
    # image_file = os.path.join(root, 'Images/images.npy')
    # # mask_file = os.path.join(root, 'Masks/masks.npy')
    # images = np.load(image_file) # b h w 3, float, 0-255
    # # masks = np.load(mask_file) # b h w 6
    # # images.shape[0] == masks.shape[0]
    # for idx, img in enumerate(images):
    #     np.save(os.path.join(root, f'imgs/{idx}.npy'), img)

    def visualize_dataset(data_dict):
        image_id = data_dict['image_id']
        get_image_fn = MetadataCatalog.get(DATASET_NAME).get('get_image_fn')
        # get_mask_fn = MetadataCatalog.get(DATASET_NAME).get('get_mask_fn')
        image = get_image_fn(image_id=image_id) 
        image =  Image.fromarray((image * 255).astype('uint8'))
        # mask = get_mask_fn(image_id=image_id) # -1/0,1,2, H, W
        # unique_labels = mask.unique()
        binary_masks = [image] # n h w
        W, H = image.size

        new_im = Image.new('RGB', (W*len(binary_masks), H))

        x_offset = 0
        for im in binary_masks:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save('./test.png')
        return {'None': None}

    dataset_dicts = DatasetFromList(DatasetCatalog.get(DATASET_NAME), copy=True, serialize=True)
   
    dataset = MapDataset(dataset_dicts, visualize_dataset)
   
    for idx in range(10):
        item = dataset.__getitem__(idx)

