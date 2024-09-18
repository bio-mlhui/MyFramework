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
    image = np.load(os.path.join(path, f'{image_id}.npy'))
    # h w 3 / h w, uint8
    if len(image.shape) == 2:
        image = torch.from_numpy(image)[..., None].repeat(1, 1, 3).numpy() # h w 3
    else:
        assert len(image.shape) == 3
    image = Image.fromarray(image, mode='RGB')

    return image

def get_image_mask(path, image_id):
    mask = np.load(os.path.join(path, f'{image_id}.npy')) # uint8/uint16; 0 is background
    # h w
    mask = torch.from_numpy(mask).long()
    
    mask = mask - 1 # 0-label
    return mask


# npz meta:
{'PET': {'autoPET': 345}, 'Dermoscopy': {'ISIC2018': 3694}, 
    'Mammography': {'CDD-CESM': 1233}, 
    'US': {'Breast-Ultrasound': 647, 'hc18': 999}, 
    'Endoscopy': {'m2caiSeg': 1807, 'Kvasir-SEG': 1000, 'CholecSeg8k': 40636}, 
    'Fundus': {'PAPILA': 976, 'IDRiD': 81}, 
    'XRay': {'COVID-19-Radiography-Database': 21165, 
            'COVID-QU-Ex-lungMask_Lung': 5826, 
            'Pneumothorax-Masks': 2669, 
            'Chest-Xray-Masks-and-Labels': 702, 
            'COVID-QU-Ex-lungMask_CovidInfection': 2913}, 
    'Microscopy': {'NeurIPS22CellSeg': 1000}, 
    'MR': {'SpineMR': 172, 'QIN-PROSTATE-Prostate': 90, 
        'AMOSMR': 60, 'BraTS_T1CE': 899, 'crossmoda': 227, 
        'ProstateT2': 338, 'Heart': 20, 'QIN-PROSTATE-Lesion': 65, 
        'BraTS_FLAIR': 954, 'WMH_FLAIR': 170, 'CervicalCancer': 7, 'WMH_T1': 170, 
        'ISLES2022_DWI': 235, 'totalseg_mr': 263, 'ProstateADC': 285, 'ISLES2022_ADC': 235, 
        'BraTS_T1': 954}, 
    'CT': {'TotalSeg_muscles': 1218, 'COVID-19-20-CT-SegChallenge': 199, 
        'AMOS': 300, 'CT-ORG': 135, 'LIDC-IDRI': 6844, 'COVID19-CT-Seg-Bench': 20, 
        'LungLesions': 141, 'AbdomenCT1K': 1000, 'TotalSeg_organs': 1176, 
        'TotalSeg_cardiac': 1174, 'LungMasks': 465, 'CT_AbdTumor': 1890, 'TCIA-LCTSC': 60}, 
    'OCT': {'Intraretinal-Cystoid-Fluid': 1436}}

DATASET_PATH = os.environ.get('DATASET_PATH')

    # npz_dir = join(DATASET_PATH, 'med_sam_workshop_data')
    # modalities_to_tasks_to_numbers = {}
    # modalities = [foo for foo in os.listdir(npz_dir) if os.path.isdir(join(npz_dir, foo))]
    # for modal in modalities:
    #     task_to_numbers = {key: len(os.listdir(join(npz_dir, modal, key)))  for key in os.listdir(join(npz_dir, modal))}
    #     modalities_to_tasks_to_numbers[modal] = task_to_numbers
    # print(modalities_to_tasks_to_numbers)
    # exit() 
    # 每个模态选10个, 3D的
visualize_meta_idxs = defaultdict(list)
visualize_meta_idxs['cutler'] = []

# 不考虑类别
cocostuff27_meta = {
    'thing_classes': [0],
    'thing_colors': [(156, 31, 23)],
    'class_labels': [0],
    'num_classes': 1,
}

root = os.path.join(DATASET_PATH, 'med_sam_workshop_data/train_npy')

tep_meta = dcopy(cocostuff27_meta)
tep_meta.update({'mode': 'all', 'name': 'cutler',})
tep_meta.update({
    'get_image_fn': partial(get_image, path=os.path.join(root, 'imgs',)), 
    'get_mask_fn': partial(get_image_mask, path=os.path.join(root, 'gts',),),        
})
def cutler_meta():
    all_files = os.listdir(os.path.join(root, 'imgs'))
    # remove mr_totalseg_mr
    all_files = [foo for foo in all_files if 'MR_totalseg_mr' not in foo]
    # randomperm
    g = torch.Generator()
    g.manual_seed(10)
    rand_perm_idxs = torch.randperm(n=len(all_files), generator=g)[:300] #  729437,  425969,  287708
    used_files = [all_files[rand_perm] for rand_perm in rand_perm_idxs]
    return [{'image_id': image_id[:-4], 'meta_idx': idx} for idx, image_id in enumerate(used_files)]
DatasetCatalog.register('cutler', cutler_meta)    
MetadataCatalog.get('cutler').set(**tep_meta, 
                                   visualize_meta_idxs=visualize_meta_idxs['cutler']) 

