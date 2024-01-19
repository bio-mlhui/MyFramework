import torch.utils.data as torch_data
import torch
import torchvision.transforms as transforms
import os
import pandas
from glob import glob

import json
import numpy as np
import torchvision.io as video_io
from pycocotools.mask import encode, area
import torchvision.transforms.functional as F
import random
from scipy.io import loadmat

# jhmdb_sentences/ 
#     ├── Rename_Images/  (frame images)
#     │   └── */ (action dirs)
#     ├── puppet_mask/  (mask annotations)
#     │   └── */ (action dirs)
#     └── jhmdb_annotation.txt  (text annotations)
#     └── jhmdb_sentences_samples_metadata.json

def read_jhmdb(root):
    # video_id, video_dir, masks_dir, num_frames, text_query
    with open(os.path.join(root, 'jhmdb_sentences_samples_metadata.json'), 'r') as f:
        json.loads(f)
    

class JHMDB_Sentences_Dataset(torch_data.DataSet):
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, idx):
        pass
    

class JHMDB_Transforms:
    def __init__(self) -> None:
        pass
        
    def __call__(self):
        pass
    

@register_loaders
def jhmdb_mttr():
    pass