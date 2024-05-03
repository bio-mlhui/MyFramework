import os
import cv2
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
import kiui
from data_schedule.render.apis import Scene_Meta
from tqdm import tqdm
import numpy as np
import logging
from functools import partial
import json

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Camera:
    def __init__(self, zfar, znear, fovY, height, width, radius, c2w):
        # 这是某次渲染的时候的相机内参, 固定不变
        self.zfar = zfar
        self.znear = znear
        self.fovY = fovY
        self.height = height
        self.width = width
        self.radius = radius

        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(fovY))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        self.proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
        self.proj_matrix[2, 3] = 1

        # extrinstic
        self.c2w = c2w


def get_rendering_fn(scene_path=None,  scene_id=None, view_id=None, return_alpha=True, **kwargs):
    image_path = os.path.join(scene_path, scene_id, 'rgb', f'{view_id:03d}.png')
    # list[dict, view_camera:str, rendering_path:str]
    image = np.frombuffer(image_path, np.uint8)
    image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
    image = image.permute(2, 0, 1) # [4, 512, 512]
    mask = image[3:4].squeeze(0) # [1, 512, 512]
    image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
    image = image[[2,1,0]].contiguous() # bgr to rgb
    if return_alpha:
        return image,mask
    else: 
        return image

def get_camera_fn(scene_path=None,  
                  scene_id=None, 
                  view_id=None, 
                  camera_zfar=None,
                  camera_znear=None,
                  camera_fovY=None,
                  camera_height=None,
                  camera_width=None,
                  camera_radius=None,
                  **kwargs):
    camera_path = os.path.join(scene_path, scene_id, 'pose', f'{view_id:03d}.txt')
    # list[dict, view_camera:str, rendering_path:str]
    with open(camera_path, 'r') as f:
        c2w = [float(t) for t in f.readline().decode().strip().split(' ')]
    c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

    return Camera(zfar=camera_zfar,
                  znear=camera_znear,
                  fovY=camera_fovY,
                  height=camera_height,
                  width=camera_width,
                  cam_radius=camera_radius,
                  c2w=c2w)


def original_objaverse_meta(ov_root, 
                            subset_version='kiuiv1',
                            register_name=None):
    if subset_version == 'kiuiv1':
        pass
    else:
        raise ValueError()

    
    scene_files = []
    if filtered_version == 'kiuiv1':
        with open(os.path.join(ov_root, 'kiuisobj_v1.txt'), 'r') as f:
            for line in f.readlines():
                scene_files.append(line.strip())
    else:
        raise ValueError()
    
    Scene_Meta
    metas = []
    for idx, scene_uid in tqdm(enumerate(scene_files)):
        metas.append({
            'scene_id': scene_uid,
            'metalog_name': register_name,
            'meta_idx': len(metas)
        })
    logging.debug(f'{register_name} Total metas: [{len(metas)}]')
    return metas


# root
    # original
    # zero123_rendered
    # lgm_rendered
    # kiuiv1.txt
dataset_root = os.path.join(os.getenv('DATASET_PATH'), 'objaverse')

DatasetCatalog.register('objaverse_original_kiuiv1', partial(original_objaverse_meta,
                                                             ov_root=dataset_root,
                                                             filtered_version='kiuiv1',
                                                             register_name='objaverse_original_kiuiv1')) 

MetadataCatalog.get('objaverse_original_kiuiv1').set(white_background=False,
                                                    get_rendering_fn=partial(get_rendering_fn,
                                                                             scene_path=os.path.join(dataset_root, 'original'),
                                                                             return_alpha=True),
                                                    get_camera_fn=partial(get_camera_fn,
                                                                          scene_path=dataset_root,
                                                                          only_extrinsic=True,), 
                                                    visualize_meta_idxs=[])


