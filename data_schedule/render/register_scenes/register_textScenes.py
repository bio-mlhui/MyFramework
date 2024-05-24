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
import math
from collections import defaultdict
from data_schedule.render.scene_utils.cameras import OrbitCamera
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_camera_fn(scene_path=None,  
                  view_id=None, 
                  only_c2w=True,
                  world_format='opengl',
                  camera_format='opengl',
                  **kwargs):
    """
    6个view_id对应6个c2w
    return proj/c2w;
    """
    if not only_c2w:
        raise NotImplementedError()

    w2c = os.path.join(scene_path, 'zero123_rendered/views_release', scene_id, f'{view_id:03d}.npy')
    w2c = np.load(w2c)  # blender camera
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
    w2c = torch.tensor(w2c).float().reshape(4, 4)

    c2w = torch.inverse(w2c)
    # blender world + camera
    # [[Right_x, Up_x, Forward_x, Position_x],
    #  [Right_y, Up_y, Forward_y, Position_y],
    #  [Right_z, Up_z, Forward_z, Position_z],
    #  [0,       0,    0,         1         ]]

    if world_format == 'opengl':
        c2w[1] *= -1  # reverse y
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
    elif world_format == 'colmap' or world_format == 'opencv':
        c2w[1] *= -1  # reverse y
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
        c2w[[1, 2]] *= -1 # reverse y and z
    elif world_format == 'blender':
        pass
    elif world_format == 'unity' or world_format == 'directx':
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
    else:
        raise ValueError()

    if camera_format == 'opengl' or camera_format == 'blender':
        pass
    elif camera_format == 'colmap' or camera_format == 'opencv':
        c2w[:, 1:3] *= -1 # reverse up and forward
    else:
        raise ValueError()
    return c2w


# 每个text就是一个数据集，并且只有一个meta, 是(text, views)

# text-to-3D的相机内参永远不变


def text_scene(text,
               register_name=None):
    camera_intrin = OrbitCamera(
        W=512,
        H=512,
        r=2,
        fovy=None,
        near=None,
        far=None,
    )
    MetadataCatalog.get(register_name).set(camera_intrin=camera_intrin)
    Scene_Meta
    # 只有一个meta, (text, view_cameras)
    metas = []
    view_ids = list(range(4))
    eval_meta_keys = {
        text: list(range(4))
    }
    # 数据集_Scene_id
    metas.append({
        'scene_id': f'{register_name}_0',
        'scene_text': text,
        'metalog_name': register_name,
        'view_cameras': view_ids,  # list[object], get_camera_fn(object) -> camera
        'meta_idx': len(metas)
    })
    logging.debug(f'{register_name} Total metas: [{len(metas)}]')
    MetadataCatalog.get(register_name).set(eval_meta_keys=eval_meta_keys)
    return metas


input_texts = [
    'Donald Trump is holding a puppy',
]

# images only
dataset_root = os.path.join(os.getenv('DATASET_PATH'), 'TaskDataset/Text3D')

for text in input_texts: # 每个text就是一个数据集, 每个数据集
    register_name = f'{text}_text23d'

    maybe_image_prompt = os.path.join(dataset_root, f'{text}.png')
    DatasetCatalog.register(register_name, 
                            partial(text_scene,
                                    text=text,
                                    register_name=register_name,))
    

    MetadataCatalog.get(register_name).set(white_background=False,
                                            mode='all',
                                            text_3d={
                                                'input_text': text,
                                                'input_negative_text': None,
                                                'image_prompt': maybe_image_prompt if os.path.exists(maybe_image_prompt) else None
                                            },

                                            gaussian_initialize=None,
                                            get_rendering_fn=None,
                                            get_camera_fn=get_camera_fn,
                                            visualize_meta_idxs=[])
