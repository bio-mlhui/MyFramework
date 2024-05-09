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

from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
from data_schedule.render.scene_utils.cameras import MiniMiniCam
   

def get_rendering_fn(scene_path=None,  scene_id=None, view_id=None, return_alpha=True, **kwargs):
    image_path = os.path.join(scene_path, 'zero123_rendered/views_release', scene_id,  f'{view_id:03d}.png')
    image = np.asarray(Image.open(image_path).convert('RGBA')).astype(np.float32) / 255
    image = torch.from_numpy(image) # [512, 512, 4] in [0, 1]
    image = image.permute(2, 0, 1) # [4, 512, 512]
    mask = image[3:4].squeeze(0) # [1, 512, 512]
    image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
    image = image.contiguous()
    if return_alpha:
        return image, mask
    else: 
        return image


def get_camera_fn(scene_path=None,  
                  scene_id=None, 
                  view_id=None, 
                  only_c2w=True,
                  world_format='opengl',
                  camera_format='opengl',
                  **kwargs):
    """
    return proj/c2w ;
    """
    assert only_c2w
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


def zero123_rendered_objaverse_meta(ov_root, 
                                    subset_version='kiuiv1',
                                    register_name=None):
    # root
    #   filter1
    #   filter2
    #   zero123_rendered/
    #       intrin.json
    #       metas_filter1.json
    #       views_release
    if subset_version == 'kiuiv1':
        def get_scene_ids():
            scene_ids = []
            with open(os.path.join(ov_root, 'kiuisobj_v1_merged_80K.csv'), 'r') as f:
                for line in f.readlines():
                    scene_ids.append(line.strip().split(',')[-1])
            return scene_ids

    else:
        raise ValueError()

    camera_intrin = MiniMiniCam(
        zfar=100,
        znear=0.01,
        fovY=49.1,
        height=512,
        width=512,
        radius=1.5, # to better use [-1, 1]^3 space
        c2w=None 
    )
    MetadataCatalog.get(register_name).set(camera_intrin=camera_intrin)
    
    meta_file = os.path.join(ov_root, f'zero123_rendered/metas_{subset_version}.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            logging.debug(f'loading {register_name} metas')
            json_file = json.load(f)
            metas, eval_meta_keys = json_file['metas'], json_file['eval_meta_keys']

    else:
        scene_ids = get_scene_ids()
        Scene_Meta
        metas = []
        eval_meta_keys = defaultdict(dict)
        logging.debug(f'generating {register_name} metas')
        for idx, scene_uid in tqdm(enumerate(scene_ids)):
            if not os.path.exists(os.path.join(ov_root, 'zero123_rendered/views_release',  scene_uid)):
                continue
            view_ids = os.listdir(os.path.join(ov_root, 'zero123_rendered/views_release',  scene_uid))
            view_ids = list(set([int(haosen[:-4]) for haosen in view_ids]))
            assert len(view_ids) == 12
            for vid in view_ids:
                assert os.path.join(ov_root, 'zero123_rendered/views_release',  scene_uid,  f'{vid:03d}.npy')
                assert os.path.join(ov_root, 'zero123_rendered/views_release',  scene_uid,  f'{vid:03d}.png')
            eval_meta_keys[scene_uid] = view_ids
            metas.append({
                'scene_id': scene_uid,
                'metalog_name': register_name,
                'view_cameras': view_ids,  # list[object], get_camera_fn(object) -> camera
                'meta_idx': len(metas)
            })
        with open(meta_file, 'w') as f:
            logging.debug(f'saving {register_name} metas')
            json.dump({'metas': metas, 'eval_meta_keys': dict(eval_meta_keys)}, f)

    logging.debug(f'{register_name} Total metas: [{len(metas)}]')
    MetadataCatalog.get('register_name').set(eval_meta_keys=eval_meta_keys)
    return metas

# root/objaverse
#   original
#   zeros123_rendered:
#       views_release/scene_uid/0.npy, 0.png, 11.npy, 11.pn
#       camera_intrin
#   kiuiv1.txt
#   kiuiv2.txt
dataset_root = os.path.join(os.getenv('DATASET_PATH'), 'objaverse')
DatasetCatalog.register('objaverse_zero123_kiuiv1', partial(zero123_rendered_objaverse_meta, 
                                                            ov_root=dataset_root, subset_version='kiuiv1', register_name='objaverse_zero123_kiuiv1'))

MetadataCatalog.get('objaverse_zero123_kiuiv1').set(white_background=False,
                                                    mode='all',
                                                    get_rendering_fn=partial(get_rendering_fn, scene_path=dataset_root),
                                                    get_camera_fn=partial(get_camera_fn, scene_path=dataset_root,),
                                                    visualize_meta_idxs=[])

DatasetCatalog.register('objaverse_zero123_kiuiv1_test', partial(zero123_rendered_objaverse_meta, 
                                                            ov_root=dataset_root, subset_version='kiuiv1', register_name='objaverse_zero123_kiuiv1_test'))

MetadataCatalog.get('objaverse_zero123_kiuiv1_test').set(white_background=False,
                                                    mode='all',
                                                    get_rendering_fn=partial(get_rendering_fn, scene_path=dataset_root),
                                                    get_camera_fn=partial(get_camera_fn, scene_path=dataset_root,),
                                                    visualize_meta_idxs=[])

# root/objaverse
#   original
#   zeros123_rendered:
#       views_release/scene_uid/0.npy, 0.png, 11.npy, 11.pn
#       camera_intrin
#   kiuiv1.txt
#   kiuiv2.txt