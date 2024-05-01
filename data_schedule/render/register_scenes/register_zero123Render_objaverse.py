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


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
from data_schedule.render.scene_utils.cameras import MiniMiniCam
   

def get_rendering_fn(scene_path=None,  scene_id=None, view_id=None, return_alpha=True, **kwargs):
    image_path = os.path.join(scene_path, scene_id,  f'{view_id:03d}.png')
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
                  camera_radius=None,
                  only_extrinsic=False,
                  **kwargs):
    camera_path = os.path.join(scene_path, scene_id, f'{view_id:03d}.npy')
    c2w = np.load(camera_path)

    # list[dict, view_camera:str, rendering_path:str]
    with open(camera_path, 'r') as f:
        c2w = [float(t) for t in f.readline().decode().strip().split(' ')]
    c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

    # TODO: you may have a different camera system
    # blender world + opencv cam --> opengl world & cam
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1 # invert up and forward direction

    # scale up radius to fully use the [-1, 1]^3 space!
    c2w[:3, 3] *= camera_radius / 1.5 # 1.5 is the default scale

    assert only_extrinsic
    return c2w


def get_T(self, target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T



def zero123_rendered_objaverse_meta(ov_root, 
                                    subset_version='kiuiv1',
                                    register_name=None):

    if subset_version == 'kiuiv1':
        pass
    else:
        raise ValueError()
    
    # 渲染时候的相机参数:
    with open(os.path.join(dataset_root, f'{subset_version}/camera_intrin.json'), 'r') as f:
        camera_intrin = json.load(f)

    camera_intrin = MiniMiniCam(
        zfar=camera_intrin['zfar'],
        znear=camera_intrin['znear'],
        fovY=camera_intrin['fovY'],
        height=camera_intrin['height'],
        width=camera_intrin['width'],
        radius=camera_intrin['radius'],
        c2w=None 
    )
    MetadataCatalog.get('register_name').set(camera_intrin=camera_intrin)
    
    meta_file = os.path.join(ov_root, f'{subset_version}/metas.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            logging.debug(f'loading {register_name} metas')
            metas = json.load(f)
    else:
        scene_ids = []
        with open(os.path.join(ov_root, f'{subset_version}/filter.txt'), 'r') as f:
            for line in f.readlines():
                scene_ids.append(line.strip())
        Scene_Meta
        metas = []
        for idx, scene_uid in tqdm(enumerate(scene_ids)):
            view_ids = os.listdir(os.path.join(ov_root, 'views_release',  scene_uid))
            view_ids = set([int(haosen[:-4]) for haosen in view_ids])
            assert len(view_ids) == 12
            for vid in view_ids:
                assert os.path.join(ov_root, 'views_release',  scene_uid,  f'{vid:03d}.npy')
                assert os.path.join(ov_root, 'views_release',  scene_uid,  f'{vid:03d}.png')
            metas.append({
                'scene_id': scene_uid,
                'metalog_name': register_name,
                'view_cameras': view_ids,  # list[object], get_camera_fn(object) -> camera
                'meta_idx': idx
            })
        with open(meta_file, 'w') as f:
            logging.debug(f'saving {register_name} metas')
            json.dump(metas, f)

    logging.debug(f'{register_name} Total metas: [{len(metas)}]')
    return metas

# root/zeros123_rendered:
#       views_release/scene_uid/0.npy, 0.png, 11.npy, 11.png
#       kiuiv1/
#       kiuiv2/
dataset_root = os.path.join(os.getenv('DATASET_PATH'), 'objaverse', 'zero123_render')
DatasetCatalog.register('objaverse_zero123_kiuiv1', partial(zero123_rendered_objaverse_meta,
                                                             ov_root=dataset_root,
                                                             subset_version='kiuiv1',
                                                             register_name='objaverse_zero123_kiuiv1'))

MetadataCatalog.get('objaverse_zero123_kiuiv1').set(white_background=False,
                                                    get_rendering_fn=partial(get_rendering_fn,
                                                                             scene_path=os.path.join(dataset_root, 'views_release'),
                                                                             return_alpha=True),
                                                    get_camera_fn=partial(get_camera_fn,
                                                                          scene_path=os.path.join(dataset_root, 'views_release'),
                                                                          only_extrinsic=True,),  # 因为相机都是相同的
                                                    visualize_meta_idxs=[])


# root/lgm_rendered:
#       views_release/scene_uid/0.npy, 0.png
#       kiuiv1/
#       kiuiv2/