
import os
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import torch.distributed as dist
import detectron2.utils.comm as comm
from glob import glob
from PIL import Image
import cv2

import json
import logging
from data_schedule.render.scene.dataset_readers import sceneLoadTypeCallbacks
from functools import partial
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_schedule.render.apis import Scene_Dataset, Scene_Terminology, Scene_Mapper


Scene_Dataset

scene_ids = ['deepblend_drjohnson', 'deepblend_playroom', 'deepblend_train', 'deepblend_truck']

scene_id_to_dir = {
    'deepblend_drjohnson': 'video_4D/deepblend/db/drjohnson',
    'deepblend_playroom': 'video_4D/deepblend/db/playroom',   
    'deepblend_train': 'video_4D/deepblend/tandt/train',
    'deepblend_truck': 'video_4D/deepblend/tandt/truck'
}

scene_id_to_sparse_tt = {
    'deepblend_drjohnson': 2,
    'deepblend_playroom': 2,   
    'deepblend_train': False,
    'deepblend_truck': False
}

scene_id_to_wbcg = {
    'deepblend_drjohnson': False,
    'deepblend_playroom': False,   
    'deepblend_train': False,
    'deepblend_truck': False   
}


scene_id_to_iseval = {
    'deepblend_drjohnson': False,
    'deepblend_playroom': False,   
    'deepblend_train': False,
    'deepblend_truck': False    
}

scene_id_to_imagedir = {
    'deepblend_drjohnson': False,
    'deepblend_playroom': False,   
    'deepblend_train': False,
    'deepblend_truck': False    
}


def get_rendering_fn(root=None,
                     scene_id=None, 
                     view_camera=None,
                     scene_video_id=None,
                     **kwargs):
    Scene_Terminology
    if view_camera is not None:
        # 根据view_camera 得到
        return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')).convert('RGB') for f in frames]
    elif view is not None:
        return None
    else:
        raise ValueError()
    
def get_scene_info(scene_path, 
                    sparse_or_tt=None,
                    image_dir=None, 
                    white_background=None, 
                    eval=None):
    if sparse_or_tt == 'sparse':
        assert os.path.exists(os.path.join(scene_path, "sparse"))
        scene_info = sceneLoadTypeCallbacks["Colmap"](scene_path, image_dir, eval)
    elif sparse_or_tt == 'tt':
        assert os.path.exists(os.path.join(scene_path, "transforms_train.json"))
        logging.info("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](scene_path, white_background, eval)
    else:
        assert False, "Could not recognize scene type!"
    return scene_info

class GetSceneInfo:
    def __init__(self, 
                 scene_dir=None,
                 sparse_or_tt=None,
                 image_dir=None, 
                 white_background=None, 
                 is_eval=None) -> None:
        self.scene_info = None
        self.scene_dir = scene_dir
        self.sparse_or_tt = sparse_or_tt
        self.image_dir = image_dir
        self.wbcg = white_background
        self.is_eval = is_eval


    def __call__(self, 
                 mode, 
                 register_name,
                 condense=False,) -> json.Any:
        
        if self.scene_info is None:
            self.scene_info = get_scene_info(scene_path=self.scene_dir,
                                             sparse_or_tt=self.sparse_or_tt,
                                             image_dir=self.image_dir,
                                             white_background=self.wbcg,
                                             eval=self.is_eval)           
        if mode == 'train':
            view_cameras = self.scene_info.train_cameras
        elif mode == 'test':
            view_cameras = self.scene_info.test_cameras
        else:
            raise ValueError()

        Scene_Mapper
        if condense:
            return [{
                'scene_id': register_name,
                'scene_text': None,
                'scene_video_id': None,
                'view_camera': haosen,
            } for haosen in view_cameras]
        else:
            return [{
                'scene_id': register_name,
                'scene_text': None,
                'scene_video_id': None,
                'view_cameras': view_cameras,
            }]



for scene_id in scene_ids:
    scene_dir = os.path.join(os.getenv('DATASET_PATH'), scene_id_to_dir[scene_id])
    sparse_or_tt = scene_id_to_sparse_tt[scene_id]
    wbcg = scene_id_to_wbcg[scene_id]
    is_eval = scene_id_to_iseval[scene_id]
    image_dir = scene_id_to_imagedir[scene_id]

    scene_call_fn = GetSceneInfo(scene_dir=scene_dir,
                                 sparse_or_tt=sparse_or_tt,
                                 white_background=wbcg,
                                 image_dir=image_dir,
                                 is_eval=is_eval)
    
    Scene_Terminology
    register_names   = [f'{scene_id}_train_condense', f'{scene_id}_test_condense', f'{scene_id}_train', f'{scene_id}_test']
    register_modes   = ['train', 'test', 'train', 'test']
    register_condenses = [True, True, False, False]

    for reg_name, reg_mode, reg_condense in zip(register_names, register_modes, register_condenses):
        DatasetCatalog.register(reg_name, partial(scene_call_fn, 
                                                  mode=reg_mode, 
                                                  condense=reg_condense,
                                                  register_name=reg_name))
        MetadataCatalog.get(reg_name).set(point_cloud=scene_call_fn.point_cloud,
                                                    nerf_normalization_radius=scene_call_fn.nerf_normalization['radius'],
                                                    sparse_or_tt=sparse_or_tt, 
                                                    wbcg=wbcg, 
                                                    is_eval=is_eval,
                                                    get_rendering_fn=partial(get_rendering_fn, root=scene_dir),
                                                    visualize_meta_idxs=[])
    
        
 
        
