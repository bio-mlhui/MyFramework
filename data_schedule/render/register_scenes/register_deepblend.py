
import os

from PIL import Image
import logging
from data_schedule.render.scene_utils.dataset_readers import sceneLoadTypeCallbacks
from functools import partial
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_schedule.render.apis import Scene_Meta
from data_schedule.render.scene_utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
import random
import objaverse
objaverse.load_objects()
scene_ids = ['deepblend_drjohnson', 'deepblend_playroom', 'deepblend_train', 'deepblend_truck']

scene_id_to_dir = {
    'deepblend_drjohnson': 'video_4D/deepblend/db/drjohnson',
    'deepblend_playroom': 'video_4D/deepblend/db/playroom',   
    'deepblend_train': 'video_4D/deepblend/tandt/train',
    'deepblend_truck': 'video_4D/deepblend/tandt/truck'
}

# 有两个子目录
# images/图片
# sparse/transforms_train.json

scene_id_to_wbcg = {
    'deepblend_drjohnson': False,
    'deepblend_playroom': False,   
    'deepblend_train': False,
    'deepblend_truck': False   
}

def get_rendering_fn(view_camera=None, **kwargs):
    return view_camera.original_image

def get_camera_fn(view_camera=None, **kwargs):
    raise ValueError()


class GetSceneInfo:
    def __init__(self, 
                 # scene-general
                 scene_dir=None,
                 scene_id=None,
                 white_background=None,
                 resolution_scale=1.0,
                 resolution_config=-1,
                    ) -> None:
        self.scene_info = None
        self.scene_id = scene_id
        self.white_background = white_background
        self.scene_dir = scene_dir
        self.resolution_scale = resolution_scale
        self.resolution_config = resolution_config



    def build_scene_info(self,
                         split_train_test,
                         llffhold,):
        if os.path.exists(os.path.join(self.scene_dir, 'sparse')):
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](self.scene_dir, 'images', split_train_test, llffhold=llffhold) # 每隔llffhold选一个test camera
        elif os.path.exists(os.path.join(self.scene_dir, 'transforms_train.json')):
            logging.info("Found transforms_train.json file, assuming Blender data set!")
            # path, white_background, eval, extension=".png"
            self.scene_info = sceneLoadTypeCallbacks["Blender"](self.scene_dir, self.white_background, split_train_test,)
        else:
            assert False, "Could not recognize scene type!"



    def __call__(self,  
                 # split-specific
                 split_train_test=None, # # is_eval=True -> train/test按照llffhold进行分割, 否则test都放到train里
                 llffhold=None,
                 split=None, # 如果self.condense是false的话 
                 condense=None,
                 register_name=None,
                 shuffle=True,):
        
        # mode: train/test/all: 要哪个split
        if self.scene_info is None:
            self.build_scene_info(split_train_test=split_train_test, llffhold=llffhold)  
            if shuffle:
                random.shuffle(self.scene_info.train_cameras) # # Multi-res consistent random shuffling
                random.shuffle(self.scene_info.test_cameras)  

        # 使用才注册的attributes
        MetadataCatalog.get(register_name).set(point_cloud=self.scene_info.point_cloud,
                                               cameras_extent=self.scene_info.nerf_normalization['radius'],
                                               )

        if split == 'train':
            assert split_train_test
            view_cameras = cameraList_from_camInfos(self.scene_info.train_cameras, 
                                                    resolution_scale=self.resolution_scale, resolution_config=self.resolution_config)
        elif split == 'test':
            assert split_train_test
            view_cameras = cameraList_from_camInfos(self.scene_info.test_cameras, 
                                                    resolution_scale=self.resolution_scale, resolution_config=self.resolution_config) 
            camera_ids = [haosen.uid for haosen in view_cameras]
            MetadataCatalog.get(register_name).set(eval_meta_keys={self.scene_id: camera_ids})

        elif split == 'all':
            assert (not split_train_test) and (llffhold == None) and (len(self.scene_info.test_cameras) == 0)
            view_cameras = cameraList_from_camInfos(self.scene_info.train_cameras, 
                                                    resolution_scale=self.resolution_scale, resolution_config=self.resolution_config)
            camera_ids = [haosen.uid for haosen in view_cameras]
            MetadataCatalog.get(register_name).set(eval_meta_keys={self.scene_id: camera_ids})

        else:
            raise ValueError()

        Scene_Meta
        if not condense:
            return [{
                'scene_id': self.scene_id,
                'scene_video_id': None,
                'scene_text': None,
                'view_camera': haosen,

                'metalog_name': register_name,
                'meta_idx': idx,

            } for idx, haosen in enumerate(view_cameras)]
        else:
            return [{
                'scene_id': self.scene_id,
                'scene_video_id': None,
                'scene_text': None,
                'view_cameras': view_cameras,

                'metalog_name': register_name,
                'meta_idx': 0,
            }]
            

for scene_id in scene_ids:
    Scene_Meta
    register_names = [f'{scene_id}_all', f'{scene_id}_train', f'{scene_id}_test', f'{scene_id}_all_condense'] # 所有视角, 训练视角, 测试视角, 
    # 每个scene 一般特征
    wbcg = scene_id_to_wbcg[scene_id]
    scene_dir = os.path.join(os.getenv('DATASET_PATH'), scene_id_to_dir[scene_id])
    scene_call_fn = [GetSceneInfo(scene_dir=scene_dir, scene_id=scene_id, white_background=wbcg, resolution_config=-1, resolution_scale=1.0), 
                     GetSceneInfo(scene_dir=scene_dir, scene_id=scene_id, white_background=wbcg, resolution_config=-1, resolution_scale=1.0),  # 共享, 保证build是一样的
                     GetSceneInfo(scene_dir=scene_dir, scene_id=scene_id, white_background=wbcg, resolution_config=-1, resolution_scale=1.0)]
    register_call_fn_idx = [0, 1, 1, 2]

    # 每个split的 特数特征
    register_split = ['all', 'train', 'test', 'all']
    register_mode = ['all', 'train', 'evaluate', 'all']
    register_llffhold = [None, 8, 8, None]

    register_condenses = [False, False, False, True]
    register_split_traintest = [False, True, True, False]
    


    for reg_name, reg_split, reg_condense, call_fn_idx, reg_split_tt, reg_mode, reg_llf in zip(register_names, register_split, register_condenses,
                                                                             register_call_fn_idx, register_split_traintest, register_mode, register_llffhold):
        DatasetCatalog.register(reg_name, partial(scene_call_fn[call_fn_idx],
                                                  
                                                  split_train_test=reg_split_tt,
                                                  llffhold=reg_llf,

                                                  split=reg_split,
                                                  condense=reg_condense,
                                                  register_name=reg_name,))
        
        MetadataCatalog.get(reg_name).set(scene_id=scene_id, # scene-general
                                          white_background=wbcg,
                                          get_rendering_fn=get_rendering_fn,
                                          get_extrinstic_fn=get_extrinstic_fn,
                                          get_intrinstic_fn=get_intrinstic_fn,
                                          
                                          mode=reg_mode, # split-specific
                                          condense=reg_condense,
                                          visualize_meta_idxs=[])
    


 
        
