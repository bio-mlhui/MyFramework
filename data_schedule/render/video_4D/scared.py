
# import os
# import numpy as np
# from joblib import Parallel, delayed
# import multiprocessing
# import torch.distributed as dist
# import detectron2.utils.comm as comm
# from glob import glob
# from PIL import Image
# import cv2

# import json
# import logging
# from data_schedule.render.scene.dataset_readers import sceneLoadTypeCallbacks
# from functools import partial
# from detectron2.data import DatasetCatalog, MetadataCatalog

# # images: 代表multi-view图像
# # sparse: cameras.bin, images.bin, points3D.bin...

# dataset_ids = ['scared',]

# dataset_dirs = {
#     'scared': 'medical/tandb',
#     'db_deepblend': 'deepblend/db',
# }

# dataset_prefix = {
#     'db_tandt': 'tandt',
#     'db_deepblend': 'deepblend',
# }

# dataset_scene_num = {
#     'db_tandt': 2,
#     'db_deepblend': 2,
# }

# dataset_sparse_or_tt = {
#     'db_tandt': 2,
#     'db_deepblend': 2,   
# }

# dataset_wbcg = {
#     'db_tandt': 2,
#     'db_deepblend': 2,   
# }

# dataset_iseval = {
#     'db_tandt': 2,
#     'db_deepblend': 2,   
# }

# dataset_imagedir = {
#     'db_tandt': None,
#     'db_deepblend': None
# }


# def get_rendering_fn(scene_id=None, 
#                      view_camera=None,
#                      time_step=None,
#                     **kwargs):
#     # 根据view_camera 得到
#     return [Image.open(os.path.join(frames_path, video_id, f'{f}.jpg')).convert('RGB') for f in frames]

# def get_scene_info(scene_path, 
#                     sparse_or_tt=None,
#                     image_dir=None, 
#                     white_background=None, 
#                     eval=None):
#     if sparse_or_tt == 'sparse':
#         assert os.path.exists(os.path.join(scene_path, "sparse"))
#         scene_info = sceneLoadTypeCallbacks["Colmap"](scene_path, image_dir, eval)
#     elif sparse_or_tt == 'tt':
#         assert os.path.exists(os.path.join(scene_path, "transforms_train.json"))
#         logging.info("Found transforms_train.json file, assuming Blender data set!")
#         scene_info = sceneLoadTypeCallbacks["Blender"](scene_path, white_background, eval)
#     else:
#         assert False, "Could not recognize scene type!"
#     return scene_info

# class GetSceneInfo:
#     def __init__(self, 
#                  scene_path=None,
#                  sparse_or_tt=None,
#                  image_dir=None, 
#                  white_background=None, 
#                  is_eval=None) -> None:
#         self.scene_info = None
#         self.scene_path = scene_path
#         self.sparse_or_tt = sparse_or_tt
#         self.image_dir = image_dir
#         self.wbcg = white_background
#         self.is_eval = is_eval


#     def __call__(self, mode) -> json.Any:
#         if self.scene_info is None:
#             self.scene_info = get_scene_info(scene_path=self.scene_path,
#                                              sparse_or_tt=self.sparse_or_tt,
#                                              image_dir=self.image_dir,
#                                              white_background=self.wbcg,
#                                              eval=self.is_eval)                  
#         if mode == 'train':
#             return self.scene_info.train_cameras
#         elif mode == 'test':
#             return self.scene_info.test_cameras           
#         else:
#             raise ValueError()


# for dataset_id in dataset_ids:
#     dataset_root = os.path.join(os.getenv('DATASET_PATH'), dataset_dirs[dataset_id])
#     scene_names = os.listdir(dataset_root)
#     assert len(scene_names) == dataset_scene_num[dataset_id]
#     sparse_or_tt = dataset_sparse_or_tt[dataset_id]
#     wbcg = dataset_wbcg[dataset_id]
#     is_eval = dataset_iseval[dataset_id]
#     image_dir = dataset_imagedir[dataset_id]
    
#     for haosen in scene_names:
#         scene_id =  f'{dataset_id}_{haosen}'
#         scene_root = os.path.join(dataset_root, haosen)
#         scene_call_fn = GetSceneInfo(scene_path=scene_root,
#                                     sparse_or_tt=sparse_or_tt,
#                                     white_background=wbcg,
#                                     image_dir=image_dir,
#                                     is_eval=is_eval)


#         DatasetCatalog.register(f'{scene_id}_train', partial(scene_call_fn, mode='train', register_name=f'{scene_id}_train'))

#         MetadataCatalog.get(f'{scene_id}_train').set(point_cloud= scene_call_fn.point_cloud,
#                                                     nerf_normalization_radius=scene_call_fn.nerf_normalization['radius'],
#                                                     sparse_or_tt=sparse_or_tt, 
#                                                     wbcg=wbcg, 
#                                                     is_eval=is_eval,
#                                                     get_rendering_fn=get_rendering_fn)
    
#         DatasetCatalog.register(f'{scene_id}_test', partial(scene_call_fn, mode='test', register_name=f'{scene_id}_test'))
#         MetadataCatalog.get(f'{scene_id}_test').set(point_cloud= scene_call_fn.point_cloud,
#                                                     nerf_normalization_radius=scene_call_fn.nerf_normalization['radius'],
#                                                     sparse_or_tt=sparse_or_tt, 
#                                                     wbcg=wbcg, 
#                                                     is_eval=is_eval,
#                                                     get_rendering_fn=get_rendering_fn)    
        
