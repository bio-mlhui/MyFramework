
import json
import os
from typing import List
import copy
from functools import partial
import random
import numpy as np
import torch
import logging
from einops import rearrange
from data_schedule.utils.segmentation import bounding_box_from_mask
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import Render_TrainMapper, Render_EvalMapper
from .render_view_sampler import RENDER_VIEWS_SAMPLER_REGISTRY
from data_schedule.render.apis import Multiview3D_Optimize_Mapper, Scene_Meta, SingleView_3D_Mapper, Text_3D_Mapper
import torch.nn.functional as F
from data_schedule.render.scene_utils.cameras import MiniMiniCam

@MAPPER_REGISTRY.register()
class Image3D_TrainMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'train':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_camera, metalog_name = data_dict['scene_id'], data_dict['view_camera'], data_dict['metalog_name']

        rendering = self.get_rendering_fn(scene_id=scene_id, 
                                          view_camera=view_camera)
        data_dict['rendering'] = rendering
        data_dict = self.augmentation(data_dict)
        Multiview3D_Optimize_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name']
            },
            'view_dict': {
                'view_camera': data_dict['view_camera']
            },
            'rendering_dict': {
                'rendering': data_dict['rendering']
            }
        }


@MAPPER_REGISTRY.register()
class Image3D_EvalMapper(Render_EvalMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'evaluate':
            logging.warning('this mapper should be used on evaluate datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        data_dict = self.augmentation(data_dict)
        callback_fns = data_dict.pop('callback_fns', [])[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        Multiview3D_Optimize_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name'],
                'callback_fns': callback_fns
            },
            'view_dict': {
                'view_camera': data_dict['view_camera'],
            },
        }


@MAPPER_REGISTRY.register()
class SingleView_3D_TrainMapper_BlenderSameIntrin(Render_TrainMapper):
    """
    同一个sample, camera的内参完全一样
    """ 
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'train':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        
        self.multiview_sampler = RENDER_VIEWS_SAMPLER_REGISTRY.get(mapper_config['multiview_sampler']['name']\
                                                                   )(sampler_configs=mapper_config['multiview_sampler'],
                                                                    dataset_meta=dataset_meta)
        self.camera = MetadataCatalog.get(dataset_name).get('camera_intrin') # camera

        self.input_view_size = mapper_config['input_view_size']
        self.output_view_size = mapper_config['output_view_size']
        self.get_c2w_fn = partial(self.get_camera_fn, 
                                  only_c2w=True, world_format='opengl', camera_format='opengl')
        
        self.get_rendering_fn = partial(self.get_rendering_fn, return_alpha=True)

    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_cameras, metalog_name = data_dict['scene_id'], data_dict['view_cameras'], data_dict['metalog_name']
        
        input_views, output_views = self.multiview_sampler(all_cameras=view_cameras)
        sampled_views = input_views + output_views
        num_input_views = len(input_views)

        rgbs, alphas = list(zip(*[self.get_rendering_fn(scene_id=scene_id, view_id=haosen) for haosen in sampled_views]))

        c2ws = [self.get_c2w_fn(scene_id=scene_id, view_id=haosen,) for haosen in sampled_views]

        rgbs = torch.stack(rgbs, dim=0) # [V, C, H, W]
        alphas = torch.stack(alphas, dim=0) # [V, H, W]
        c2ws = torch.stack(c2ws, dim=0) # [V, 4, 4] 
        
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.camera.radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(c2ws[0])
        c2ws = transform.unsqueeze(0) @ c2ws  # [V, 4, 4]

        # [V_in, C, H, W]
        in_rendering_rgbs = F.interpolate(rgbs[:num_input_views].clone(), 
                                          size=(self.input_view_size, self.input_view_size), mode='bilinear', align_corners=False)
        in_cam_poses = c2ws[:num_input_views].clone()
        in_dict = {
            'rendering_rgbs': in_rendering_rgbs, # V_in C H W
            'extrin': in_cam_poses, # V_in 4 4 
            'intrin': self.camera
        }
        aug_in_dict = self.augmentation(in_dict)
        SingleView_3D_Mapper
        return {
            'scene_dict':{
                'scene_id': scene_id,
                'metalog_name': metalog_name,
            },
            # 输入的views
            'inviews_dict':{
                'intrin': self.camera,
                'extrin': aug_in_dict['extrin'],
                'rendering_rgbs': aug_in_dict['rendering_rgbs'],
            },
            # 输出的views
            'outviews_dict':{
                'intrin': self.camera,
                'extrin': c2ws,
                # resize render ground-truth images, range still in [0, 1]
                'rendering_rgbs': F.interpolate(rgbs,  size=(self.output_view_size, self.output_view_size), mode='bilinear', align_corners=False), 
                                                # [V, C, output_size, output_size],
                'rendering_alphas': F.interpolate(alphas.unsqueeze(1), size=(self.output_view_size, self.output_view_size), mode='bilinear',align_corners=False)
                                                # [V, 1, output_size, output_size
            }
        }
                  

@MAPPER_REGISTRY.register()
class SingleView_3D_EvalMapper_BlenderSameIntrin(Render_EvalMapper):
    """
    同一个sample, camera的内参完全一样
    """ 
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'evaluate':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        
        self.multiview_sampler = RENDER_VIEWS_SAMPLER_REGISTRY.get(mapper_config['multiview_sampler']['name']\
                                                                   )(sampler_configs=mapper_config['multiview_sampler'],
                                                                    dataset_meta=dataset_meta)
        self.camera = MetadataCatalog.get(dataset_name).get('camera_intrin') # camera

        self.input_view_size = mapper_config['input_view_size']
        self.output_view_size = mapper_config['output_view_size']
        self.get_extrinsic_fn = partial(self.get_camera_fn, 
                                        camera_radius=self.camera.radius,
                                        only_extrinsic=True)

    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_cameras, metalog_name = data_dict['scene_id'], data_dict['view_cameras'], data_dict['metalog_name']
        
        input_views, output_views = self.multiview_sampler(all_cameras=view_cameras)
        sampled_views = input_views + output_views
        num_input_views = len(input_views)

        rgbs, alphas = list(zip(*[self.get_rendering_fn(haosen) for haosen in sampled_views]))

        cam_poses = [self.get_extrinsic_fn(haosen) for haosen in sampled_views]

        rgbs = torch.stack(rgbs, dim=0) # [V, C, H, W]
        alphas = torch.stack(alphas, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4] 
        
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.camera.radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # [V_in, C, H, W]
        in_rendering_rgbs = F.interpolate(rgbs[:num_input_views].clone(), 
                                          size=(self.input_view_size, self.input_view_size), mode='bilinear', align_corners=False)
        in_cam_poses = cam_poses[:num_input_views].clone()
        in_dict = {
            'rendering_rgbs': in_rendering_rgbs, # V_in C H W
            'extrin': in_cam_poses, # V_in 4 4 
            'intrin': self.camera
        }
        aug_in_dict = self.augmentation(in_dict)
        SingleView_3D_Mapper
        return {
            'scene_dict':{
                'scene_id': scene_id,
                'metalog_name': metalog_name,
            },
            # 输入的views
            'inviews_dict':{
                'intrin': self.camera,
                'extrin': aug_in_dict['extrin'],
                'rendering_rgbs': aug_in_dict['rendering_rgbs'],
            },
            # 输出的views
            'outviews_dict':{
                'intrin': self.camera,
                'extrin': cam_poses,
                # resize render ground-truth images, range still in [0, 1]
                'rendering_rgbs': F.interpolate(rgbs, 
                                                size=(self.output_view_size, self.output_view_size), mode='bilinear', align_corners=False), 
                                                # [V, C, output_size, output_size],
                'rendering_alphas': F.interpolate(alphas.unsqueeze(1), 
                                                  size=(self.output_view_size, self.output_view_size), mode='bilinear',align_corners=False)
                                                # [V, 1, output_size, output_size
            }
        }
        


@MAPPER_REGISTRY.register()
class Text3D_Optimize_TrainMapper_SameIntrin(Render_TrainMapper):
    # 只有一个meta
    # meta里只有text
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'train':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        self.get_c2w_fn = partial(self.get_camera_fn, 
                                  only_c2w=True, world_format='opengl', camera_format='opengl')


    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_cameras, metalog_name = data_dict['scene_id'], data_dict['view_cameras'], data_dict['metalog_name']
        scene_text = data_dict['scene_text']

        Text_3D_Mapper
        return {
            'scene_dict':{
                'scene_id': scene_id,
                'metalog_name': metalog_name,
                'scene_text': scene_text
            },
            'text_dict':{
                'text': scene_text,
            },
            # 输出的views
            'outviews_dict':{
            }
        }
                  

@MAPPER_REGISTRY.register()
class Text3D_Optimize_EvalMapper_SameIntrin(Render_EvalMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'evaluate':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        
        self.get_c2w_fn = partial(self.get_camera_fn, 
                                  only_c2w=True, world_format='opengl', camera_format='opengl')

    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_cameras, metalog_name = data_dict['scene_id'], data_dict['view_cameras'], data_dict['metalog_name']
        scene_text = data_dict['scene_text']

        # 4个test views
        # c2ws = [self.get_c2w_fn(haosen) for haosen in view_cameras]

        Text_3D_Mapper
        return {
            'scene_dict':{
                'scene_id': scene_id,
                'metalog_name': metalog_name,
                'scene_text': scene_text
            },
            'text_dict':{
                'text': scene_text,
            },
            # 输出的views
            'outviews_dict':{
                # 'extrin': c2ws,
            }
        }



# class Text4D_Condense_TrainMapper
# class Text4D_TrainMapper

@MAPPER_REGISTRY.register()
class Video4D_TrainMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        rendering = self.get_rendering_fn(scene_id=data_dict['scene_id'], 
                                           scene_video_id=data_dict['scene_video_id'],
                                           view_camera=data_dict['view_camera'])
        
        data_dict['rendering'] = rendering

        data_dict = self.augmentation(data_dict)

        return {
            'view_dict': {'viewcamera': data_dict['view_camera'] },
            'targets': {
                'rendering': data_dict['rendering']
            }
        }


@MAPPER_REGISTRY.register()
class Video4D_EvalMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        rendering = self.get_rendering_fn(scene_id=data_dict['scene_id'], 
                                           scene_video_id=data_dict['scene_video_id'],
                                           view_camera=data_dict['view_camera'])
        
        data_dict['rendering'] = rendering

        data_dict = self.augmentation(data_dict)
        callback_fns = data_dict.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        return {
            'view_dict': {'viewcamera': data_dict['view_camera'] },
            'meta': {
                'scene_id': data_dict['scene_id'],
                'scene_video_id': data_dict['scene_video_id'],
                'callback_fns': callback_fns
            }
        }



