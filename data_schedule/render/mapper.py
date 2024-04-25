
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
from data_schedule.render.apis import Scene_Mapper, Scene_Meta


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
        Scene_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name']
            },
            'view_dict': {
                'viewcamera': data_dict['view_camera']
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
        Scene_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name'],
                'callback_fns': callback_fns
            },
            'view_dict': {
                'viewcamera': data_dict['view_camera']
            },
        }



# class Text4D_Condense_TrainMapper
# class Text4D_TrainMapper


