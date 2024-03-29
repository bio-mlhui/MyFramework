
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
from data_schedule.render.apis import Scene_Dataset, Scene_Mapper, Scene_Terminology

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
        Scene_Dataset
        Scene_Mapper
        Scene_Terminology

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
        Scene_Dataset
        Scene_Mapper
        Scene_Terminology
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


# class Text4D_Condense_TrainMapper


# class Text4D_TrainMapper


