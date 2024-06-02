
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
from .mapper_utils import Optimize_EvalMapper, Learn_TrainMapper
from data_schedule.videnoise.apis import VIDenoise_Meta, VIDenoise_LearnMapper, VIDenoise_OptimizeMapper

@MAPPER_REGISTRY.register()
class VIDenoiseOptimize_EvalMapper(Optimize_EvalMapper):
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
    VIDenoise_Meta
    def _call(self, data_dict):
        ret = {}
        ret['video_dict'] = {}
        ret['metas'] = data_dict
        return ret

@MAPPER_REGISTRY.register()
class VIDenoiseOptimize_TrainMapper(Optimize_EvalMapper):
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('name') == dataset_name
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict):
   
        ret = {}
        ret['video_dict'] = {}
        ret['targets'] = {}
        return ret







