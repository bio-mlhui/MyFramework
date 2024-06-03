
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
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import VIDVID_Mapper

@MAPPER_REGISTRY.register()
class Optimize_Mapper(VIDVID_Mapper):
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
    def _call(self, data_dict):
        return data_dict






