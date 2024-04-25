from .render_aug_utils import RENDER_EVAL_AUG_REGISTRY, RENDER_TRAIN_AUG_REGISTRY
import torch
from copy import deepcopy as dcopy
import cv2 as cv
from data_schedule.registry import Mapper
import copy
from data_schedule.render.apis import Scene_Meta

import torch.nn as nn

class Render_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        Scene_Meta
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.get_rendering_fn = dataset_meta.get('get_rendering_fn')

class Render_TrainMapper(Render_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        if 'augmentation' in mapper_config:  
            self.augmentation = RENDER_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        else:
            self.augmentation = lambda x: x
    

class Render_EvalMapper(Render_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        if 'augmentation' in mapper_config:
            self.augmentation = RENDER_EVAL_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        else:
            self.augmentation = lambda x: x
        


