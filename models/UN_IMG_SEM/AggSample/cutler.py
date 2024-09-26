"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import os
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
import logging
import torchvision.transforms.functional as Trans_F
import copy
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import math
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
class OptimizeModel(nn.Module):
    """
    optimize_setup:
        optimizer, scheduler都是标准类
        log_lr_idx随着训练不改变
        
    optimize:
        backward, optimzier_step, optimizer_zero_grad, scheduler_step
        
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.log_lr_group_idx: Dict = None

    def optimize_setup(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.data import MetadataCatalog
import cv2
from argparse import Namespace

from models.UN_IMG_SEM.divide_and_conquer.divide_conquer_utils import smallest_square_containing_mask, coverage, resize_mask, NMS, generate_feature_matrix, setup_cfg, DefaultPredictor, area
from models.UN_IMG_SEM.divide_and_conquer.iterative_merging import iterative_merge
from PIL import Image
from models.UN_IMG_SEM.divide_and_conquer.coco_annotator import create_annotation_info, output, category_info
from models.UN_IMG_SEM.divide_and_conquer.cascadepsp import postprocess
from pycocotools import mask as mask_utils

class Cutler(OptimizeModel):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        import segmentation_refinement as refine
        parser_args = configs['model']['parser_args']
        model_configs = configs['model']
        namespace_parser = Namespace(**parser_args)
        self.refiner = refine.Refiner(device='cuda:0')
        cfg = setup_cfg(namespace_parser)

        # divide
        self.predictor = DefaultPredictor(cfg)

        assert model_configs['backbone']['type'] == 'dinov1_vitb8'
        self.backbone = BACKBONE_REGISTRY.get(model_configs['backbone']['name'])(model_configs['backbone'])
        self.backbone.eval().cuda().to(torch.float16)
      
        namespace_parser.feature_dim = 768
        namespace_parser.backbone_size = 'base'
        namespace_parser.patch_size = 8

        self.parser_args = namespace_parser
    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        segmentation_id = 1
        assert not self.training
        self.backbone.eval()
        image = batch_dict['images'][0].permute(1,2,0) # h w 3, 0-1float

        # input: bgr numpy
        image = np.copy((image * 255).numpy().astype(np.uint8)[:, :, ::-1])
        predictions = self.predictor(image)

        return {'pred_masks': [predictions["instances"].get("pred_masks")], # n h w, bool
                'pred_scores': [predictions["instances"].get("scores")]} # n, 0-1

        
    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None
    
@register_model
def cutler_pt(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))

    model = Cutler(configs)
    model.to(device)

    return model, train_loader, eval_function

