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

class Cutler(OptimizeModel):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        parser_args = configs['model']['parser_args']

        from .train_net import setup, Trainer, verify_results
        from detectron2.checkpoint import DetectionCheckpointer
        args = Namespace(config_file="/home/xuhuihui/workspace/rvos_encoder/models/UN_IMG_SEM/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
                            test_dataset='',
                            train_dataset='',
                            no_segm=False,
                            resume=False,
                            eval_only=True,
                            opts=['MODEL.WEIGHTS','/home/xuhuihui/workspace/UnSAM/cutler_cascade_final.pth'])
        cfg = setup(args)
        self.model = Trainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        

    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        # list[dict]
        assert not self.training
        ret = self.model(batch_dict)
        predictions = ret['model_output']
        total_time = ret['process_time']
        return predictions        

# [10/11 15:24:34 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.evaluation.coco_evaluation coco_evaluation.py]: Evaluation results for segm: 
# |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl   |
# |:-----:|:------:|:------:|:-----:|:-----:|:------:|
# | 9.783 | 18.923 | 9.198  | 2.437 | 8.772 | 24.298 |
# [10/11 15:24:34 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.engine.defaults defaults.py]: Evaluation results for cls_agnostic_coco in csv format:
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: Task: bbox
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: Task: bbox
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: 12.3283,21.9745,11.9013,3.6529,12.7211,29.5985
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: 12.3283,21.9745,11.9013,3.6529,12.7211,29.5985
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: Task: segm
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: Task: segm
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: 9.7830,18.9234,9.1977,2.4367,8.7717,24.2982
# [10/11 15:24:34 d2.evaluation.testing]: copypaste: 9.7830,18.9234,9.1977,2.4367,8.7717,24.2982

        
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

class Divide_Conquer(OptimizeModel):
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
      
        namespace_parser.feature_dim = 768
        namespace_parser.backbone_size = 'base'
        namespace_parser.patch_size = 8

        self.parser_args = namespace_parser
    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        # list[dict]
        from data_schedule.unsupervised_image_semantic_seg import UNSEG_Eval_GenPseudoMask_API
        UNSEG_Eval_GenPseudoMask_API
        assert not self.training
        image = batch_dict[0]['image'].permute(1,2,0) # h w 3, 0-255
        H, W, _ = image.shape
        image = image.numpy()[:, :, ::-1]
        predictions, total_time = self.predictor(image)

        sample_ret = [{
            'instances': predictions["instances"]
        }]
        return sample_ret        

    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None
    
@register_model
def divide_conquer(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))

    model = Cutler(configs)
    model.to(device)

    return model, train_loader, eval_function