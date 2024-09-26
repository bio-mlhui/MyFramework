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

class AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                # list[3 3 h w] -> b 3 3 h w
                'images': torch.stack([item['image'] for item in batch_dict], dim=0),
                'instance_masks': [item['instance_mask'] for item in batch_dict],
                'image_ids': [item['image_id'] for item in batch_dict],
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict]
            }
        elif mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                },
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'instance_masks': [item['instance_mask'] for item in batch_dict], # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()

from .divide_conquer_utils import smallest_square_containing_mask, coverage, resize_mask, NMS, generate_feature_matrix, setup_cfg, DefaultPredictor, area
from .iterative_merging import iterative_merge
from PIL import Image
from .coco_annotator import create_annotation_info, output, category_info
from .cascadepsp import postprocess
from pycocotools import mask as mask_utils

class Divide_and_Conquer(OptimizeModel):
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

        # coco format annotator initialization
        divide_conquer_masks = []
        output["image"], output["annotations"] = {}, []

        # Divide phase
        # input: bgr numpy
        image = np.copy((image * 255).numpy().astype(np.uint8)[:, :, ::-1])
        predictions = self.predictor(image)
        divide_masks_tensor = predictions["instances"].get("pred_masks")
        divide_masks = []
        for i in range(divide_masks_tensor.shape[0]):
            divide_masks.append(divide_masks_tensor[i,:,:].cpu().numpy())
        divide_conquer_masks.extend(divide_masks)

        # Conquer phase
        for divide_mask in divide_masks:
            conquer_masks = []
            # find the bounding box and resize the original images
            ymin, ymax, xmin, xmax = smallest_square_containing_mask(divide_mask)
            if (ymax-ymin) <= 0 or (xmax-xmin) <= 0: continue
            local_image = image[ymin:ymax, xmin:xmax]
            resized_local_image = Image.fromarray(local_image).resize([self.parser_args.local_size, self.parser_args.local_size])
            # h w c
            feature_matrix = generate_feature_matrix(self.backbone, resized_local_image, self.parser_args.feature_dim, self.parser_args.local_size//self.parser_args.patch_size)
            merging_masks = iterative_merge(feature_matrix, self.parser_args.thetas)
            
            for layer in merging_masks:
                if layer.shape[0] == 0: continue

                for i in range(layer.shape[0]):
                    mask = layer[i, :, :]
                    mask = resize_mask(mask, [xmax-xmin, ymax-ymin])
                    mask = (mask > 0.5 * 255).astype(int)

                    if coverage(mask, divide_mask[ymin:ymax, xmin:xmax]) <= self.parser_args.kept_thresh: continue
                    enlarged_mask = np.zeros_like(divide_mask)
                    enlarged_mask[ymin:ymax, xmin:xmax] = mask
                    conquer_masks.append(enlarged_mask)

            conquer_masks = NMS(conquer_masks, self.parser_args.NMS_iou, self.parser_args.NMS_step)
            divide_conquer_masks.extend(conquer_masks)

        for m in divide_conquer_masks:
            # create coco-style annotation info 
            annotation_info = create_annotation_info(
                segmentation_id, 0, category_info, m.astype(np.uint8), None)
            if annotation_info is not None:
                output["annotations"].append(annotation_info)
                segmentation_id += 1
        
        # postprocess CascadePSP
        if self.parser_args.postprocess:    
            refined_annotations = postprocess(self.parser_args, self.refiner, output, image)
            output["annotations"] = refined_annotations["annotations"]
        
        visualized_masks = []
        for mask_encoded in output["annotations"]:
            mask = mask_utils.decode(mask_encoded['segmentation'])
            visualized_masks.append(mask)
        sorted_masks = sorted(visualized_masks, key=lambda m: area(m), reverse=True)
        sorted_masks = [torch.from_numpy(foo) for foo in sorted_masks]
        sorted_masks = torch.stack(sorted_masks, dim=0).bool()[None, ...] # N h w 
        return {'pred_masks': sorted_masks}

        
    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None
    
@register_model
def divide_and_conquer(configs, device):
    train_dataset_name = list(configs['data']['train'].keys())[0]
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))

    model = Divide_and_Conquer(configs)
    model.to(device)

    return model, train_loader, eval_function

