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


from tqdm import tqdm



from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY

class AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                # list[3 3 h w] -> b 3 3 h w
                'images': torch.stack([item['image'] for item in batch_dict], dim=0),
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict],
                'image_ids':[item['image_id'] for item in batch_dict],
            }
        elif mode == 'evaluate':
            return {
                'metas': {
                    'image_ids': [item['image_id'] for item in batch_dict],
                    'meta_idxs': [item['meta_idx'] for item in batch_dict],
                },
                'images': torch.stack([item['image'] for item in batch_dict], dim=0), # b 3 h w
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()


@register_model
def extract_features(configs, device):
    # 假设只有一个训练数据集
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    pixel_mean = torch.tensor([0.485, 0.456, 0.406],).cuda().view(-1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225],).cuda().view(-1, 1, 1)
    backbone = BACKBONE_REGISTRY.get(configs['model']['backbone']['name'])(configs['model']['backbone'])
    backbone.eval()
    backbone.cuda() 
    backbone = torch.compile(backbone)
    target_file = os.path.join(configs['out_dir'], f"features")
    if os.path.exists(target_file):
        logging.warning('has been')
        exit()
    else:
        os.makedirs(target_file, exist_ok=True)
        with torch.no_grad():
            for i, batch_dict in tqdm(enumerate(train_loader)):
                image_ids = batch_dict['image_ids']
                images = batch_dict['images'].to('cuda:0')
                images = (images - pixel_mean) / pixel_std
                with torch.autocast('cuda'):
                    features = backbone(images) # b c h w
                for img_id, feat in zip(image_ids, features):
                    torch.save(feat.cpu(), os.path.join(target_file, f'{img_id}.bin'))
        exit()


@register_model
def nearest_neighbors(configs, device):
    # 假设只有一个训练数据集
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    pixel_mean = torch.tensor([0.485, 0.456, 0.406],).cuda().view(-1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225],).cuda().view(-1, 1, 1)
    backbone = BACKBONE_REGISTRY.get(configs['model']['backbone']['name'])(configs['model']['backbone'])
    backbone.eval()
    backbone.cuda() 
    backbone = torch.compile(backbone)
    target_file = os.path.join(configs['out_dir'], f"{configs['config']}.npz")
    if os.path.exists(target_file):
        logging.warning('has been')
        exit()
    else:
        os.makedirs(target_file, exist_ok=True)
        with torch.no_grad():
            all_feats = []
            all_image_ids = []
            for i, batch_dict in tqdm(enumerate(train_loader)):
                image_ids = batch_dict['image_ids']
                images = batch_dict['images'].to('cuda:0')
                images = (images - pixel_mean) / pixel_std
                all_image_ids.extend(image_ids)
                features = backbone(images) # b c h w
                features = torch.nn.functional.normalize(features.mean([2, 3]), dim=1) # pos信息会影响语义信息
                all_feats.append(features)

            all_feats = torch.cat(all_feats, dim=0) 
            all_nns = {}
            for imag_id, feat in zip(all_image_ids, all_feats):
                pairwise_sims = torch.einsum("f,mf->m", feat, all_feats) 
                top30_idxs = torch.topk(pairwise_sims, 30)[1].cpu().tolist()
                top30_img_ids = [all_image_ids[idx] for idx in top30_idxs]

                all_nns[imag_id] = top30_img_ids                
            torch.save(all_nns, target_file)
        exit()