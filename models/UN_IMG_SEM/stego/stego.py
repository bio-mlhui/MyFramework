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
                'images_pos': torch.stack([item['image_pos'] for item in batch_dict], dim=0),
                # 'image_feats': torch.stack([item['image_feat'] for item in batch_dict], dim=0),
                # 'images_pos_feats': torch.stack([item['image_pos_feat'] for item in batch_dict], dim=0),
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

class Featurizer(nn.Module):

    def __init__(self, 
                 configs,
                 patch_size,
                 embed_dim):
        super().__init__()
        self.distill_dim = configs['distill_dim']
        self.dropout = configs['dropout']

        self.feat_type = configs['dino_feat_type']
        self.proj_type = configs['projection_type']
        self.dropout = torch.nn.Dropout2d(p=.1)

        self.patch_size = patch_size
        self.n_feats = embed_dim


        self.cluster1 = self.make_clusterer(self.n_feats)
        
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.distill_dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.distill_dim, (1, 1)))

    @property
    def device(self):
        return self.cluster1[0].weight.device

    def forward(self, image_feat):
        code = self.cluster1(self.dropout(image_feat))
        if self.proj_type == "nonlinear":
            code += self.cluster2(self.dropout(image_feat))
    
        return self.dropout(image_feat), code

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


from .stegeo_modules import ClusterLookup, ContrastiveCorrelationLoss, ContrastiveCRFLoss
class Stego(OptimizeModel):
    def __init__(
        self,
        configs,
        num_classes,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        model_configs = configs['model']
        from transformers import AutoImageProcessor, AutoModel
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        # backbone = AutoModel.from_pretrained('facebook/dinov2-giant')        
        backbone = BACKBONE_REGISTRY.get(model_configs['backbone']['name'])(model_configs['backbone'])
        backbone.eval()
        backbone = torch.compile(backbone)
        self.backbone = backbone
        self.net = Featurizer(model_configs, backbone.patch_size, backbone.embed_dim)
        self.num_classes = num_classes

        self.patch_size = backbone.patch_size
        self.embed_dim = self.net.distill_dim

        self.num_queries = model_configs['num_queries']
        if self.num_queries == 'same_to_num_class':
            self.num_queries = self.num_classes
        assert self.num_queries >= self.num_classes

        self.cluster_probe = ClusterLookup(self.embed_dim, self.num_queries)

        self.loss_weight = {
            'correspondence_weight': model_configs['loss']['correspondence_weight'],
            'neg_inter_weight': model_configs['loss']['neg_inter_weight'],
            'pos_inter_weight': model_configs['loss']['pos_inter_weight'],
            'pos_intra_weight': model_configs['loss']['pos_intra_weight'],
            'neg_inter_shift': model_configs['loss']['neg_inter_shift'],
            'pos_inter_shift': model_configs['loss']['pos_inter_shift'],
            'pos_intra_shift': model_configs['loss']['pos_intra_shift'],
        }
        self.use_salience = model_configs['loss']['use_salience']
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(model_configs['loss'])
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return self.pixel_mean.device
        
    def optimize_setup(self, configs):
        main_params = list(self.net.parameters())
        self.net_optim = torch.optim.Adam(main_params, lr=configs['lr'])
        self.cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)


    def forward_backward(self, batch_dict):
        assert self.training
        # b 3 3 h w
        img = batch_dict['images'].to(self.device) 
        img = img - self.pixel_mean / self.pixel_std

        img_pos = batch_dict['images_pos'].to(self.device) 
        img_pos = img_pos - self.pixel_mean / self.pixel_std
        
        with torch.autocast(device_type="cuda"):
            if self.use_salience:
                salience = batch_dict["mask"].to(torch.float32).squeeze(1)
                salience_pos = batch_dict["mask_pos"].to(torch.float32).squeeze(1)
            else:
                salience = None
                salience_pos = None

            with torch.no_grad():
                image_feats = self.backbone(img)
                image_pos_feats = self.backbone(img_pos)

            feats, code = self.net(image_feats)
            if self.loss_weight['correspondence_weight'] > 0:
                feats_pos, code_pos = self.net(image_pos_feats)
        
            signal = feats
            signal_pos = feats_pos

            loss = 0

            if self.loss_weight['correspondence_weight'] > 0:
                (
                    pos_intra_loss, pos_intra_cd,
                    pos_inter_loss, pos_inter_cd,
                    neg_inter_loss, neg_inter_cd,
                ) = self.contrastive_corr_loss_fn(
                    signal, signal_pos,
                    salience, salience_pos,
                    code, code_pos,
                )

                neg_inter_loss = neg_inter_loss.mean()
                pos_intra_loss = pos_intra_loss.mean()
                pos_inter_loss = pos_inter_loss.mean()

                loss += (self.loss_weight['pos_inter_weight'] * pos_inter_loss +
                        self.loss_weight['pos_intra_weight'] * pos_intra_loss +
                        self.loss_weight['neg_inter_weight'] * neg_inter_loss) * self.loss_weight['correspondence_weight']

            detached_code = torch.clone(code.detach())       
            cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
            loss += cluster_loss
        
        loss_dict = {
            'pos_inter_loss': pos_inter_loss.cpu().item(),
            'pos_intra_loss': pos_intra_loss.cpu().item(),
            'neg_inter_loss': neg_inter_loss.cpu().item(),
            'cluster_loss': cluster_loss.cpu().item(),
            'loss': loss.cpu().item(),
        }

        self.net_optim.zero_grad()
        self.cluster_probe_optim.zero_grad()
        loss.backward()
        self.net_optim.step()
        self.cluster_probe_optim.step()

        return loss_dict

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        images = batch_dict['images']
        images = images.to(self.device) 
        images = images - self.pixel_mean / self.pixel_std
        H, W = images.shape[-2:]

        feats = self.backbone(images)
        feats, code = self.net(feats)
        code = F.interpolate(code, (H, W), mode='bilinear', align_corners=False)
        cluster_loss, cluster_preds = self.cluster_probe(code, None)

        return  {
            'pred_masks': cluster_preds,
        }

    def optimize_state_dict(self,):
        return {
            'net_optim': self.net_optim.state_dict(),
            'cluster_probe_optim': self.cluster_probe_optim.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.net_optim.load_state_dict(state_dict=state_dict['net_optim'])
        self.cluster_probe_optim.load_state_dict(state_dict=state_dict['cluster_probe_optim'])


    def get_lr_group_dicts(self, ):
        return  {f'lr_net': self.net_optim.param_groups[0]["lr"],
                 f'lr_cluster': self.cluster_probe_optim.param_groups[0]["lr"]}

from tqdm import tqdm
@register_model
def stego(configs, device):
    # 假设只有一个训练数据集
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    model = Stego(configs, num_classes=num_classes)
    model.to(device)
    #model.net.backbone = torch.compile(model.net.backbone)
    model.optimize_setup(configs['optim'])
    
    # import torch_tensorrt
    # model.net.backbone = torch_tensorrt.compile(model.net.backbone, 
    #         inputs= [torch_tensorrt.Input((1, 3, 896, 896))],
    #         enabled_precisions= { torch_tensorrt.dtype.half} # Run with FP16
    # )
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

