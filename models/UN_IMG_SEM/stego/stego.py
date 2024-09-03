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

    @torch.no_grad()
    def sample_point_similarities(self, backbone_features, code_features, num_points):
        # b c h w, num_points
        H_P, W_P = code_features.shape[-2:]
        H, W = H_P * self.patch_size, W_P * self.patch_size
        sampled_points = torch.rand(num_points, 2)
        sampled_points[:, 0] = sampled_points[:, 0] * H_P
        sampled_points[:, 1] = sampled_points[:, 1] * W_P
        sampled_points = sampled_points.long()
        sampled_points[:, 0].clamp_(0, H_P-1)
        sampled_points[:, 1].clamp_(0, W_P-1)
        similarities = []
        for point in sampled_points:
            query = code_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(code_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            similarities.append(sim)

        backbone_similarities = []
        for point in sampled_points:
            query = backbone_features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(backbone_features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            backbone_similarities.append(sim)

        sampled_points = sampled_points * self.patch_size
        return sampled_points, similarities, backbone_similarities


    @torch.no_grad()
    def self_cluster(self, features, gt_masks):
        from models.UN_IMG_SEM.kmeans.kmeans import kmeans
        # b c h w -> b
        _, _, H_P, W_P = features.shape
        assert features.shape[0] == 1
        if self.kmeans_strategy == 'adaptive':
            num_image_classes = len(set(gt_masks.unique().tolist()) - set([-1]))
        else:
            raise ValueError()
        features = features.permute(0, 2,3,1).flatten(0,2) # bhw c
        _, cluster_centers = kmeans(X=features, num_clusters=num_image_classes, device=self.device) # num c
        cluster_logits = torch.einsum('sc,nc->sn', 
                                    F.normalize(features, dim=-1, eps=1e-10),
                                    F.normalize(cluster_centers.to(self.device), dim=-1, eps=1e-10))
        # 把和cluster_center最近的点标注出来
        cluster_logits = rearrange(cluster_logits, '(b h w) n -> b n h w', b=1,h=H_P, w=W_P)
        cluster_logits = F.interpolate(cluster_logits, size=(H_P*self.patch_size, W_P*self.patch_size), mode='bilinear', align_corners=False)
        cluster_ids = cluster_logits.max(dim=1)[1].cpu()
        return cluster_ids, cluster_logits, num_image_classes


def build_optimizer(main_params, linear_params, cluster_params, opt: dict, ):
    # opt = opt["optimizer"]
    net_optimizer_type = opt["net"]["name"].lower()
    if net_optimizer_type == "adam":
        net_optimizer = torch.optim.Adam(main_params, lr=opt["net"]["lr"])
    elif net_optimizer_type == "adamw":
        net_optimizer = torch.optim.AdamW(main_params, lr=opt["net"]["lr"], weight_decay=opt["net"]["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer type {net_optimizer_type}.")

    linear_probe_optimizer_type = opt["linear"]["name"].lower()
    if linear_probe_optimizer_type == "adam":
        linear_probe_optimizer = torch.optim.Adam(linear_params, lr=opt["linear"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer type {linear_probe_optimizer_type}.")

    cluster_probe_optimizer_type = opt["cluster"]["name"].lower()
    if cluster_probe_optimizer_type == "adam":
        cluster_probe_optimizer = torch.optim.Adam(cluster_params, lr=opt["cluster"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer type {cluster_probe_optimizer_type}.")


    return net_optimizer, linear_probe_optimizer, cluster_probe_optimizer


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
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict],
                'image_ids':[item['image_id'] for item in batch_dict],
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0),
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

class DinoFeaturizer(nn.Module):

    def __init__(self, 
                 configs,):
        super().__init__()
        self.distill_dim = configs['dim']

        self.feat_type = configs['pretrained']['dino_feat_type']
        self.proj_type = configs['pretrained']['projection_type']

        self.dropout = torch.nn.Dropout2d(p=.1)

        
        self.backbone = BACKBONE_REGISTRY.get(configs['pretrained']['name'])(configs['pretrained'])
        self.backbone.eval().cuda()

        self.patch_size = self.backbone.patch_size
        self.n_feats = self.backbone.embed_dim


        self.cluster1 = self.make_clusterer(self.n_feats)
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

    def forward(self, img, n=1):
        self.backbone.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            features = self.backbone(img, n=n)['features']
            feat = features[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))
            
        code = self.cluster1(self.dropout(image_feat))
        if self.proj_type == "nonlinear":
            code += self.cluster2(self.dropout(image_feat))

        return self.dropout(image_feat), code
    

from .stegeo_modules import ClusterLookup, ContrastiveCorrelationLoss
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

        self.net = DinoFeaturizer(model_configs,)
        self.num_classes = num_classes
        self.patch_size = self.net.patch_size
        self.embed_dim = self.net.distill_dim
        self.num_queries = num_classes
        self.cluster_probe = ClusterLookup(self.embed_dim, num_classes)
        self.linear_probe = nn.Conv2d(self.embed_dim, num_classes, (1, 1))

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(model_configs['loss'])
        self.loss_weight = {
            'neg_inter_weight': model_configs['loss']['neg_inter_weight'],
            'pos_inter_weight': model_configs['loss']['pos_inter_weight'],
            'pos_intra_weight': model_configs['loss']['pos_intra_weight'],
        }
        self.kmeans_strategy = 'adaptive'
    @property
    def device(self):
        return self.pixel_mean.device
        
    def optimize_setup(self, configs):
        self.net_optim, self.linear_probe_optim, self.cluster_probe_optim = build_optimizer(
            main_params=self.net.parameters(),
            linear_params=self.linear_probe.parameters(),
            cluster_params=self.cluster_probe.parameters(),
            opt=configs['optimizer'],)

    def forward_backward(self, batch_dict):
        assert self.training
        # b 3 3 h w
        img = batch_dict['images'].to(self.device) 
        img = (img - self.pixel_mean) / self.pixel_std

        img_pos = batch_dict['images_pos'].to(self.device) 
        img_pos = (img_pos - self.pixel_mean) / self.pixel_std

        label = batch_dict['masks'].to(self.device)
        
        with torch.autocast(device_type="cuda"):
            salience = None
            salience_pos = None

            feats, code = self.net(img)
            feats_pos, code_pos = self.net(img_pos)
        
            signal = feats
            signal_pos = feats_pos

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

            loss = self.loss_weight['pos_inter_weight'] * pos_inter_loss +\
                    self.loss_weight['pos_intra_weight'] * pos_intra_loss +\
                    self.loss_weight['neg_inter_weight'] * neg_inter_loss

            flat_label = label.reshape(-1)
            mask = (flat_label >= 0) & (flat_label < self.num_classes)

            detached_code = torch.clone(code.detach())

            linear_logits = self.linear_probe(detached_code)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            linear_loss = F.cross_entropy(linear_logits[mask], flat_label[mask]).mean()
            loss += linear_loss
    
            cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
            loss += cluster_loss
            
        loss_dict = {
            'pos_inter_loss': pos_inter_loss.cpu().item(),
            'pos_intra_loss': pos_intra_loss.cpu().item(),
            'neg_inter_loss': neg_inter_loss.cpu().item(),
            'cluster_loss': cluster_loss.cpu().item(),
            'linear_loss': linear_loss.cpu().item(),
            'loss': loss.cpu().item(),
        }
        self.net_optim.zero_grad()
        self.linear_probe_optim.zero_grad()
        self.cluster_probe_optim.zero_grad()

        loss.backward()
        self.net_optim.step()
        self.cluster_probe_optim.step()
        self.linear_probe_optim.step()

        return loss_dict

    @torch.no_grad()
    def sample(self, batch_dict, visualize_all=False):
        assert not self.training
        images = batch_dict['images']
        images = images.to(self.device) 
        images = images - self.pixel_mean / self.pixel_std
        H, W = images.shape[-2:]
        label: torch.Tensor = batch_dict['masks'].to(self.device, non_blocking=True)
        feats, code = self.net(images)

        sampled_points, similarities, backbone_similarities = None, None, None
        kmeans_preds, num_kmeans_classes, kmeans_preds_backbone = None, None, None,
        if visualize_all:
            sampled_points, similarities, backbone_similarities = self.sample_point_similarities(code_features=code,
                                                                                                 backbone_features=feats, 
                                                                                                 num_points=10)
            kmeans_preds, _ , num_kmeans_classes = self.self_cluster(code, label)
            kmeans_preds_backbone, _ , _ = self.self_cluster(feats, label)


        code = F.interpolate(code, images.shape[-2:], mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast(enabled=True):
            linear_preds = self.linear_probe(code)
            cluster_loss, cluster_preds = self.cluster_probe(code, None)

        return  {
            'linear_preds': linear_preds,
            'cluster_preds': cluster_preds,

            'sampled_points': sampled_points,
            'similarities': similarities,
            'backbone_similarities': backbone_similarities,

            'kmeans_preds': kmeans_preds,
            'kmeans_preds_bb': kmeans_preds_backbone,
            'num_kmeans_classes': num_kmeans_classes,
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
    model.optimize_setup(configs)
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

