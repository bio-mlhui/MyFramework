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
from torch.optim import Adam, AdamW
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

def build_optimizer(main_params, linear_params, cluster_params, opt: dict, model_type: str):
    # opt = opt["optimizer"]
    model_type = model_type.lower()

    if "stego" in model_type:
        net_optimizer_type = opt["net"]["name"].lower()
        if net_optimizer_type == "adam":
            net_optimizer = Adam(main_params, lr=opt["net"]["lr"])
        elif net_optimizer_type == "adamw":
            net_optimizer = AdamW(main_params, lr=opt["net"]["lr"], weight_decay=opt["net"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {net_optimizer_type}.")

        linear_probe_optimizer_type = opt["linear"]["name"].lower()
        if linear_probe_optimizer_type == "adam":
            linear_probe_optimizer = Adam(linear_params, lr=opt["linear"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {linear_probe_optimizer_type}.")

        cluster_probe_optimizer_type = opt["cluster"]["name"].lower()
        if cluster_probe_optimizer_type == "adam":
            cluster_probe_optimizer = Adam(cluster_params, lr=opt["cluster"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {cluster_probe_optimizer_type}.")


        return net_optimizer, linear_probe_optimizer, cluster_probe_optimizer

    else:
        raise ValueError("No model: {} found".format(model_type))




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
                'img': torch.stack([item['image'] for item in batch_dict], dim=0),
                'label': torch.stack([item['mask'] for item in batch_dict], dim=0),
                'img_aug': torch.stack([item['img_aug'] for item in batch_dict], dim=0),
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

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)
from models.UN_IMG_SEM.hidden_positive.dino.DinoFeaturizer import DinoFeaturizer
from models.UN_IMG_SEM.stego.utils.layer_utils import ClusterLookup
from models.UN_IMG_SEM.stego.utils.common_utils import freeze_bn, zero_grad_bn
from models.UN_IMG_SEM.stego.utils.seg_utils import UnsupervisedMetrics, batched_crf, get_metrics
from .LambdaLayer import LambdaLayer
from .loss import StegoLoss

class STEGOmodel(nn.Module):
    # opt["model"]
    def __init__(self,
                 opt: dict,
                 n_classes:int
                 ):
        super().__init__()
        self.opt = opt
        self.n_classes= n_classes


        if not opt["continuous"]:
            dim = n_classes
        else:
            dim = opt["dim"]

        if opt["arch"] == "dino":
            self.net = DinoFeaturizer(dim, opt)
        else:
            raise ValueError("Unknown arch {}".format(opt["arch"]))

        self.cluster_probe = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.cluster_probe2 = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe2 = nn.Conv2d(dim, n_classes, (1, 1))

        self.cluster_probe3 = ClusterLookup(dim, n_classes + opt["extra_clusters"])
        self.linear_probe3 = nn.Conv2d(dim, n_classes, (1, 1))


    def forward(self, x: torch.Tensor):
        return self.net(x)[1]

    @classmethod
    def build(cls, opt, n_classes):
        # opt = opt["model"]
        m = cls(
            opt = opt,
            n_classes= n_classes
        )
        print(f"Model built! #params {m.count_params()}")
        return m

    def count_params(self) -> int:
        count = 0
        for p in self.parameters():
            count += p.numel()
        return count

def build_model(opt: dict, n_classes: int = 27, is_direct: bool = False):
    model_type = opt["model_type"].lower()

    if "stego" in model_type:
        model = STEGOmodel.build(
            opt=opt,
            n_classes=n_classes
        )
        net_model = model.net
        linear_model = model.linear_probe
        cluster_model = model.cluster_probe
        
    elif model_type == "dino":
        model = nn.Sequential(
            DinoFeaturizer(20, opt),
            LambdaLayer(lambda p: p[0])
        )

    else:
        raise ValueError("No model: {} found".format(model_type))

    bn_momentum = opt.get("bn_momentum", None)
    if bn_momentum is not None:
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.momentum = bn_momentum

    bn_eps = opt.get("bn_eps", None)
    if bn_eps is not None:
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eps = bn_eps

    if "stego" in model_type:
        return net_model, linear_model, cluster_model
    elif model_type == "dino":
        return model

def build_criterion(n_classes: int, opt: dict):
    # opt = opt["loss"]
    loss_name = opt["name"].lower()
    if "stego" in loss_name:
        loss = StegoLoss(n_classes=n_classes, cfg=opt, corr_weight=opt["correspondence_weight"])
    else:
        raise ValueError(f"Unsupported loss type {loss_name}")

    return loss
from .make_reference_pool import renew_reference_pool
class HP(OptimizeModel):
    def __init__(
        self,
        configs,
        num_classes,
        train_loader_memory,
        device,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        model_configs = configs['model']

        self.net_model, self.linear_model, self.cluster_model = build_model(opt=model_configs,
                                                                            n_classes=num_classes,
                                                                            is_direct=configs["eval"]["is_direct"]) 
        self.patch_size = self.net_model.patch_size
        self.criterion = build_criterion(n_classes=num_classes, opt=configs["loss"]) 
        self.net_model.to(device)
        self.linear_model.to(device)
        self.cluster_model.to(device)
               
        self.project_head = nn.Linear(model_configs['dim'], model_configs['dim'])
        self.project_head.to(device)
        self.head_optimizer = Adam(self.project_head.parameters(), lr=configs['optimizer']["net"]["lr"])

        from .loss import SupConLoss
        self.supcon_criterion = SupConLoss(temperature=configs["tau"])
        self.pd = nn.PairwiseDistance()
        self.feat_dim =  self.net_model.n_feats 
        self.configs = configs
        self.maxiter = len(train_loader_memory) * configs['optim']['epochs']   # 45795
        self.train_loader_memory = train_loader_memory
        self.freeze_encoder_bn = configs["train"]["freeze_encoder_bn"]
        self.freeze_all_bn = configs["train"]["freeze_all_bn"]
        self.grad_norm = configs["train"]["grad_norm"]
        self.num_queries = num_classes
        self.kmeans_strategy = 'adaptive'

    @property
    def device(self):
        return self.pixel_mean.device
        
    def optimize_setup(self, configs):
        self.net_optimizer, self.linear_probe_optimizer, self.cluster_probe_optimizer = build_optimizer(
            main_params=self.net_model.parameters(),
            linear_params=self.linear_model.parameters(),
            cluster_params=self.cluster_model.parameters(),
            opt=configs['optimizer'],
            model_type=configs["model"]["model_type"])

    def forward_backward(self, batch_dict):
        trainingiter = batch_dict['num_iterations']
        current_epoch = batch_dict['num_epoch']
        device = self.device
        assert self.training
        
        if trainingiter <= self.configs["model"]["warmup"]:
            lmbd = 0
        else:
            lmbd = (trainingiter - self.configs["model"]["warmup"]) / (self.maxiter - self.configs["model"]["warmup"])

        if trainingiter % self.configs["renew_interval"] == 0 and trainingiter!= 0:
            self.Pool_sp = renew_reference_pool(self.net_model, self.train_loader_memory, self.configs, self.device,
                                                pixel_mean=self.pixel_mean, pixel_std=self.pixel_std)
            
        img: torch.Tensor = batch_dict['img'].to(device, non_blocking=True)
        label: torch.Tensor = batch_dict['label'].to(device, non_blocking=True)

        img_aug = batch_dict['img_aug'].to(device, non_blocking=True)

        with torch.no_grad():
            img = (img - self.pixel_mean) / self.pixel_std

        if self.freeze_encoder_bn:
            freeze_bn(self.net_model.model)
        if 0 < self.freeze_all_bn <= current_epoch:
            freeze_bn(self.net_model)

        batch_size = img.shape[0]
        self.net_optimizer.zero_grad(set_to_none=True)
        self.linear_probe_optimizer.zero_grad(set_to_none=True)
        self.cluster_probe_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)

        model_input = (img, label)

        with torch.cuda.amp.autocast(enabled=True):
            model_output = self.net_model(img, train=True)
            model_output_aug = self.net_model(img_aug)

        modeloutput_f = model_output[0].clone().detach().permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        modeloutput_f = F.normalize(modeloutput_f, dim=1)

        modeloutput_s = model_output[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        modeloutput_s_aug = model_output_aug[1].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])

        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z = self.project_head(modeloutput_s)
            modeloutput_z_aug = self.project_head(modeloutput_s_aug)
        modeloutput_z = F.normalize(modeloutput_z, dim=1)
        modeloutput_z_aug = F.normalize(modeloutput_z_aug, dim=1)

        loss_consistency = torch.mean(self.pd(modeloutput_z, modeloutput_z_aug))

        modeloutput_s_mix = model_output[3].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z_mix = self.project_head(modeloutput_s_mix)
        modeloutput_z_mix = F.normalize(modeloutput_z_mix, dim=1)

        modeloutput_s_pr = model_output[2].permute(0, 2, 3, 1).reshape(-1, self.configs["model"]["dim"])
        modeloutput_s_pr = F.normalize(modeloutput_s_pr, dim=1)

        loss_supcon = self.supcon_criterion(modeloutput_z, modeloutput_s_pr=modeloutput_s_pr, modeloutput_f=modeloutput_f,
                                Pool_ag=self.Pool_ag, Pool_sp=self.Pool_sp,
                                opt=self.configs, lmbd=lmbd, modeloutput_z_mix=modeloutput_z_mix)


        detached_code = torch.clone(model_output[1].detach())
        with torch.cuda.amp.autocast(enabled=True):
            linear_output = self.linear_model(detached_code)
            cluster_output = self.cluster_model(detached_code, None, is_direct=False)

            loss, loss_dict, corr_dict = self.criterion(model_input=model_input,
                                                    model_output=model_output,
                                                    linear_output=linear_output,
                                                    cluster_output=cluster_output
                                                    )

            loss = loss + loss_supcon + loss_consistency * self.configs['alpha'] 


        self.scaler.scale(loss).backward()

        if self.freeze_encoder_bn:
            zero_grad_bn(self.net_model)
        if 0 < self.freeze_all_bn <= current_epoch:
            zero_grad_bn(self.net_model)

        self.scaler.unscale_(self.net_optimizer)

        g_norm = nn.utils.clip_grad_norm_(self.net_model.parameters(), self.grad_norm)
        self.scaler.step(self.net_optimizer)

        self.scaler.step(self.linear_probe_optimizer)
        self.scaler.step(self.cluster_probe_optimizer)
        self.scaler.step(self.head_optimizer)

        self.scaler.update()


        loss_dict['loss'] = loss.cpu().item()
        loss_dict['loss_supcon'] = loss_supcon.cpu().item()
        loss_dict['loss_consistency'] = loss_consistency.cpu().item()
        return loss_dict

    @torch.no_grad()
    def sample(self, batch_dict, visualize_all=False):
        assert not self.training
        img: torch.Tensor = batch_dict['images'].to(self.device, non_blocking=True)
        img = (img - self.pixel_mean) / self.pixel_std
        label: torch.Tensor = batch_dict['masks'].to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            output = self.net_model(img)

        backbone_feats = output[0] # b c h w
        head_code = output[1].float() # b c h w

        sampled_points, similarities, backbone_similarities = None, None, None
        kmeans_preds, num_kmeans_classes, kmeans_preds_backbone = None, None, None,
        if visualize_all:
            sampled_points, similarities, backbone_similarities = self.sample_point_similarities(code_features=head_code,
                                                                                                 backbone_features=backbone_feats, 
                                                                                                 num_points=10)
            kmeans_preds, _ , num_kmeans_classes = self.self_cluster(head_code, label)
            kmeans_preds_backbone, _ , _ = self.self_cluster(backbone_feats, label)

        head_code = F.interpolate(head_code, label.shape[-2:], mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=True):
            linear_preds = self.linear_model(head_code)
        with torch.cuda.amp.autocast(enabled=True):
            _, cluster_preds = self.cluster_model(head_code, None, is_direct=self.configs["eval"]["is_direct"])

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
            'net_optim': self.net_optimizer.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.net_optimizer.load_state_dict(state_dict=state_dict['net_optim'])


    def get_lr_group_dicts(self, ):
        return  {f'lr_net': self.net_optimizer.param_groups[0]["lr"],
                 f'lr_cluster': self.cluster_probe_optimizer.param_groups[0]["lr"],
                 f'lr_linear': self.linear_probe_optimizer.param_groups[0]["lr"]}

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

@register_model
def hp(configs, device):
    # 假设只有一个训练数据集
    scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    train_loader_memory, _ = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    model = HP(configs, num_classes=num_classes, train_loader_memory=train_loader_memory, device=device)
    model.to(device)
    model.optimize_setup(configs)
    from .make_reference_pool import initialize_reference_pool
    model.Pool_ag, model.Pool_sp = initialize_reference_pool(model.net_model, 
                                                            train_loader_memory, configs, 
                                                            model.feat_dim, device, model.pixel_mean, model.pixel_std)
    model.scaler = scaler
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

