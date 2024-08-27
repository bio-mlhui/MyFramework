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
import torch.distributed as dist
from .make_reference_pool import renew_reference_pool
from detectron2.data import MetadataCatalog, DatasetCatalog
from .dino.DinoFeaturizer import DinoFeaturizer

from models.UN_IMG_SEM.loss import StegoLoss, SupConLoss
def build_criterion(n_classes: int, opt: dict):
    # opt = opt["loss"]
    loss_name = opt["name"].lower()
    if "stego" in loss_name:
        loss = StegoLoss(n_classes=n_classes, cfg=opt, corr_weight=opt["correspondence_weight"])
    else:
        raise ValueError(f"Unsupported loss type {loss_name}")

    return loss

class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs

class STEGOmodel(nn.Module):

    def __init__(self,
                 opt: dict,
                 n_classes:int
                 ):
        super().__init__()
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

        self.feat_dim = self.net.model.embed_dim

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



class HiddenPositive(OptimizeModel):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.configs = configs
        val_num_classes = MetadataCatalog.get(MetadataCatalog.get('global_dataset').get('subset_list')[0]).get('num_classes')
        model = STEGOmodel.build(opt=configs['model'], n_classes=val_num_classes)
        bn_momentum = configs['model']["bn_momentum"]
        if bn_momentum is not None:
            for module_name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.momentum = bn_momentum

        bn_eps = configs['model']["bn_eps"] 
        if bn_eps is not None:
            for module_name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    module.eps = bn_eps
        self.net_model = model.net
        self.linear_model = model.linear_probe
        self.cluster_model = model.cluster_probe
        
        self.criterion = build_criterion(n_classes=val_num_classes, opt=configs['model']["loss"])
        
        self.project_head = nn.Linear(configs['model']['dim'], configs['model']['dim'])
        self.supcon_criterion = SupConLoss(temperature=configs['model']["tau"])
        self.pd = nn.PairwiseDistance()   
        
        self.feat_dim = model.feat_dim
        self.out_dim = configs['model']['dim']
        
        # self.pool_ag, self.pool_sp
        self.Pool_ag, self.Pool_sp, self.dataset_memory = None, None, None
        
        self.renew_interval = configs['model']['renew_interval']
        self.warmup_iterations = configs["model"]["warmup"]
        self.maxiter = configs['model']['maxiter']
    
    @property
    def device(self):
        return self.project_head.weight.device
        
    def optimize_setup(self, configs):
        from torch.backends import cudnn
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)
        cudnn.benchmark = True
        
        self.head_optimizer = torch.optim.Adam(self.project_head.parameters(), lr=configs['optim']['net']['lr'])
        net_optimizer_type = configs['optim']["net"]["name"].lower()
        if net_optimizer_type == "adam":
            self.net_optimizer = torch.optim.Adam(self.net_model.parameters(), lr=configs['optim']["net"]["lr"])
        elif net_optimizer_type == "adamw":
            self.net_optimizer = torch.optim.AdamW(self.net_model.parameters(), lr=configs['optim']["net"]["lr"], 
                                              weight_decay=configs['optim']["net"]["weight_decay"])
        else:
            raise ValueError(f"Unsupported optimizer type {net_optimizer_type}.")

        linear_probe_optimizer_type = configs['optim']["linear"]["name"].lower()
        if linear_probe_optimizer_type == "adam":
            self.linear_probe_optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=configs['optim']["linear"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {linear_probe_optimizer_type}.")

        cluster_probe_optimizer_type = configs['optim']["cluster"]["name"].lower()
        if cluster_probe_optimizer_type == "adam":
            self.cluster_probe_optimizer = torch.optim.Adam(self.cluster_model.parameters(), lr=configs['optim']["cluster"]["lr"])
        else:
            raise ValueError(f"Unsupported optimizer type {cluster_probe_optimizer_type}.")
        
        self.grad_norm = configs['optim']['grad_norm']
        # self.freeze_encoder_bn = configs['optim']['freeze_encoder_bn']
        # self.freeze_all_bn =  configs['optim']['freeze_all_bn']
        self.alpha = configs['model']["alpha"]


    def forward(self, batch_dict):
        assert self.training
        num_iterations = batch_dict['num_iterations']
        if num_iterations <= self.warmup_iterations:
            lmbd = 0
        else:
            lmbd = (num_iterations - self.warmup_iterations) / (self.maxiter - self.warmup_iterations)
        
        if num_iterations % self.renew_interval == 0 and num_iterations!= 0:
            self.Pool_sp = renew_reference_pool(self.net_model, self.dataset_memory, self.configs, self.device)
            comm.synchronize()
        img: torch.Tensor = batch_dict['images'].to(self.device, non_blocking=True) # b 3 h w
        label: torch.Tensor = batch_dict['masks'].to(self.device, non_blocking=True) # b h w, -1 for background, 0, 1, 2, 3
        img_aug = batch_dict['pho_aug_image'].to(self.device, non_blocking=True) 
           
        model_input = (img, label)
        with torch.cuda.amp.autocast(enabled=True):
            model_output = self.net_model(img, train=True)
            model_output_aug = self.net_model(img_aug)               
        modeloutput_f = model_output[0].clone().detach().permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        modeloutput_f = F.normalize(modeloutput_f, dim=1)

        modeloutput_s = model_output[1].permute(0, 2, 3, 1).reshape(-1, self.out_dim)

        modeloutput_s_aug = model_output_aug[1].permute(0, 2, 3, 1).reshape(-1, self.out_dim)
        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z = self.project_head(modeloutput_s)
            modeloutput_z_aug = self.project_head(modeloutput_s_aug)
        modeloutput_z = F.normalize(modeloutput_z, dim=1)
        modeloutput_z_aug = F.normalize(modeloutput_z_aug, dim=1)

        loss_consistency = torch.mean(self.pd(modeloutput_z, modeloutput_z_aug))     
           
        modeloutput_s_mix = model_output[3].permute(0, 2, 3, 1).reshape(-1, self.out_dim)
        with torch.cuda.amp.autocast(enabled=True):
            modeloutput_z_mix = self.project_head(modeloutput_s_mix)
        modeloutput_z_mix = F.normalize(modeloutput_z_mix, dim=1)

        modeloutput_s_pr = model_output[2].permute(0, 2, 3, 1).reshape(-1, self.out_dim)
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
                                                        cluster_output=cluster_output)

            loss = loss + loss_supcon + loss_consistency*self.alpha
            # loss = loss / num_accum
        loss_dict.update({'supcon': loss_supcon.item(), 'consistency': loss_consistency.item(), 'loss': loss})
        return loss_dict, {'linear': 1, 'cluster': 1, 'supcon': 1, 'consistency': self.alpha}, None 


    def optimize(self,
                loss_weight=None,
                loss_dict_unscaled=None,
                closure=None,
                num_iterations=None,
                **kwargs):
        loss = loss_dict_unscaled['loss']
        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        
        self.net_optimizer.zero_grad(set_to_none=True)
        self.linear_probe_optimizer.zero_grad(set_to_none=True)
        self.cluster_probe_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)
               
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.net_optimizer)
        g_norm = nn.utils.clip_grad_norm_(self.net_model.parameters(), self.grad_norm)
        self.scaler.step(self.net_optimizer)

        self.scaler.step(self.linear_probe_optimizer)
        self.scaler.step(self.cluster_probe_optimizer)
        self.scaler.step(self.head_optimizer)
        self.scaler.update()  


    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training

        img: torch.Tensor = batch_dict['images'].to(self.device, non_blocking=True) # b 3 h w

        with torch.cuda.amp.autocast(enabled=True):
            output = self.net_model(img)
        feats = output[0]
        head_code = output[1]

        head_code = F.interpolate(head_code, img.shape[-2:], mode='bilinear', align_corners=False)

        # with torch.cuda.amp.autocast(enabled=True):
        #     linear_preds = torch.log_softmax(self.linear_model(head_code), dim=1)

        # with torch.cuda.amp.autocast(enabled=True):
        #     cluster_loss, cluster_preds = self.cluster_model(head_code, 2, log_probs=True, is_direct=False)
        # linear_preds = batched_crf(img, linear_preds).argmax(1).cuda()
        # cluster_preds = batched_crf(img, cluster_preds).argmax(1).cuda()

        with torch.cuda.amp.autocast(enabled=True):
            linear_preds = self.linear_model(head_code).argmax(1)

        with torch.cuda.amp.autocast(enabled=True):
            cluster_loss, cluster_preds = self.cluster_model(head_code, None, is_direct=False)
        cluster_preds = cluster_preds.argmax(1)

        return {
            'cluster_loss': cluster_loss,
            'linear_preds': linear_preds, 
            'cluster_preds': cluster_preds, 
        }



    def optimize_state_dict(self,):
        return {
            'head_optimizer': self.head_optimizer.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.head_optimizer.load_state_dict(state_dict=state_dict['head_optimizer'])
        self.scheduler.load_state_dict(state_dict=state_dict['scheduler'])


    def get_lr_group_dicts(self, ):
        return  {f'lr_group_head': self.head_optimizer.param_groups[0]["lr"]}


import torchvision.transforms.functional as VF
MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)

    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)

class AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                'images': torch.stack([item['image'] for item in batch_dict], dim=0),
                'pho_aug_image': torch.stack([item['pho_aug_image'] for item in batch_dict], dim=0),
                'geo_aug_coord': torch.stack([item['geo_aug_coord'] for item in batch_dict], dim=0),
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'meta_idxs': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict]
            }
        elif mode == 'evaluate':
            return {
                'metas': [item['image_id'] for item in batch_dict],
                'images': torch.stack([item['image'] for item in batch_dict], dim=0),
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'metas': [item['meta_idx'] for item in batch_dict],
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()
    

@register_model
def alignseg(configs, device):
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    model = HiddenPositive(configs)
        
    model.to(device)
    model.optimize_setup(configs)
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

