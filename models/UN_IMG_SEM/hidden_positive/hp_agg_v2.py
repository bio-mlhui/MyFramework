"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import os
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from functools import partial
import logging
from models.registry import register_model
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from torch.optim import Adam, AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from models.UN_IMG_SEM.stego.utils.layer_utils import ClusterLookup
from models.UN_IMG_SEM.stego.utils.common_utils import freeze_bn, zero_grad_bn
from .LambdaLayer import LambdaLayer
from .loss import StegoLoss
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)



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



class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):  # cfg["pretrained"]
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.feat_type = self.cfg["pretrained"]["dino_feat_type"]
        dino_configs = self.cfg["pretrained"]
        self.sigma = 28
        self.dropout = torch.nn.Dropout2d(p=.1)

        self.model = BACKBONE_REGISTRY.get(dino_configs['name'])(dino_configs)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()

        self.n_feats = self.model.embed_dim
        self.patch_size = self.model.patch_size

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg["pretrained"]["projection_type"]
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        self.ema_model1 = self.make_clusterer(self.n_feats)
        self.ema_model2 = self.make_nonlinear_clusterer(self.n_feats)

        for param_q, param_k in zip(self.cluster1.parameters(), self.ema_model1.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model1.cuda()
        self.ema_model1.eval()

        for param_q, param_k in zip(self.cluster2.parameters(), self.ema_model2.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        self.ema_model2.cuda()
        self.ema_model2.eval()

        sz = cfg["spatial_size"]

        self.index_mask = torch.zeros((sz*sz, sz*sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz*sz), dtype=torch.long)
        for _im in range(sz*sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im+1, _im+sz, _im+(sz+1)])
            elif _im==(sz-1):
                index_set = torch.tensor([_im-1, _im, _im+(sz-1), _im+sz])
            elif _im==(sz*sz-sz):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1])
            elif _im==(sz*sz-1):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im])

            elif ((1 <= _im) and (_im <= (sz-2))):
                index_set = torch.tensor([_im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            elif (((sz*sz-sz+1) <= _im) and (_im <= (sz*sz-2))):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im-sz, _im-(sz-1), _im, _im+1, _im+sz, _im+(sz+1)])
            elif ((_im+1) % sz == 0):
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-1, _im, _im+(sz-1), _im+sz])
            else:
                index_set = torch.tensor([_im-(sz+1), _im-sz, _im-(sz-1), _im-1, _im, _im+1, _im+(sz-1), _im+sz, _im+(sz+1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num[_im] = index_set.size(0)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()


    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer_layer3(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    @torch.no_grad()
    def ema_model_update(self, model, ema_model, ema_m):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
            param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

        for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def forward(self, img, n=1, return_class_feat=False, train=False):
        self.model.eval()
        batch_size = img.shape[0]

        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            rets = self.model(img, n=n)
            # b cls+hw c, 
            feat, attn, qkv = rets['features'], rets['attentions'], rets['qkvs']
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            if train==True:
                attn = attn[:, :, 1:, 1:]
                attn = torch.mean(attn, dim=1)
                attn = attn.type(torch.float32)
                attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
                attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
                attn = torch.max(torch.min(attn, attn_max), attn_min)

                attn = attn.softmax(dim=-1)
                attn = attn*self.sigma
                attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                raise ValueError() # 6?
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            code_ema = self.ema_model1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
                code_ema += self.ema_model2(self.dropout(image_feat))
        else:
            code = image_feat

        if train==True:
            attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            code_clone = code.clone()
            code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
            code_clone = code_clone.permute(0,2,1)

            code_3x3_all = []
            for bs in range(batch_size):
                code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
                code_3x3 = torch.sum(code_3x3, dim=1)
                code_3x3 = code_3x3 / self.divide_num
                code_3x3_all.append(code_3x3)
            code_3x3_all = torch.stack(code_3x3_all)
            code_3x3_all = code_3x3_all.permute(0,2,1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        if train==True:
            with torch.no_grad():
                self.ema_model_update(self.cluster1, self.ema_model1, self.cfg["ema_m"])
                self.ema_model_update(self.cluster2, self.ema_model2, self.cfg["ema_m"])

        if train==True:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema), self.dropout(code_3x3_all)
            else:
                return image_feat, code, code_ema, code_3x3_all
        else:
            if self.cfg["pretrained"]["dropout"]:
                return self.dropout(image_feat), code, self.dropout(code_ema)
            else:
                return image_feat, code, code_ema



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

