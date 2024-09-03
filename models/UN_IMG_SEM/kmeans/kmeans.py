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
class AUXMapper:
    def mapper(self, data_dict, mode,):
        return data_dict
    def collate(self, batch_dict, mode):
        if mode == 'train':
            return {
                # list[3 3 h w] -> b 3 3 h w
                'images': torch.stack([item['image'] for item in batch_dict], dim=0),
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0),
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
                'masks': torch.stack([item['mask'] for item in batch_dict], dim=0), # b h w
                'visualize': [item['visualize'] for item in batch_dict]
            }
        else:
            raise ValueError()

from torchvision.transforms import GaussianBlur
from skimage.measure import label
def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.6, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k
            if np.sum(mask) <= 2:
                th_attn[j, 0][np.squeeze(mask, 0)] = 0
    return th_attn.detach()

# 单图kmeans, 
# adaptive: 聚类数量是unique() - 255
# fixed: 聚类数量是固定的
# 只能使用 alignseg 的 evaluate 方式, 但是会有 label leak
# kmeans/EM/minibatchKmeans
class KMeans_SingleImage(OptimizeModel):
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
        self.backbone = BACKBONE_REGISTRY.get(model_configs['backbone']['name'])(model_configs['backbone'])
        self.backbone.eval()
        
        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim

        self.num_classes = num_classes

        self.cluster_strategy = model_configs['cluster_strategy']
        if self.cluster_strategy == 'fixed':
            self.num_queries = model_configs['num_queries']
    
    def get_backbone_features(self, images):
        images = images.to(self.device) 
        images = (images - self.pixel_mean) / self.pixel_std
        B, _, H, W = images.shape
        # Extract feature
        with torch.no_grad():
            features = self.backbone(images) # b 3 h w -> b c h//patch w//patch
            features = features['features'][0][:, 1:, :] # b cls_hw c
            features = features.reshape(B, H//self.patch_size, W//self.patch_size, features.shape[-1])
        return features
            
    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        self.backbone.eval()
        images = batch_dict['images']
        _, _, H, W = images.shape
        gt_masks = batch_dict['masks']
        features = self.get_backbone_features(images) # b c h w
        cluster_ids, cluster_logits, num_image_classes = self.self_cluster(features, gt_masks)
        sampled_points, similarities = self.sample_point_similarities(features, num_points=5)
        return {
            'pred_masks': cluster_logits, # 要测试kmeans的cluster miou
            'cluster_ids': cluster_ids,
            'num_image_classes': num_image_classes,
            'sampled_points': sampled_points,
            'similarities': similarities
        }
        
    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None

    def train(self, mode):
        super().train(mode)
        self.backbone.eval()
    
    @torch.no_grad()
    def self_cluster(self, features, gt_masks):
        # b c h w -> b
        _, _, H_P, W_P = features.shape
        assert features.shape[0] == 1
        if self.cluster_strategy == 'adaptive':
            num_image_classes = len(set(gt_masks.unique().tolist()) - set([255]))
        elif self.cluster_strategy == 'fixed':
            num_image_classes = self.num_queries
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

    @torch.no_grad()
    def sample_point_similarities(self, features, num_points):
        H_P, W_P = features.shape[-2:]
        H, W = H_P * self.patch_size, W_P * self.patch_size
        sampled_points = torch.rand(num_points, 2)
        sampled_points[:, 0] = sampled_points[:, 0] * H_P
        sampled_points[:, 1] = sampled_points[:, 1] * W_P
        sampled_points = sampled_points.long()
        sampled_points[:, 0].clamp_(0, H_P-1)
        sampled_points[:, 1].clamp_(0, W_P-1)
        similarities = []
        for point in sampled_points:
            query = features[:, :, point[0], point[1]] # 1 c
            sim = torch.einsum('c,chw->hw',
                                F.normalize(query[0], dim=0, eps=1e-10),
                                F.normalize(features[0], dim=0, eps=1e-10),).cpu() # -1, 1
            sim = F.interpolate(sim[None, None, ...], size=(H, W), align_corners=False, mode='bilinear').clamp(-1, 1)[0, 0]
            similarities.append(sim)
        sampled_points = sampled_points * self.patch_size
        return sampled_points, similarities

@register_model
def kmeans_singleImage(configs, device):
    train_dataset_name = list(configs['data']['train'].keys())[0]
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    model = KMeans_SingleImage(configs,num_classes=num_classes)
    model.to(device)
    eval_train = configs['model']['eval_train']
    eval_number = 100
    if eval_train:
        from models.utils.visualize_sem_seg import visualize_cluster
        from models.utils.visualize_cos_similarity import visualize_cos_similarity
        from PIL import Image
        save_dir = os.path.join(configs['out_dir'], 'cluster_trainset', )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(train_loader)):
                images = batch['images'].to(device)
                _, _, H, W = images.shape
                image_id = batch['image_ids'][0]
                gt_masks = batch['masks'].to(device)
                img_path = os.path.join(save_dir, f'{image_id}.jpg')
                features = model.get_backbone_features(images)
                cluster_ids, _, num_image_classes = model.self_cluster(features, gt_masks)
                sampled_points, similarities = model.sample_point_similarities(features, num_points=5)
                cluster_image = visualize_cluster(images[0].cpu(), 
                                  gt=gt_masks[0].cpu(), 
                                  pred=cluster_ids[0], 
                                  num_image_classes=num_image_classes,
                                  num_gt_classes=num_classes) # H W*3 3
                sim_image = visualize_cos_similarity(image=images[0].cpu(), sampled_points=sampled_points, similarities=similarities,) #  H W*5 3
                cluster_image = F.pad(cluster_image, pad=(0, 0, 0, 2*W, 0, 0), value=0)
                whole_image = torch.cat([cluster_image, sim_image], dim=0) # 2H W*5 3
                Image.fromarray(whole_image.numpy()).save(img_path)
                if i > eval_number:
                    exit()
    return model, train_loader, eval_function


def kmeans(X, num_clusters, device, tol=1e-4):
    from kmeans_pytorch import initialize,pairwise_cosine
    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_cosine(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state.cpu()

# 所有图minibatch_kmeans
# 使用 stegeo 的 evaluate 方式, 但是会有 label leak
# kmeans/EM/minibatchKmeans

class MiniBatchKMeans_AllEval(OptimizeModel):
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
        self.backbone = BACKBONE_REGISTRY.get(model_configs['backbone']['name'])(model_configs['backbone'])
        self.backbone.eval()
        
        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim

        self.num_classes = num_classes
        from models.UN_IMG_SEM.stego.stegeo_modules import ClusterLookup
        self.num_queries = model_configs['num_queries']
        self.cluster_probe = ClusterLookup(self.embed_dim, self.num_queries)
        assert self.num_queries >= self.num_classes

    def forward_backward(self, batch_dict):
        assert self.training
        # b 3 3 h w
        images = batch_dict['images'].to(self.device) 
        features = self.get_backbone_features(images)

        cluster_loss, cluster_probs = self.cluster_probe(features, None)
        loss = cluster_loss
        
        loss_dict = {
            'cluster_loss': cluster_loss.cpu().item(),
            'loss': loss.cpu().item(),
        }

        self.cluster_probe_optim.zero_grad()
        loss.backward()
        self.cluster_probe_optim.step()

        return loss_dict

    def get_backbone_features(self, images):
        images = images.to(self.device) 
        images = (images - self.pixel_mean) / self.pixel_std
        # Extract feature
        with torch.no_grad():
            features = self.backbone(images) # b 3 h w -> b c h//patch w//patch
        return features
            
    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training

        images = batch_dict['images']
        _, _, H, W = images.shape
        view_image = images[0]
        gt_masks = batch_dict['masks']
        num_image_classes = len(set(gt_masks.unique().tolist()) - set([255]))
        features = self.get_backbone_features(images) # b c h w
        code = F.interpolate(features, (H, W), mode='bilinear', align_corners=False)
        cluster_loss, cluster_preds = self.cluster_probe(code, None)
        return {
            'pred_masks': cluster_preds
        }
        
    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None

    def train(self, mode):
        super().train(mode)
        self.backbone.eval()

    def optimize_setup(self, configs):
        self.cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)


@register_model
def minibatch_kmeans_allEval(configs, device):
    train_dataset_name = list(configs['data']['train'].keys())[0]
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    model = KMeans_MiniBatch(configs,num_classes=num_classes)
    model.to(device)
    model.optimize_setup(configs)

    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function
