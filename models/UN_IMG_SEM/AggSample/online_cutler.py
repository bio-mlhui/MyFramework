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
from models.UN_IMG_SEM.AggSample.code import aggo_whole_batch
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from torchvision.transforms.functional import normalize as torchv_Normalize
from detectron2.structures import Instances, BitMasks

def area(mask: Tensor):
    # h w, bool
    return torch.count_nonzero(mask) / mask.numel()

def iou(mask1, mask2):
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    union = torch.count_nonzero(mask1) + torch.count_nonzero(mask2) - intersection
    if union == 0: return 0
    return intersection / union

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

    @property
    def device(self):
        return self.pixel_mean.device

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])

    def get_lr_group_dicts(self, ):
        return self.optimizer.get_lr_group_dicts()

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

    @torch.no_grad()
    def forward_backbone(self, img):
        """
        img: b 3 h w, normalize之后的
        return: b c h w
        """
        self.backbone.eval()
        B, _, H, W = img.shape
        # b 3 h w -> b c h w
        rets = self.backbone(img)
        # b cls+hw c, 3 b head cls_hw head_dim
        feat, qkv = rets['features'][0], rets['qkvs'][0]
        if self.backbone_feat_type == "feat":
            image_feat = feat[:, 1:, :].reshape(B, H//self.backbone_patch_size, W//self.backbone_patch_size, -1).permute(0, 3, 1, 2)
        elif self.backbone_feat_type == "key":
            image_feat = qkv[1, :, :, 1:, :].reshape(B, self.backbone_nheads, 
                                                     H//self.backbone_patch_size, W//self.backbone_patch_size, -1)
            image_feat = image_feat.permute(0, 1, 4, 2, 3).flatten(1,2) # b c h w
        else:
            raise ValueError("Unknown feat type:{}".format(self.feat_type))
        return image_feat
    

class ProbeDetachLoss(nn.Module):

    def __init__(self,
                 n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.linear_loss = nn.CrossEntropyLoss()

    def get_linear(self, linear_logits: torch.Tensor, label: torch.Tensor, n_classes: int):
        # b k h w, logits
        flat_label = label.reshape(-1) # bhw
        mask = (flat_label >= 0) & (flat_label < n_classes)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        linear_loss = self.linear_loss(linear_logits[mask], flat_label[mask]).mean()
        return linear_loss

    def forward(self, 
                label,
                linear_output, # b k h w, logits
                cluster_loss) \
            -> Tuple[torch.Tensor, Dict[str, float]]:
        linear_loss = self.get_linear(linear_output, label, self.n_classes)
        return {
            'linear_loss': linear_loss,
            'cluster_loss': cluster_loss
        }

class ClusterLookup(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x,):
        # logits -> K_prob
        # TODO: 距离换成dot-product / 
        normed_clusters = F.normalize(self.clusters, dim=1) # K d
        normed_features = F.normalize(x, dim=-1) # N d
        inner_products = torch.einsum("nd,kd->nk", normed_features, normed_clusters) # N k

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=-1), self.clusters.shape[0]).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        in_channels = configs['in_dim']
        out_dim = configs['out_dim']
        self.is_identity = configs['is_identity']
        if self.is_identity:
            self.encoder = nn.Identity()
        else:
            # TODO: ablation不同的architecture
            self.conv11 = torch.nn.Conv2d(in_channels, out_dim, (1, 1))
            self.non_linear = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels, out_dim, (1, 1)))
            
            # zero_initialize
            # TODO: 利用 deformable_encoder
            if configs['zero_initialize']:
                with torch.no_grad():
                    from models.layers.utils import zero_module
                    self.conv11 = zero_module(self.conv11)
                    self.non_linear = zero_module(self.non_linear)
            # TODO: 添加residual_scaling
            # TODO: 先8转换然后
            # TODO: 加上mask_adapter
    def forward(self, x):
        if self.is_identity:
            return self.encoder(x)
        else:
            x_2 = self.conv11(x) + self.non_linear(x)
            x = x + x_2
            return x


class SimpleFeaturePyramid(nn.Module):
    def __init__(
        self,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()


        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool import ClusterPooling
from torch_geometric.utils import (
    dense_to_sparse,
    one_hot,
    to_dense_adj,
    to_scipy_sparse_matrix,
)
from PIL import Image
import time
from models.UN_IMG_SEM.AggSample.agg_sample import visualize_cutler_thre_masks
class Online_Cutler(OptimizeModel):
    def __init__(
        self,
        configs,
        num_classes,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],
        batch_size=None,
        train_size=None,
        eval_size=None):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        model_configs = configs['model']
        self.backbone = BACKBONE_REGISTRY.get(model_configs["backbone"]['name'])(model_configs["backbone"])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.backbone_feat_type = model_configs["backbone"]["dino_feat_type"]
        self.backbone_dim = self.backbone.embed_dim # n_feats
        self.backbone_patch_size = self.backbone.patch_size # 8 / 16
        self.backbone_nheads = self.backbone.num_heads
        self.clustering_threshold_list = model_configs['clustering_threshold_list']
        self.clustering_area_threshold = model_configs['clustering_area_threshold']

        encoder_configs = model_configs['encoder']
        encoder_configs['in_dim'] = self.backbone_dim
        encoder_configs['out_dim'] = self.backbone_dim if encoder_configs['out_dim'] == 'same' else encoder_configs['out_dim']
        # TODO: 先看看只有dino没有任何转换能不能学到什么(metric: 和gt的拟合程度)
        # A unsupervised image segmentation model
        self.encoder = Encoder(encoder_configs)
        self.hidden_dim = encoder_configs['out_dim']

        eval_size = model_configs['eval_size']
        self.init_graph_utils(batch_size, train_size, eval_size)
        decoder_configs = model_configs['decoder']
        decoder_configs['d_model'] = self.hidden_dim
        decoder_configs['attn']['dim_feedforward'] = self.hidden_dim * 4
        decoder_configs['attn']['nheads'] = self.backbone.num_heads # TODO: 不一致的
        self.eval_size = (eval_size, eval_size)
        # TODO: multiscale
        # TODO: loss不是Point_sample
        from models.backbone.utils import ImageMultiscale_Shape
        decoder_multiscale_shape = {'res3': ImageMultiscale_Shape(self.backbone_patch_size, dim=self.backbone_dim)}
        self.decoder = META_ARCH_REGISTRY.get(decoder_configs['name'])(decoder_configs, multiscale_shapes = decoder_multiscale_shape)
        
        # probes
        self.cluster_model = ClusterLookup(self.hidden_dim, num_classes)
        self.linear_model = nn.Linear(self.hidden_dim, num_classes)
        self.probe_loss = ProbeDetachLoss(n_classes=num_classes) 
        
        # for evaluator
        self.num_queries = num_classes
        self.kmeans_strategy = 'adaptive'

    def optimize_setup(self, configs):
        # TODO: embeding特殊对待
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)
        optim_configs = configs['optim']
        weight_decay_norm = optim_configs['weight_decay_norm']
        weight_decay_embed = optim_configs['weight_decay_embed']

        net_config = {'lr': optim_configs['base_lr'], 'weight_decay': optim_configs['base_wd']}
        linear_config = { 'lr': optim_configs['linear_lr'], 'weight_decay': optim_configs['linear_weight_decay']} 
        cluster_config = { 'lr': optim_configs['cluster_lr'], 'weight_decay': optim_configs['cluster_weight_decay']}  

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'encoder_decoder':None, 'linear':None, 'cluster': None}

        for module_name, module in self.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                if "linear_model" in module_name:
                    hyperparams = copy.copy(linear_config)
                    if log_lr_group_idx['linear'] is None:
                        log_lr_group_idx['linear'] = len(params)             
                if 'cluster_model' in module_name:
                    hyperparams = copy.copy(cluster_config)
                    if log_lr_group_idx['cluster'] is None:
                        log_lr_group_idx['cluster'] = len(params)
                else:
                    hyperparams = copy.copy(net_config)
                    if log_lr_group_idx['encoder_decoder'] is None:
                        log_lr_group_idx['encoder_decoder'] = len(params)
                                     
                # pos_embed, norm, embedding的weight decay特殊对待
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    logging.debug(f'setting weight decay of {module_name}.{module_param_name} to zero')
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        to_train_num_parameters = len([n for n, p in self.named_parameters() if p.requires_grad])
        assert len(params) == to_train_num_parameters, \
            f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params)}个参数没有列在params_group里'
        self.optimizer = get_optimizer(params, configs)
        # self.scheduler = build_scheduler(configs=configs, optimizer=self.optimizer)
        self.optimizer.get_lr_group_dicts =  partial(lambda x: {f'lr_group_{key}': self.optimizer.param_groups[log_lr_group_idx]["lr"] \
                                                                    if value is not None else 0 for key, value in x.items()},
                                                                x=log_lr_group_idx)


    def forward_backward(self, batch_dict):
        trainingiter = batch_dict['num_iterations']
        assert self.training
        img: torch.Tensor = batch_dict['img'].to(self.device, non_blocking=True)
        label: torch.Tensor = batch_dict['label'].to(self.device, non_blocking=True)
        img_aug = batch_dict['img_aug'].to(self.device, non_blocking=True)
        batch_size, _, H, W = img.shape
        with torch.no_grad():
            img = torchv_Normalize(img, self.pixel_mean, self.pixel_std, False)
            img_aug = torchv_Normalize(img_aug, self.pixel_mean, self.pixel_std, False)
        with torch.cuda.amp.autocast(enabled=True):
            # b c h w (patch_size)
            image_feats = self.forward_backbone(img)
            # TODO: 利用何凯明的multiscale输入到decoder中
            image_feats = self.encoder(image_feats)
            q_feats, q_masks = self.forward_online_agg_clustering(image_feats, orig_img=batch_dict['img'], iteration_num=trainingiter)
            p_feats, p_masks = self.decoder(image_feats) # b n c, b n h w
            # TODO: geometric & photometric 不变性
            pq_loss, pq_loss_dict = self.decoder.compute_loss(p_masks, q_masks)
    
            detached_rois = p_feats.clone().detach()
            linear_output = self.linear_model(detached_rois) #  b n k, logits
            cluster_loss, cluster_probs = self.cluster_model(detached_rois, log_probs=False) # b n k, prob(0-1)
            prob_loss, prob_loss_dict = self.probe_loss(label=label,
                                                        linear_output=linear_output,
                                                        cluster_loss=cluster_loss)
            loss = pq_loss + prob_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        log_dict = pq_loss_dict.update(prob_loss_dict)
        log_dict['loss'] = loss.item()
        return log_dict

    # class-agnostic instance segmentation
    @torch.no_grad()
    def sample_oap(self, batch_dict):
        assert not self.training
        img = batch_dict[0]['image']
        height, width, image_id = batch_dict[0]['height'], batch_dict[0]['width'], batch_dict[0]['image_id']

        img = torch.from_numpy(np.asarray(Image.fromarray(img.permute(1,2,0).numpy()).resize(self.eval_size)))
        img = (img.float() / 255).permute(2, 0, 1).to(self.device) # 3 h w, [0,1]
        img = torchv_Normalize(img, self.pixel_mean, self.pixel_std, False)  
        with torch.cuda.amp.autocast(enabled=True):
            # 探讨不同的scale对它的影响, scale-invariant, 只和图片中instance数量有关
            image_feats = self.forward_backbone(img[None, ...]) # b c h w, patch_size
            q_feats, q_masks, q_scores = self.forward_online_agg_clustering_withSimi(image_feats,) # list[ni c]
        # list[nqi h_p w_p] -> list[nqi h w]
        q_masks = [F.interpolate(foo[None, ...].float(), size=(height,width), 
                                    align_corners=False, mode='bilinear')[0] > 0.5 for foo in q_masks]
        pred_masks = q_masks[0]
        N, image_size = pred_masks.shape[0],pred_masks.shape[-2:]
        scores = torch.ones(N).float() # N
        pred_masks = BitMasks(pred_masks)
        pred_boxes = pred_masks.get_bounding_boxes()
        pred_classes = torch.zeros(N).int()

        return [{
            'instances': Instances(image_size=image_size, pred_masks=pred_masks,scores=scores,
                                   pred_boxes=pred_boxes, pred_classes=pred_classes)
        }]     


    @torch.no_grad()
    def forward_online_agg_clustering_cntMaxArea(self, image_feats, orig_image=None):
        """
        image_feats: b c h w
        return: 
            list[ni c], batch, 特征
            list[ni h w(bool)], batch, 特征mask
            大小和image_feats的大小一致，都是缩放patch_size之后的
        """
        # b c h w -> bhw c
        batch_size = image_feats.shape[0]
        nodes_feature = image_feats.permute(0, 2, 3, 1).flatten(0, 2) # N c
        clustering_threshold_list = self.clustering_threshold_list
        clustering_area_threshold = self.clustering_area_threshold
        edge_index= self.train_edge_index if self.training else self.eval_edge_index # 2 E
        node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids # N
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # cnt_registered_area = self.train_cnt_registered_area if self.training else self.eval_cnt_registered_area # N N
        # node_num_patches=self.train_num_patches if self.training else self.eval_num_patches # N
        token_masks = self.train_token_masks if self.training else self.eval_token_masks # N N, 0-1
        edge_score = self.get_edge_score(nodes_feature, edge_index) # E, -1/1
        HW = token_masks.shape[1] // batch_size
        batch_clusters_feats = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni c], threshold] batch
        batch_clusters_masks = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni h w], threshold] batch
        for thre_idx, threshold in enumerate(clustering_threshold_list):
            while (edge_score > threshold).any():
                # Nt c, 2 Et, Nt, Nt N(0-1)
                nodes_feature, edge_index, node_batch_tensor, token_masks \
                                                        = self.hc_graph(x=nodes_feature, # N c
                                                                        edge_index=edge_index, # 2 E
                                                                        batch=node_batch_tensor, # N
                                                                        threshold=threshold,
                                                                        edge_score=edge_score,# E
                                                                        token_masks=token_masks,) # Nt N
                # Nt-1 N
                # Nt Nt-1, -> Nt N # A,B,C注册的3个的最大值 
                edge_score = self.get_edge_score(nodes_feature, edge_index)

            counted_cluster = set()
            node_num_patches = (token_masks > 0).int().sum(-1) # N
            # edge_index肯定能遍历一遍所有node, 因为每个Node强制和Neighbor相连
            # 始终是是一个connected graph
            # same as UNSAM :)
            for node_idx in range(nodes_feature.shape[0]):
                if node_idx not in counted_cluster:
                    counted_cluster.add(node_idx)
                    # N' N
                    first_condition = node_num_patches[node_idx] >= clustering_area_threshold
                    # N, N_cnt
                    second_condition = iou(token_masks[node_idx, :], cnt_registered_area[node_idx, :]) < self.clustering_nms_iou
                    if first_condition and second_condition: # 如果新的cluster和旧的差的非常多，那么append
                        this_node_batch_idx = node_batch_tensor[node_idx]
                        this_node_feat = nodes_feature[node_idx]
                        this_node_mask = token_masks[node_idx, (HW*this_node_batch_idx):((this_node_batch_idx+1)*HW)] # B*HW -> HW
                        batch_clusters_feats[this_node_batch_idx][thre_idx].append(this_node_feat)
                        batch_clusters_masks[this_node_batch_idx][thre_idx].append(this_node_mask)
                        cnt_registered_area[node_idx] = token_masks[node_idx, :]
                    # 0.9
                    if first_condition and (not second_condition): # TODO: 如果新的cluster(包含原来的)只更新了一点点, 那么更新旧的cluster, 相当于MNMS的排序
                        pass
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # visualize:
        for batch_idx in range(batch_size):
            visualize_cutler_thre_masks(image=orig_image[batch_idx], cluster_masks=batch_clusters_feats[batch_idx],patch_size=self.backbone_patch_size)
        batch_clusters_feats = [torch.cat(foo, dim=0) for foo in batch_clusters_feats] # list[ni c], batch
        batch_clusters_masks = [torch.cat(foo, dim=0) for foo in batch_clusters_masks] # list[ni h w], batch
        
        # NMS
        # TODO: NMS
        # NMS的step和list的长度一样
        # TODO2:  每个节点维护到目前为止它的子树中注册的最大面积的hw
        # TODO: visualize是否运行的好
        # from models.UN_IMG_SEM.AggSample.agg_sample import visualize_cutler_onlyAttn
        # for batch_idx in range(batch_size):
        #     visualize_cutler_onlyAttn(image=orig_image[batch_idx].float(),
        #                             cluster_attn=image_attns[batch_idx].float(), # h w nq
        #                             patch_size=self.patch_size,
        #                             lengths=threshold_lenghts[batch_idx],
        #                             image_path='./tmp',
        #                             image_name=f'test_iter{iteration_num}_bs{batch_idx}.png')
        # TODO: mask的获得是通过graph, 还是通过attention得到mask(利用hp/alignseg的attention->mask的方式)
        return batch_clusters_feats, batch_clusters_masks


    @torch.no_grad()
    def forward_online_agg_clustering_nms(self, image_feats, orig_image=None, image_ids=None):
        """
        image_feats: b c h w
        return: 
            list[ni c], batch, 特征
            list[ni h w(bool)], batch, 特征mask
            大小和image_feats的大小一致，都是缩放patch_size之后的
        """
        # b c h w -> bhw c
        batch_size, _, H, W = image_feats.shape
        nodes_feature = image_feats.permute(0, 2, 3, 1).flatten(0, 2) # N c
        clustering_threshold_list = [0.8, 0.7, 0.6, 0.5, 0.4] # [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # clustering_threshold_list = [0.9, 0.8, 0.7, 0.6, 0.5] # [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # clustering_threshold_list = torch.linspace(0.93, 0.53, steps=8).tolist()
        # /home/xuhuihui/workspace/rvos_encoder/output/UN_IMG_SEM/cocostuff27/kmeans_singleImage/dinov1_vitb8_adaptive/epc[0_00]_iter[0]_sap[0]/eval_cocostuff27-IIC_eval/cluster_testset/000000011122.jpg
        def instanceness_function(foo_x, start=0.25, cnt=0.35, cnt_value=0.2):
            # 0到cnt是从1到cnt_value, cnt到1是从cnt_value到0
            if foo_x < start:
                return 1.
            elif foo_x < cnt:
                return 1 - (1-cnt_value) / (cnt-start) * (foo_x-start)
            elif foo_x >= cnt:
                return cnt_value - (cnt_value) / (1 - cnt) * (foo_x - cnt)
            else:
                raise ValueError()
            
        clustering_area_threshold = 2 # TODO:一个物体的大小4*32像素
        # TODO: feat * feature = semantic mask (alignseg att->mask); online k-means -> semantic class
        # Kmeans获得semantic mask非常慢，通过attention即可
        edge_index= self.train_edge_index if self.training else self.eval_edge_index # 2 E
        node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids # N
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # cnt_registered_area = self.train_cnt_registered_area if self.training else self.eval_cnt_registered_area # N N
        # node_num_patches=self.train_num_patches if self.training else self.eval_num_patches # N
        token_masks = self.train_token_masks if self.training else self.eval_token_masks # N N, 0-1
        edge_score = self.get_edge_score(nodes_feature, edge_index) # E, -1/1
        HW = token_masks.shape[1] // batch_size
        assert HW == H * W
        batch_clusters_feats = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni c], threshold] batch
        batch_clusters_masks = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni h w], threshold] batch
        batch_clusters_instanceness = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni], threshold] batch
        for thre_idx, threshold in enumerate(clustering_threshold_list):
            while (edge_score > threshold).any():
                # Nt c, 2 Et, Nt, Nt N(0-1)
                nodes_feature, edge_index, node_batch_tensor, token_masks \
                                                        = self.hc_graph(x=nodes_feature, # N c
                                                                        edge_index=edge_index, # 2 E
                                                                        batch=node_batch_tensor, # N
                                                                        threshold=threshold,
                                                                        edge_score=edge_score,# E
                                                                        token_masks=token_masks,) # Nt N
                edge_score = self.get_edge_score(nodes_feature, edge_index)
                assert len(token_masks.unique()) == 2
            # TODO: 还有0.799的边:)
            connect_adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_score, max_num_nodes=nodes_feature.shape[0])[0]
            connect_adj = connect_adj + connect_adj.t() # Nt Nt
            node_max_simlarity, _ = connect_adj.max(dim=1) # Nt
            node_instanceness = [instanceness_function(foo) for foo in node_max_simlarity] # N

            node_num_patches = (token_masks > 0).int().sum(-1) # Nt
            for node_idx in range(nodes_feature.shape[0]):
                first_condition = node_num_patches[node_idx] >= clustering_area_threshold
                if first_condition:
                    this_node_batch_idx = node_batch_tensor[node_idx]
                    this_node_feat = nodes_feature[node_idx]
                    this_node_mask = token_masks[node_idx, (H*W*this_node_batch_idx):((this_node_batch_idx+1)*H*W)].reshape(H, W).bool() # B*HW -> HW
                    this_node_instancess = node_instanceness[node_idx]
                    batch_clusters_feats[this_node_batch_idx][thre_idx].append(this_node_feat)
                    batch_clusters_masks[this_node_batch_idx][thre_idx].append(this_node_mask)
                    batch_clusters_instanceness[this_node_batch_idx][thre_idx].append(this_node_instancess)
            for batch_idx in range(batch_size):
                batch_clusters_feats[batch_idx][thre_idx] = torch.stack(batch_clusters_feats[batch_idx][thre_idx], dim=0)
                batch_clusters_masks[batch_idx][thre_idx] = torch.stack(batch_clusters_masks[batch_idx][thre_idx], dim=0)
                batch_clusters_instanceness[batch_idx][thre_idx] = torch.tensor(batch_clusters_instanceness[batch_idx][thre_idx])
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # visualize:
        # for batch_idx in range(batch_size):
        #     visualized_image = visualize_cutler_thre_masks(image=orig_image[batch_idx], cluster_masks=batch_clusters_masks[batch_idx], patch_size=self.backbone_patch_size)
        #     Image.fromarray(visualized_image.numpy()).save(f'./tmp/test_{int(image_ids[batch_idx])}.png')
        batch_clusters_feats = [torch.cat(foo, dim=0) for foo in batch_clusters_feats] # list[ni c], batch
        batch_clusters_masks = [torch.cat(foo, dim=0) for foo in batch_clusters_masks] # list[ni h w], batch, bool
        batch_clusters_instanceness = [torch.cat(foo) for foo in batch_clusters_instanceness] # list[ni], batch, float
        # NMS
        clustering_nms_iou = 0.9
        clustering_nms_step = len(clustering_threshold_list)
        nms_feats = [] 
        nms_masks = []
        nms_instanceness = [] # list[ni]
        for pool, pool_feat, pool_ins in zip(batch_clusters_masks,batch_clusters_feats, batch_clusters_instanceness) :
            pool = pool.unbind(0) # list[h w], ni
            mask_areas = torch.tensor([area(foo) for foo in pool])
            _, sorted_idxs = torch.sort(mask_areas, descending=True, dim=0)
            sorted_masks = [pool[foo_idx] for foo_idx in sorted_idxs]
            sorted_feats = [pool_feat[foo_idx] for foo_idx in sorted_idxs]
            sorted_ins = [pool_ins[foo_idx] for foo_idx in sorted_idxs] # list[float]
            masks_kept_indices = list(range(len(pool)))
            for i in range(len(sorted_masks)):
                if i in masks_kept_indices:
                    for j in range(i+1, min(len(sorted_masks), i+clustering_nms_step)):
                        if iou(sorted_masks[i], sorted_masks[j]) > clustering_nms_iou:
                            masks_kept_indices.remove(j) if j in masks_kept_indices else None
            nms_pool = torch.stack([sorted_masks[i] for i in masks_kept_indices], dim=0)
            nms_pool_feats = torch.stack([sorted_feats[i] for i in masks_kept_indices], dim=0)
            nms_pool_ins = torch.tensor([sorted_ins[i] for i in masks_kept_indices]) # ni
            nms_feats.append(nms_pool_feats)
            nms_masks.append(nms_pool)
            nms_instanceness.append(nms_pool_ins)
        
        return nms_feats, nms_masks, nms_instanceness
        # NMS
        
        # TODO: NMS
        # NMS的step和list的长度一样
        # TODO2:  每个节点维护到目前为止它的子树中注册的最大面积的hw
        # TODO: visualize是否运行的好
        # from models.UN_IMG_SEM.AggSample.agg_sample import visualize_cutler_onlyAttn
        # for batch_idx in range(batch_size):
        #     visualize_cutler_onlyAttn(image=orig_image[batch_idx].float(),
        #                             cluster_attn=image_attns[batch_idx].float(), # h w nq
        #                             patch_size=self.patch_size,
        #                             lengths=threshold_lenghts[batch_idx],
        #                             image_path='./tmp',
        #                             image_name=f'test_iter{iteration_num}_bs{batch_idx}.png')
        # TODO: mask的获得是通过graph, 还是通过attention得到mask(利用hp/alignseg的attention->mask的方式)

    @torch.no_grad()
    def batch_feat_attns(self, feat_attns, temperature):
        # b hw hw [-1,1] -> bhw bhw
        dtype = feat_attns.dtype
        batch_size, HW, _ = feat_attns.shape
        feat_attns = feat_attns / temperature
        batch_attns = torch.zeros([batch_size*HW, batch_size*HW], dtype=dtype).to(feat_attns.device)
        for batch_idx in range(batch_size):
            batch_attns[(batch_idx*HW): (batch_idx+1)*HW, (batch_idx*HW): (batch_idx+1)*HW].copy_(feat_attns[batch_idx])
        return batch_attns

    
    @torch.no_grad()
    def forward_online_agg_clustering_orig_dino(self, image_feats, 
                                                orig_image=None,
                                                image_ids=None):
        """
        image_feats: b c h w
        feats_attns: b hw hw, [-1, 1]
        return: 
            list[ni c], batch, 特征
            list[ni h w(bool)], batch, 特征mask
            大小和image_feats的大小一致, 都是缩放patch_size之后的
        """
        # b c h w -> bhw c
        batch_size, _, H, W = image_feats.shape
        clustering_threshold_list = [0.8, 0.7, 0.6, 0.5, 0.4]
        feature_merge_temperature = 0.07
        clustering_area_threshold = 4 # TODO:一个物体的大小4*32像素
        # TODO: self.的attention永远是1， 可不可以用dino的attn, self.就不是1了
        feats_attns = torch.einsum('bcs,bcd->bsd', F.normalize(image_feats.flatten(2), dim=1),  F.normalize(image_feats.flatten(2), dim=1))
        def instanceness_function(foo_x, start=0.25, cnt=0.35, cnt_value=0.2):
            # 0到cnt是从1到cnt_value, cnt到1是从cnt_value到0
            if foo_x < start:
                return 1.
            elif foo_x < cnt:
                return 1 - (1-cnt_value) / (cnt-start) * (foo_x-start)
            elif foo_x >= cnt:
                return cnt_value - (cnt_value) / (1 - cnt) * (foo_x - cnt)
            else:
                raise ValueError()
        
        bhw_attns = self.batch_feat_attns(feat_attns=feats_attns, temperature=feature_merge_temperature) # N N
        bhw_features = image_feats.permute(0, 2, 3, 1).flatten(0, 2) # N c
        assert bhw_attns.shape[0] == bhw_features.shape[0] # N N
        
        edge_index= self.train_edge_index if self.training else self.eval_edge_index # 2 E
        node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids # N
        batch_mask = self.train_batch_mask if self.training else self.eval_batch_mask
        token_masks = self.train_token_masks if self.training else self.eval_token_masks # N N, 0-1
        edge_score = self.get_edge_score(bhw_features, edge_index) # E, -1/1
        batch_clusters_feats = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni c], threshold] batch
        batch_clusters_masks = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni h w], threshold] batch
        batch_clusters_instanceness = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni], threshold] batch
        for thre_idx, threshold in enumerate(clustering_threshold_list):
            while (edge_score > threshold).any():
                # Nt c,  2 Et, Nt, Nt N(0-1)
                nodes_feature, edge_index, node_batch_tensor, token_masks = self.hc_graph_orig_dino(edge_index=edge_index, # 2 Et
                                                                                    batch=node_batch_tensor, # Nt
                                                                                    threshold=threshold,
                                                                                    edge_score=edge_score,# Et
                                                                                    token_masks=token_masks,
                                                                                    bhw_features=bhw_features,
                                                                                    bhw_attns=bhw_attns,
                                                                                    batch_mask=batch_mask) # Nt N
                edge_score = self.get_edge_score(nodes_feature, edge_index)
                assert len(token_masks.unique()) == 2
            # TODO: 还有0.799的边:)
            connect_adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_score, max_num_nodes=nodes_feature.shape[0])[0]
            connect_adj = connect_adj + connect_adj.t() # Nt Nt
            node_max_simlarity, _ = connect_adj.max(dim=1) # Nt
            node_instanceness = [instanceness_function(foo) for foo in node_max_simlarity] # N

            node_num_patches = (token_masks > 0).int().sum(-1) # Nt
            for node_idx in range(nodes_feature.shape[0]):
                first_condition = node_num_patches[node_idx] >= clustering_area_threshold
                if first_condition:
                    this_node_batch_idx = node_batch_tensor[node_idx]
                    this_node_feat = nodes_feature[node_idx]
                    this_node_mask = token_masks[node_idx, (H*W*this_node_batch_idx):((this_node_batch_idx+1)*H*W)].reshape(H, W).bool() # B*HW -> HW
                    this_node_instancess = node_instanceness[node_idx]
                    batch_clusters_feats[this_node_batch_idx][thre_idx].append(this_node_feat)
                    batch_clusters_masks[this_node_batch_idx][thre_idx].append(this_node_mask)
                    batch_clusters_instanceness[this_node_batch_idx][thre_idx].append(this_node_instancess)
            if len(batch_clusters_feats[this_node_batch_idx][thre_idx]) != 0:
                for batch_idx in range(batch_size): 
                    batch_clusters_feats[batch_idx][thre_idx] = torch.stack(batch_clusters_feats[batch_idx][thre_idx], dim=0)
                    batch_clusters_masks[batch_idx][thre_idx] = torch.stack(batch_clusters_masks[batch_idx][thre_idx], dim=0)
                    batch_clusters_instanceness[batch_idx][thre_idx] = torch.tensor(batch_clusters_instanceness[batch_idx][thre_idx])
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # visualize:
        # for batch_idx in range(batch_size):
        #     visualized_image = visualize_cutler_thre_masks(image=orig_image[batch_idx], cluster_masks=batch_clusters_masks[batch_idx], patch_size=self.backbone_patch_size)
        #     Image.fromarray(visualized_image.numpy()).save(f'./tmp_after_attn_512/test_{int(image_ids[batch_idx])}.png')
        batch_clusters_feats = [torch.cat(foo, dim=0) for foo in batch_clusters_feats] # list[ni c], batch
        batch_clusters_masks = [torch.cat(foo, dim=0) for foo in batch_clusters_masks] # list[ni h w], batch, bool
        batch_clusters_instanceness = [torch.cat(foo) for foo in batch_clusters_instanceness] # list[ni], batch, float
        # NMS
        clustering_nms_iou = 0.9
        clustering_nms_step = len(clustering_threshold_list)
        nms_feats = [] 
        nms_masks = []
        nms_instanceness = [] # list[ni]
        for pool, pool_feat, pool_ins in zip(batch_clusters_masks,batch_clusters_feats, batch_clusters_instanceness) :
            pool = pool.unbind(0) # list[h w], ni
            mask_areas = torch.tensor([area(foo) for foo in pool])
            _, sorted_idxs = torch.sort(mask_areas, descending=True, dim=0)
            sorted_masks = [pool[foo_idx] for foo_idx in sorted_idxs]
            sorted_feats = [pool_feat[foo_idx] for foo_idx in sorted_idxs]
            sorted_ins = [pool_ins[foo_idx] for foo_idx in sorted_idxs] # list[float]
            masks_kept_indices = list(range(len(pool)))
            for i in range(len(sorted_masks)):
                if i in masks_kept_indices:
                    for j in range(i+1, min(len(sorted_masks), i+clustering_nms_step)):
                        if iou(sorted_masks[i], sorted_masks[j]) > clustering_nms_iou:
                            masks_kept_indices.remove(j) if j in masks_kept_indices else None
            nms_pool = torch.stack([sorted_masks[i] for i in masks_kept_indices], dim=0)
            nms_pool_feats = torch.stack([sorted_feats[i] for i in masks_kept_indices], dim=0)
            nms_pool_ins = torch.tensor([sorted_ins[i] for i in masks_kept_indices]) # ni
            nms_feats.append(nms_pool_feats)
            nms_masks.append(nms_pool)
            nms_instanceness.append(nms_pool_ins)

        # TODO: crf postprocess
        # 每个ni的分数: 前景attn的平均值
        # pred_scores = [] # list[ni], batch
        # for batch_idx in range(B):
        #     q_attns = torch.einsum('chw,nc->nhw', 
        #                            F.normalize(image_feats[batch_idx], dim=0), F.normalize(q_feats[batch_idx], dim=-1))
        #     # 前景的attn的平均值
        #     q_score: Tensor = (q_masks[batch_idx].float() * q_attns).sum([1,2]) / (q_masks[batch_idx].float().sum([1,2])) # (ni h w * ni h w)
        #     q_score = q_score.clamp(min=0.)
        #     pred_scores.append(q_score)

        return nms_feats, nms_masks, nms_instanceness
        # NMS
        
        # TODO: NMS
        # NMS的step和list的长度一样
        # TODO2:  每个节点维护到目前为止它的子树中注册的最大面积的hw
        # TODO: visualize是否运行的好
        # from models.UN_IMG_SEM.AggSample.agg_sample import visualize_cutler_onlyAttn
        # for batch_idx in range(batch_size):
        #     visualize_cutler_onlyAttn(image=orig_image[batch_idx].float(),
        #                             cluster_attn=image_attns[batch_idx].float(), # h w nq
        #                             patch_size=self.patch_size,
        #                             lengths=threshold_lenghts[batch_idx],
        #                             image_path='./tmp',
        #                             image_name=f'test_iter{iteration_num}_bs{batch_idx}.png')
        # TODO: mask的获得是通过graph, 还是通过attention得到mask(利用hp/alignseg的attention->mask的方式)

    def hc_graph_orig_dino(self,edge_index: Tensor, # 2 Et, 每个edge从小到大, 但是是一个无向图
                                batch: Tensor, # Nt
                                threshold: float,
                                edge_score, # Et, float, -1/1
                                token_masks, # Nt N
                                bhw_features, # N c
                                bhw_attns, # N N, -inf
                                batch_mask # N N
                    ):
        """edge_index_out, batch_out, token_masks
        edge_index 2 Et, long()
        batch: Nt, int
        threshold: float
        edge_score: Et
        token_masks: Nt N, float0/1
        """
        (NT, N), device = token_masks.shape, token_masks.device
        edge_contract = edge_index[:, edge_score > threshold]
        adj = to_scipy_sparse_matrix(edge_contract, num_nodes=NT)
        adj = adj + adj.T
        # NT NT, 原始节点之间是否被选中
        _, cluster_np = connected_components(adj, directed=False)
        # NT
        cluster = torch.tensor(cluster_np, dtype=torch.long, device=device)
        # NT N'  #新节点的归属问题, sum(-1)都是1
        C = one_hot(cluster) 
        _, NT_NEW = C.shape
        # NT NT, 原始节点之间是否连接
        A = to_dense_adj(edge_index, max_num_nodes=NT).squeeze(0)
        A = A + A.T

        # N' Nt, Nt N -> N' N
        # union of masks of previous nodes
        token_masks = C.t() @ token_masks
        assert token_masks.max() <= 1
        # 肯定不会超过1, 因为union的时候不会有重叠的区域, 但有可能全是1

        # TODO: 先每个softmax,然后加权; 先加，然后一起softmax
        node_attns:Tensor = token_masks @ bhw_attns
        # node_batch_mask = (token_masks @ (batch_mask.float())).bool()
        # node_attns.masked_fill_(node_batch_mask, value=torch.finfo())
        # N' N @ N c -> N' c
        node_features = node_attns.softmax(-1) @ bhw_features

        # N' N @ N N @ N N' -> N' N'
        new_new_adj = (C.T @ A @ C).fill_diagonal_(0)
        edge_index_out, _ = dense_to_sparse(torch.triu(new_new_adj)) # 保持小的在前

        # N'
        # batch_out[cluster[i]]=batch[i], i=1...N
        batch_out = batch.new_empty(NT_NEW).scatter_(0, cluster, batch)
        return node_features, edge_index_out, batch_out, token_masks


    @torch.no_grad()
    def forward_online_agg_clustering_withSimi(self, image_feats,):
        """
        image_feats: b c h w
        feats_attns: b hw hw, [-1, 1]
        return: 
            list[ni c], batch, 特征
            list[ni h w(bool)], batch, 特征mask
            大小和image_feats的大小一致, 都是缩放patch_size之后的
        """
        # b c h w -> bhw c
        batch_size, _, H, W = image_feats.shape
        feature_merge_temperature = 0.07
        clustering_threshold_list = [0.8, 0.7, 0.6, 0.5, 0.4]
        clustering_area_threshold = 4 # TODO:一个物体的大小4*32像素
        # TODO: self.的attention永远是1， 可不可以用dino的attn, self.就不是1了
        feats_attns = torch.einsum('bcs,bcd->bsd', 
                                   F.normalize(image_feats.flatten(2), dim=1),  
                                   F.normalize(image_feats.flatten(2), dim=1)) # -1, 1
        
        bhw_attns = self.batch_feat_attns(feat_attns=feats_attns, temporature=feature_merge_temperature) # N N, /temperature
        bhw_features = image_feats.permute(0, 2, 3, 1).flatten(0, 2) # N c
        assert bhw_attns.shape[0] == bhw_features.shape[0] # N N
        
        edge_index= self.train_edge_index if self.training else self.eval_edge_index # 2 E
        node_batch_tensor=self.train_node_batch_ids if self.training else self.eval_node_batch_ids # N
        batch_mask = self.train_batch_mask if self.training else self.eval_batch_mask
        token_masks = self.train_token_masks if self.training else self.eval_token_masks # N N, 0-1
        edge_score = self.get_edge_score(bhw_features, edge_index) # E, -1/1
        batch_clusters_feats = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni c], threshold] batch
        batch_clusters_masks = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni h w], threshold] batch
        batch_clusters_instanceness = [[[] for _ in range(len(clustering_threshold_list))] for _ in range(batch_size)]# list[list[ni], threshold] batch
        for thre_idx, threshold in enumerate(clustering_threshold_list):
            while (edge_score > threshold).any():
                # Nt c,  2 Et, Nt, Nt N(0-1)
                nodes_feature, edge_index, node_batch_tensor, token_masks = self.hc_graph_orig_dino(edge_index=edge_index, # 2 Et
                                                                                    batch=node_batch_tensor, # Nt
                                                                                    threshold=threshold,
                                                                                    edge_score=edge_score,# Et
                                                                                    token_masks=token_masks,
                                                                                    bhw_features=bhw_features,
                                                                                    bhw_attns=bhw_attns,
                                                                                    batch_mask=batch_mask) # Nt N
                edge_score = self.get_edge_score(nodes_feature, edge_index)
                assert len(token_masks.unique()) == 2
            # TODO: 还有0.799的边:)
            connect_adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_score, max_num_nodes=nodes_feature.shape[0])[0]
            connect_adj = connect_adj + connect_adj.t() # Nt Nt
            node_max_simlarity, _ = connect_adj.max(dim=1) # Nt
            node_instanceness = [instanceness_function(foo) for foo in node_max_simlarity] # N

            node_num_patches = (token_masks > 0).int().sum(-1) # Nt
            for node_idx in range(nodes_feature.shape[0]):
                first_condition = node_num_patches[node_idx] >= clustering_area_threshold
                if first_condition:
                    this_node_batch_idx = node_batch_tensor[node_idx]
                    this_node_feat = nodes_feature[node_idx]
                    this_node_mask = token_masks[node_idx, (H*W*this_node_batch_idx):((this_node_batch_idx+1)*H*W)].reshape(H, W).bool() # B*HW -> HW
                    this_node_instancess = node_instanceness[node_idx]
                    batch_clusters_feats[this_node_batch_idx][thre_idx].append(this_node_feat)
                    batch_clusters_masks[this_node_batch_idx][thre_idx].append(this_node_mask)
                    batch_clusters_instanceness[this_node_batch_idx][thre_idx].append(this_node_instancess)
            if len(batch_clusters_feats[this_node_batch_idx][thre_idx]) != 0:
                for batch_idx in range(batch_size): 
                    batch_clusters_feats[batch_idx][thre_idx] = torch.stack(batch_clusters_feats[batch_idx][thre_idx], dim=0)
                    batch_clusters_masks[batch_idx][thre_idx] = torch.stack(batch_clusters_masks[batch_idx][thre_idx], dim=0)
                    batch_clusters_instanceness[batch_idx][thre_idx] = torch.tensor(batch_clusters_instanceness[batch_idx][thre_idx])
        # TODO: NMS和N_cnt进行比较，看哪个比较好
        # visualize:
        # for batch_idx in range(batch_size):
        #     visualized_image = visualize_cutler_thre_masks(image=orig_image[batch_idx], cluster_masks=batch_clusters_masks[batch_idx], patch_size=self.backbone_patch_size)
        #     Image.fromarray(visualized_image.numpy()).save(f'./tmp_after_attn_512/test_{int(image_ids[batch_idx])}.png')
        batch_clusters_feats = [torch.cat(foo, dim=0) for foo in batch_clusters_feats] # list[ni c], batch
        batch_clusters_masks = [torch.cat(foo, dim=0) for foo in batch_clusters_masks] # list[ni h w], batch, bool
        batch_clusters_instanceness = [torch.cat(foo) for foo in batch_clusters_instanceness] # list[ni], batch, float
        # NMS
        clustering_nms_iou = 0.9
        clustering_nms_step = len(clustering_threshold_list)
        nms_feats = [] 
        nms_masks = []
        nms_instanceness = [] # list[ni]
        for pool, pool_feat, pool_ins in zip(batch_clusters_masks,batch_clusters_feats, batch_clusters_instanceness) :
            pool = pool.unbind(0) # list[h w], ni
            mask_areas = torch.tensor([area(foo) for foo in pool])
            _, sorted_idxs = torch.sort(mask_areas, descending=True, dim=0)
            sorted_masks = [pool[foo_idx] for foo_idx in sorted_idxs]
            sorted_feats = [pool_feat[foo_idx] for foo_idx in sorted_idxs]
            sorted_ins = [pool_ins[foo_idx] for foo_idx in sorted_idxs] # list[float]
            masks_kept_indices = list(range(len(pool)))
            for i in range(len(sorted_masks)):
                if i in masks_kept_indices:
                    for j in range(i+1, min(len(sorted_masks), i+clustering_nms_step)):
                        if iou(sorted_masks[i], sorted_masks[j]) > clustering_nms_iou:
                            masks_kept_indices.remove(j) if j in masks_kept_indices else None
            nms_pool = torch.stack([sorted_masks[i] for i in masks_kept_indices], dim=0)
            nms_pool_feats = torch.stack([sorted_feats[i] for i in masks_kept_indices], dim=0)
            nms_pool_ins = torch.tensor([sorted_ins[i] for i in masks_kept_indices]) # ni
            nms_feats.append(nms_pool_feats)
            nms_masks.append(nms_pool)
            nms_instanceness.append(nms_pool_ins)
        
        return nms_feats, nms_masks, nms_instanceness

    def hc_graph_orig_withSimi(self,
                               edge_index: Tensor, # 2 Et, 每个edge从小到大, 但是是一个无向图
                                batch: Tensor, # Nt
                                threshold: float,
                                edge_score, # Et, float, -1/1
                                token_masks, # Nt N
                                bhw_features, # N c
                                bhw_attns, # N N, -inf
                                batch_mask # N N
                    ):
        """edge_index_out, batch_out, token_masks
        edge_index 2 Et, long()
        batch: Nt, int
        threshold: float
        edge_score: Et
        token_masks: Nt N, float0/1
        """
        (NT, N), device = token_masks.shape, token_masks.device
        edge_contract = edge_index[:, edge_score > threshold]
        adj = to_scipy_sparse_matrix(edge_contract, num_nodes=NT)
        adj = adj + adj.T
        # NT NT, 原始节点之间是否被选中
        _, cluster_np = connected_components(adj, directed=False)
        # NT
        cluster = torch.tensor(cluster_np, dtype=torch.long, device=device)
        # NT N'  #新节点的归属问题, sum(-1)都是1
        C = one_hot(cluster) 
        _, NT_NEW = C.shape
        # NT NT, 原始节点之间是否连接
        A = to_dense_adj(edge_index, max_num_nodes=NT).squeeze(0)
        A = A + A.T

        # N' Nt, Nt N -> N' N
        # union of masks of previous nodes
        token_masks = C.t() @ token_masks
        assert token_masks.max() <= 1
        # 肯定不会超过1, 因为union的时候不会有重叠的区域, 但有可能全是1

        # TODO: 先每个softmax,然后加权; 先加，然后一起softmax
        node_attns:Tensor = token_masks @ bhw_attns
        # N' N @ N c -> N' c
        node_features = node_attns.softmax(-1) @ bhw_features

        # N' N @ N N @ N N' -> N' N'
        new_new_adj = (C.T @ A @ C).fill_diagonal_(0)
        edge_index_out, _ = dense_to_sparse(torch.triu(new_new_adj)) # 保持小的在前

        # N'
        # batch_out[cluster[i]]=batch[i], i=1...N
        batch_out = batch.new_empty(NT_NEW).scatter_(0, cluster, batch)
        return node_features, edge_index_out, batch_out, token_masks



    def hc_graph(self,
                    x: Tensor, # N c
                    edge_index: Tensor, # 2 E, 每个edge从小到大, 但是是一个无向图
                    batch: Tensor, # N
                    threshold: float,
                    edge_score, # E, float, -1/1
                    token_masks, # N N
                    ):
        """x_out, edge_index_out, batch_out, token_masks
        x: N c
        edge_index 2 E, long()
        batch: N, int
        token_masks: N N, float0/1
        """
        edge_contract = edge_index[:, edge_score > threshold] # 2 E_chosen
        # A_contract = to_dense_adj(edge_contract,max_num_nodes=x.size(0)).squeeze(0)
        # A_contract = A_contract + A_contract.T
        adj = to_scipy_sparse_matrix(edge_contract, num_nodes=x.size(0))
        adj = adj + adj.T
        # N N, 原始节点之间是否被选中
        # A_contract = torch.from_numpy(adj.toarray()).float().to(edge_contract.device)
        # N 每个节点之后的index
        _, cluster_np = connected_components(adj, directed=False)
        cluster = torch.tensor(cluster_np, dtype=torch.long, device=x.device)#
        C = one_hot(cluster) # N N' 新节点的归属问题, sum(-1)都是1

        # N N, 原始节点之间是否连接
        A = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
        A = A + A.T
        # N N, 原始节点之间的相似度, 不相连的两个节点之间是0
        # S = to_dense_adj(edge_index, edge_attr=edge_score, max_num_nodes=x.size(0)).squeeze(0)
        # S = S + S.T
        # # 单个节点自成一派的话就是1
        # nodes_single = ((A_contract.sum(dim=-1) + A_contract.sum(dim=-2)) == 0).nonzero()
        # S[nodes_single, nodes_single] = 1.0
        
        # N N * N N' -> N' N @ N c -> N' c
        # x_out = (S @ C).t() @ x


        # N' N @ N c -> N' c 
        x_out = C.t() @ x # TODO:nomalize方式
        # x_out = x_out / node_num_patches[:, None]

        # # N' Nt * 1 Nt
        # feature_weights = C.t() * (token_masks.sum(dim=1)[None, :])
        # # N' Nt / N' 1
        # feature_weights = feature_weights / (feature_weights.sum(dim=1, keepdim=True)) # * node数量
        # x_out = feature_weights @ x

        # N' N, N N -> N' N
        # union of masks of previous nodes
        token_masks = C.t() @ token_masks
        assert len(token_masks.unique()) == 2
        # 肯定不会超过1, 因为union的时候不会有重叠的区域

        # N' N @ N N @ N N' -> N' N'
        new_new_adj = (C.T @ A @ C).fill_diagonal_(0)
        edge_index_out, _ = dense_to_sparse(torch.triu(new_new_adj))

        # N'
        # batch_out[cluster[i]]=batch[i], i=1...N
        batch_out = batch.new_empty(x_out.size(0)).scatter_(0, cluster, batch)
        return x_out, edge_index_out, batch_out, token_masks

    def get_edge_score(self, nodes_feature, edge_index):
        # N c, 2 E
        src_nodes_feats = nodes_feature[edge_index[0]] # e c
        tgt_nodes_feats = nodes_feature[edge_index[1]] # e c
        edge_score = torch.einsum('ec,ec->e', 
                                  F.normalize(src_nodes_feats, dim=-1, eps=1e-10),  F.normalize(tgt_nodes_feats, dim=-1, eps=1e-10))
        return edge_score

    def init_graph_utils(self, train_batch_size, train_size, eval_size):
        # test=1
        H = W = (train_size // self.backbone_patch_size) 
        num_train_nodes = H*W
        train_edge_index = []
        train_masks = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    train_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    train_edge_index.append([cluster_idx-W, cluster_idx])
                token_mask = torch.zeros(H*W).to(torch.bfloat16)
                token_mask[cluster_idx] = 1.
                train_masks.append(token_mask)
                cluster_idx += 1
        train_masks = torch.stack(train_masks, dim=0) # N HW
        train_edge_index = torch.tensor(train_edge_index).permute(1, 0).contiguous().long() # 2 N

        H = W = (eval_size // self.backbone_patch_size)
        num_test_nodes = H*W
        eval_edge_index = []
        eval_masks = []
        cluster_idx = 0
        for y in range(H):
            for x in range(W):
                assert cluster_idx == y * H + x
                if (cluster_idx % W) != 0: # 不是第一列的token, 定义它和它前面的那个token的相似度
                    eval_edge_index.append([cluster_idx-1, cluster_idx])
                if (cluster_idx - W) >= 0: # 不是第一行的token, 定义它和它上一行的那个token的相似度
                    eval_edge_index.append([cluster_idx-W, cluster_idx])
                token_mask = torch.zeros(H*W).to(torch.bfloat16)
                token_mask[cluster_idx] = 1.
                eval_masks.append(token_mask)
                cluster_idx += 1
        eval_masks = torch.stack(eval_masks, dim=0) # N HW
        eval_edge_index = torch.tensor(eval_edge_index).permute(1, 0).contiguous().long()

        from torch_geometric.data import Batch, Data
        train_graph = Data(edge_index=train_edge_index,)
        train_graph.num_nodes = num_train_nodes
        eval_graph = Data(edge_index=eval_edge_index)
        eval_graph.num_nodes = num_test_nodes

        whole_train_graph = [train_graph.clone() for _ in range(train_batch_size)]
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in whole_train_graph]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in whole_train_graph]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        whole_train_graph = Batch.from_data_list(whole_train_graph)

        # HW HW -> N bhw
        train_masks = F.pad(train_masks, [0, (train_batch_size-1)*num_train_nodes, 0, 0], value=0.)
        all_train_masks = []
        for batch_idx in range(train_batch_size):
            all_train_masks.append(train_masks.roll(shifts=num_train_nodes*batch_idx, dims=1))
        all_train_masks = torch.cat(all_train_masks, dim=0) # bhw bhw

        self.register_buffer('train_node_batch_ids', torch.tensor(nodes_batch_ids).int())
        self.register_buffer('train_edge_batch_ids', torch.tensor(edges_batch_ids).int())
        self.register_buffer('train_edge_index', whole_train_graph.edge_index.long())
        self.register_buffer('train_token_masks', all_train_masks)
        self.register_buffer('train_cnt_registered_area', torch.zeros(num_train_nodes, num_train_nodes).int()) # BHW BHW

        # 到目前位置的

        self.register_buffer('eval_node_batch_ids', torch.zeros(eval_graph.num_nodes).int())
        self.register_buffer('eval_edge_batch_ids', torch.zeros(eval_graph.num_edges).int())
        self.register_buffer('eval_edge_index', eval_graph.edge_index.long())
        self.register_buffer('eval_token_masks', eval_masks)
        self.register_buffer('eval_cnt_registered_area', torch.zeros(num_test_nodes, num_test_nodes).int()) # HW HW


        train_batch_mask = torch.ones([train_batch_size*num_train_nodes, train_batch_size*num_train_nodes])
        for batch_idx in range(train_batch_size):
            train_batch_mask[(batch_idx*num_train_nodes):((batch_idx+1)*num_train_nodes), 
                             batch_idx*num_train_nodes:((batch_idx+1)*num_train_nodes)].fill_(0)
        train_batch_mask = train_batch_mask.bool()

        eval_batch_mask = torch.zeros([num_test_nodes, num_test_nodes])
        eval_batch_mask = eval_batch_mask.bool()

        self.register_buffer('train_batch_mask', train_batch_mask)
        self.register_buffer('eval_batch_mask', eval_batch_mask)    


@register_model
def online_cutler_evalCluster(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    eval_dataset_name = list(configs['data']['evaluate'].keys())[0]
    model = Online_Cutler(configs, 
                          num_classes=num_classes,
                          batch_size=configs['optim']['batch_size'],
                          train_size=configs['data']['train'][train_dataset_name]['mapper']['res'],
                           eval_size=configs['data']['evaluate'][eval_dataset_name]['mapper']['res'])
    model.to(device)
    model.optimize_setup(configs)
    model.sample = model.sample_oap
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function


# 同一个module, 不同的研究流程
@register_model
def online_cutler_evalDino(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    eval_dataset_name = list(configs['data']['evaluate'].keys())[0]
    model = Online_Cutler(configs, 
                          num_classes=num_classes,
                          batch_size=configs['optim']['batch_size'],
                          train_size=configs['data']['train'][train_dataset_name]['mapper']['res'],
                           eval_size=configs['data']['evaluate'][eval_dataset_name]['mapper']['res'])
    model.to(device)
    model.optimize_setup(configs)
    model.sample = model.sample_eval_onlineCluster
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

@register_model
def online_cutler_evalEncoder(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    from detectron2.data import MetadataCatalog
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_dataset_name = list(configs['data']['train'].keys())[0]
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    num_classes = MetadataCatalog.get(train_dataset_name).get('num_classes')
    eval_dataset_name = list(configs['data']['evaluate'].keys())[0]
    model = Online_Cutler(configs, 
                          num_classes=num_classes,
                          batch_size=configs['optim']['batch_size'],
                          train_size=configs['data']['train'][train_dataset_name]['mapper']['res'],
                           eval_size=configs['data']['evaluate'][eval_dataset_name]['mapper']['res'])
    model.to(device)
    model.optimize_setup(configs)
    model.sample = model.sample_eval_onlineCluster
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function