"""
Modified from DETR https://github.com/facebookresearch/detr
"""
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
from einops.layers.torch import Rearrange
from data_schedule import build_schedule
from torch import einsum
import math
from typing import List, Optional
from utils.misc import NestedTensor
import numpy as np
from models.layers.mamba.ss2d import SS2D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm import Mamba
from torch.cuda.amp import autocast
import logging
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
import torchvision.transforms.functional as Trans_F
import copy
from models.layers.mamba.vss_layer_3d import VSSLayer_3d
from models.layers.matching import dice_loss, ce_mask_loss

from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from models.layers.utils import _get_clones
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

from models.backbone.video_swin import PatchEmbed3D, PatchMerging
from models.backbone.utils import VideoMultiscale_Shape

class MedSam(nn.Module):
    def __init__(
        self,
        configs=None,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        from models.VIS.outers.segment_anything import sam_model_registry
        self.medsam_model = sam_model_registry["vit_b"](checkpoint='/home/xuhuihui/pt/medsam/medsam_vit_b.pth')

    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict,):
        # b 3 t h w
        batch_size, _, nf = videos.shape[:3]
        orig_size = videos.shape[-2:]
        videos = videos.permute(0, 2, 1, 3, 4).flatten(0, 1) # bt 3 h w
        videos = F.interpolate(videos, size=(1024, 1024), mode='bilinear', align_corners=False)

        box_torch = torch.tensor([0, 0, 1024, 1024]).float().to(self.device) # 4
        box_torch = box_torch.unsqueeze(0).unsqueeze(1).repeat(videos.shape[0], 1, 1) # bt 1 4
        
        low_res_logits_by_image = []
        for batch_idx in range(videos.shape[0]):
            img = videos[[batch_idx]] # 1 3 h w
            box_t = box_torch[[batch_idx]] # 1 1 4
            image_embedding = self.medsam_model.image_encoder(img)  # bt c h w # 16

            sparse_embeddings, dense_embeddings = self.medsam_model.prompt_encoder(
                points=None,
                boxes=box_t,
                masks=None,
            )
            low_res_logits, _ = self.medsam_model.mask_decoder(
                image_embeddings=image_embedding,  
                image_pe=self.medsam_model.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embeddings, 
                dense_prompt_embeddings=dense_embeddings,  
                multimask_output=False,
            ) # bt 1 h w
            low_res_logits_by_image.append(low_res_logits)

        low_res_logits_by_image = torch.cat(low_res_logits_by_image, dim=0) # bt 1 h w
        low_res_pred = F.interpolate(
            low_res_logits_by_image,
            size=orig_size,
            mode="bilinear",
            align_corners=False,
        )   # bt 1 H W
        # b n c -> b t 1 c, probability
        pred_class = torch.tensor([1., 0.]).to(self.device)
        pred_class = repeat(pred_class, 'c -> b t 1 c',b=batch_size, t=nf)
        return {'pred_masks': rearrange(low_res_pred, '(b t) 1 h w -> b t 1 h w',t=nf,b=batch_size),
                'pred_class': pred_class}

        # pred_masks: b n t h w;
        # pred_class: b t n c

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        # videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        # 如果是List的话, 那么取最后一层
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear') > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

@register_model
def medsam(configs, device):
    from models.VIS.aux_mapper import AUXMapper_v1
    model = MedSam(configs)
    from torch.optim import SGD
    from torch.optim.lr_scheduler import MultiStepLR
    model.to(device)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])
    optimizer = SGD(model.parameters(), lr=1e-3,)
    lr_scheduler = MultiStepLR(optimizer=optimizer,milestones=[3,10], gamma=0.1, verbose=False)


    train_samplers, train_loaders, eval_function, dataset_features = build_schedule(configs, model_input_mapper.mapper, 
                                                                                    partial(model_input_mapper.collate, max_stride=16))

    # dataset_specific initialization

    return model, optimizer, lr_scheduler,  train_samplers, train_loaders, {'none': 0}, eval_function

