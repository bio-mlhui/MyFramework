from typing import Any, Optional, List, Dict, Set
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as Trans_F
from einops import repeat, reduce, rearrange
from utils.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from functools import partial
from utils.misc import to_device
from models.utils.visualize_amr import save_model_output
from models.registry import register_model
from data_schedule.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from models.optimization.scheduler import build_scheduler 
from detectron2.config import configurable
from models.registry import register_model
import detectron2.utils.comm as comm
import copy
from models.optimization.utils import get_total_grad_norm
from models.optimization.optimizer import get_optimizer
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import matplotlib.pyplot as plt
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video, VIS_Aug_CallbackAPI
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann
import time
import torchvision.transforms as torch_trans

class HeatmapGenerator():
    def __init__(self, output_res=352, num_joints=1, sigma=10):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.to_tensor = torch_trans.ToTensor()
    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(joints):
            if pt[2] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        
        return self.to_tensor(hms).permute(1, 0, 2)


def heatmap_loss(pred, gt):
    assert pred.size() == gt.size()
    loss = (pred - gt)**2
    loss = loss.mean()
    return loss

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

from models.VIS.outers.flanet.segmentation_models_pytorch import create_model

class FlaNET(nn.Module):
    def __init__(self,                 
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.model = create_model(arch='unet', 
                        encoder_name="timm-res2net50_26w_4s",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=1,)
        self.loss_weight = {'structure': 1, 'hm': 0.5}
        self.max_stride = 32
        self.heat_map_generator = HeatmapGenerator()

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        # 看成instance个数是1的instance segmentation
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        H, W = videos.shape[-2:]
        
        gts = batch_dict['targets']['masks'] # list[n t' h w]
        gts = torch.stack([haosen.squeeze(0)[1] for haosen in gts], dim=0).unsqueeze(1).float() # b h w
        # w, h = bbox[2] - bbox[0], bbox[3] - bbox[1] # b 4
        # center_x, center_y = bbox[0] + w//2, bbox[1] + h//2
        # ratio = gt.size[0] / self.trainsize, gt.size[1] / self.trainsize
        # center_x, center_y = int(center_x/ratio[0]), int(center_y/ratio[1])
        boxes = batch_dict['targets']['boxes'] # list[n t' 4]
        boxes = torch.stack([haosen.squeeze(0)[1] for haosen in boxes], dim=0) # b 4
        center_x, center_y = boxes[:, :2].unbind(-1) 
        center_x, center_y = center_x * W, center_y * H

        hm_by_batch = []
        for batch_idx in range(len(videos)):
            hm_by_batch.append(self.heat_map_generator([[center_x[batch_idx], center_y[batch_idx], 1]]))
        hm_by_batch = torch.stack(hm_by_batch, dim=0).to(self.device)
        pre_res, hm_preds  = self.model(videos) 
        loss    = structure_loss(pre_res, gts) 
        hm_loss = heatmap_loss(hm_preds[-1].squeeze(),  hm_by_batch.squeeze())
        return {'structure': loss, 'hm': hm_loss}, self.loss_weight
            
    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        H, W = videos.shape[-2:]
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]

        pre_res, _  = self.model(videos) # b 1 h w, 
    
        pred_mask = F.interpolate(pre_res, size=(H, W), mode='bilinear') > 0 # b 1 h w
        # h w
        pred_mask = pred_mask[0][0]
        # 2
        pred_class = torch.tensor([1, 0]).to(self.device)

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        pred_mask = pred_mask[:orig_h, :orig_w] # h w
        pred_masks = pred_mask.unsqueeze(0) # 1 h w
        pred_class = pred_class.unsqueeze(0) # 1 c

        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])

        return {
            'video': [orig_video.cpu()], # [t 3 h w], 1
            'pred_masks': [[None, pred_masks.cpu(), None]], # [list[1 h w], t, bool], 1
            'pred_class': [[None, pred_class.cpu(), None]], # [list[1 c], t, probability], 1
        }


@register_model
def fla_net(configs, device):
    from models.VIS.aux_mapper import AUXMapper_v1
    model = FlaNET(configs)
    model.to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=2e-4)

    foo = {
        'optim': {
            'scheduler': {'name': 'static'},
        }
    }
    scheduler = build_scheduler(configs=foo, optimizer=optimizer) # static

    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])
    
    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        {'base': 0}

