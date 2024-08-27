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

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def exclude_from_wt_decay(named_params, weight_decay: float, lr: float):
    params = []
    excluded_params = []
    query_param = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        # do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            excluded_params.append(param)
        elif 'clsQueries' in name:
            query_param.append(param)
        else:
            params.append(param)
    return [{'params': params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': excluded_params, 'weight_decay': 0., 'lr': lr},
            {'params': query_param, 'weight_decay': 0., 'lr': lr * 1}]

def configure_optimizers(model, train_config):
    # Separate Decoder params from ViT params
    # only train Decoder
    decoder_params_named = []
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            param.requires_grad = False
        elif train_config['fix_prototypes'] and 'clsQueries' in name:
            param.requires_grad = False
        else:
            decoder_params_named.append((name, param))

    # Prepare param groups. Exclude norm and bias from weight decay if flag set.
    if train_config['exclude_norm_bias']:
        params = exclude_from_wt_decay(decoder_params_named,
                                       weight_decay=train_config["weight_decay"],
                                       lr=train_config['lr_decoder'])
    else:
        decoder_params = [param for _, param in decoder_params_named]
        params = [{'params': decoder_params, 'lr': train_config['lr_decoder']}]

    # Init optimizer and lr schedule
    optimizer = torch.optim.AdamW(params, weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    return optimizer, scheduler

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

# 使用alignseg的evaluate方式
# 但是当queries >= num_classes并且 有classifier的时候就可以用stegeo的评估方式
class AlignSeg(OptimizeModel):
    def __init__(
        self,
        configs,
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
        self.ffn_factor = model_configs['decoder']['ffn_factor']
        self.num_decode_layers = model_configs['decoder']['num_layers']
        self.num_queries = model_configs['decoder']['num_queries']
        self.num_heads = model_configs['decoder']['num_heads']
        hidden_dim = self.embed_dim * self.ffn_factor
        self.clsQueries = nn.Embedding(self.num_queries, self.embed_dim)

        # simple Transformer Decoder with num_decoder_layers
        self.decoder_cross_attention_layers = nn.ModuleList()
        self.decoder_self_attention_layers = nn.ModuleList()
        self.decoder_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decode_layers):
            self.decoder_cross_attention_layers.append(
                CrossAttentionLayer(d_model=self.embed_dim, nhead=self.num_heads)
            )
            self.decoder_self_attention_layers.append(
                SelfAttentionLayer(d_model=self.embed_dim, nhead=self.num_heads)
            )
            self.decoder_ffn_layers.append(
                FFNLayer(d_model=self.embed_dim, dim_feedforward=hidden_dim)
            )
        from .criterion import AlignCriterion
        self.criterion = AlignCriterion(patch_size=self.patch_size,
                                num_queries=self.num_queries,
                                nmb_crops=model_configs['loss']['nmb_crops'],
                                roi_align_kernel_size=model_configs['loss']['roi_align_kernel_size'],
                                ce_temperature=model_configs['loss']['ce_temperature'],
                                negative_pressure=model_configs['loss']['negative_pressure'],
                                last_self_attention=model_configs['loss']['last_self_attention'])

    def train(self, mode):
        super().train(mode)
        self.backbone.eval()

    @property
    def device(self):
        return self.pixel_mean.device
        
    def optimize_setup(self, configs):
        optim_config = configs['optim']
        decoder_params_named = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if optim_config['fix_prototypes'] and 'clsQueries' in name:
                    param.requires_grad = False
                else:
                    decoder_params_named.append((name, param))

        # Prepare param groups. Exclude norm and bias from weight decay if flag set.
        if optim_config['exclude_norm_bias']:
            params = exclude_from_wt_decay(decoder_params_named,
                                        weight_decay=optim_config["weight_decay"],
                                        lr=optim_config['lr_decoder'])
        else:
            decoder_params = [param for _, param in decoder_params_named]
            params = [{'params': decoder_params, 'lr': optim_config['lr_decoder']}]

        # Init optimizer and lr schedule
        self.optimizer = torch.optim.AdamW(params, weight_decay=optim_config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)


    def model_outputs(self, inputs):
        if self.training:
            num_crops = [1,2]
            last_self_attention = True
            inputs = list(inputs.unbind(1)) # b 3 3 h w -> list[b 3 h w]
        else:
            num_crops = [1, 0]
            last_self_attention = False
            inputs = [inputs]
        
        B = inputs[0].size(0)

        # b N c
        outQueries = self.clsQueries.weight.unsqueeze(0).repeat(B, 1, 1)
        posQueries = pos = None

        # Extract feature
        with torch.no_grad():
            outputs = self.backbone.get_alignseg_feats(inputs=inputs, nmb_crops=num_crops, last_self_attention=last_self_attention)
            
        if last_self_attention:
            outputs, attentions = outputs  # outputs=[B*N(196+36), embed_dim], attentions(only global)=[B, heads, 196]


        # calculate gc and lc resolutions. Split output in gc and lc embeddings
        gc_res_w = inputs[0].size(2) / self.patch_size
        gc_res_h = inputs[0].size(3) / self.patch_size
        assert gc_res_w.is_integer() and gc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        assert gc_res_w == gc_res_h, f"Only supporting square images not {inputs[0].size(2)}x{inputs[0].size(3)}"
        gc_spatial_res = int(gc_res_w)
        lc_res_w = inputs[-1].size(2) / self.patch_size
        assert lc_res_w.is_integer(), "Image dims need to be divisible by patch size"
        lc_spatial_res = int(lc_res_w)
        gc_spatial_output, lc_spatial_output = outputs[:B * num_crops[0] * gc_spatial_res ** 2], \
            outputs[B * num_crops[0] * gc_spatial_res ** 2:]  # bhw c, 2bhw c
        # (B*N, C) -> (B, N, C)
        gc_spatial_output = gc_spatial_output.reshape(B, -1, self.embed_dim) # b hw c
        if num_crops[-1] != 0:
            lc_spatial_output = lc_spatial_output.reshape(B, num_crops[-1], lc_spatial_res**2, self.embed_dim) # B 2 HW C

        # merge attention heads and threshold attentions
        attn_hard = None
        if last_self_attention:
            attn_smooth = sum(attentions[:, i] * 1 / attentions.size(1) for i in range(attentions.size(1)))
            attn_smooth = attn_smooth.reshape(B * sum(num_crops), 1, gc_spatial_res, gc_spatial_res) # 3b 1 h w
            # attn_hard is later served as 'foreground' hint, use attn_hard.bool()
            attn_hard = process_attentions(attn_smooth, gc_spatial_res, threshold=0.6, blur_sigma=0.6)
            attn_hard = attn_hard.squeeze(1)

        # Align Queries to each image crop's features with decoder, assuming only 1 global crop
        all_queries = []
        for i in range(sum(num_crops)):
            if i == 0:
                features = gc_spatial_output
            else:
                features = lc_spatial_output[:, i-1]
            for j in range(self.num_decode_layers):
                # attention: cross-attention first
                queries = self.decoder_cross_attention_layers[j](
                    outQueries, features, pos=pos, query_pos=posQueries)
                # self-attention
                queries = self.decoder_self_attention_layers[j](
                    queries, query_pos=posQueries)
                # FFN
                queries = self.decoder_ffn_layers[j](queries)

            all_queries.append(queries)

        return all_queries, gc_spatial_output, lc_spatial_output, attn_hard, gc_spatial_res, lc_spatial_res


    def forward_backward(self, batch_dict):
        assert self.training
        self.backbone.eval()
        # b 3 h w
        images = batch_dict['images'].to(self.device) 
        images = images - self.pixel_mean / self.pixel_std
        
        results = self.model_outputs(images)

        self.optimizer.zero_grad()
        loss, loss_dict = self.criterion(results)
        loss.backward()
        self.optimizer.step()
        

        loss_dict.update({'loss': loss.cpu().item()})
        return loss_dict


    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training

        images = batch_dict['images'].to(self.device, non_blocking=True) # b 3 h w
        H, W = images.shape[-2:]
        images = images - self.pixel_mean / self.pixel_std
        all_queries, tokens, _, _, res, _ = self.model_outputs(images)
        pred_masks = torch.einsum("bnc,bqc->bnq", F.normalize(tokens, dim=-1, eps=1e-10), 
                                  F.normalize(all_queries[0], dim=-1, eps=1e-10))
        pred_masks = rearrange(pred_masks, 'b (h w) nq -> b nq h w', h=H//self.patch_size, w=W//self.patch_size)
        pred_masks = F.interpolate(pred_masks, size=(H,W), mode='bilinear', align_corners=False) # b nq h w
        return {
            'pred_masks': pred_masks,
        }

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict=state_dict['scheduler'])


    def get_lr_group_dicts(self, ):
        return  {f'lr_decoder': self.optimizer.param_groups[0]["lr"]}


@register_model
def alignseg(configs, device):
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    model = AlignSeg(configs)
        
    model.to(device)
    model.optimize_setup(configs)
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

