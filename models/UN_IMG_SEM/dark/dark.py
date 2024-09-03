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

from .layers import OptimizeModel, CrossAttentionLayer, SelfAttentionLayer, FFNLayer, PositionEmbeddingSine


# 使用alignseg的evaluate方式
# 但是当queries >= num_classes并且 有classifier的时候就可以用stegeo的评估方式
class DarkModel(OptimizeModel):
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
        
        decoder_configs = configs['decoder']
        num_queries = decoder_configs['num_queries']
        nheads = decoder_configs['nheads']
        ffn_factor = decoder_configs['ffn_factor']
        dec_layers = decoder_configs['dec_layers']
        pre_norm = decoder_configs['pre_norm']

        hidden_dim = self.backbone.embed_dim

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=ffn_factor * hidden_dim,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes) # 假设unsupervised没有背景或者背景占比非常小
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.num_feature_levels = 1

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

    def model_outputs(self, x, num_crops, last_self_attention=True):
        # list[b c h w], 

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(x[i].flatten(2))

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


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
def dark_model(configs, device):
    aux_mapper = AUXMapper()
    from data_schedule import build_singleProcess_schedule
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))
    model = AlignSeg(configs)
        
    model.to(device)
    model.optimize_setup(configs)
    
    logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
    logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model, train_loader, eval_function

