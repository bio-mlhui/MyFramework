# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import META_ARCH_REGISTRY
from models.layers.position_encoding import PositionEmbeddingSine
from models.layers.utils import _get_clones, _get_activation_fn
from models.layers.decoder_layers import SelfAttentionLayer, FFNLayer
from einops import repeat, rearrange
import copy
from einops import rearrange
from models.layers.utils import _get_clones
from models.layers.position_encoding import build_position_encoding

from models.encoder.ops.modules.deform_selective_scan_mamba_scan import Simplify_S6_ScanFeat
import torch.nn as nn
from models.layers.decoder_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.layers.anyc_trans import MLP
import torch.nn.functional as F
import torch
import copy
from models.layers.utils import zero_module, _get_clones
from models.layers.position_encoding import build_position_encoding
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting
from models.layers.utils import zero_module
from utils.misc import is_dist_avail_and_initialized
from collections import defaultdict
from detectron2.projects.point_rend.point_features import point_sample
from torch.cuda.amp import autocast
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


class Encoder_Mask_loss(nn.Module):
    def __init__(self, 
                 loss_config,
                 num_classes,) -> None:
        super().__init__()
        self.num_classes = num_classes # n=1 / n=0 / n>1
        self.matching_metrics = loss_config['matching_metrics'] # mask: mask/dice; point_sample_mask: ..
        self.losses = loss_config['losses'] 

        self.aux_layer_weights = loss_config['aux_layer_weights']  # int/list
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = loss_config['background_cls_eos']
        self.register_buffer('empty_weight', empty_weight)
        self._warmup_iters = 2000
        self.register_buffer("_iter", torch.zeros([1]))
        self.loss_key_prefix = loss_config['loss_key_prefix']

    @property
    def device(self,):
        return self.empty_weight.device
    
    def compute_loss(self, 
                     model_outs, 
                     targets,
                     video_aux_dict,
                     **kwargs):
        # list[n t' h w], batch
        if 'masks' in targets:
            num_objs = sum([haosen.flatten(1).any(-1).int().sum().item() for haosen in targets['masks']])
        else:
            raise ValueError('targets里没有boxes/masks, 需要确定数量')
        
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / comm.get_world_size(), min=1).item()
        if isinstance(self.aux_layer_weights, list):
            assert len(self.aux_layer_weights) == (len(model_outs) - 1)
        else:
            self.aux_layer_weights = [self.aux_layer_weights] * (len(model_outs) - 1)

        layer_weights = self.aux_layer_weights + [1.]

        loss_values = {
            f'{self.loss_key_prefix}_mask_dice':0., f'{self.loss_key_prefix}_mask_ce':0.,
        }
        
        if ('mask_dice_ce' in self.matching_metrics) or ('mask_dice_ce' in self.losses):
            # mask interpolate
            tgt_mask_shape = targets['masks'][0].shape[-2:] # list[n t H W], b
            for layer_idx in range(len(model_outs)):
                # b t nq h w
                batch_size, nf = model_outs[layer_idx]['pred_masks'].shape[:2]
                model_outs[layer_idx]['pred_masks'] = rearrange(F.interpolate(model_outs[layer_idx]['pred_masks'].flatten(0, 1),
                                                                size=tgt_mask_shape, mode='bilinear', align_corners=False),
                                                                    '(b t) n h w -> b t n h w',b=batch_size, t=nf)
        for taylor, layer_out in zip(layer_weights, model_outs):
            if taylor != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    loss_extra_param = self.losses[loss]
                    if loss == 'mask_dice_ce' :
                        loss_dict = self.loss_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                           loss_extra_param=loss_extra_param)
                    else:
                        raise ValueError()
                    
                    for key, value in loss_dict.items():
                        loss_values[key] = loss_values[key] + value
    
        return loss_values      

    @torch.no_grad()
    def matching(self, layer_out, targets):
        batch_size = len(targets['masks']) if 'masks' in targets else len(targets['boxes'])
        indices = [] 
        has_ann = targets['has_ann']
        for i in range(batch_size):
            C = 0.

            if 'mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i][has_ann[i]].permute(1, 0, 2, 3).contiguous()  # nq t' h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni t' H W
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) 
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
                C += self.matching_metrics['mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['mask_dice_ce']['dice'] * cost_dice

            
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def loss_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        has_ann = targets['has_ann'] # b t
        src_masks = outputs['pred_masks'].permute(0, 2, 1, 3, 4).contiguous() # b nq t h w
        tgt_masks = targets['masks'] # list[n t' h w]
        # list[nq t' h w] -> n_sigma t' h w
        src_masks = torch.cat([t[J][:, haosen] for t, (J, _), haosen in zip(src_masks, indices, has_ann)],dim=0) 
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            f"{self.loss_key_prefix}_mask_ce": ce_mask_loss(src_masks.flatten(0, 1).flatten(1), tgt_masks.flatten(0, 1).flatten(1), num_boxes=num_objs),
            f"{self.loss_key_prefix}_mask_dice": dice_loss(src_masks.flatten(0, 1).flatten(1), tgt_masks.flatten(0, 1).flatten(1), num_boxes=num_objs),
        }
        return losses   

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class QueryScanEncoder(nn.Module):
    def __init__(self, 
                 d_model=256, 
                 num_encoder_layers=6, 
                 dim_feedforward=1024, 
                 dropout=0.1,
                 activation="relu",
                 num_feature_levels=4,
                 qs_configs=None,
                ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_encoder_layers
        n_scan_queries = qs_configs['n_scan_queries']
        self.scan_queries = nn.Embedding(n_scan_queries, embedding_dim=d_model,) 
        self.scan_poses = nn.Embedding(n_scan_queries, embedding_dim=d_model,)
        encoder_layer = QueryScanEncoderLayer(d_model = d_model,
                                              d_ffn=dim_feedforward,
                                              dropout=dropout,
                                              activation=activation,
                                              n_levels=num_feature_levels,
                                              qs_configs= qs_configs)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        normal_(self.level_embed)

    def forward(self, 
                srcs=None,  # list[b c t h w]
                pos_embeds=None, # list[b c t h w]
                video_aux_dict=None, # 
                **kwargs):
        scan_queries = repeat(self.scan_queries.weight, 'nq c -> b nq c', b=srcs[0].shape[0])
        scan_poses = repeat(self.scan_poses.weight, 'nq c -> b nq c', b=srcs[0].shape[0])
        pos_embeds = [haosen + self.level_embed[lvl_idx].view(1,-1,1,1,1)  for lvl_idx, haosen in enumerate(pos_embeds)]
        all_masks = []
        for _, layer in enumerate(self.layers):
            srcs = [haosen + haosen_pos for haosen, haosen_pos in zip(srcs, pos_embeds)]
            scan_queries = scan_queries + scan_poses
            srcs, scan_queries, query_masks = layer(srcs=srcs, 
                                       scan_queries = scan_queries,
                                       video_aux_dict=video_aux_dict)
            all_masks.append(query_masks)
        return srcs, all_masks


class QueryScanEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, 
                 qs_configs=None):
        super().__init__()
        # self attention
        self.d_model = d_model
        self.scan_topk = qs_configs['scan_topk']
        self.topk_sorted = qs_configs['topk_sorted']

        # value conv2d
        self.value_d_conv = qs_configs['value_d_conv']
        assert self.value_d_conv > 1
        self.value_proj = nn.Conv3d(
            in_channels=self.d_model,
            out_channels=self.d_model, # head * head_dim
            bias=True,
            kernel_size=self.value_d_conv,
            groups=self.d_model,
            padding=(self.value_d_conv - 1) // 2,
        )
        # query * 
        # topk 
        # scan
        # feat_self
        self.feat_scan = Simplify_S6_ScanFeat(self.d_model,
                                              d_state=qs_configs['d_state'],
                                              dt_rank=qs_configs['dt_rank'],
                                              d_conv=qs_configs['d_conv'],
                                              d_conv_bias=qs_configs['d_conv_bias'],)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feat_ffn = FFNLayer(d_model=d_model,
                                  dim_feedforward=qs_configs['attn']['dim_feedforward'],
                                dropout=qs_configs['attn']['dropout'],
                                activation=qs_configs['attn']['activation'],
                                normalize_before=False)
        
    
        self.query_self = SelfAttentionLayer(d_model=d_model,
                                             nhead=qs_configs['attn']['nheads'],
                                             dropout=qs_configs['attn']['dropout'],
                                             activation=qs_configs['attn']['activation'],
                                             normalize_before=False)
        self.query_ffn = FFNLayer(d_model=d_model,
                                  dim_feedforward=qs_configs['attn']['dim_feedforward'],
                                dropout=qs_configs['attn']['dropout'],
                                activation=qs_configs['attn']['activation'],
                                normalize_before=False)
        self.query_norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    # 每一帧的前topk个token; 
    def forward(self, 
                srcs=None,  # list[b c t h w]
                scan_queries =None, # b nq c
                video_aux_dict=None):
        ori_src_flatten = torch.cat([rearrange(haosen, 'b c t h w -> (b t) (h w) c') for haosen in srcs], dim=1) # b thw_sigma c
        batch_size, dim, nf, *_ = srcs[0].shape  
        Nq = scan_queries.shape[1]      
        # list[(h w)]
        scale_shapes = [haosen.shape[-2:] for haosen in srcs] 
        # list[int]
        topk_each_frame_by_scale = [int(math.floor(self.scan_topk * haosen[0] * haosen[1])) for haosen in scale_shapes]
        # list[b n t h w] logits
        query_masks = [torch.einsum('bnc,bcthw->bnthw', 
                                    self.query_norm(scan_queries), haosen) for haosen in srcs]
        
        # list[b n t k] -> blis[b n t k c]
        if self.topk_sorted:
            scan_idxs = [torch.topk(haosen.flatten(3), k=topk_num, dim=-1, sorted=True)[1].flip(-1).unsqueeze(-1).repeat(1, 1, 1, 1, dim) \
                     for haosen, topk_num in zip(query_masks, topk_each_frame_by_scale)]
        else:
            scan_idxs = [torch.topk(haosen.flatten(3), k=topk_num, dim=-1, sorted=False)[1].unsqueeze(-1).repeat(1, 1, 1, 1, dim) \
                     for haosen, topk_num in zip(query_masks, topk_each_frame_by_scale)]
        # b c t h w 
        srcs = [self.value_proj(haosen) for haosen in srcs]
        # b c t h w -> b c t hw -> b n c t hw -> b n t hw c
        scan_src = [repeat(haosen, 'b c t h w -> b n t (h w) c ',n=Nq) for haosen in srcs]
        # b n t hw c; b n t k c -> b n t k_l c
        scan_feats = [torch.gather(haosen_feat, dim=3, index=haosen_idx) for haosen_idx, haosen_feat in zip(scan_idxs, scan_src)]
        # b n t k_l+1 c
        scan_feats = [torch.cat([haosen, repeat(scan_queries, 'b nq c -> b nq t 1 c', t=nf)], dim=3) for haosen in scan_feats]

        # b n t L(k_l+1) c
        scan_feats = torch.cat(scan_feats, dim=3)
        scale_token_sum = scan_feats.shape[3]
        # b n t L(k_l+1)+1 c
        scan_feats = torch.cat([scan_feats, repeat(scan_queries, 'b nq c -> b nq t 1 c', t=nf)], dim=3)
        scan_feats =  rearrange(scan_feats, 'b nq t L c -> (b nq) c (t L)') # b*n c t*(L(k_l+1)+1)
        # b*n c t*(L(k_l+1)+1) -> b nq c nf k_l_sum+1
        scan_feats = self.feat_scan(scan_feats).view(batch_size, Nq, -1, nf, scale_token_sum+1).contiguous()
        temporal_summarizes = scan_feats[..., -1].contiguous() # b nq c nf
        # b nq c nf L(k_l+1)
        scan_feats = scan_feats[..., :-1].contiguous() # b nq c t L(k_l+1)

        scale_splits = [val for pair in zip(topk_each_frame_by_scale, [1] * len(topk_each_frame_by_scale)) for val in pair]
        assert sum(scale_splits) == scan_feats.shape[-1]
        scan_feats = scan_feats.split(scale_splits, -1) # list[b nq c t k_l/1], # scale 1 scale 1 scale 1
        multiscale_summarizes = torch.cat(scan_feats[1::2], dim=-1) # list[b nq c t 1] 3 -> b nq c t 3
        scale_feats = scan_feats[0::2] # list[b nq c t k_l]

        # b nq c 
        scan_queries = scan_queries + temporal_summarizes.mean(-1) + multiscale_summarizes.mean((-2, -1))

        # list[b c t h w]
        summed_scale_feats = []
        # list[b n t k c], list[b nq c t k_l]
        for l_idx, (query_idx, query_scan) in enumerate(zip(scan_idxs, scale_feats)):
            H, W = scale_shapes[l_idx]
            sum_feats = query_scan.new_zeros([batch_size, Nq, dim, nf, H*W]) 
            query_idx = rearrange(query_idx, 'b n t k c -> b n c t k')
            assert query_scan.shape == query_idx.shape
            sum_feats.scatter_add_(dim=-1, index=query_idx, src=query_scan)
            sum_feats = sum_feats.sum(1) # b c t hw
            sum_feats = rearrange(sum_feats, 'b c t (h w) -> b c t h w',h=H, w=W)

            summed_scale_feats.append(sum_feats)
        
        # list[b c t h w]
        srcs = [srcs[l_idx] + summed_scale_feats[l_idx] for l_idx in range(len(srcs))] # list[b c t h w]
        src_flatten = torch.cat([rearrange(haosen, 'b c t h w -> (b t) (h w) c') for haosen in srcs], dim=1) # bt hw_sigma c
        src_flatten = ori_src_flatten + self.dropout1(src_flatten)
        src_flatten = self.norm1(src_flatten)
        src_flatten = self.feat_ffn(src_flatten)

        srcs = src_flatten.split([haosen[0] * haosen[1] for haosen in scale_shapes], dim=1)
        srcs = [rearrange(haosen, '(b t) (h w) c -> b c t h w',b=batch_size, t=nf,h=haosen_shape[0],w=haosen_shape[1])\
                 for haosen, haosen_shape in zip(srcs, scale_shapes)]

        scan_queries = self.query_self(scan_queries)
        scan_queries = self.query_ffn(scan_queries)
        return srcs, scan_queries, query_masks


@META_ARCH_REGISTRY.register()
class QueryScan_MultiscaleEncoder(nn.Module):
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
    ):
        super().__init__()
        d_model = configs['d_model']
        self.d_model = d_model
        fpn_norm = configs['fpn_norm'] # fpn的norm
        nlayers = configs['nlayers']
        
        # 4, 8, 16, 32
        self.multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))
        self.encoded_scales = sorted(configs['encoded_scales'], 
                                     key=lambda x:self.multiscale_shapes[x].spatial_stride) # res3, res4, res5
        
        # 4 -> 8 -> 16 -> 32    
        self.scale_dims = [val.dim for val in multiscale_shapes.values()]
        self.video_projs = META_ARCH_REGISTRY.get(configs['video_projs']['name'])(configs=configs['video_projs'],
                                                                            multiscale_shapes=multiscale_shapes, out_dim=d_model)

        self.pos_3d = build_position_encoding(position_embedding_name='3d', hidden_dim=d_model)

        layer_configs = configs['layer_configs']
        self.transformer = QueryScanEncoder(
            d_model=d_model,
            dropout=layer_configs['dropout'],
            dim_feedforward=layer_configs['dim_ff'],
            activation=layer_configs['activation'],
            num_encoder_layers=nlayers,
            num_feature_levels=len(self.encoded_scales),
            qs_configs=layer_configs['qs_configs'],
        )

        min_encode_stride = self.multiscale_shapes[self.encoded_scales[0]].spatial_stride # 8
        min_stride = list(self.multiscale_shapes.values())[0].spatial_stride # 4
        self.num_fpn_levels = int(np.log2(min_encode_stride) - np.log2(min_stride))
        lateral_convs = [] 
        output_convs = []
        use_bias = fpn_norm == ""
        for idx, in_channels in enumerate(self.scale_dims[:self.num_fpn_levels]):
            lateral_norm = get_norm(fpn_norm, d_model)
            output_norm = get_norm(fpn_norm, d_model)

            lateral_conv = Conv2d(in_channels, d_model, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)

            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1] # 8 4
        self.output_convs = output_convs[::-1] # 8 4
        
        self.loss_module = Encoder_Mask_loss(configs['loss'], num_classes=1)

    def forward(self, 
                multiscales=None, # b c t h w
                video_aux_dict=None, # dict{}
                **kwargs):
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        video_aux_dict['nf'] = nf
        multiscales = self.video_projs(multiscales) 
        assert set(list(multiscales.keys())).issubset(set(list(self.multiscale_shapes.keys())))
        assert set(list(self.multiscale_shapes.keys())).issubset(set(list(multiscales.keys())))

        srcs = []
        poses = [] # 32, 16, 8
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            x = multiscales[scale_name] # b c t h w
            srcs.append(x)
            # b t c h w -> b c t h w
            poses.append(self.pos_3d(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4))

        # list[b c t h w], list[list[b n t h w, scale] layer
        memory_features, query_masks = self.transformer(srcs=srcs, 
                                            pos_embeds=poses,
                                            video_aux_dict=video_aux_dict)

        for idx, f in enumerate(list(self.multiscale_shapes.keys())[:self.num_fpn_levels][::-1]):
            x = multiscales[f].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous() # bt c h w
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(memory_features[-1].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous(), size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[idx](y)
            memory_features.append(rearrange(y, '(b t) c h w -> b c t h w',b=batch_size, t=nf))

        assert len(memory_features) == len(list(self.multiscale_shapes.keys()))

        ret = {}
        for key, out_feat in zip(list(self.multiscale_shapes.keys()), memory_features[::-1]):
            ret[key] = out_feat
            
        return ret, query_masks

    def compute_loss(self, query_masks, targets, **kwargs):
        assert self.training
        # list[list[b n t h w], scale], L
        num_scales = len(query_masks[0])
        pred_by_scale = [] # list[list[b t n h w], L], scale
        for scale_idx in range(num_scales):
            pred_by_scale.append([{'pred_masks': haosen[scale_idx].permute(0, 2, 1, 3, 4).contiguous()} for haosen in query_masks])
        
        loss_dict_by_scales = [self.loss_module.compute_loss(pred_by_scale[scale_idx], targets, video_aux_dict=None) for scale_idx in range(num_scales)]
        
        loss_keys = list(loss_dict_by_scales[0].keys()) # 
        
        loss_dict = {}
        for loss_key in loss_keys:
            loss_value = 0.
            for scale_idx in range(num_scales):
                loss_value += loss_dict_by_scales[scale_idx][loss_key]
            loss_dict[loss_key] = loss_value
        return loss_dict

class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out