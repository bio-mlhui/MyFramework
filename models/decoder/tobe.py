
# multi-scale features, b c h w -> module -> obj queries, predictions, b nq c
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
from .mask2former import Image_SetMatchingLoss

# def scores_loss(self, layer_gscore_output, matching_indices,  targets):
#     pass
#     #     is_valid = targets['isvalid'] # list[ni], batch
#     #     referent_idx = targets['gt_referent_idx'] # list[int], batch
#     #     ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
#     #     num_refs = (ref_is_valid.int().sum())
#     #     match_as_gt_indices = [] # list[int], bt
#     #     for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
#     #         sel_idx = src_idx.tolist().index(ref_idx)
#     #         match_as_gt_idx = tgt_idx[sel_idx]
#     #         match_as_gt_indices.append(match_as_gt_idx.item())
#     #     match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
#     #     choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
#     #     return {'objdecoder_reason': choose_loss.sum() / num_refs}

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x) # B C hw
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Video_SetMatchingLoss(nn.Module):
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
        # self.register_buffer('small_obj_weight', torch.tensor(loss_config['small_obj_weight']).float())

    @property
    def device(self,):
        return self.empty_weight.device
    
    def compute_loss(self, model_outs, targets):
        if 'masks' in targets:
            num_objs = torch.stack(targets['masks'], dim=0).flatten(1).any(-1).int().sum() 
        elif 'boxes' in targets:
            gt_boxes = torch.stack(targets['boxes'], dim=0) # bn 4
            num_objs = (gt_boxes[:, 2:] > 0).all(-1).int().sum()
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
            'mask_dice':0., 'mask_ce':0.,
            'box_l1': 0., 'box_giou': 0.,
            'class_ce':0.,

            'mask_dice_smobj':0., 'mask_ce_smobj':0.,
            'boxMask_dice':0., 'boxMask_ce':0.,
        }
        
        if ('mask_ce_dice' in self.matching_metrics) or ('mask_ce_dice' in self.losses):
            # mask interpolate
            tgt_mask_shape = targets['masks'][0].shape[-2:] # list[n H W], b
            for layer_idx in range(len(model_outs)):
                # b nq h w
                model_outs[layer_idx]['pred_masks'] = F.interpolate(model_outs[layer_idx]['pred_masks'],
                                                                    size=tgt_mask_shape, mode='bilinear', align_corners=False)
        for taylor, layer_out in zip(layer_weights, model_outs):
            if taylor != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    loss_extra_param = self.losses[loss]
                    if loss == 'mask_dice_ce' :
                        loss_dict = self.loss_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                           loss_extra_param=loss_extra_param)
                    elif loss == 'boxMask_dice_ce':
                        pass
                    elif loss == 'box_l1_giou':
                        loss_dict = self.loss_box_l1_giou(layer_out, targets, matching_indices, num_objs,
                                                          loss_extra_param=loss_extra_param)
                    elif loss == 'class_ce':
                        loss_dict = self.loss_class_ce(layer_out, targets, matching_indices, num_objs,
                                                       loss_extra_param=loss_extra_param)
                    elif loss == 'point_mask_dice_ce':
                        loss_dict = self.loss_point_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
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
        for i in range(batch_size):
            C = 0.

            if 'class_prob' in self.matching_metrics:
                out_cls = layer_out['pred_class'][i].softmax(-1) # nq c
                tgt_cls = targets['classes'][i] # n
                cost_class = - out_cls[:, tgt_cls] # nq n
                C += self.matching_metrics['class_prob']['prob'] * cost_class

            if 'mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) 
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
                C += self.matching_metrics['mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['mask_dice_ce']['dice'] * cost_dice

            if 'box_l1_giou' in self.matching_metrics:
                out_box = layer_out['pred_boxes'][i].sigmoid() # nq 4
                tgt_bbox = targets['boxes'][i] # ni 4 
                cost_l1 = torch.cdist(out_box, tgt_bbox, p=1) 
                cost_giou = 1 - box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_box),
                                                        box_ops.box_cxcywh_to_xyxy(tgt_bbox))
                C += self.matching_metrics['box_l1_giou']['l1'] * cost_l1 + \
                      self.matching_metrics['box_l1_giou']['giou'] + cost_giou
 
            if 'point_mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.matching_metrics['point_mask_dice_ce']['num_points'],
                                           2, device=self.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float() # nq num_points
                    tgt_mask = tgt_mask.float()
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                C += self.matching_metrics['point_mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['point_mask_dice_ce']['dice'] * cost_dice
            
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def loss_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "mask_ce": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
            "mask_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
        }
        return losses   

    def loss_point_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = tgt_masks.to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = tgt_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                loss_extra_param['num_points'],
                loss_extra_param['oversample_ratio'],
                loss_extra_param['importance_sample_ratio'],
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "mask_dice": ce_mask_loss(point_logits, point_labels, num_objs),
            "mask_ce": dice_loss(point_logits, point_labels, num_objs),
        }

        del src_masks
        del target_masks
        return losses        

    def loss_class_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_class"].float() # b nq c

        idx = self._get_src_permutation_idx(indices)
        # list[n], b -> bn
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['classes'], indices)]) 
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"class_ce": loss_ce}
        return losses

    def loss_box_l1_giou(self, outputs, targets, indices, num_objs, loss_extra_param): 
        tgt_boxes = targets['boxes'] # list[n 4], b

        src_boxes = outputs['pred_boxes'].sigmoid() # b nq 4

        src_boxes = torch.cat([t[J] for t, (J, _) in zip(src_boxes, indices)], dim=0) # n_sum 4
        tgt_boxes = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes, indices)], dim=0) # n_sum 4
        
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='none') # n_sum 4

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
        return {
            'box_l1': loss_l1.sum(-1).sum() / num_objs,
            'box_giou': loss_giou.sum() / num_objs
        }

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





# t nqf -> cros -> ss2d -> cross
# 只有video-level的loss
class Video_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.frame_nqueries = configs['frame_nqueries'] # 20
        self.video_nqueries = configs['video_nqueries'] # 10
        self.nlayers = configs['nlayers'] 
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']
        self.mask_spatial_stride = multiscale_shapes[self.mask_scale].spatial_stride
        num_classes = configs['num_classes']

        inputs_projs = configs['inputs_projs']
        self.inputs_projs = nn.Sequential()
        if inputs_projs is not None:
            self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                             multiscale_shapes=multiscale_shapes,
                                                                             out_dim=d_model)
        self.frame_query_poses =  nn.Embedding(self.frame_nqueries, d_model)
        self.frame_query_feats = nn.Embedding(self.frame_nqueries, d_model)
        self.temporal_query_poses = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_feats = nn.Embedding(self.video_nqueries, d_model)

        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)
        assert self.nlayers % len(self.memory_scales) == 0

        self.frame_cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                              self.nlayers)
        
        temporal_self_layer = META_ARCH_REGISTRY.get(configs['temporal_self_layer']['name'])(configs['temporal_self_layer'])
        self.temporal_self_layers = _get_clones(temporal_self_layer, self.nlayers)

        temporal_cross_layer = META_ARCH_REGISTRY.get(configs['temporal_cross_layer']['name'])(configs['temporal_cross_layer'])
        self.temporal_cross_layers = _get_clones(temporal_cross_layer, self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.frame_query_norm = nn.LayerNorm(d_model)
        self.temporal_query_norm = nn.LayerNorm(d_model)
        self.pos_3d = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d') # b t c h w
        
        self.head_outputs = configs['head_outputs']
        assert 'mask' in self.head_outputs
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.head_outputs:
            self.query_class = nn.Linear(d_model, num_classes + 1)    

        self.vid_loss_module = Video_SetMatchingLoss(loss_config=configs['temporal_loss'], num_classes=num_classes)

    @property
    def device(self,):
        return self.frame_query_poses.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c t h w 
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_3d(mem.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) for mem in memories]
        memories = [rearrange(mem, 'b c t h w -> (h w) (b t) c').contiguous() for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c t h w -> (h w) (b t) c').contiguous() for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c t h w

        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales, # b c t h w
                ):
        multiscales = self.inputs_projs(multiscales)
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]

        # hw bt c; bt c h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(multiscales)
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]

        # nqf bt c
        frame_query_poses = self.frame_query_poses.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)
        frame_query_feats = self.frame_query_feats.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)

        # nq b c
        temporal_query_poses = self.temporal_query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporal_query_feats = self.temporal_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)


        frame_ret = []
        vid_ret = []
        # bt nqf class, bt nqf 4, bt nqf h w, bt*head nqf hw
        # b nq class, b nq h w
        frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
            self.forward_heads(frame_query_feats=frame_query_feats, 
                               temporal_query_feats=temporal_query_feats,
                               mask_features=mask_features, attn_mask_target_size=size_list[0]) # first sight you re not human
        frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
        vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})

        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            frame_cross_attn_mask[torch.where(frame_cross_attn_mask.sum(-1) == frame_cross_attn_mask.shape[-1])] = False 

            frame_query_feats = self.frame_cross_layers[i](
                tgt=frame_query_feats, # nqf bt c
                memory=memories[level_index], # hw bt c
                memory_mask=frame_cross_attn_mask, 
                memory_key_padding_mask=None,
                pos=memories_poses[level_index], 
                query_pos=frame_query_poses,
            )
            frame_query_feats = self.temporal_self_layers[i](
                frame_query_feats=frame_query_feats, 
                frame_query_poses=frame_query_poses,
            )

            temporal_query_feats = self.temporal_cross_layers[i](
                temporal_query_feats=temporal_query_feats, 
                temporal_query_poses=temporal_query_poses,

                frame_query_feats=frame_query_feats,
                frame_query_poses=frame_query_poses,
            )

            frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
                self.forward_heads(frame_query_feats=frame_query_feats, 
                                    temporal_query_feats=temporal_query_feats,
                                    mask_features=mask_features, attn_mask_target_size=size_list[0])
            frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
            vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})
    
        if self.training:
            return frame_ret, vid_ret
        else:
            return vid_ret

    def forward_heads(self, frame_query_feats, temporal_query_feats,  mask_features, attn_mask_target_size):
        frame_query_feats = self.frame_query_norm(frame_query_feats) # nqf bt c
        frame_query_feats = frame_query_feats.transpose(0, 1).contiguous() # bt nqf c
        frame_mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features.permute(0, 2,1,3,4).flatten(0,1))  #bt nq h w

        # b nq h w
        attn_mask = frame_mask_logits.detach().clone() 
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # b*head nq hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).
# multi-scale features, b c h w -> module -> obj queries, predictions, b nq c
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
from .mask2former import Image_SetMatchingLoss

# def scores_loss(self, layer_gscore_output, matching_indices,  targets):
#     pass
#     #     is_valid = targets['isvalid'] # list[ni], batch
#     #     referent_idx = targets['gt_referent_idx'] # list[int], batch
#     #     ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
#     #     num_refs = (ref_is_valid.int().sum())
#     #     match_as_gt_indices = [] # list[int], bt
#     #     for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
#     #         sel_idx = src_idx.tolist().index(ref_idx)
#     #         match_as_gt_idx = tgt_idx[sel_idx]
#     #         match_as_gt_indices.append(match_as_gt_idx.item())
#     #     match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
#     #     choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
#     #     return {'objdecoder_reason': choose_loss.sum() / num_refs}

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x) # B C hw
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Video_SetMatchingLoss(nn.Module):
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
        # self.register_buffer('small_obj_weight', torch.tensor(loss_config['small_obj_weight']).float())

    @property
    def device(self,):
        return self.empty_weight.device
    
    def compute_loss(self, model_outs, targets):
        if 'masks' in targets:
            num_objs = torch.stack(targets['masks'], dim=0).flatten(1).any(-1).int().sum() 
        elif 'boxes' in targets:
            gt_boxes = torch.stack(targets['boxes'], dim=0) # bn 4
            num_objs = (gt_boxes[:, 2:] > 0).all(-1).int().sum()
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
            'mask_dice':0., 'mask_ce':0.,
            'box_l1': 0., 'box_giou': 0.,
            'class_ce':0.,

            'mask_dice_smobj':0., 'mask_ce_smobj':0.,
            'boxMask_dice':0., 'boxMask_ce':0.,
        }
        
        if ('mask_ce_dice' in self.matching_metrics) or ('mask_ce_dice' in self.losses):
            # mask interpolate
            tgt_mask_shape = targets['masks'][0].shape[-2:] # list[n H W], b
            for layer_idx in range(len(model_outs)):
                # b nq h w
                model_outs[layer_idx]['pred_masks'] = F.interpolate(model_outs[layer_idx]['pred_masks'],
                                                                    size=tgt_mask_shape, mode='bilinear', align_corners=False)
        for taylor, layer_out in zip(layer_weights, model_outs):
            if taylor != 0:
                matching_indices = self.matching(layer_out, targets)
                for loss in self.losses:
                    loss_extra_param = self.losses[loss]
                    if loss == 'mask_dice_ce' :
                        loss_dict = self.loss_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
                                                           loss_extra_param=loss_extra_param)
                    elif loss == 'boxMask_dice_ce':
                        pass
                    elif loss == 'box_l1_giou':
                        loss_dict = self.loss_box_l1_giou(layer_out, targets, matching_indices, num_objs,
                                                          loss_extra_param=loss_extra_param)
                    elif loss == 'class_ce':
                        loss_dict = self.loss_class_ce(layer_out, targets, matching_indices, num_objs,
                                                       loss_extra_param=loss_extra_param)
                    elif loss == 'point_mask_dice_ce':
                        loss_dict = self.loss_point_mask_dice_ce(layer_out, targets, matching_indices, num_objs,
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
        for i in range(batch_size):
            C = 0.

            if 'class_prob' in self.matching_metrics:
                out_cls = layer_out['pred_class'][i].softmax(-1) # nq c
                tgt_cls = targets['classes'][i] # n
                cost_class = - out_cls[:, tgt_cls] # nq n
                C += self.matching_metrics['class_prob']['prob'] * cost_class

            if 'mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W
                cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) 
                cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))
                C += self.matching_metrics['mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['mask_dice_ce']['dice'] * cost_dice

            if 'box_l1_giou' in self.matching_metrics:
                out_box = layer_out['pred_boxes'][i].sigmoid() # nq 4
                tgt_bbox = targets['boxes'][i] # ni 4 
                cost_l1 = torch.cdist(out_box, tgt_bbox, p=1) 
                cost_giou = 1 - box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_box),
                                                        box_ops.box_cxcywh_to_xyxy(tgt_bbox))
                C += self.matching_metrics['box_l1_giou']['l1'] * cost_l1 + \
                      self.matching_metrics['box_l1_giou']['giou'] + cost_giou
 
            if 'point_mask_dice_ce' in self.matching_metrics:
                out_mask = layer_out['pred_masks'][i]  # nq h w
                tgt_mask = targets['masks'][i].to(out_mask) # ni H W

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.matching_metrics['point_mask_dice_ce']['num_points'],
                                           2, device=self.device)
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float() # nq num_points
                    tgt_mask = tgt_mask.float()
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                C += self.matching_metrics['point_mask_dice_ce']['ce'] * cost_mask + \
                     self.matching_metrics['point_mask_dice_ce']['dice'] * cost_dice
            
            C = C.cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def loss_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0)
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0)
        tgt_masks = tgt_masks.to(src_masks)
        losses = {
            "mask_ce": ce_mask_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
            "mask_dice": dice_loss(src_masks.flatten(1), tgt_masks.flatten(1), num_boxes=num_objs),
        }
        return losses   

    def loss_point_mask_dice_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        src_masks = outputs['pred_masks'] # b nq h w
        tgt_masks = targets['masks']
        src_masks = torch.cat([t[J] for t, (J, _) in zip(src_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = torch.cat([t[J] for t, (_, J) in zip(tgt_masks, indices)], dim=0) # n_sigma h w
        tgt_masks = tgt_masks.to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = tgt_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                loss_extra_param['num_points'],
                loss_extra_param['oversample_ratio'],
                loss_extra_param['importance_sample_ratio'],
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "mask_dice": ce_mask_loss(point_logits, point_labels, num_objs),
            "mask_ce": dice_loss(point_logits, point_labels, num_objs),
        }

        del src_masks
        del target_masks
        return losses        

    def loss_class_ce(self, outputs, targets, indices, num_objs, loss_extra_param):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs["pred_class"].float() # b nq c

        idx = self._get_src_permutation_idx(indices)
        # list[n], b -> bn
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['classes'], indices)]) 
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=self.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"class_ce": loss_ce}
        return losses

    def loss_box_l1_giou(self, outputs, targets, indices, num_objs, loss_extra_param): 
        tgt_boxes = targets['boxes'] # list[n 4], b

        src_boxes = outputs['pred_boxes'].sigmoid() # b nq 4

        src_boxes = torch.cat([t[J] for t, (J, _) in zip(src_boxes, indices)], dim=0) # n_sum 4
        tgt_boxes = torch.cat([t[J] for t, (_, J) in zip(tgt_boxes, indices)], dim=0) # n_sum 4
        
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='none') # n_sum 4

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
        return {
            'box_l1': loss_l1.sum(-1).sum() / num_objs,
            'box_giou': loss_giou.sum() / num_objs
        }

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




# t nqf -> cros -> ss2d -> cross
class Video_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.frame_nqueries = configs['frame_nqueries'] # 20
        self.video_nqueries = configs['video_nqueries'] # 10
        self.nlayers = configs['nlayers'] 
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']
        self.mask_spatial_stride = multiscale_shapes[self.mask_scale].spatial_stride
        num_classes = configs['num_classes']

        inputs_projs = configs['inputs_projs']
        self.inputs_projs = nn.Sequential()
        if inputs_projs is not None:
            self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                             multiscale_shapes=multiscale_shapes,
                                                                             out_dim=d_model)
        self.frame_query_poses =  nn.Embedding(self.frame_nqueries, d_model)
        self.frame_query_feats = nn.Embedding(self.frame_nqueries, d_model)
        self.temporal_query_poses = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_feats = nn.Embedding(self.video_nqueries, d_model)

        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)
        assert self.nlayers % len(self.memory_scales) == 0

        self.frame_cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                              self.nlayers)
        
        temporal_self_layer = META_ARCH_REGISTRY.get(configs['temporal_self_layer']['name'])(configs['temporal_self_layer'])
        self.temporal_self_layers = _get_clones(temporal_self_layer, self.nlayers)

        temporal_cross_layer = META_ARCH_REGISTRY.get(configs['temporal_cross_layer']['name'])(configs['temporal_cross_layer'])
        self.temporal_cross_layers = _get_clones(temporal_cross_layer, self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.frame_query_norm = nn.LayerNorm(d_model)
        self.temporal_query_norm = nn.LayerNorm(d_model)
        self.pos_3d = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d') # b t c h w
        
        self.frame_head_outputs = configs['frame_head_outputs']
        assert 'mask' in self.frame_head_outputs
        self.frame_query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.frame_head_outputs:
            self.frame_query_class = nn.Linear(d_model, num_classes + 1)
        if 'box' in self.frame_head_outputs:
            self.frame_query_box = MLP(d_model, d_model, 4, 3)

        self.temporal_head_outputs = configs['temporal_head_outputs']
        assert 'mask' in self.temporal_head_outputs
        self.temporal_query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.temporal_head_outputs:
            self.temporal_query_class = nn.Linear(d_model, num_classes + 1)    

        self.vid_loss_module = Video_SetMatchingLoss(loss_config=configs['temporal_loss'], num_classes=num_classes)
        self.frame_loss_module = Image_SetMatchingLoss(loss_config=configs['frame_loss'], num_classes=num_classes)

        self.vid_mask_features = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.frame_mask_features = nn.Conv2d(d_model, d_model, kernel_size=1)
        
    @property
    def device(self,):
        return self.frame_query_poses.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c t h w 
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_3d(mem.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) for mem in memories]
        memories = [rearrange(mem, 'b c t h w -> (h w) (b t) c').contiguous() for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c t h w -> (h w) (b t) c').contiguous() for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c t h w

        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales, # b c t h w
                ):
        multiscales = self.inputs_projs(multiscales)
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]

        # hw bt c; bt c h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(multiscales)
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]

        # nqf bt c
        frame_query_poses = self.frame_query_poses.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)
        frame_query_feats = self.frame_query_feats.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)

        # nq b c
        temporal_query_poses = self.temporal_query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporal_query_feats = self.temporal_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)


        frame_ret = []
        vid_ret = []
        # bt nqf class, bt nqf 4, bt nqf h w, bt*head nqf hw
        # b nq class, b nq h w
        frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
            self.forward_heads(frame_query_feats=frame_query_feats, 
                               temporal_query_feats=temporal_query_feats,
                               mask_features=mask_features, attn_mask_target_size=size_list[0]) # first sight you re not human
        frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
        vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})

        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            frame_cross_attn_mask[torch.where(frame_cross_attn_mask.sum(-1) == frame_cross_attn_mask.shape[-1])] = False 

            frame_query_feats = self.frame_cross_layers[i](
                tgt=frame_query_feats, # nqf bt c
                memory=memories[level_index], # hw bt c
                memory_mask=frame_cross_attn_mask, 
                memory_key_padding_mask=None,
                pos=memories_poses[level_index], 
                query_pos=frame_query_poses,
            )
            frame_query_feats = self.temporal_self_layers[i](
                frame_query_feats=frame_query_feats, 
                frame_query_poses=frame_query_poses,
            )

            temporal_query_feats = self.temporal_cross_layers[i](
                temporal_query_feats=temporal_query_feats, 
                temporal_query_poses=temporal_query_poses,

                frame_query_feats=frame_query_feats,
                frame_query_poses=frame_query_poses,
            )

            frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
                self.forward_heads(frame_query_feats=frame_query_feats, 
                                    temporal_query_feats=temporal_query_feats,
                                    mask_features=mask_features, attn_mask_target_size=size_list[0])
            frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
            vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})
    
        if self.training:
            return frame_ret, vid_ret
        else:
            return vid_ret

    def forward_heads(self, frame_query_feats, temporal_query_feats,  mask_features, attn_mask_target_size):
        decoder_output = self.frame_query_norm(frame_query_feats) # nqf bt c
        decoder_output = decoder_output.transpose(0, 1).contiguous()

        box_logits = self.frame_query_box(decoder_output) if 'box' in self.frame_head_outputs else None # bt n 4
        class_logits = self.frame_query_class(decoder_output) if 'class' in self.frame_head_outputs else None # bt n class+1
        mask_embeds = self.frame_query_mask(decoder_output)  # bt n c
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features.permute(0, 2, 1, 3, 4).flatten(0,1).contiguous())  

        # b nq h w
        attn_mask = mask_logits.detach().clone() 
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # b*head nq hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        
        mask_logits = F.interpolate(mask_logits, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)

        if self.training:
            return class_logits, box_logits, mask_logits, attn_mask, decoder_output # b nq h w
        else:
            return class_logits.softmax(-1) if class_logits is not None else None,\
                  box_logits.sigmoid() if box_logits is not None else None, mask_logits, attn_mask, decoder_output # b nq h w
    

    def compute_loss(self, outputs, targets):
        assert self.training
        return self.loss_module.compute_loss(outputs, targets)

        temporal_query_feats = self.temporal_query_norm(temporal_query_feats) # nq b c
        temporal_query_feats = frame_query_feats.transpose(0, 1).contiguous() # b nq c

        class_logits = self.query_class(temporal_query_feats) if 'class' in self.head_outputs else None # b n class+1
        mask_embeds = self.query_mask(temporal_query_feats)  # b n c
        mask_logits = torch.einsum("bqc,bcthw->bqthw", mask_embeds, mask_features) 

        mask_logits = F.interpolate(mask_logits, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)

        if self.training:
            return class_logits, mask_logits, attn_mask
        else:
            return class_logits.softmax(-1) if class_logits is not None else None, mask_logits, attn_mask
    

    def compute_loss(self, outputs, targets):
        assert self.training
        return self.loss_module.compute_loss(outputs, targets)
