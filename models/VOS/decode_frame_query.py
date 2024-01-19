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
from util.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from functools import partial
from util.misc import to_device
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

from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.registry import VOS_TrainAPI_clipped_video, VOS_EvalAPI_output
from data_schedule.registry import VOS_EvalAPI_clipped_video_request_ann
__all__ = ['decode_frame_query']


from util.misc import nested_tensor_from_videos_list_with_stride
class AUXMapper_v1:
    VOS_TrainAPI_clipped_video
    """ VOS_TrainAPI_clipped_video 的 aux mapper和collator
    对于API是RVOS_TrainAPI_referent_text_clipped_video的model,
        !!! 只能添加不能更改 
        如果他有需要添加新的input/targets, 
            可以再video_dict, targets里添加, 
            如果这些key不能够表示新添加的知识, 就需要添加新的dict, 
            修改对应的mapper, collate函数
        examples:
            添加video的optical flow作为新的input, 可以添加到video dict里
            添加使用外部model的video数据增强作为新的input, 可以添加到video_dict
    train
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'targets': {
            'has_ann': b t (bool)
            'masks': list[t' h w] (bool)
        }
        'frame_targets':{
            'masks': list[n h w], bt'
            'boxes': list[n 4], bt'
        }
        'meta_idx': int
        'visualize': True/False
    """
    def __init__(self, aux_configs):
        video_auxes = aux_configs['video_auxes']

        video_auxes_names = [config['name'] for config in video_auxes]
        assert len(list(set(video_auxes_names))) == len(video_auxes_names), '每个aux的名字必须不一样'
        self.video_auxes_names = video_auxes_names
        self.video_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in video_auxes]

        self.targets_auxes = None

    def mapper(self, old_data_dict, mode,):
        data_dict = dcopy(old_data_dict)
        if mode == 'train':
            VOS_TrainAPI_clipped_video
            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video)
        
        elif mode == 'evaluate':
            VOS_EvalAPI_clipped_video_request_ann
            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video)      
        return data_dict

    def collate(self, batch_dict, mode, max_stride):
        if mode == 'train':
            VOS_TrainAPI_clipped_video
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride)
            targets = [sample['targets'] for sample in batch_dict]
            frame_targets = [sample['frame_targets'] for sample in batch_dict]

            _, pad_T, _, pad_H, pad_W = video_dict['videos'].shape
            collated_targets = self.collate_targets(old_targets=targets, pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            collated_frame_targets = self.collate_frame_targets(old_frame_targets=frame_targets, 
                                                                old_clip_targets=targets,
                                                                pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            
            ret = {
                'video_dict': video_dict,
                'targets': collated_targets,
                'frame_targets': collated_frame_targets,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],               
            }   
                        
        elif mode == 'evaluate':
            VOS_EvalAPI_clipped_video_request_ann
            assert len(batch_dict) == 1
            orig_t, _, orig_h, orig_w = batch_dict[0]['video_dict']['video'].shape # t 3 h w
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride) # 不pad
            metas = [sample['meta'] for sample in batch_dict]

            collated_metas = {}
            for key in metas[0].keys():
                collated_metas[key] = [mt[key] for mt in metas]
            ret = {
                'video_dict': video_dict,
                'metas': collated_metas,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],  
                'orig_size': [[orig_t, orig_h, orig_w]] # model的sample是按照batch写的, 但是有assert
            }  
        debug_data = False
        if debug_data:
            self.visualize_input_target_for_debug_data(ret) # ./test.png
        return ret

    def collate_video_dict(self, old_batch_dict, max_stride):
        batch_dict = dcopy(old_batch_dict)
        videos = [sample['video_dict']['video'] for sample in batch_dict]  # list[ti 3 hi wi] -> b T 3 H W
        if type(max_stride) == int: # temporal max stride 为1, spatial max stride
            pad_stride = [1, max_stride]
        if (type(max_stride) == list) and (len(max_stride) == 2):
            pad_stride = max_stride
        videos = nested_tensor_from_videos_list_with_stride(videos, max_stride=pad_stride).tensors # b t c h w
        video_dicts = {'videos': videos}
        return video_dicts

    def collate_frame_targets(self, old_frame_targets, old_clip_targets, pad_H, pad_W, pad_T): # 
        frame_targets = dcopy(old_frame_targets)
        clip_targets = dcopy(old_clip_targets)
        ret = {}
        # frame_targets的mask表示 是 for each t: nq c * h w c, padding帧不考虑; has_ann padding的value是0
        frame_has_ann = [clip_tgt['has_ann'] for clip_tgt in clip_targets] # list[t]
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=0.).bool() for ha in frame_has_ann], dim=0).flatten() # bT
        ret['has_ann'] = has_ann
        masks = [ftarget['masks'] for sample in frame_targets for ftarget in sample] # list[Ni h w], bt'
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2])).bool() for m in masks] # list[Ni H W], bt'
        ret['masks'] = masks # list[N h w], bt'

        boxes = [ftarget['boxes'] for sample in frame_targets for ftarget in sample] # list[N 4], x1y1x2y2, bt'
        boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
        boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=bx.dtype) for bx in boxes] # 0-1
        ret['boxes'] = boxes # list[N 4], bt' 
        
        return ret

    def collate_targets(self, old_targets, pad_H, pad_W, pad_T):
        # padding越大, 正确率高的越不可信
        # 但是这个问题完全可以通过在训练的时候控制padding较小就行
        targets = dcopy(old_targets)
        has_ann = [sample['has_ann'] for sample in targets] # list[t], bool
        pad_Ts = [pad_T - len(taylor_swift) for taylor_swift in has_ann]
        # targets的mask表示 是 nq c * t h w c, 所以padding的部分也算上, padding的value是1
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=1.).bool() for ha in has_ann], dim=0) # b T
        
        masks = [sample['masks'] for sample in targets] # list[t' h w]
        # 不能给没有annotation的指定ground truth mask
        # padding的部分的ground truth都是0, h,w,t都是
        masks = [F.pad(m.unsqueeze(0).float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2], 0, taylor_swift), value=0.)[0].bool() \
                 for m, taylor_swift in zip(masks, pad_Ts)] # list[T' H W]

        # 把mask放缩到H/4, W/4
        # for btc_idx in range(batch_size):
        #     start = int(self.temporal_decoder_mask_out_stride // 2)
        #     im_h, im_w = tgt_masks[btc_idx].shape[-2:]
        #     tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
        #     assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
        #     assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        targets = {'masks': masks, # list[T'_i h w]
                   'has_ann': has_ann, # b T
        } # list[Ni]
        return targets

    def visualize_input_target_for_debug_data(self, ret):
        videos = ret['video_dict']['videos'] # b T 3 H W
        pass

class Decode_FrameQuery_VOS111(nn.Module): # frame decoder做object segmentation, temporal decoder做referent segmentation
    # model知道整个module-submodule结构
    def __init__(self,
                 configs,
                 pixel_mean = [0.485, 0.456, 0.406],
                 pixel_std = [0.229, 0.224, 0.225],
                ) -> None:
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # encoder的对比学习损失, decoder的mask, box损失
        self.loss_weight = configs['model']['loss_weights']
        # video_dict -> multiscale
        video_backbone_configs = configs['model']['video_backbone'] 
        # (multiscale, text_dict) 在帧间parsing 
        frame_encoder_configs = configs['model']['frame_encoder']
        # frame query聚合每帧的信息
        frame_decoder_configs = configs['model']['frame_decoder']

        # temporal encoder使用的frame decoder的层数
        self.num_frame_decoder_layers_used = configs['model']['num_frame_decoder_layers_used']

        # frame query每帧的信息进行交流
        temporal_encoder_configs = configs['model']['temporal_encoder']
        # temporal query聚合整个video的信息
        temporal_decoder_configs = configs['model']['temporal_decoder']

        # video_backbone.video_backbone/text_backbone
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride

        frame_encoder_cls = META_ARCH_REGISTRY.get(frame_encoder_configs['name'])
        self.frame_encoder = frame_encoder_cls(frame_encoder_configs,
                                                multiscale_shapes=self.video_backbone.multiscale_shapes,)
        
        frame_decoder_cls = META_ARCH_REGISTRY.get(frame_decoder_configs['name'])
        self.frame_decoder = frame_decoder_cls(frame_decoder_configs,)
        
        temporal_encoder_cls = META_ARCH_REGISTRY.get(temporal_encoder_configs['name'])
        self.temporal_encoder = temporal_encoder_cls(temporal_encoder_configs)
        
        temporal_decoder_cls = META_ARCH_REGISTRY.get(temporal_decoder_configs['name'])
        self.temporal_decoder = temporal_decoder_cls(temporal_decoder_configs)

        self.decoder_mask_stride = self.frame_decoder.mask_stride

 
    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['weight_decay']

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
        log_lr_group_idx = {'backbone':None, 'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if log_lr_group_idx['backbone'] is None:
                        log_lr_group_idx['backbone'] = len(params)

                else:
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                                     
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
   
        return params, log_lr_group_idx

    def forward(self, batch_dict):
        VOS_TrainAPI_clipped_video
        videos = batch_dict.pop('video_dict')['videos'] # 0-1, float, b t 3 h w
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = batch_dict.pop('targets')
        frame_targets = batch_dict.pop('frame_targets')
        # biparte matching
        tgt_masks = targets['masks'] # list[t' H W], batch
        batch_size = len(tgt_masks)
        for btc_idx in range(batch_size):
            start = int(self.decoder_mask_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, start::self.decoder_mask_stride, start::self.decoder_mask_stride] 
            assert tgt_masks[btc_idx].size(1) * self.decoder_mask_stride == im_h
            assert tgt_masks[btc_idx].size(2) * self.decoder_mask_stride == im_w
        targets['masks'] = tgt_masks
        
        frame_tgt_masks = frame_targets['masks'] # list[ni H W], bt'
        batch_size = len(frame_tgt_masks)
        for btc_idx in range(batch_size):
            start = int(self.decoder_mask_stride // 2)
            im_h, im_w = frame_tgt_masks[btc_idx].shape[-2:]
            frame_tgt_masks[btc_idx] = frame_tgt_masks[btc_idx][:, start::self.decoder_mask_stride, start::self.decoder_mask_stride] 
            assert frame_tgt_masks[btc_idx].size(1) * self.decoder_mask_stride == im_h
            assert frame_tgt_masks[btc_idx].size(2) * self.decoder_mask_stride == im_w
        frame_targets['masks'] = frame_tgt_masks
        
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch

        batch_size, T, _, H, W = videos.shape
        # 抽特征
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4).contiguous())  # b c t h w
        # 对比学习
        # self.video_backbone.compute_loss(multiscales, text_inputs)
        fencoded_multiscales = self.frame_encoder(multiscales={scale_name:scale_feat.clone() \
                                                               for scale_name, scale_feat in multiscales.items()})
        # bt c h w -> {'queries': list[b t nqf c], 'pred_masks': b t n h w, 'pred_boxes': b t n 4, 'pred_class': b t n class+1}
        frame_decoder_output = \
            self.frame_decoder(video_multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in fencoded_multiscales.items()},)
        frame_queries_by_layer = [layer_output['frame_queries'] for layer_output in frame_decoder_output]
        frame_queries_by_layer = frame_queries_by_layer[(-1 * self.num_frame_decoder_layers_used):] # list[ b t nq c ]
        
        # b t nqf c -> L b t nqf c -> Lb t nqf c
        frame_queries = torch.stack(frame_queries_by_layer, dim=0).flatten(0, 1).contiguous()

        # Lb t nqf c, 
        tencoded_frame_queries = self.temporal_encoder(frame_queries)
        # Lb t nqf c -> {'queries': list[Lb nq c], 'predictions': }
        temporal_decoder_output = self.temporal_decoder(frame_query=tencoded_frame_queries, 
                                                        multiscales=fencoded_multiscales)
        frame_decoder_losses = self.frame_decoder.compute_loss(frame_decoder_output, frame_targets)
        frame_decoder_losses = {f'frame_decoder_{key}': value for key, value in frame_decoder_losses.items()}
        temporal_decoder_losses = self.temporal_decoder.compute_loss(temporal_decoder_output, targets)
        temporal_decoder_losses = {f'temporal_decoder_{key}': value for key, value in temporal_decoder_losses.items()}
        assert len(set(list(frame_decoder_losses.keys())) & set(list(temporal_decoder_losses.keys()))) == 0
        loss_value_dict = {}
        loss_value_dict.update(frame_decoder_losses)
        loss_value_dict.update(temporal_decoder_losses)
        assert set(list(self.loss_weight.keys())).issubset(set(list(loss_value_dict.keys())))
        assert set(list(loss_value_dict.keys())).issubset(set(list(self.loss_weight.keys())))
        loss = sum([loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()])
        if not math.isfinite(loss.item()):
            logging.debug("Loss is {}, stopping training".format(loss.item()))
            raise RuntimeError()
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        # from models.backbone.swin import compute_mask
        # compute_mask.cache_clear()        
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample(self, batch_dict):
        VOS_EvalAPI_clipped_video_request_ann
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # B T 3 H W
        assert len(videos) == 1
        visualize_paths = batch_dict.pop('visualize_paths') 
        orig_t, orig_h, orig_w = batch_dict.pop('orig_size')[0]

        batch_size, T, _, H, W = videos.shape
        
        multiscales = self.video_backbone(x=videos.permute(0, 2, 1, 3, 4),)  # b c t h w

        fencoded_multiscales = self.frame_encoder(multiscales={scale_name:scale_feat.clone() \
                                                               for scale_name, scale_feat in multiscales.items()})

        frame_decoder_output = \
            self.frame_decoder(video_multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in fencoded_multiscales.items()},)

        frame_queries = frame_decoder_output[-1]['frame_queries'] # b t nqf c
        tencoded_frame_queries = self.temporal_encoder(frame_queries)
        temporal_decoder_output = self.temporal_decoder(frame_query=tencoded_frame_queries, 
                                                        multiscales=fencoded_multiscales)

        pred_masks = temporal_decoder_output[-1]['pred_masks'] # b n t h w
        pred_obj = temporal_decoder_output[-1]['pred_class'].softmax(-1)[:, :, 0] # b n 
        assert len(pred_masks) == 1
        assert len(pred_obj) == 1
        # unpad
        padded_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        pred_masks = pred_masks[0][:, :orig_t, :orig_h, :orig_w] # nq t h w
        pred_obj = pred_obj[0] # nq

        # unnormalize
        orig_video = Trans_F.normalize(padded_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
        VOS_EvalAPI_output
        return {
            'video': [orig_video], # t 3 h w
            'pred_masks': [pred_masks.unbind(1)], # list[nq h w], t
            'pred_obj': [pred_obj.unsqueeze(0).repeat(orig_t, 1).unbind(0)] # list[nq], t
        }

        # prob = temporal_decoder_output[-1]['pred_class'].softmax(-1)[0][:, 0] # b nq c -> nq
        # pred_masks = temporal_decoder_output[-1]['pred_masks'][0].sigmoid() > 0.5 # nq t h w
        # max_query_idx = prob.argmax(-1) # b nq
        # pred_masks = pred_masks[max_query_idx] # t h w
        # import matplotlib.pyplot as plt
        # plt.imsave('./test.png', pred_masks[0].cpu())

    @torch.no_grad()
    def tta_sample(self, asembles, visualize_dir=None):
        # test time agumentation 
        preds_by_aug = []
        for sample in asembles:
            preds_by_aug.append(self.sample(asembles, visualize_dir=visualize_dir))
        return preds_by_aug


@register_model
def decode_frame_query_vos111(configs, device):
    model =  Decode_FrameQuery_VOS111(configs)
    model.to(device)
    params_group, log_lr_group_idx = Decode_FrameQuery_VOS111.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx

