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

from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.rvos.apis import RVOS_TrainAPI_ForEachRefer_clipped_video, RVOS_EvalAPI_referent_text_clipped_video_request_ann

from utils.misc import nested_tensor_from_videos_list_with_stride

class AUXMapper_v1: # ForEachRefer
    RVOS_TrainAPI_ForEachRefer_clipped_video
    """ RVOS_TrainAPI_ForEachRefer_clipped_video 的 aux mapper和collator
    对于API是RVOS_TrainAPI_ForEachRefer_clipped_video的model,
        !!! 只能添加不能更改 
        如果他有需要添加新的input/targets, 
            可以再video_dict, refer_dict, video_refer_dict, targets里添加, 
            如果这些key不能够表示新添加的知识, 就需要添加新的dict, 
            修改对应的mapper, collate函数
        examples:
            添加CLIP的visual-text cross attention作为targets进行指导, 可以添加到targets/frame_targets
            添加video的optical flow作为新的input, 可以添加到video dict里
            添加使用外部model的video数据增强作为新的input, 可以添加到video_dict
    train
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'refer_dict': {
            'text': str,
            'aux': None
        }
        'video_refer_dict':{
            'aux': None
        }
        'targets': {
            'has_ann': b t (bool)
            'boxes': list[N t' 4], x1y1x2y2
            'masks': list[N t' h w] (bool)
            'class_labels': list[N],
            'referent_objs': list[int]
        }
        'frame_targets':{
            'masks': list[n h w], bt'
            'boxes': list[n 4], bt'
            'class_labels': list[n], bt'
            'referent_objs': list[list[int]], bt'
        }
        'meta_idx': int
        'visualize': True/False
    eval
        输入输出: RVOS_EvalAPI_referent_text_clipped_video_request_ann
    """
    def __init__(self, aux_configs):
        refer_auxes = aux_configs['refer_auxes']

        refer_auxes_names = [config['name'] for config in refer_auxes]
        assert len(list(set(refer_auxes_names))) == len(refer_auxes_names), '每个aux的名字必须不一样'
        self.refer_auxes_names = refer_auxes_names
        self.refer_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in refer_auxes]

        # 添加任何dict, 
        self.video_auxes = []
        self.video_refer_auxes = []
        self.video_auxes_names = []
        self.targets_auxes = []

    def mapper(self, data_dict, mode,):
        if mode == 'train':
            RVOS_TrainAPI_ForEachRefer_clipped_video
            refer_text = data_dict['refer_dict']['text']
            for aux, aux_name in zip(self.refer_auxes, self.refer_auxes_names):
                data_dict['refer_dict'][aux_name] = aux.mapper(refer_text)

            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video)
        
        elif mode == 'evaluate':
            RVOS_EvalAPI_referent_text_clipped_video_request_ann
            refer_text = data_dict['refer_dict']['text']
            for aux, aux_name in zip(self.refer_auxes, self.refer_auxes_names):
                data_dict['refer_dict'][aux_name] = aux.mapper(refer_text)  

            video = data_dict['video_dict']['video']
            for aux, aux_name in zip(self.video_auxes, self.video_auxes_names):
                data_dict['video_dict'][aux_name] = aux.mapper(video)

        return data_dict

    def collate(self, batch_dict, mode, max_stride):
        if mode == 'train':
            RVOS_TrainAPI_ForEachRefer_clipped_video
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride)
            refer_dict = self.collate_refer_dict(batch_dict)
            targets = [sample['targets'] for sample in batch_dict]
            frame_has_ann = [clip_tgt['has_ann'] for clip_tgt in targets] # list[t], b
            frame_targets = [sample['frame_targets'] for sample in batch_dict]

            _, pad_T, _, pad_H, pad_W = video_dict['videos'].shape
            collated_targets = self.collate_targets(old_targets=targets, pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            collated_frame_targets = self.collate_frame_targets(frame_targets=frame_targets, 
                                                                frame_has_ann=frame_has_ann ,
                                                                pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            
            ret = {
                'video_dict': video_dict,
                'refer_dict': refer_dict,
                'targets': collated_targets,
                'frame_targets': collated_frame_targets,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],               
            }   
                        
        elif mode == 'evaluate':
            RVOS_EvalAPI_referent_text_clipped_video_request_ann
            assert len(batch_dict) == 1
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride) # 不pad
            refer_dict = self.collate_refer_dict(batch_dict)
            metas = [sample['meta'] for sample in batch_dict]

            collated_metas = {}
            for key in metas[0].keys():
                collated_metas[key] = [mt[key] for mt in metas]
            
            ret = {
                'video_dict': video_dict,
                'refer_dict': refer_dict,
                'metas': collated_metas,
                'meta_idxs':  [sample['meta_idx'] for sample in batch_dict],
                'visualize': [sample['visualize'] for sample in batch_dict],  
            }  
        debug_data = False
        if debug_data:
            self.visualize_input_target_for_debug_data(ret) # ./test.png
        return ret

    def collate_video_dict(self, batch_dict, max_stride):
        videos = [sample['video_dict']['video'] for sample in batch_dict]  # list[ti 3 hi wi] -> b T 3 H W
        orig_sizes = [list(vid.shape) for vid in videos] # t 3 h w
        if type(max_stride) == int: # temporal max stride 为1, spatial max stride
            pad_stride = [1, max_stride]
        if (type(max_stride) == list) and (len(max_stride) == 2):
            pad_stride = max_stride
        videos = nested_tensor_from_videos_list_with_stride(videos, max_stride=pad_stride).tensors
        video_dicts = {'videos': videos, 'orig_sizes': orig_sizes}
        return video_dicts

    def collate_refer_dict(self, batch_dict):
        refer_dicts = {
            'texts': [sample['refer_dict']['text'] for sample in batch_dict]
        }  
        for aux_name, aux in zip(self.refer_auxes_names, self.refer_auxes):
            auxes = [sample['refer_dict'][aux_name] for sample in batch_dict] # list[dict] / list[tensor]
            collated_auxes = aux.collate(auxes) # list[dict]
            if isinstance(auxes[0], dict):
                keys = collated_auxes.keys()
                for key in keys:
                    assert key not in refer_dicts
                    refer_dicts[key] = collated_auxes[key]
            else:
                refer_dicts[aux_name] = collated_auxes

        return refer_dicts

    def collate_frame_targets(self, frame_targets, frame_has_ann, pad_H, pad_W, pad_T): # 
        ret = {}
        # frame_targets的mask表示 是 for each t: nq c * h w c, padding帧不考虑; has_ann padding的value是0
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=0.).bool() for ha in frame_has_ann], dim=0).flatten() # bT
        ret['has_ann'] = has_ann
        masks = [ftarget['masks'] for sample in frame_targets for ftarget in sample] # list[Ni h w], bt'
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2])).bool() for m in masks] # list[Ni H W], bt'
        ret['masks'] = masks # list[N h w], bt'

        boxes = [ftarget['boxes'] for sample in frame_targets for ftarget in sample] # list[N 4], x1y1x2y2, bt'
        boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
        boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=bx.dtype) for bx in boxes] # 0-1
        ret['boxes'] = boxes # list[N 4], bt'
        ret['classes'] = [ftarget['classes'] for sample in frame_targets for ftarget in sample] 
        
        if 'referent_objs' in frame_targets[0][0]:
            ret['referent_objs'] = [ftarget['referent_objs'] for sample in frame_targets for ftarget in sample], # list[list[int]], bt
        
        return ret

    def collate_targets(self, targets, pad_H, pad_W, pad_T):
        has_ann = [sample['has_ann'] for sample in targets] # list[t], bool
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=0.).bool() for ha in has_ann], dim=0) # b T
        
        # padding部分没有annotation
        # list[ni T' h w] -> list[ni T' H W]
        masks = [sample['masks'] for sample in targets] 
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2]), value=0.).bool() \
                 for m in masks] # list[ni T' H W]

        # 把mask放缩到H/4, W/4
        # for btc_idx in range(batch_size):
        #     start = int(self.temporal_decoder_mask_out_stride // 2)
        #     im_h, im_w = tgt_masks[btc_idx].shape[-2:]
        #     tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
        #     assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
        #     assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        boxes = [sample['boxes'] for sample in targets] # list[N t' 4], x1y1x2y2
        boxes = [sample['boxes'] for sample in targets] # list[ni T' 4], x1y1x2y2
        boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
        boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=torch.float) for bx in boxes] # 0-1

        targets = {'masks': masks, # list[Ni T'_i h w]
                   'boxes': boxes, # list[Ni T'_i 4]
                   'has_ann': has_ann, # b T
                   'referent_objs': [sample['referent_objs'] for sample in targets], # list[list[int], ]
                   'classes': [sample['classes'] for sample in targets]
        }
        return targets

    def collate_video_refer(self, batch_dict):
        # 如果要加入新的 video_refer_dict, 
        pass

    def visualize_input_target_for_debug_data(self, ret):
        videos = ret['video_dict']['videos'] # b T 3 H W
        pass
