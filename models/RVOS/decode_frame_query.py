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
from data_schedule.registry import RVOS_TrainAPI_referent_text_clipped_video
from data_schedule.registry import RVOS_EvalAPI_referent_text_clipped_video_request_ann
__all__ = ['decode_frame_query']


from util.misc import nested_tensor_from_videos_list_with_stride
class AUXMapper_v1:
    RVOS_TrainAPI_referent_text_clipped_video
    """ RVOS_TrainAPI_referent_text_clipped_video 的 aux mapper和collator
    对于API是RVOS_TrainAPI_referent_text_clipped_video的model,
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
        self.video_auxes = None
        self.video_refer_auxes = None
        self.targets_auxes = None

    def mapper(self, old_data_dict, mode,):
        RVOS_TrainAPI_referent_text_clipped_video
        data_dict = dcopy(old_data_dict)
        if mode == 'train':
            RVOS_TrainAPI_referent_text_clipped_video
            refer_text = data_dict['refer_dict']['text']
            for aux, aux_name in zip(self.refer_auxes, self.refer_auxes_names):
                data_dict['refer_dict'][aux_name] = aux.mapper(refer_text)
        
        elif mode == 'evaluate':
            RVOS_EvalAPI_referent_text_clipped_video_request_ann
            refer_text = data_dict['refer_dict']['text']
            for aux, aux_name in zip(self.refer_auxes, self.refer_auxes_names):
                data_dict['refer_dict'][aux_name] = aux.mapper(refer_text)        
        return data_dict

    def collate(self, batch_dict, mode, max_stride):
        if mode == 'train':
            RVOS_TrainAPI_referent_text_clipped_video
            video_dict = self.collate_video_dict(batch_dict, max_stride=max_stride)
            refer_dict = self.collate_refer_dict(batch_dict)
            targets = [sample['targets'] for sample in batch_dict]
            frame_targets = [sample['frame_targets'] for sample in batch_dict]

            _, pad_T, _, pad_H, pad_W = video_dict['videos'].shape
            collated_targets = self.collate_targets(old_targets=targets, pad_H=pad_H, pad_W=pad_W, pad_T=pad_T)
            collated_frame_targets = self.collate_frame_targets(old_frame_targets=frame_targets, 
                                                                old_clip_targets=targets,
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
            orig_t, _, orig_h, orig_w = batch_dict[0]['video_dict']['video'].shape # t 3 h w
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
                'orig_size': [[orig_t, orig_h, orig_w]]
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
        videos = nested_tensor_from_videos_list_with_stride(videos, max_stride=pad_stride).tensors
        video_dicts = {'videos': videos}
        return video_dicts

    def collate_refer_dict(self, old_batch_dict):
        batch_dict = dcopy(old_batch_dict)
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
        ret['class_labels'] = [ftarget['class_labels'] for sample in frame_targets for ftarget in sample] 
        
        if 'referent_objs' in frame_targets[0][0]:
            ret['referent_objs'] = [ftarget['referent_objs'] for sample in frame_targets for ftarget in sample], # list[list[int]], bt
        
        return ret

    def collate_targets(self, old_targets, pad_H, pad_W, pad_T):
        # padding越大, 正确率高的越不可信
        # 但是这个问题完全可以通过在训练的时候控制padding较小就行
        targets = dcopy(old_targets)
        has_ann = [sample['has_ann'] for sample in targets] # list[t], bool
        pad_Ts = [pad_T - len(taylor_swift) for taylor_swift in has_ann]
        # targets的mask表示 是 nq c * t h w c, 所以padding的部分也算上, padding的value是1
        has_ann = torch.stack([F.pad(ha.float(), pad=(0, pad_T - len(ha)), value=1.).bool() for ha in has_ann], dim=0) # b T
        
        masks = [sample['masks'] for sample in targets] # list[N t' h w]
        # 不能给没有annotation的指定ground truth mask
        # padding的部分的ground truth都是0, h,w,t都是
        masks = [F.pad(m.float(), pad=(0, pad_W-m.shape[-1], 0, pad_H-m.shape[-2], 0, taylor_swift), value=0.).bool() \
                 for m, taylor_swift in zip(masks, pad_Ts)] # list[N T' H W]

        # 把mask放缩到H/4, W/4
        # for btc_idx in range(batch_size):
        #     start = int(self.temporal_decoder_mask_out_stride // 2)
        #     im_h, im_w = tgt_masks[btc_idx].shape[-2:]
        #     tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
        #     assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
        #     assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        boxes = [sample['boxes'] for sample in targets] # list[N t' 4], x1y1x2y2
        boxes = [F.pad(bx, pad=(0, 0, 0, taylow_swift), value=0.) for bx, taylow_swift in zip(boxes, pad_Ts)] # list[N T' 4]
        boxes = [box_xyxy_to_cxcywh(bx) for bx in boxes]
        boxes = [bx / torch.tensor([pad_W, pad_H, pad_W, pad_H], dtype=torch.float) for bx in boxes] # 0-1

        targets = {'masks': masks, # list[Ni T'_i h w]
                   'boxes': boxes, # list[Ni T'_i 4]
                   'has_ann': has_ann, # b T
                   'referent_objs': [sample['referent_objs'] for sample in targets], # list[list[int], ]
                   'class_labels': [sample['class_labels'] for sample in targets]
        } # list[Ni]
        return targets

    def collate_video_refer(self, batch_dict):
        # 如果要加入新的 video_refer_dict, 
        pass

    def visualize_input_target_for_debug_data(self, ret):
        videos = ret['video_dict']['videos'] # b T 3 H W
        pass

class Decode_FrameQuery(nn.Module): # frame decoder做object segmentation, temporal decoder做referent segmentation
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
        # video, text_dict -> multiscale, text_dict/AMRData
        video_text_backbone_configs = configs['model']['video_text_backbone'] 
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

        fusion_module_configs = configs['model']['fusion_module'] # multiscale/frame_query, text_dict <->
        reason_module_configs = configs['model']['reason_module'] # temporal_query, text_dict -> 每个query被refer的概率

        # 共享的modules
        # forward: 输入query特征, 文本特征, 输出每个query 被Refer的概率
        # compute_loss: 输出的refer的概率, matching的结果
        self.fusion_module = None # META_ARCH_REGISTRY.get(configs['model']['fusion_module']['name'])(configs['model']['fusion_module'])
        self.reason_module = None # META_ARCH_REGISTRY.get(configs['model']['reason_module']['name'])(configs['model']['reason_module'])

        # video_text_backbone.video_backbone/text_backbone
        video_text_backbone_cls = BACKBONE_REGISTRY.get(video_text_backbone_configs['name'])
        self.video_text_backbone = video_text_backbone_cls(video_text_backbone_configs)
        self.max_stride = self.video_text_backbone.max_stride

        frame_encoder_cls = META_ARCH_REGISTRY.get(frame_encoder_configs['name'])
        self.frame_encoder = frame_encoder_cls(frame_encoder_configs,
                                                multiscale_shapes=self.video_text_backbone.multiscale_shapes,
                                                text_dim=self.video_text_backbone.text_dim,
                                                fusion_module=self.fusion_module)
        
        frame_decoder_cls = META_ARCH_REGISTRY.get(frame_decoder_configs['name'])
        self.frame_decoder = frame_decoder_cls(frame_decoder_configs,)
        
        temporal_encoder_cls = META_ARCH_REGISTRY.get(temporal_encoder_configs['name'])
        self.temporal_encoder = temporal_encoder_cls(temporal_encoder_configs, fusion_module=self.fusion_module)
        
        temporal_decoder_cls = META_ARCH_REGISTRY.get(temporal_decoder_configs['name'])
        self.temporal_decoder = temporal_decoder_cls(temporal_decoder_configs, reason_module=self.reason_module)
 
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
        log_lr_group_idx = {'backbone':[], 'base':[]}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_text_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if len(log_lr_group_idx['backbone']) == 0:
                        log_lr_group_idx['backbone'].append(len(params))

                else:
                    if len(log_lr_group_idx['base']) == 0:
                        log_lr_group_idx['base'].append(len(params))  
                                     
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
        RVOS_TrainAPI_referent_text_clipped_video
        videos = to_device(batch_dict.pop('video_dict')['videos'], self.device) # 0-1, float, b t 3 h w
        videos = (videos - self.pixel_mean) / self.pixel_std
        text_dict = to_device(batch_dict.pop('refer_dict'), self.device)
        targets = to_device(batch_dict.pop('targets'), self.device)
        frame_targets = to_device(batch_dict.pop('frame_targets'), self.device)
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch

        batch_size, T, _, H, W = videos.shape
        
        # 抽特征
        multiscales, text_inputs = self.video_text_backbone(videos=videos,  text_dict=text_dict)  # b c t h w
        # 对比学习
        # self.video_text_backbone.compute_loss(multiscales, text_inputs)

        fencoded_multiscales, fencoded_text_inputs = \
            self.frame_encoder(multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in multiscales.items()}, 
                               text_inputs=text_inputs.clone())

        # bt c h w -> {'queries': list[b t nqf c], 'pred_masks': b t n h w, 'pred_boxes': b t n 4, 'pred_class': b t n class+1}
        frame_decoder_output, _ = self.frame_decoder(video_multiscales={scale_name:scale_feat.clone() for scale_name, scale_feat in fencoded_multiscales.items()}, 
                                                                        text_inputs=fencoded_text_inputs.clone())

        frame_queries_by_layer = frame_decoder_output['frame_queries'][(-1 * self.num_frame_decoder_layers_used):] # list[ b t nq c ]
        
        # b t nqf c -> L b t nqf c -> Lb t nqf c
        frame_queries = torch.stack(frame_queries_by_layer, dim=0).flatten(0, 1)

        # Lb t nqf c, 
        frame_queries, tencoded_text_inputs = self.temporal_encoder(frame_queries.clone(), 
                                                                    fencoded_text_inputs.repeat(self.num_frame_decoder_layers_used))
        # Lb t nqf c -> {'queries': list[Lb nq c], 'predictions': }
        temporal_decoder_output, text_inputs = self.temporal_decoder(frame_queries, text_inputs=text_inputs, multiscales=multiscales)


        loss_value_dict = self.frame_decoder.compute_loss(frame_decoder_output, frame_targets)
        loss_value_dict.update(self.temporal_decoder.compute_loss(temporal_decoder_output, targets))

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
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
        RVOS_EvalAPI_referent_text_clipped_video_request_ann
        videos = to_device(batch_dict.pop(['video_dict'])['videos'], self.device) # B T 3 H W
        assert len(videos) == 1
        text_dict = to_device(batch_dict.pop(['refer_dict']), self.device)
        visualize_paths = batch_dict.pop('visualize_paths') # list[True/False], batch
        orig_t, orig_h, orig_w = batch_dict.pop('orig_size')[0]

        batch_size, T, _, H, W = videos.shape
        # {'res2', 'res3', 'res4', 'res5'}
        # b t c h w
        multiscales, text_dict = self.video_text_backbone(videos=videos, text_dict=text_dict) 
        # b t c h w
        multiscales, text_inputs = self.frame_encoder(multiscales=multiscales, text_inputs=text_inputs,)

        # bt c h w -> {'queries': list[bt nqf c], 'predictions': }
        frame_decoder_output, text_inputs \
            = self.frame_decoder(multiscales={key:value.clone().flatten(0,1) for key, value in multiscales.items()}, 
                                       text_inputs=text_inputs)
        
        frame_queries = frame_decoder_output[-1]['queries']
        frame_queries = rearrange(frame_queries, '(b t) nqf c -> b t nqf c', b=batch_size, t=T)

        frame_queries, text_dict = self.temporal_encoder(frame_queries, text_dict)

        # Lb t nqf c -> {'queries': list[b nq c], 'predictions': ['masks': b nq t h w } }
        temporal_decoder_output, text_inputs = self.temporal_decoder(frame_queries, text_inputs=text_inputs, multiscales=multiscales)        

        pred_boxes = temporal_decoder_output['predictions']['boxes'][0] # nq t 4, cxcywh, 0-1
        pred_boxes = pred_boxes * (torch.tensor([W, H, W, H])[None, None, :]) # cxcywh
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes) # x1y1x2y2
        
        # unpad
        padded_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        pred_masks = temporal_decoder_output['predictions']['masks'][0][:, :orig_t, :orig_h, :orig_w] # nq t h w
        pred_boxes = temporal_decoder_output['predictions']['boxes'][0][:, :orig_t, :] # nq t 4, x1y1x2y2, 0-1
        pred_boxes[:, :, 0::2].clamp_(min=0, max=orig_w) # unpad对于box来说就相当于clip
        pred_boxes[:, :, 1::2].clamp_(min=0, max=orig_h) # clip box

        # unnormalize
        orig_video = Trans_F.normalize(padded_video, [0, 0, 0], 1 / self.std)
        orig_video = Trans_F.normalize(orig_video, -self.mean, [1, 1, 1])
        
        # 输出应该进入aug callback的接口
        return {
            'video': orig_video,
            'pred_masks': pred_masks,
            'pred_boxes': pred_boxes,
        }

    @torch.no_grad()
    def tta_sample(self, asembles, visualize_dir=None):
        # test time agumentation 
        preds_by_aug = []
        for sample in asembles:
            preds_by_aug.append(self.sample(asembles, visualize_dir=visualize_dir))
        return preds_by_aug


@register_model
def decode_frame_query(configs, device):
    model =  Decode_FrameQuery(configs)
    model.to(device)
    params_group, log_lr_group_idx = Decode_FrameQuery.get_optim_params_group(model=model, configs=configs)
    optimizer = get_optimizer(params_group, configs)
    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx


# model_aux_mapper(mode=train/evaluate, )
# model_aux_collate_fn(mode=train/evaluate, )

class AMR_Grounding_3DObj(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                 work_dir=None,
                 mode=None,
                loss_weight={},
                tasks = { 'tempdecoder':{}},
                pixel_mean = [0.485, 0.456, 0.406],
                pixel_std = [0.229, 0.224, 0.225],
                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {},
                 temporal_decoder = {},
                reason_module_3d={},
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.mode = mode
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        from models.pretrained_video_instance_decoder import pt_3d_obj_decoder_entrypoint
        create_temporal_decoder = pt_3d_obj_decoder_entrypoint(temporal_decoder['name'])
        self.temporal_decoder = create_temporal_decoder(temporal_decoder, pt_dir, work_dir)
        self.temporal_decoder_num_layers = self.temporal_decoder.num_layers # 4层
        self.temporal_decoder_mask_out_stride = self.temporal_decoder.mask_out_stride
        self.temporal_decoder_mask_threshold = self.temporal_decoder.mask_threshold
        if mode == 'rvos测试预训练vis':
            pass
        elif mode == '只训练rvos':
            self.reason_3d_choose = reason_module_3d['choose_who']
            self.reason_3d_layer_if_reason = reason_module_3d['layer_if_reason'] # obj_decoder的每层是否reason
            assert self.reason_3d_layer_if_reason[-1]
            assert len(self.reason_3d_layer_if_reason) == self.temporal_decoder_num_layers
            from .layer_graph import graphLayer_entrypoint
            create_reason_module = graphLayer_entrypoint(reason_module_3d['graph']['name'])
            self.reason_module_3d = create_reason_module(reason_module_3d['graph'])
        else:
            raise ValueError()
        
    @property
    def device(self):
        return self.pixel_mean.device

    def encode_text(self, text_queries, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_auxiliary['text_token_ids'] # b smax
        text_tok_splits = text_auxiliary['text_token_splits'] # list[list[int]], batch
        text_feats = self.amrbart_wordEmbedding(text_tokens) # b smax c
        text_feats = self.amrtext_wordEmbedding_proj(text_feats) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        node_alignments = text_auxiliary['node_alignments']
        return amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_ = samples.tensors.shape
        device = samples.tensors.device

        amrs, amr_token_feats, amr_token_seg_ids, text_feats, text_pad_masks, node_alignments = self.encode_text(text_queries, auxiliary, device)
        # list[bt nq c], num_layers,  obj_queries
        # list[bt nq h w], num_layers,  pred_masks
        # TODO: 添加text 特征变换
        temporal_decoder_output, amr_token_feats, text_feats = self.temporal_decoder(samples, 
                                                                                     
                                                                                    amrs=amrs, 
                                                                                    amr_token_feats=amr_token_feats,
                                                                                    amr_token_seg_ids=amr_token_seg_ids, 
                                                                                    text_feats=text_feats, 
                                                                                    text_pad_masks=text_pad_masks)
         # l b nq c, l b t nqf c, l b nq T nqf
         # l b t nq h w,
        temporal_queries_by_layer, frame_queries_by_layer, cross_attn_weights_by_layer,\
              pred_masks_by_layer, multiscale_feats = temporal_decoder_output['video_queries'], temporal_decoder_output['frame_queries'], \
                                                            temporal_decoder_output['cross_attn_weights'],\
                                                         temporal_decoder_output['pred_masks'], temporal_decoder_output['multiscale_feats']

        if self.mode == 'rvos测试预训练vis':
            return {'tempdecoder': {'pred_masks': pred_masks_by_layer,}}
        elif self.mode == '只训练rvos':  
            grounding_score_by_layer = []
            for layer_idx, (frame_queries, temporal_queries, cross_attn_weights) in enumerate(zip(frame_queries_by_layer, 
                                                                                                  temporal_queries_by_layer, 
                                                                                                  cross_attn_weights_by_layer)):
                if self.reason_3d_layer_if_reason[layer_idx]:
                    grounding_score = self.reason_module_3d(temporal_queries=temporal_queries, 
                                                            frame_queries=frame_queries,
                                                             cross_attn_weights=cross_attn_weights, 
                                                             amrs=amrs,
                                                             amr_token_feats=amr_token_feats,
                                                             amr_token_seg_ids=amr_token_seg_ids,
                                                             node_alignments=node_alignments,
                                                             text_feats=text_feats,
                                                             text_pad_masks=text_pad_masks) # list[vi nq]
                    
                    if self.reason_3d_choose == '第一个':
                        grounding_score = torch.stack([lg[0] for lg in grounding_score], dim=0)
                    else:
                        raise ValueError()
                    grounding_score_by_layer.append(torch.stack(grounding_score, dim=0))
                else:
                    grounding_score_by_layer.append(None)

            return {'temporal_decoder': {'pred_masks': pred_masks_by_layer, # list[b nq h w]
                                   'reason_3d': grounding_score_by_layer} ,} # list[b nq h w], num_layers

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary)
        # b nq T h w
        out_mask_logits = decoder_layer_preds['objdecoder_objseg'].permute(0,2,1,3,4)
        if self.is_pretraining_seg:
            for idx in range(batch_size):
                h, w = targets[idx]['masks'].shape[-2:]
                # n t h w -> n t H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
            # list[n t h w]
            tgt_masks = [targets[idx]['masks'] for idx in range(batch_size)]
            for btc_idx in range(batch_size):
                start = int(self.obj_decoder_mask_out_stride // 2)
                im_h, im_w = tgt_masks[btc_idx].shape[-2:]
                tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
                assert tgt_masks[btc_idx].size(2) * self.obj_decoder_mask_out_stride == im_h
                assert tgt_masks[btc_idx].size(3) * self.obj_decoder_mask_out_stride == im_w

            gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
            _, matching_result = self.obj_decoder_objseg_loss(out_mask_logits, perFrame_has_ann, tgt_masks)
            # list[t h w] -> b t h w
            out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]][has_ann]
                                            for out_mask, gt_ref_idx, (tgt_idx, src_idx), has_ann in zip(out_mask_logits, gt_referent_idx, matching_result,
                                                                                                         perFrame_has_ann)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1) # bt h w
        else:
            refdecoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
            ref_last_layer_preds = refdecoder_layer_preds[f'layer{self.decoder_trans_nlayers-1}_preds']
            ref_last_layer_gscore = ref_last_layer_preds['grounding_score']  # b nq 
            argmax_query_idx = ref_last_layer_gscore.argmax(-1)
            # b T h w
            out_mask_logits = torch.stack([out_mask[max_query] for out_mask, max_query in zip(out_mask_logits, argmax_query_idx)], dim=0)
            out_mask_logits = out_mask_logits.flatten(0,1)[perFrame_has_ann.flatten()]
        # # # bt 1 h w
        query_pred_masks = F.interpolate(out_mask_logits.unsqueeze(1), 
                                         scale_factor=self.obj_decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.obj_decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
  

    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape

        new_targets = self.rvos_targets_handler(targets, pad_h=H, pad_W=W)
        model_outs = self.model_outputs(samples, text_queries, auxiliary) 
        loss_value_dict = self.temporal_decoder_loss(model_outs, new_targets)

        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    def rvos_targets_handler(self, targets, pad_T, pad_H, pad_W):
        batch_size = len(targets)
        tgt_masks = [] 
        # list[ni t' h w] -> list[ni t' H W]
        for idx in range(batch_size):
            _, _, h, w = targets[idx]['masks'].shape # ni t' h w
            tgt_masks.append(F.pad(targets[idx]['masks'].float(), pad=(0, pad_W-w, 0, pad_H-h)).bool())

        for btc_idx in range(batch_size):
            start = int(self.temporal_decoder_mask_out_stride // 2)
            im_h, im_w = tgt_masks[btc_idx].shape[-2:]
            tgt_masks[btc_idx] = tgt_masks[btc_idx][:, :, start::self.temporal_decoder_mask_out_stride, start::self.temporal_decoder_mask_out_stride] 
            assert tgt_masks[btc_idx].size(2) * self.temporal_decoder_mask_out_stride == im_h
            assert tgt_masks[btc_idx].size(3) * self.temporal_decoder_mask_out_stride == im_w

        gt_referent_idx = [targets[idx]['referent_idx'] for idx in range(batch_size)] # list[int], b
        
        # list[ni]
        is_valid = [tgt_m.flatten(1).any(-1) for tgt_m in tgt_masks]

        perFrame_has_ann = [t['has_ann'] for t in targets] # list[t_video_i]
        # list[t_video_i] -> list[T]
        perFrame_has_ann = [F.pad(t.float(), pad=(0, pad_T-len(t))).bool() for t in perFrame_has_ann]  

        return {
            'masks': tgt_masks,
            'is_valid': is_valid,
            'referent_idx': gt_referent_idx,
            'has_ann': perFrame_has_ann
        }

    def temporal_decoder_loss(self, model_outs, targets):
        loss_layer_weights = self.tasks['temporal_decoder']['loss_layer_weights']
        tgt_masks = targets['masks'] # list[ni t' H W]
        referent_idxs = targets['referent_idx'] # list[int]
        is_valid = targets['is_valid'] # list[ni]
        num_objs = sum([is_v.int().sum() for is_v in is_valid])
        num_objs = torch.as_tensor([num_objs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objs)
        num_objs = torch.clamp(num_objs / get_world_size(), min=1).item()

        num_refs = sum([is_v[ref_idx].int() for is_v, ref_idx in zip(is_valid, referent_idxs)])
        num_refs = torch.as_tensor([num_refs], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_refs)
        num_refs = torch.clamp(num_refs / get_world_size(), min=1).item()
        
        loss_value = {'tempdecoder_mask': torch.tensor(0, device=self.device).float(), 
                      'tempdecoder_dice': torch.tensor(0, device=self.device).float(),
                      'tempdecoder_reason': torch.tensor(0, device=self.device).float(),}
        
        out_mask_logits = model_outs['temporal_decoder']['pred_masks'] # list[b nq T h w], num_layers
        out_gscores = model_outs['temporal_decoder']['reason_3d'] # list[b nq], num_layers     

        for layer_idx, (layer_mask_output, layer_gscore_output, layer_loss_weight) in enumerate(zip(out_mask_logits, out_gscores, loss_layer_weights)):
            if layer_loss_weight != 0:
                matching_indices = self.temporal_decoder_matching(layer_mask_output, targets)
                if self.loss_weight['tempdecoder_mask'] != 0 or self.loss_weight['tempdecoder_dice'] !=0:
                    assert layer_mask_output is not None
                    masks_losses = self.temporal_decoder_masks_loss(layer_mask_output, targets, matching_indices, num_objs)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_loss_weight * masks_losses[k]
                if (self.loss_weight['tempdecoder_reason'] != 0) and (layer_gscore_output is not None):
                    reason_loss = self.temporal_reason_loss(layer_gscore_output, targets, matching_indices, num_refs)
                    for k in reason_loss.keys():
                        loss_value[k] += layer_loss_weight * reason_loss[k]
        return loss_value     

    @torch.no_grad()
    def temporal_decoder_matching(self, out_mask_logits, targets):
        perFrame_has_ann = targets['has_ann'] # list[T]
        tgt_masks = targets['masks'] # list[ni t' H W]
        src_masks_logits = out_mask_logits  # b nq T h w
        batch_size, nq, T, h, w = src_masks_logits.shape 
        indices = [] 
        for i in range(batch_size):
            out_mask = src_masks_logits[i]  # nq T h w
            out_mask = out_mask[:, perFrame_has_ann[i]] # nq t' h w
            tgt_mask = tgt_masks[i].to(out_mask) # ni t' H W
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq ni

            C = self.tasks['temporal_decoder']['objseg_matching_costs']['mask'] * cost_mask + \
                self.tasks['temporal_decoder']['objseg_matching_costs']['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def temporal_decoder_masks_loss(self, out_mask_logits, targets, matching_indices, num_objs):
        has_anntation = targets['has_ann'].bool() # list[T]
        is_valid = targets['is_valid'].bool() # list[ni]
        # b nq T H W -> list[ni t' H W]
        src_masks = [t[J][:, has_ann] for t, (J, _), has_ann in zip(out_mask_logits, matching_indices, has_anntation)]
        
        # list[ni t' H W], b 
        tgt_masks = [t[J] for t, (_, J) in zip(targets['masks'], matching_indices)]
        
        # 每个视频可能有不同的annotation数量
        # list[ni] -> n_sigma
        masks_losses = torch.cat([self.binary_cross_entropy_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)
        dice_losses = torch.cat([self.dice_mask_loss(src_m[is_v], tgt_m[is_v]) for src_m, tgt_m, is_v in zip(src_masks, tgt_masks, is_valid)], dim=0)

        losses = {
            "tempdecoder_mask": masks_losses.sum() / num_objs,
            "tempdecoder_dice": dice_losses.sum() / num_objs,
        }
        return losses    

    def binary_cross_entropy_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, T, -> list[cross_entropy], n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1) # n thw

        ce_loss = F.binary_cross_entropy_with_logits(src_masks, tgt_masks.float(), reduction="none")
        ce_loss = ce_loss.mean(-1) # n
        return ce_loss
    
    def dice_mask_loss(self, src_masks, tgt_masks):
        # n T h w, n t h w, -> n
        src_masks = src_masks.flatten(1) # n thw
        tgt_masks = tgt_masks.flatten(1).float() # n thw

        src_masks = src_masks.sigmoid()
        numerator = 2 * ((src_masks * tgt_masks).sum(1))
        denominator = src_masks.sum(-1) + tgt_masks.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss


    def temporal_reason_loss(self, layer_gscore_output, targets, matching_indices, num_refs):
        # b nq
        referent_idx = targets['referent_idx'] # list[int], batch
        is_valid = targets['is_valid'].bool() # list[ni]
        ref_is_valid = torch.cat([is_v[ref_idx] for is_v, ref_idx in zip(is_valid, referent_idx)]) # b
        match_as_gt_indices = [] # list[int], b
        for ref_idx, (src_idx, tgt_idx) in zip(referent_idx,  matching_indices): # b
            sel_idx = tgt_idx.tolist().index(ref_idx)
            match_as_gt_idx = src_idx[sel_idx]
            match_as_gt_indices.append(match_as_gt_idx.item())
        match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(self.device) # b
        choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
        return {'tempdecoder_reason': choose_loss.sum() / num_refs}


