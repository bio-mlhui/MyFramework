from typing import Optional, Union
import os
from glob import glob
from tqdm import tqdm
import shutil
from functools import partial
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.distributed as dist
import detectron2.utils.comm as comm
from utils.misc import is_dist_avail_and_initialized, all_gather, to_device
import logging
from detectron2.data import  MetadataCatalog
from data_schedule.registry import EVALUATOR_REGISTRY
import time
from .evaluator_utils import vis_metric_entrypoint
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann, VIS_Aug_CallbackAPI, VIS_Dataset, VIS_Evaluator_OutAPI_EvalFn_API
import json
from collections import defaultdict
# TODO: 添加Test-TIme augmentation
@EVALUATOR_REGISTRY.register()
class VIS_Evaluator_FrameFast:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        self.dataset_name = dataset_name
        self.loader = data_loader
        frame_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['frame_metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.frame_metric_fns = []
        for metric_name, metric_config in frame_metrics:
            metric_fn = vis_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.frame_metric_fns.append(metric_fn)
            
        self.eval_meta_keys = dataset_meta.get('eval_meta_keys') 

        metrics_aggregator = configs['data']['evaluate'][dataset_name]['evaluator']['metrics_aggregator']
        self.eval_meta_keys = dataset_meta.get('eval_meta_keys')  # { video_id: list[fnames] }
        self.metrics_aggregator = partial(vis_metric_entrypoint(metrics_aggregator[0]),
                                                dataset_meta=dataset_meta,
                                                eval_meta_keys=self.eval_meta_keys,
                                                **metrics_aggregator[1])

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        macs, params = None, None
        metrics_by_video_id_frame = defaultdict(dict) 
        for batch_dict in tqdm(self.loader):
            VIS_EvalAPI_clipped_video_request_ann
            eval_metas = batch_dict.pop('metas')
            request_anns = eval_metas['request_ann'][0] # t, bool tensor
            frame_strs = eval_metas['frames'][0] # t', list[str]
            video_id = eval_metas['video_id'][0] # str
            assert request_anns.int().sum() == len(frame_strs)
            callback_fns = eval_metas['callback_fns'][0] # list[fn]
            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'], visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model')) # 模型的可视化
            batch_dict['visualize_paths'] = visualize_path
            batch_dict = to_device(batch_dict, device=model.device)
            VIS_Aug_CallbackAPI
            # if macs is None:
            #     from detectron2.utils.analysis import (
            #         FlopCountAnalysis,
            #     )
            #     flops = FlopCountAnalysis(model, batch_dict, inference_func=lambda model, *inputs: model.sample(*inputs))
            #     total_flops = flops.total()
            #     # counts = flops.by_operator()
            #     logging.debug(f'macs: {total_flops/ (10**9) / len(request_anns)}')
            model_outputs = model.sample(batch_dict) 
            predictions = {
                'video': model_outputs['video'][0], # t 3 h w  
                'pred_masks': [haosen for idx, haosen in enumerate(model_outputs['pred_masks'][0]) if request_anns[idx]], # list[nt h w], t'
                'pred_class': [haosen for idx, haosen in enumerate(model_outputs['pred_class'][0]) if request_anns[idx]], # list[nt c], t', 
            }  
            if 'pred_boxes' in model_outputs: # nt代表的是objects而不是semantics 
                predictions.update({'pred_boxes':  [haosen for idx, haosen in enumerate(model_outputs['pred_boxes'][0]) if request_anns[idx]]}) # # list[nt 4], t,
            for cardib in callback_fns:
                predictions = cardib(predictions) 
            pred_masks = predictions['pred_masks']
            pred_class = predictions['pred_class']
            assert len(frame_strs) == len(pred_masks)

            for idx, (fname, fmk, fclass) in enumerate(zip(frame_strs, pred_masks, pred_class)):
                VIS_Evaluator_OutAPI_EvalFn_API
                frame_pred = {'masks': fmk, 'classes': fclass.tolist(), 'video_id': video_id, 'frame_name': fname}
                if 'pred_boxes' in predictions:
                    frame_pred.update({'boxes': predictions['pred_boxes'][idx]})

                meta_key_metrics = {}                
                for metric_fn in self.frame_metric_fns:
                    metric_values = metric_fn(frame_pred=frame_pred, output_dir=evaluator_path)
                    for key, value in metric_values.items():
                        assert key not in meta_key_metrics
                        meta_key_metrics[key] = value

                assert fname not in metrics_by_video_id_frame[video_id]
                metrics_by_video_id_frame[video_id][fname] = meta_key_metrics

        metrics_by_video_id_frame = comm.gather(dict(metrics_by_video_id_frame), dst=0)
        eval_metrics = {}
        if comm.is_main_process():
            metrics_by_video = {} # video:frame : value
            for video_id in tqdm(self.eval_meta_keys.keys(), desc='gathering different processes'):
                # list[{fname: predictions}]
                video_id_metrics = [haosen[video_id] for haosen in metrics_by_video_id_frame if video_id in haosen]

                video_id_frame_names = [list(haosen.keys()) for haosen in video_id_metrics]
                merged_video_id_frame_names = [item for sublist in video_id_frame_names for item in sublist]
                assert len(set(merged_video_id_frame_names)) == len(merged_video_id_frame_names),'保证frame没有重合'
                assert set(merged_video_id_frame_names).issubset(set(self.eval_meta_keys[video_id]))
                assert set(self.eval_meta_keys[video_id]).issubset(set(merged_video_id_frame_names))

                # perframe metrics frame: predictions
                vid_frame_metrics = video_id_metrics[0]
                for haosen in video_id_metrics[1:]:
                    vid_frame_metrics.update(haosen)
                metrics_by_video[video_id] = vid_frame_metrics

            eval_metrics = self.metrics_aggregator(metrics_by_video)
        comm.synchronize() 
        return eval_metrics


import torch.nn.functional as F
@EVALUATOR_REGISTRY.register()
class Card_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        self.dataset_name = dataset_name
        self.loader = data_loader
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.get_frame_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        all_dices = []
        all_ious = []
        # batch_size = 1
        for batch_dict in tqdm(self.loader):
            VIS_EvalAPI_clipped_video_request_ann
            meta_info = batch_dict['metas'] 
            video_id, frames, request_ann = meta_info['video_id'][0], meta_info['frames'][0], meta_info['request_ann'][0]
            assert request_ann[len(request_ann) // 2]
            assert len(frames) == len(request_ann)
            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'], visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model')) # 模型的可视化
            batch_dict['visualize_paths'] = visualize_path
            batch_dict = to_device(batch_dict, device=model.device)
            gt_masks = self.get_frame_mask_fn(video_id=video_id, mid_frame=frames[len(frames) // 2]) # K h w
            
            model_outputs = model.sample(batch_dict) 
            
            pred_masks = [haosen for idx, haosen in enumerate(model_outputs['pred_masks'][0]) if request_ann[idx]] # list[nt h w], t'
            pred_scores = [haosen for idx, haosen in enumerate(model_outputs['pred_class'][0]) if request_ann[idx]] # list[nt K], t',
            
            assert len(pred_masks) == 1 and len(pred_scores) == 1
            pred_masks = pred_masks[0] # nt h w, logits
            pred_scores = pred_scores[0] # nt K
            # semantic inference
            # K h w
            pred_masks = F.interpolate(pred_masks[None, ...], size=gt_masks.shape[-2:], mode='biliear', align_corners=False)[0]
            
            # K h w, K h w
            dice, iou = computer_dice_iou(pred_masks, gt_masks)
            all_dices.append(dice)
            all_ious.append(iou)

        return {'dice': torch.tensor(all_dices).mean(),
                'iou': torch.tensor(all_ious).mean()}


def computer_dice_iou(pred, gt):
    # K hw
    # K hw， 每个类别的平均值, float
    dices = []
    ious = []
    for cls_id in range(len(pred)):
        inter, union = (pred[cls_id]*gt[cls_id]).sum(), (pred[cls_id]+gt[cls_id]).sum()
        dice = (2*inter+1)/(union+1)
        iou = (inter+1)/(union-inter+1)

        dices.append(dice)
        ious.append(iou)
    
    return torch.tensor(dices).mean(), torch.tensor(ious).mean()