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
from collections import defaultdict
import logging
from detectron2.data import  MetadataCatalog
from data_schedule.registry import EVALUATOR_REGISTRY
import time
from .evaluator_utils import vis_metric_entrypoint
from data_schedule.vis.apis import VIS_EvalAPI_clipped_video_request_ann, VIS_Aug_CallbackAPI, VIS_Dataset, VIS_Evaluator_OutAPI_EvalFn_API
import json
# TODO: 添加Test-TIme augmentation
# 存mask -> 主进程读mask,
@EVALUATOR_REGISTRY.register()
class VIS_Evaluator:
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
        
        video_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['video_metrics']
        self.video_metric_fns = []
        for metric_name, metric_config in video_metrics:
            metric_fn = vis_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.video_metric_fns.append(metric_fn)    

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
        
        predictions_by_video_id_frame = defaultdict(dict)
        # video_id : video_metric1 / frame_name: metric_name1
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
            model_outputs = model.sample(batch_dict) 
            predictions = {
                'video': model_outputs['video'][0], # t 3 h w  
                'pred_masks': model_outputs['pred_masks'][0], # list[nt h w], t
                'pred_class': model_outputs['pred_class'][0], # list[nt c] t,
            }
            if 'pred_boxes' in model_outputs: # nt代表的是objects而不是semantics 
                predictions.update({'pred_boxes':  model_outputs['pred_boxes'][0]}) # # list[nt 4], t,
            for cardib in callback_fns:
                predictions = cardib(predictions) 
            pred_class = [pred_p for idx, pred_p in enumerate(predictions['pred_class']) if request_anns[idx]] # # list[nt c], t'  
            pred_masks = [pred_mk for idx, pred_mk in enumerate(predictions['pred_masks']) if request_anns[idx]] # list[nt h w], t'
            if 'pred_boxes' in predictions:
                pred_boxes = [pred_bx for idx, pred_bx in enumerate(predictions['pred_boxes']) if request_anns[idx]] # list[nt 4], t'
            assert len(frame_strs) == len(pred_masks)

            for idx, (fname, fmk, fclass) in enumerate(zip(frame_strs, pred_masks, pred_class)):
                VIS_Evaluator_OutAPI_EvalFn_API
                frame_pred = {'masks': fmk, 'classes': fclass.tolist(), 'video_id': video_id, 'frame_name': fname}
                if 'pred_boxes' in predictions:
                    frame_pred.update({'boxes': pred_boxes[idx]})
                predictions_by_video_id_frame[video_id][fname] = frame_pred
            
        predictions_by_video_id_frame = comm.gather(dict(predictions_by_video_id_frame), dst=0)
        eval_metrics = {}
        if comm.is_main_process():
            metrics_by_video = {} # video:frame : value
            for video_id in tqdm(self.eval_meta_keys.keys(), desc='gathering different processes'):
                # list[{fname: predictions}]
                video_id_predictions = [taylor[video_id] for taylor in predictions_by_video_id_frame if video_id in taylor]

                video_id_frame_names = [list(taylor.keys()) for taylor in video_id_predictions]
                merged_video_id_frame_names = [item for sublist in video_id_frame_names for item in sublist]
                assert len(set(merged_video_id_frame_names)) == len(merged_video_id_frame_names),'保证frame没有重合'
                assert set(merged_video_id_frame_names).issubset(set(self.eval_meta_keys[video_id]))
                assert set(self.eval_meta_keys[video_id]).issubset(set(merged_video_id_frame_names))

                # perframe metrics frame: predictions
                vid_frame_predictions = video_id_predictions[0]
                for taylor in video_id_predictions[1:]:
                    vid_frame_predictions.update(taylor)
                
                # 默认vid_metrics是空
                metrics_by_frame_vid_metrics = {
                    'vid_metrics': {}
                }
                for fname, frame_pred in vid_frame_predictions.items(): 
                    frame_all_metrics = {}    
                    for metric_fn in self.frame_metric_fns:
                        metric_values = metric_fn(frame_pred=frame_pred, output_dir=evaluator_path)
                        for key, value in metric_values.items():
                            assert key not in frame_all_metrics
                            frame_all_metrics[key] = value
                    metrics_by_frame_vid_metrics[fname] = frame_all_metrics
                
                video_metrics = {}
                for video_metric_fn in self.video_metric_fns:
                    metric_values = video_metric_fn(video_predictions=vid_frame_predictions,
                                                     output_dir=evaluator_path)
                    for key, value in metric_values:
                        assert key not in video_metrics
                        video_metrics[key] = value
                metrics_by_frame_vid_metrics['vid_metrics'] = video_metrics
                metrics_by_video[video_id] = metrics_by_frame_vid_metrics
                
            eval_metrics = self.metrics_aggregator(metrics_by_video)
        comm.synchronize() 
        # del os.environ['PYTORCH_CUDA_ALLOC_CONF']
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class VIS_Evaluator_TTA:
    pass

# multi-model ensemble
class VIS_Evaluator_MM:
    pass
