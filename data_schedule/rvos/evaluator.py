from typing import Optional, Union
import os
from glob import glob
from tqdm import tqdm
import shutil
from functools import partial
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import detectron2.utils.comm as comm
import logging
from utils.misc import is_dist_avail_and_initialized, all_gather, to_device
from detectron2.data import  MetadataCatalog
from data_schedule.registry import EVALUATOR_REGISTRY
from .evaluator_utils import rvos_metric_entrypoint
from data_schedule.rvos.apis import RVOS_Aug_CallbackAPI, RVOS_EvalAPI_referent_text_clipped_video_request_ann
from collections import defaultdict


# 添加Test-TIme augmentation
@EVALUATOR_REGISTRY.register()
class RVOS_Evaluator:
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
            metric_fn = rvos_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.frame_metric_fns.append(metric_fn)

        self.connect_vidText_fn = dataset_meta.get('connect_vidText_fn')
        self.eval_meta_keys = dataset_meta.get('eval_meta_keys') # videoText: frame

        metrics_aggregator = configs['data']['evaluate'][dataset_name]['evaluator']['metrics_aggregator']
        self.metrics_aggregator = partial(rvos_metric_entrypoint(metrics_aggregator[0]),
                                                                dataset_meta=dataset_meta,
                                                                eval_meta_keys=self.eval_meta_keys,
                                                                **metrics_aggregator[1])
    
    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        metric_dict_by_vidText_frame = defaultdict(dict) # vidText: frame1: {'iou':, ...}
        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas')
            request_anns = eval_metas['request_ann'][0] # t, bool tensor
            frame_strs = eval_metas['frames'][0] # t', list[str]
            video_id = eval_metas['video_id'][0] # str
            refer_text = batch_dict['refer_dict']['texts'][0]
            exp_id = eval_metas['exp_id'][0] #  str
            videoText_key = self.connect_vidText_fn(video_id, exp_id)
            assert request_anns.int().sum() == len(frame_strs)
            callback_fns = eval_metas['callback_fns'][0] # list[fn]
            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'], visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model')) 
            batch_dict['visualize_paths'] = visualize_path
            batch_dict = to_device(batch_dict, device=model.device)
            model_outputs = model.sample(batch_dict)  # {'video', 'pred_masks', 'pred_boxes', 'pred_logits'} # t 3 h w, nq t h w, nq t 4, nq t
            predictions = {
                'video': model_outputs['video'][0], # t 3 h w  
                'pred_masks': [haosen for idx, haosen in enumerate(model_outputs['pred_masks'][0]) if request_anns[idx]], # list[nq h w], t'
                'pred_class': [haosen for idx, haosen in enumerate(model_outputs['pred_class'][0]) if request_anns[idx]], # list[nq c], t', 
            }  
            if 'pred_boxes' in model_outputs:
                predictions.update({'pred_boxes':  [haosen for idx, haosen in enumerate(model_outputs['pred_boxes'][0]) if request_anns[idx]]}) # # list[nq 4], t,
            for cardib in callback_fns:
                predictions = cardib(predictions) 
            pred_masks = predictions['pred_masks']
            pred_class = predictions['pred_class']
            assert len(frame_strs) == len(pred_masks)

            for idx, (fname, fmk, fclass) in enumerate(zip(frame_strs, pred_masks, pred_class)):
                frame_pred = {'masks': fmk, 'classes': fclass.tolist(), 'video_id': video_id, 'frame_name': fname, 'exp_id': exp_id,
                              'text': refer_text}
                if 'pred_boxes' in predictions:
                    frame_pred.update({'boxes': predictions['pred_boxes'][idx]})

                meta_key_metrics = {}                
                for metric_fn in self.frame_metric_fns:
                    metric_values = metric_fn(frame_pred=frame_pred, output_dir=evaluator_path)
                    for key, value in metric_values.items():
                        assert key not in meta_key_metrics
                        meta_key_metrics[key] = value

                # assert fname not in metric_dict_by_vidText_frame[videoText_key]
                metric_dict_by_vidText_frame[videoText_key][fname] = meta_key_metrics

        metric_dict_by_vidText_frame = comm.gather(dict(metric_dict_by_vidText_frame), dst=0)
        eval_metrics = {}
        if comm.is_main_process():  # 合并多个进程的结果
            metrics_by_videoText = {} # videotext:frame : value
            for videoText_id in tqdm(self.eval_meta_keys.keys(), desc='gathering different processes'):
                videoTextid_metrics = [haosen[video_id] for haosen in metric_dict_by_vidText_frame if videoText_id in haosen]

                videoTextid_frame_names = [list(haosen.keys()) for haosen in videoTextid_metrics]
                merged_videoTextid_frame_names = [item for sublist in videoTextid_frame_names for item in sublist]
                assert len(set(merged_videoTextid_frame_names)) == len(merged_videoTextid_frame_names),'request annotation overlayed'
                # assert set(merged_videoTextid_frame_names).issubset(set(self.eval_meta_keys[video_id]))
                assert set(self.eval_meta_keys[videoText_key]).issubset(set(merged_videoTextid_frame_names))

                vid_frame_metrics = videoTextid_metrics[0]
                for haosen in videoTextid_metrics[1:]:
                    vid_frame_metrics.update(haosen)  # frame1: iou,  frame2: iou
                # filter out unrequested frames
                vid_frame_metrics = {key: value for key, value in vid_frame_metrics if key in self.eval_meta_keys[videoText_key]}
                metrics_by_videoText[videoText_key] = vid_frame_metrics   

            eval_metrics = self.metrics_aggregator(metrics_by_videoText, output_dir=evaluator_path)
        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class RVOS_Evaluator_TTA:
    pass

# multi-model ensemble
class RVOS_Evaluator_MM:
    pass
