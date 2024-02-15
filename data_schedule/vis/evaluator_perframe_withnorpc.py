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
# TODO: 添加Test-TIme augmentation
@EVALUATOR_REGISTRY.register()
class VIS_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        self.dataset_name = dataset_name
        self.loader = data_loader
        metrics = configs['data']['evaluate'][dataset_name]['evaluator']['metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.metric_fns = []
        for metric_name, metric_config in metrics:
            metric_fn = vis_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.metric_fns.append(metric_fn)
            
        self.eval_meta_keys = dataset_meta.get('eval_meta_keys') 

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        
        metric_dict_by_vid_frame = {} # meta_key: {metric_name: scalar, metric_name2: scaler}
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
            predictions = model.sample(batch_dict) 
            predictions = {
                'video': predictions['video'][0], # t 3 h w  
                'pred_masks': predictions['pred_masks'][0], # list[nt h w], t
                'pred_class': predictions['pred_class'][0], # list[nt c] t,
            }
            if 'pred_boxes' in predictions: # nt代表的是objects而不是semantics 
                predictions.update({'pred_boxes':  predictions['pred_boxes'][0]}) # # list[nt 4], t,
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

                meta_key = f'{video_id}_{fname}'
                meta_key_metrics = {}                
                for metric_fn in self.metric_fns:
                    metric_values = metric_fn(frame_pred=frame_pred, output_dir=evaluator_path)
                    for key, value in metric_values.items():
                        assert key not in meta_key_metrics
                        meta_key_metrics[key] = value

                assert meta_key not in metric_dict_by_vid_frame
                metric_dict_by_vid_frame[meta_key] = meta_key_metrics

        metric_dict_by_vid_frame = comm.gather(metric_dict_by_vid_frame, dst=0)
        comm.synchronize()
        eval_metrics = {}
        if comm.is_main_process():
            # list[dict]
            gathered_ret = {}
            for process_pred in metric_dict_by_vid_frame:
                assert len(set(list(process_pred.keys())) & set(list(gathered_ret.keys()))) == 0
                gathered_ret.update(process_pred)
            metric_dict_by_vid_frame = gathered_ret
            assert len(set(list((metric_dict_by_vid_frame.keys()))) - set(self.eval_meta_keys)) == 0
            assert len(set(self.eval_meta_keys) - set(list(metric_dict_by_vid_frame.keys()))) == 0

            eval_metrics = {}
            metric_names = list(list(metric_dict_by_vid_frame.values())[0].keys())
            for taylor_swift in metric_names:
                eval_metrics[taylor_swift] = torch.tensor([metric_dict[taylor_swift] for metric_dict in metric_dict_by_vid_frame.values()]).mean()

        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class VIS_Evaluator_TTA:
    pass

# multi-model ensemble
class VIS_Evaluator_MM:
    pass
