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
from util.misc import is_dist_avail_and_initialized, all_gather, to_device
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
        metric_fns = []
        metric_names = []
        for metric_name, metric_config in metrics:
            metric_fn = vis_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, 
                                dataset_meta=dataset_meta,
                                **metric_config)
            metric_fns.append(metric_fn)
            metric_names.append(metric_name)
        self.metric_fns = metric_fns
        self.metric_names = metric_names

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        model_preds = []
        for batch_dict in tqdm(self.loader):
            VIS_EvalAPI_clipped_video_request_ann
            # meta
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
                rle_masks = [mask_util.encode(np.array(mask.numpy()[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                        for mask in fmk]
                for mask_idx in range(len(rle_masks)):
                    rle_masks[mask_idx]['counts'] = rle_masks[mask_idx]['counts'].decode('utf-8')
                VIS_Evaluator_OutAPI_EvalFn_API
                frame_pred = {
                    'video_id': video_id,
                    'frame_name': fname,
                    'masks': rle_masks, # list[rle], nt
                    'classes': fclass.tolist(), # nt c
                }
                if 'pred_boxes' in predictions:
                    frame_pred.update({'boxes': pred_boxes[idx]}), # nt 4, x1y1x2y2})
                model_preds.append(frame_pred)
        # logging.debug('Gathering...')
        # start = time.time()
        # model_preds = comm.gather(model_preds, dst=0)   
        # logging.debug(f'Gather done, using {time.time() - start} s')
        with open(os.path.join(evaluator_path, f'preds_rank_{comm.get_rank()}.json'), 'w') as f:
            json.dump(model_preds, f)
        comm.synchronize()
        eval_metrics = {}
        if comm.is_main_process():
            logging.debug('Gathering ...')
            start = time.time()
            all_model_preds = []
            for rank in range(comm.get_world_size()):
                with open(os.path.join(evaluator_path, f'preds_rank_{rank}.json'), 'r') as f:
                    rank_preds = json.load(f)
                    all_model_preds.append(rank_preds)
                os.remove(os.path.join(evaluator_path, f'preds_rank_{rank}.json'))
            logging.debug(f'Gather done, using {time.time() - start} s')

            all_model_preds = [itm for item in all_model_preds for itm in item ]
            for metric_fn in self.metric_fns:
                eval_metrics.update(metric_fn(model_preds=all_model_preds, 
                                              output_dir=evaluator_path))
        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class VIS_Evaluator_TTA:
    pass

# multi-model ensemble
class VIS_Evaluator_MM:
    pass
