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
from .registry import vos_metric_entrypoint
from data_schedule.registry import VOS_EvalAPI_clipped_video_request_ann, VOS_EvalAPI_output
# 添加Test-TIme augmentation
@EVALUATOR_REGISTRY.register()
class VOS_Evaluator:
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
            metric_fn = vos_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, 
                                dataset_meta=dataset_meta,
                                **metric_config)
            metric_fns.append(metric_fn)
            metric_names.append(metric_name)
        self.metric_fns = metric_fns
        self.metric_names = metric_names

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    def __call__(self, model, output_dir):
        # evaluator_path: epc1_iter500_sap8099/eval_dataset1/
        # epc1_iter500_sap8099/eval_dataset1/visualize_model/meta_0
        # epc1_iter500_sap8099/eval_dataset1/metric1/config_web_epoch.zip, images
        # epc1_iter500_sap8099/eval_dataset1/metric2/
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        model_preds = []
        for batch_dict in tqdm(self.loader):
            VOS_EvalAPI_clipped_video_request_ann
            # meta
            eval_metas = batch_dict.pop('metas')
            request_anns = eval_metas['request_ann'][0] # t, bool tensor
            frame_strs = eval_metas['frames'][0] # t', list[str]
            video_id = eval_metas['video_id'][0] # str
            assert request_anns.int().sum() == len(frame_strs)
            callback_fns = eval_metas['callback_fns'][0] # list[fn]

            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'],
                                                 visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model')) # 模型的可视化
            batch_dict['visualize_paths'] = visualize_path
            VOS_EvalAPI_output
            predictions = model.sample(batch_dict) 
            predictions = {
                'video': predictions['video'][0].cpu(), # t 3 h w
                'pred_masks': predictions['pred_masks'][0].cpu(), # list[nt h w], t, 是logits
                'pred_obj': predictions['pred_obj'][0].cpu(), # list[nt] t, 0-1
            }
            for cardib in callback_fns:
                predictions = cardib(predictions)   
            # list[nt h w], t'
            pred_masks = [pred_mk for idx, pred_mk in enumerate(predictions['pred_masks']) if request_anns[idx]]
            # list[nt], t'
            pred_obj = [pred_p for idx, pred_p in enumerate(predictions['pred_obj']) if request_anns[idx]]
            assert len(frame_strs) == len(pred_masks)

            for frame_name, frame_masks_pred, frame_prob_pred in zip(frame_strs, pred_masks, pred_obj):
                # nt h w
                frame_masks_pred = frame_masks_pred.sigmoid() > 0.5
                # list[rle], nq
                rle_masks = [mask_util.encode(np.array(mask.numpy()[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                        for mask in frame_masks_pred]
                model_preds.append(
                    {
                        'video_id': video_id,
                        'frame_name': frame_name,
                        'masks': rle_masks, # list[rle], nq
                        'scores': frame_prob_pred.tolist(), # nq
                    }
                )

        model_preds = all_gather(model_preds)
        model_preds = [itm for item in model_preds for itm in item ]
            
        eval_metrics = {}
        if comm.is_main_process():
            for metric_fn in self.metric_fns:
                eval_metrics.update(metric_fn(model_preds=model_preds, 
                                              output_dir=evaluator_path))
        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class VOS_Evaluator_TTA:
    pass

# multi-model ensemble
class VOS_Evaluator_MM:
    pass
