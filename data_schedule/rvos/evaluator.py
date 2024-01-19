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

from detectron2.data import  MetadataCatalog
from data_schedule.registry import EVALUATOR_REGISTRY
from .evaluate_utils import metric_web, metric_coco

# 添加Test-TIme augmentation
@EVALUATOR_REGISTRY.register()
class RVOS_Evaluator:
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
            if metric_name == 'web':
                metric_fn = partial(metric_web,  remove_pngs=metric_config['remove_pngs'])
            elif metric_name == 'coco':
                metric_fn = partial(metric_coco, coco_file=dataset_meta.get('coco_eval_file'))
            else:
                raise ValueError()
            metric_fns.append(metric_fn)
            metric_names.append(metric_name)
        self.metric_fns = metric_fns
        self.metric_names = metric_names

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, 'visualize_model', meta_idx) if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]

    def __call__(self, model, output_dir):
        # evaluator_path: epc1_iter500_sap8099/eval_dataset1/
        # epc1_iter500_sap8099/eval_dataset1/visualize_model/meta_0
        # epc1_iter500_sap8099/eval_dataset1/metric1/config_web_epoch.zip, images
        # epc1_iter500_sap8099/eval_dataset1/metric2/
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        model_preds = []
        for batch_dict in tqdm(self.loader):
            # meta
            request_anns = batch_dict['metas']['request_ann'][0] # t, bool tensor
            frame_strs = batch_dict['metas']['frames'][0] # t', list[str]
            video_id = batch_dict['metas']['video_ids'][0] # str
            exp_id = batch_dict['metas']['exp_ids'][0] #  str
            assert request_anns.int().sum() == len(frame_strs)
            callback_fns = batch_dict['metas']['callback_fns'][0] # list[fn]

            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'],
                                                 visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'model')) # 模型的可视化
            batch_dict['visualize_paths'] = visualize_path
            predictions = model.sample(batch_dict)  # {'video', 'pred_masks', 'pred_boxes', 'pred_logits'} # t 3 h w, nq t h w, nq t 4, nq t

            for cardib in callback_fns:
                predictions = cardib(predictions)

            pred_masks = predictions['pred_masks'][:, request_anns].sigmoid() > 0.5 # nq t' h w
            pred_refer_probs = predictions['pred_logits'][:, request_anns].sigmoid() # nq t'

            assert len(frame_strs) == len(pred_masks)
            assert len(frame_strs) == len(pred_refer_probs)

            # 每一帧多个预测
            for frame_name, frame_masks_pred, frame_prob_pred in zip(frame_strs, pred_masks.permute(0, 1), pred_refer_probs.permute(0, 1)):
                model_preds.append(
                    {
                        'video_id': video_id,
                        'exp_id': exp_id,
                        'frame_name': frame_name,
                        'masks': frame_masks_pred, # list[h w], nq
                        'scores': frame_prob_pred, # nq
                    }
                )

        model_preds = comm.all_gather(model_preds)
        eval_metrics = {}
        if comm.is_main_process():
            for metric_fn in self.metric_fns:
                eval_metrics.update(metric_fn(model_preds=model_preds, 
                                              output_dir=evaluator_path))
        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class RVOS_Evaluator_TTA:
    pass

# multi-model ensemble
class RVOS_Evaluator_MM:
    pass
