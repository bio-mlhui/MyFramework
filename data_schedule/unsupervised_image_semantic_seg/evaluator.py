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
from .evaluator_utils import metric_entrypoint
import json
from collections import defaultdict
# TODO: 添加Test-TIme augmentation
import torch.nn.functional as F
from .evaluator_utils import UnsupervisedMetrics, get_metrics, RunningAverage

@EVALUATOR_REGISTRY.register()
class UN_IMG_SEM_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        self.num_classes = dataset_meta.get('num_classes')
        eval_configs = configs['data']['evaluate'][dataset_name]['evaluator']
        self.extra_clusters = eval_configs['extra_clusters']
        self.is_direct = eval_configs['is_direct']
        self.is_crf = eval_configs['is_crf']
        assert self.is_crf
        
    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        if comm.is_main_process():
            # make sure mode is in eval mode
            model.eval()
            cluster_metrics = UnsupervisedMetrics("Cluster_", self.num_classes, self.extra_clusters, True)
            linear_metrics = UnsupervisedMetrics("Linear_", self.num_classes, 0, False) 
            eval_stats = RunningAverage()
                    
            evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
            os.makedirs(evaluator_path, exist_ok=True)
            
            for batch_dict in tqdm(self.loader):
                
                eval_metas = batch_dict.pop('metas') # image_ids: list[str]
                label: torch.Tensor = batch_dict['masks'].to(model.device, non_blocking=True)
                start = time.time()
                model_out = model.sample(batch_dict)
                print(time.time() - start)
                linear_preds, cluster_preds, cluster_loss = model_out['linear_preds'], model_out['cluster_preds'], model_out['cluster_loss']
                linear_metrics.update(linear_preds, label)
                cluster_metrics.update(cluster_preds, label)
                
                eval_stats.append(cluster_loss)             
            eval_metrics = get_metrics(cluster_metrics, linear_metrics)
            eval_metrics.update({'cluster_loss': eval_stats.avg})
        else:
            eval_metrics = {}
        comm.synchronize()
        return eval_metrics

