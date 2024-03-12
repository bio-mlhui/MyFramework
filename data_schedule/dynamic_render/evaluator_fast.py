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
from .evaluator_utils import render_metric_entrypoint
import json
from collections import defaultdict

# 模型是初始化的, 先训练然后测试
@EVALUATOR_REGISTRY.register()
class Render_Evaluator_ViewFast: # 每个view测试
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        self.dataset_name = dataset_name
        self.loader = data_loader
        # 渲染出来的图像 和 真实图像的 metrics
        view_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['view_metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.view_metric_fns = []
        for metric_name, metric_config in view_metrics:
            metric_fn = render_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.view_metric_fns.append(metric_fn)

        metrics_aggregator = configs['data']['evaluate'][dataset_name]['evaluator']['metrics_aggregator']
        self.eval_meta_keys = dataset_meta.get('eval_meta_keys')  # { scene_name: list[views] }
        self.metrics_aggregator = partial(render_metric_entrypoint(metrics_aggregator[0]),
                                                                    dataset_meta=dataset_meta,
                                                                    eval_meta_keys=self.eval_meta_keys,
                                                                    **metrics_aggregator[1])
        self.scene = None

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    def __call__(self, output_dir=None, **kwargs):
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)

        metrics_by_scene_view = defaultdict(dict) 
        for scene_dict in tqdm(self.loader): 
            scene_dict = to_device(scene_dict, device=self.scene.device)  

            scene_name = scene_dict['name']         
            # train path
            scene_train = scene_dict['train']
            scene_train['visualize_path'] = self.visualize_path(evaluator_path=os.path.join(evaluator_path, scene_name, 'train'))
            self.scene.initialize(scene_train)
            self.scene.train(scene_train)
            # test model
            scene_eval = scene_dict['eval']
            eval_views = scene_eval['view_points']
            # eval_metas = scene_eval['meta_idxs']
            eval_output_dir = self.visualize_path(evaluator_path=os.path.join(evaluator_path, scene_name, 'eval'))
            scene_eval['visualize_path'] = eval_output_dir
            view_preds = self.scene.evaluate(scene_eval)

            for idx, (view, vimg) in enumerate(zip(eval_views, view_preds)):
                view_pred = {'img': vimg, 'view': view}
                meta_key_metrics = {}                
                for metric_fn in self.eval_meta_keys:
                    metric_values = metric_fn(view_pred=view_pred, output_dir=eval_output_dir)
                    for key, value in metric_values.items():
                        assert key not in meta_key_metrics
                        meta_key_metrics[key] = value

                assert view not in metrics_by_scene_view[scene_name]
                metrics_by_scene_view[scene_name][view] = meta_key_metrics

        metrics_by_scene_view = comm.gather(dict(metrics_by_scene_view), dst=0)  # list[{scene1:, scene2:, scene3:}, {scene4, scene5}]
        eval_metrics = {}
        if comm.is_main_process():
            metrics_by_scene = {} 
            for scene_name in tqdm(self.eval_meta_keys.keys(), desc='gathering different processes'):
                # list[{view:, view:, view:}, {view:, view:, view:}]
                view_metrics = [haosen[scene_name] for haosen in metrics_by_scene_view if scene_name in haosen]
                assert len(view_metrics) == 0
                view_metrics = view_metrics[0] # {'view1', 'view2': }
                list_views = list(view_metrics.keys())
                assert set(list_views).issubset(set(self.eval_meta_keys[scene_name]))
                assert set(self.eval_meta_keys[scene_name]).issubset(set(list_views))

                metrics_by_scene[scene_name] = view_metrics

            eval_metrics = self.metrics_aggregator(metrics_by_scene)
        comm.synchronize() 
        return eval_metrics

