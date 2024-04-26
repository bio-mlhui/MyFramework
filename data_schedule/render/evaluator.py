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
from .evaluator_utils import render_metric_entrypoint
import json
from data_schedule.render.apis import Scene_Meta


@EVALUATOR_REGISTRY.register()
class Render_Evaluator:
    def __init__(self,
                 dataset_name=None,
                 dataset_loader=None,
                 configs=None,
                 **kwargs) -> None:
        """
        text_4D: text -> model.sample; metric(4D)
        video_4D: video -> model.sample; metric(4D)
        learning:
            model.sample(text)
        optimization:
            model.sample(video)
        """
        Scene_Meta
        self.dataset_loader = dataset_loader
        self.dataset_name = dataset_name

        # 每一帧的metric, psnr, ssim
        view_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['view_metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.view_metric_fns = []
        for metric_name, metric_config in view_metrics:
            metric_fn = render_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.view_metric_fns.append(metric_fn)
        
        # 整个scene的metric
        repre_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['scene_metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.repre_metric_fns = []
        for metric_name, metric_config in repre_metrics:
            metric_fn = render_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.repre_metric_fns.append(metric_fn)
        
        # # 一个数据集中多个scene的metric合并
        # metrics_aggregator = configs['data']['evaluate'][dataset_name]['evaluator']['metrics_aggregator']
        # self.eval_meta_keys = dataset_meta.get('eval_meta_keys')  # {scene_id: list of view_id}
        # self.metrics_aggregator = partial(render_metric_entrypoint(metrics_aggregator[0]),
        #                                   dataset_meta=dataset_meta,
        #                                   eval_meta_keys=self.eval_meta_keys,
        #                                   **metrics_aggregator[1])

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        # render: dataset_1/config/eval_
        # learing_render: dataset_1/config/eval_text1
        #                 dataset_1/config/eval_text2
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        metrics_by_scene_id_view_id = defaultdict(dict) # {scene_id: {view_id: {}}}}
        metrics_by_scene_id = {} # {scene_id: {}}
        
        for batch_dict in tqdm(self.dataset_loader):
            eval_metas = batch_dict.pop('metas')
            view_strs = eval_metas['views'][0] # list[str]
            scene_id = eval_metas['scene_id'][0] # str
            view_queries = eval_metas['view_queries'][0] # list[3]

            visualize_path = self.visualize_path(meta_idxs=batch_dict['meta_idxs'],
                                                  visualize=batch_dict['visualize'], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model'))
            batch_dict['visualize_paths'] = visualize_path
            batch_dict = to_device(batch_dict, device=model.device)
            scene_repre = model.sample(batch_dict)['scene_representation']

            # 每个view进行metric
            for idx, (view_id, view_query) in enumerate(zip(view_strs, view_queries)):
                view_rendering = scene_repre.render(view_query)
                view_pred = {
                    'view_id': view_id,
                    'scene_id': scene_id,
                    'rendering': view_rendering
                }
                meta_key_metrics = {}                
                for metric_fn in self.view_metric_fns:
                    metric_values = metric_fn(view_pred=view_pred, output_dir=evaluator_path)
                    for key, value in metric_values.items():
                        assert key not in meta_key_metrics
                        meta_key_metrics[key] = value

                assert view_id not in metrics_by_scene_id_view_id[scene_id]
                metrics_by_scene_id_view_id[scene_id][view_id] = meta_key_metrics

            # 对整个4D进行metric
            repre_key_metrics = {}
            for metric_fn in self.repre_metric_fns:
                metric_values = metric_fn(repre={ 'scene_id': scene_id, 'repre': scene_repre,  }, 
                                          output_dir=evaluator_path)
                for key, value in metric_values.items():
                    assert key not in repre_key_metrics
                    repre_key_metrics[key] = value
            
            metrics_by_scene_id[scene_id] = repre_key_metrics
        
        metrics_by_scene_id = comm.gather(dict(metrics_by_scene_id), dst=0)
        metrics_by_scene_id_view_id = comm.gather(dict(metrics_by_scene_id_view_id), dst=0)
        
        eval_metrics = {}
        if comm.is_main_process():
            metrics_by_scene = {} 
            # {scene_id:  {view_id: {metrics}, 'repreRRRR'  }}
            for scene_id in tqdm(self.eval_meta_keys.keys(), desc='gathering different processes'):
                scene_metrics = {}
                scene_view_metrics = [haosen[scene_id] for haosen in metrics_by_scene_id_view_id if scene_id in haosen] 
                assert len(scene_view_metrics) == 1
                scene_view_metrics = scene_view_metrics[0] # {view_id: {metrics}}
                assert set(list(scene_view_metrics.keys())).issubset(set(self.eval_meta_keys[scene_id]))
                assert set(self.eval_meta_keys[scene_id]).issubset(set(list(scene_view_metrics.keys())))
                scene_metrics.update(scene_view_metrics)

                scene_repre_metrics = [haosen[scene_id] for haosen in metrics_by_scene_id if scene_id in haosen]
                assert len(scene_repre_metrics) == 1
                assert 'repreRRR' not in scene_view_metrics
                scene_metrics['repreRRR'] = scene_repre_metrics

                metrics_by_scene[scene_id] = scene_metrics

            eval_metrics = self.metrics_aggregator(metrics_by_scene)

        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class Image3D_Optimize_Evaluator:
    """
    单个scene, 最普通的重建
    单个gpu
    """
    def __init__(self,
                 dataset_name=None,
                 data_loader=None,
                 configs=None,
                 **kwargs) -> None:
        
        Scene_Meta
        self.dataset_loader = data_loader
        self.dataset_name = dataset_name

        view_metrics = configs['data']['evaluate'][dataset_name]['evaluator']['view_metrics']
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.view_metric_fns = []
        for metric_name, metric_config in view_metrics:
            metric_fn = render_metric_entrypoint(metric_name)
            metric_fn = partial(metric_fn, dataset_meta=dataset_meta, **metric_config)
            self.view_metric_fns.append(metric_fn)
        
        # metric aggregator
        metrics_aggregator = configs['data']['evaluate'][dataset_name]['evaluator']['metrics_aggregator']
        eval_meta_keys = dataset_meta.get('eval_meta_keys')
        assert len(list(eval_meta_keys.keys())) == 1
        self.eval_meta_keys = eval_meta_keys[list(eval_meta_keys.keys())[0]]  # {list of view_id}
        self.metrics_aggregator = partial(render_metric_entrypoint(metrics_aggregator[0]),
                                          dataset_meta=dataset_meta,
                                          eval_meta_keys=self.eval_meta_keys,
                                          **metrics_aggregator[1])

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        assert comm.is_main_process()
        # render: dataset_1/config/eval_
        # learing_render: dataset_1/config/eval_text1
        #                 dataset_1/config/eval_text2
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        metrics_by_view_id = defaultdict(dict) # {scene_id: {view_id: {}}}}
        # 设置成worker>0会造成mem leak, https://github.com/pytorch/pytorch/issues/92134
        for batch_dict in tqdm(self.dataset_loader):
            # batch_size就是1
            from data_schedule.render.apis import Scene_Mapper
            scene_id = batch_dict['scene_dict']['scene_id'] # list[str]
            view_camera_uid = batch_dict['view_dict']['view_camera'].uid
            visualize_path = self.visualize_path(meta_idxs=[batch_dict['meta_idx']],
                                                  visualize=[batch_dict['visualize']], 
                                                 evaluator_path=os.path.join(evaluator_path, 'visualize_model'))
            batch_dict['visualize_paths'] = visualize_path
            batch_dict = to_device(batch_dict, device=model.device)
            pred_rendering = model.sample(batch_dict)['rendering'] # 0-1, float

            view_pred = {
                'view_id': view_camera_uid,
                'scene_id': scene_id,
                'rendering': pred_rendering,
                'view_camera': batch_dict['view_dict']['view_camera']
            }
            meta_key_metrics = {}                
            for metric_fn in self.view_metric_fns:
                metric_values = metric_fn(view_pred=view_pred, output_dir=evaluator_path)
                for key, value in metric_values.items():
                    assert key not in meta_key_metrics
                    meta_key_metrics[key] = value
            assert view_camera_uid not in metrics_by_view_id
            metrics_by_view_id[view_camera_uid] = meta_key_metrics
        
        assert set(metrics_by_view_id.keys()).issubset(set(self.eval_meta_keys))
        assert set(self.eval_meta_keys).issubset(set(metrics_by_view_id.keys()))
        eval_metrics = self.metrics_aggregator(metrics_by_view_id=metrics_by_view_id)

        comm.synchronize() 
        return eval_metrics


@EVALUATOR_REGISTRY.register()
class SLAM_4D_Render_Evaluator:
    pass




