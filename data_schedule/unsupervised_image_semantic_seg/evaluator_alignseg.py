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

import time
from typing import Tuple, List, Dict

import torch
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from collections import deque, defaultdict
from torchmetrics import Metric

# matching方法: hug/alignseg
# allimage/single_image
# fixed_query/dynamic_query


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.stats = torch.zeros(n_classes + self.extra_clusters, n_classes,
                                           dtype=torch.int64, device="cuda")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1) # 32*320*320
            preds = preds.reshape(-1) # 32*320*320

            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:  # cluster
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)  # row, col
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]

            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:  # linear
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}

from einops import rearrange
import cv2
from models.utils.visualize_cos_similarity import MyVisualizer, ColorMode
from models.utils.visualize_sem_seg import rbg_colors, generate_semseg_canvas_uou
@EVALUATOR_REGISTRY.register()
class AllImages_FixedQueries_HugMatching_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        eval_configs: dict = configs['data']['evaluate'][dataset_name]['evaluator']
        self.num_gt_classes = MetadataCatalog.get(dataset_name).get('num_classes')
        self.visualize_all = eval_configs.get('visualize_all', False)

    @torch.no_grad()
    def __call__(self, model, output_dir):
        model.eval()                    
        visualize_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(visualize_path, exist_ok=True)
        cluster_metrics = UnsupervisedMetrics("Cluster_", compute_hungarian=True,
                                              n_classes=self.num_gt_classes, extra_clusters=model.num_queries - self.num_gt_classes,) 
        linear_metrics = UnsupervisedMetrics("Linear_", self.num_gt_classes, 0, False)  
        self.num_cluster_classes = model.num_queries

        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_ids = eval_metas['image_ids']
            images = batch_dict['images'] # b 3 h w
            gt_masks = batch_dict['masks'].cuda() # b h w, -1/gt
            model_out = model.sample(batch_dict, visualize_all=self.visualize_all)  

            cluster_preds: torch.Tensor = model_out['cluster_preds'] # b nq h w, logits
            assert cluster_preds.shape[1] == self.num_cluster_classes
            cluster_preds = cluster_preds.argmax(1) # b h w
            cluster_metrics.update(cluster_preds, gt_masks)  

            linear_preds: torch.Tensor = model_out['linear_preds'] # b gt h w
            assert linear_preds.shape[1] == self.num_gt_classes
            linear_preds = linear_preds.argmax(1)
            linear_metrics.update(linear_preds, gt_masks)
 

            if self.visualize_all:
                assert len(image_ids) == 1
                image_path = os.path.join(visualize_path, f'{image_ids[0]}.jpg')
                kmeans_preds, num_kmeans_classes, kmeans_preds_bb = model_out['kmeans_preds'], model_out['num_kmeans_classes'], model_out['kmeans_preds_bb']
                sampled_points, similarities, backbone_similarities = \
                    model_out['sampled_points'], model_out['similarities'], model_out['backbone_similarities']

                # image, gt, cluster_pred, linear_pred, kmeans_pred, kmeans_pred_bb
                # H 5*W 3 
                first_row = visualize_cluster(image=images[0], gt=gt_masks[0].cpu(), linear_pred=linear_preds[0].cpu(), num_gt_classes=self.num_gt_classes,
                                              cluster_pred=cluster_preds[0].cpu(), num_cluster_classes=self.num_cluster_classes,
                                              kmeans_pred=kmeans_preds[0].cpu(), kmeans_pred_bb=kmeans_preds_bb[0].cpu(), num_kmeans_classes=num_kmeans_classes,) # H W*3 3
                # 3*H N*W 3 point, transformed_sim, backbone_sim
                sim_image = visualize_cos_similarity(image=images[0], sampled_points=sampled_points, 
                                                     similarities=similarities, backbone_similarities=backbone_similarities)
                first_row = F.pad(first_row, pad=(0, 0, 0, sim_image.shape[1]-first_row.shape[1], 0, 0),)
                whole_image = torch.cat([first_row, sim_image], dim=0) # 4H 10W 3
                Image.fromarray(whole_image.numpy()).save(image_path)  

        eval_metrics = cluster_metrics.compute()
        eval_metrics.update(linear_metrics.compute())
        return eval_metrics

class UnsupervisedMetrics_DDP(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,):
        super().__init__(dist_sync_on_step=False,
                         sync_on_compute=False,)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state('stats', 
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64, device="cuda"),
                       dist_reduce_fx='sum')
    
    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        actual = target.reshape(-1) 
        preds = preds.reshape(-1) 
        mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
        actual = actual[mask]
        preds = preds[mask]
        self.stats += torch.bincount(
            (self.n_classes + self.extra_clusters) * actual + preds,
            minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
            .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def compute(self):
        if comm.is_main_process():
            self.assignments = linear_sum_assignment(self.stats.cpu(), maximize=True)  # row, col
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]

            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)

            tp = torch.diag(self.histogram)
            fp = torch.sum(self.histogram, dim=0) - tp
            fn = torch.sum(self.histogram, dim=1) - tp

            iou = tp / (tp + fp + fn)
            prc = tp / (tp + fn)
            opc = torch.sum(tp) / torch.sum(self.histogram)

            metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item()*100,
                        self.prefix + "Accuracy": opc.item()*100}
        else:
            metric_dict = {}
        comm.synchronize()
        return metric_dict

@EVALUATOR_REGISTRY.register()
class AllImages_FixedQueries_HugMatching_Evaluator_DDP:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        eval_configs: dict = configs['data']['evaluate'][dataset_name]['evaluator']
        self.num_gt_classes = MetadataCatalog.get(dataset_name).get('num_classes')
        self.visualize_all = eval_configs.get('visualize_all', False)

    @torch.no_grad()
    def __call__(self, model, output_dir):
        model.eval()                    
        visualize_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(visualize_path, exist_ok=True)
        cluster_metrics = UnsupervisedMetrics_DDP("Cluster_", compute_hungarian=True,
                                              n_classes=self.num_gt_classes, extra_clusters=model.num_queries - self.num_gt_classes,) 
        self.num_cluster_classes = model.num_queries

        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_ids = eval_metas['image_ids']
            images = batch_dict['images'] # b 3 h w
            gt_masks = batch_dict['masks'].cuda() # b h w, -1/gt
            model_out = model.sample(batch_dict, visualize_all=self.visualize_all)  

            cluster_preds: torch.Tensor = model_out['cluster_preds'] # b nq h w, logits
            assert cluster_preds.shape[1] == self.num_cluster_classes
            cluster_preds = cluster_preds.argmax(1) # b h w
            cluster_metrics.update(cluster_preds, gt_masks)   

            if self.visualize_all:
                assert len(image_ids) == 1
                image_path = os.path.join(visualize_path, f'{image_ids[0]}.jpg')
                kmeans_preds, num_kmeans_classes, kmeans_preds_bb = model_out['kmeans_preds'], model_out['num_kmeans_classes'], model_out['kmeans_preds_bb']
                sampled_points, similarities, backbone_similarities = \
                    model_out['sampled_points'], model_out['similarities'], model_out['backbone_similarities']

                # image, gt, cluster_pred, linear_pred, kmeans_pred, kmeans_pred_bb
                # H 5*W 3 
                first_row = visualize_cluster(image=images[0], gt=gt_masks[0].cpu(), linear_pred=linear_preds[0].cpu(), num_gt_classes=self.num_gt_classes,
                                              cluster_pred=cluster_preds[0].cpu(), num_cluster_classes=self.num_cluster_classes,
                                              kmeans_pred=kmeans_preds[0].cpu(), kmeans_pred_bb=kmeans_preds_bb[0].cpu(), num_kmeans_classes=num_kmeans_classes,) # H W*3 3
                # 3*H N*W 3 point, transformed_sim, backbone_sim
                sim_image = visualize_cos_similarity(image=images[0], sampled_points=sampled_points, 
                                                     similarities=similarities, backbone_similarities=backbone_similarities)
                first_row = F.pad(first_row, pad=(0, 0, 0, sim_image.shape[1]-first_row.shape[1], 0, 0),)
                whole_image = torch.cat([first_row, sim_image], dim=0) # 4H 10W 3
                Image.fromarray(whole_image.numpy()).save(image_path)  
            # comm.synchronize()
        cluster_metrics.sync()
        eval_metrics = cluster_metrics.compute()
        return eval_metrics
    
def visualize_cluster(image, gt, linear_pred, num_gt_classes,
                    cluster_pred, num_cluster_classes,
                    kmeans_pred, kmeans_pred_bb, num_kmeans_classes
                    ):
    """
    image: 0-1 float, 3 h w
    gt & linear_pred: gt/-1, long, h w ;
    cluster_pred: cluster, long, h w
    kmeans_pred: kmeans, long, hw
    """
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    if num_gt_classes == 27:
        gt_color_name = 'color_gt_27'
    else:
        raise NotImplementedError()
    if num_cluster_classes > 27 or num_kmeans_classes > 27:
        raise NotImplementedError()
    cluster_color_name = f'cluster_color_{time.time()}'
    MetadataCatalog.get(cluster_color_name).set(stuff_classes = [str(idx) for idx in range(num_cluster_classes)],
                                                stuff_colors = rbg_colors[:num_cluster_classes])
    kmeans_color_name = f'kmeans_color_{time.time()}'
    MetadataCatalog.get(kmeans_color_name).set(stuff_classes = [str(idx) for idx in range(num_kmeans_classes)],
                                                stuff_colors = rbg_colors[:num_kmeans_classes])

    gt[gt==-1] = (num_gt_classes + 2000) # detectron2: for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
    gt_mask_image = torch.from_numpy(generate_semseg_canvas_uou(image=image, H=H, W=W, mask=gt, num_classes=num_gt_classes, 
                                                        dataset_name=gt_color_name,))  
    linear_pred_image = torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=linear_pred, num_classes=num_gt_classes, 
                                                            dataset_name=gt_color_name,))  
    cluster_pred_image = torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=cluster_pred, num_classes=num_cluster_classes, 
                                                            dataset_name=cluster_color_name,))  
    kmeans_pred_image = torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=kmeans_pred, num_classes=num_kmeans_classes, 
                                                            dataset_name=kmeans_color_name,))  
    kmeans_pred_bb_image = torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=kmeans_pred_bb, num_classes=num_kmeans_classes, 
                                                            dataset_name=kmeans_color_name,))     
    whole_image = torch.cat([torch.from_numpy(image), gt_mask_image, cluster_pred_image, linear_pred_image, kmeans_pred_image, kmeans_pred_bb_image], dim=1)
    return whole_image


def visualize_cos_similarity(image, 
                             sampled_points,
                             similarities,
                             backbone_similarities):
    """
    image: 0-1, 3 h w, float
    sampled_points: N 2, 0-H-1, H, W
    similarities: N h w, [-1,1], float, -1=blue, 1=red
    """
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    
    point_images = []
    superimposed_imgs = []
    superimposed_imgs_bb = []
    for point, sim, bb_sim in zip(sampled_points, similarities, backbone_similarities):
        # 2, h w,
        istce_canvas = MyVisualizer(img_rgb=image, metadata=None, instance_mode=ColorMode.SEGMENTATION)
        istce_canvas.draw_circle(circle_coord=point.tolist()[::-1], color=(1.0, 0, 0), radius=10)
        istce_canvas = istce_canvas.get_output()
        point_image =  torch.from_numpy(istce_canvas.get_image())  # h w 3

        heatmap = ((sim + 1) / 2).clamp(0, 1).numpy()
        heatmap = np.uint8(255 * heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        superimposed_img = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)
        superimposed_img = torch.from_numpy(np.asarray(superimposed_img)) # h w 3

        heatmap = ((bb_sim + 1) / 2).clamp(0, 1).numpy()
        heatmap = np.uint8(255 * heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
        superimposed_img_bb = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)
        superimposed_img_bb = torch.from_numpy(np.asarray(superimposed_img_bb)) # h w 3

        point_images.append(point_image) # 
        superimposed_imgs.append(superimposed_img)
        superimposed_imgs_bb.append(superimposed_img_bb)
 
        
    whole_image = torch.cat([torch.cat(point_images, dim=1), 
                             torch.cat(superimposed_imgs, dim=1),
                             torch.cat(superimposed_imgs_bb, dim=1)], dim=0)
    return whole_image
    # Image.fromarray(whole_image.numpy()).save(save_dir)

from scipy.optimize import linear_sum_assignment
class SingleImageHugMatching(Metric):
    def __init__(self,):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("iou", [])
        self.add_state("iou_excludeFirst", [])
        self.n_jobs = -1

        self.temporary_match = None
    def batch_dice_cost(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: N h w, float, 0/1
            targets: Nq h w, float 0/1
        """
        inputs = inputs.flatten(1) # N hw
        targets = targets.flatten(1) # M hw
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :] # N 1 + 1 M -> N M
        dice = 1 - (numerator + 1) / (denominator + 1)
        return dice

    def update(self, gt: torch.Tensor, pred: torch.Tensor, many_to_one=True, precision_based=True, linear_probe=False):
        # n h w, m h w, bool, 0/1
        assert pred.shape[0] >= gt.shape[0]
        cost_dice = self.batch_dice_cost(pred.float(), gt.float()) # n m
        matched_indices = linear_sum_assignment(cost_dice)
        # match_src matched_tgt
        # 看一下数据集的情况, 比如怎么讲semantic(两个不连通的)分成

        for src_idx, tgt_idx in zip(matched_indices[0], matched_indices[1]):
            src_mask = pred[src_idx] # h w
            tgt_mask = gt[tgt_idx] # h w

            inter = torch.logical_and(src_mask, tgt_mask)
            union = torch.logical_or(src_mask, tgt_mask)

            iou = inter.float().sum() / union.float().sum()

            self.iou.append(iou)

        self.temporary_match = matched_indices

    def compute(self):
        """
        Compute mIoU
        """
        mIoU = np.mean(self.iou)
        return {'miou': mIoU}


@EVALUATOR_REGISTRY.register()
class CutLER_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        eval_configs: dict = configs['data']['evaluate'][dataset_name]['evaluator']
        self.visualize = eval_configs.get('visualize_all', False)

        MetadataCatalog.get('single').set(stuff_classes = ['0'],
                                            stuff_colors = [(156, 31, 23),])

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        # 假设: 固定输入输出大小, evaluate也在固定大小上测试
        # 假设: B=1
        # 1 3 h w -> 1 nq h w, 然后单图matching
        model.eval()                    
        visualize_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(visualize_path, exist_ok=True)
        metric = SingleImageHugMatching()
        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_id = eval_metas['image_ids'][0]
            image_path = os.path.join(visualize_path, f'{image_id}.jpg')

            image = batch_dict['images'][0] # 3 h w, 0-1
            instance_gt_masks = batch_dict['instance_masks'][0] # N h w;bool
            _, H, W = image.shape

            if H >= 1000 or W >= 1000:
                continue
            model_out = model.sample(batch_dict) # dict{'pred_masks': b nq h w, logits} 
            pred_masks: torch.Tensor = model_out['pred_masks'][0] # nq h w, bool  
            num_pred_classes = pred_masks.shape[0]
            num_gt_classes = instance_gt_masks.shape[0]

            metric.update(instance_gt_masks, pred_masks)
            matched_indices = metric.temporary_match
            not_matched_src_idxs = list(set(list(range(num_pred_classes))) - set(matched_indices[0]))
            instance_gt_masks = torch.stack([instance_gt_masks[foo_idx] for foo_idx in matched_indices[1]], dim=0)
            pred_masks = torch.stack([pred_masks[foo_idx] for foo_idx in (matched_indices[0].tolist() + not_matched_src_idxs)], dim=0)

            cluster_image = visualize_cutler(image, 
                                             gt=instance_gt_masks, pred=pred_masks, 
                                             num_pred_classes=num_pred_classes,
                                             num_gt_classes=num_gt_classes) # H W*3 3
            cluster_image = cluster_image[:5000, :30000, :]
            Image.fromarray(cluster_image.numpy()).save(image_path)            
        iou_mean_over_instances = metric.compute()['miou'] * 100
        return {
            'iou_mean_over_instances': iou_mean_over_instances
        }

def visualize_cutler(image, 
                     gt, num_gt_classes,
                     pred, num_pred_classes,
                    ):
    # N h w, T/F; M h w, T/F;
    H, W = image.shape[-2:]
    image = (image.permute(1,2,0).numpy() * 255).astype('uint8')
    
    gt_plots = []
    pred_plots = []
    
    gt = gt.int() - 1
    gt[gt==-1] = 20000
    pred = pred.int() - 1
    pred[pred==-1] = 20000

    for gt_m in gt:
        gt_plots.append(torch.from_numpy(generate_semseg_canvas_uou(image=image, H=H, W=W, mask=gt_m, num_classes=num_gt_classes, dataset_name='single',)) )
    for pred_m in pred:
        pred_plots.append(torch.from_numpy(generate_semseg_canvas_uou(image=image,  H=H, W=W, mask=pred_m, num_classes=num_gt_classes, dataset_name='single',)) )
    
    gt_plots = torch.cat(gt_plots, dim=1)
    gt_plots = F.pad(gt_plots, pad=(0, 0, 0, (pred.shape[0] - gt.shape[0]) * image.shape[1]))
    pred_plots = torch.cat(pred_plots, dim=1)
    whole_image = torch.cat([gt_plots, pred_plots], dim=0)
    return whole_image


# region
# alignseg
class PredsmIoU_DynamicQueries(Metric):
    """
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """

    def __init__(self,
                 num_gt_classes: int):
        """
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.num_gt_classes = num_gt_classes
        self.add_state("iou", [])
        self.add_state("iou_excludeFirst", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor, many_to_one=True, precision_based=True, linear_probe=False):
        pred = pred.cpu().numpy().astype(int)
        gt = gt.cpu().numpy().astype(int)
        self.num_pred_classes = len(np.unique(pred))
        iou_all, iou_excludeFirst = self.compute_miou(gt, pred, self.num_pred_classes, len(np.unique(gt)),
                                            many_to_one=many_to_one, precision_based=precision_based, linear_probe=linear_probe)
        self.iou.append(iou_all)
        self.iou_excludeFirst.append(iou_excludeFirst)

    def compute(self):
        """
        Compute mIoU
        """
        mIoU = np.mean(self.iou)
        mIoU_excludeFirst = np.mean(self.iou_excludeFirst)
        print('---mIoU computed---', mIoU)
        print('---mIoU exclude first---', mIoU_excludeFirst)
        return mIoU

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt: int,
                     many_to_one=False, precision_based=False, linear_probe=False):
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt
        """
        assert pred.shape == gt.shape
        # print(f"unique semantic class = {np.unique(gt)}")
        gt_class = np.unique(gt).tolist()
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        iou = [0] * num_gt # 13个类别

        if linear_probe:
            reordered_preds = pred
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based) # gt->list[pred_cls]
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                # 1. gt>5, 但是pred因为softmax+max没有类2
                # 2. gt<5, matched到的Pred没有2
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == gt_class[i_part])
            tmp_pred = (reordered_preds == gt_class[i_part])
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            iou[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print('\tiou = ', iou, np.mean(iou[1:]))
        if len(iou) > 1:
            return np.mean(iou), np.mean(iou[1:])
        else:
            # return np.mean(iou), tp, fp, fn, reordered_preds.astype(int).tolist()
            return np.mean(iou), np.mean(iou)

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            # print('\tgt, pred = ', c1, c2, ' | precision=', prec)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        # print("Parallelizing iou computation")
        # start = time.time()
        score_mat = []
        for c2 in range(num_pred):
            for c1 in np.unique(gt):
                score_mat.append(self.get_score(pred, gt, c1, c2, precision_based=precision_based))
                
        # score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
        #                                          for c2 in range(num_pred) for c1 in np.unique(gt))
        # print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        gt_class = np.unique(gt).tolist()
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_i in range(num_gt):
                score = score_mat[gt_i, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_class[gt_i]
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k, v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        # print('original match:', gt_to_matches)
        return gt_to_matches

# from models.utils.visualize_sem_seg import visualize_cluster
# from models.utils.visualize_cos_similarity import visualize_cos_similarity
# from PIL import Image
@EVALUATOR_REGISTRY.register()
class SingleImage_Evaluator:
    """
    单图matching, 每个图输出b n h w, 和mask b h w进行matching
    """
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        self.num_classes = dataset_meta.get('num_classes')
        eval_configs: dict = configs['data']['evaluate'][dataset_name]['evaluator']
        self.visualize = eval_configs.get('visualize', False)
        self.num_gt_classes = MetadataCatalog.get(dataset_name).get('num_classes')

    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        # 假设: 固定输入输出大小, evaluate也在固定大小上测试
        # 假设: B=1
        # 1 3 h w -> 1 nq h w, 然后单图matching
        model.eval()                    
        visualize_path = os.path.join(output_dir, f'eval_{self.dataset_name}', 'cluster_testset')
        os.makedirs(visualize_path, exist_ok=True)
        metric = PredsmIoU_DynamicQueries(num_gt_classes=self.num_gt_classes)
        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_id = eval_metas['image_ids'][0]
            image = batch_dict['images'][0] # 3 h w
            _, H, W = image.shape
            gt_masks = batch_dict['masks'] # b h w, 0-27, 255, uint8
            model_out = model.sample(batch_dict) # dict{'pred_masks': b nq h w, logits} 
            image_path = os.path.join(visualize_path, f'{image_id}.jpg')
            pred_masks: torch.Tensor = model_out['pred_masks']  
            pred_masks = pred_masks.softmax(1)
            num_image_classes = pred_masks.shape[1]
            pred_masks = pred_masks.max(dim=1)[1].cpu() # b h w, 0-num_queries

            _, num_image_classes = model_out['cluster_ids'], model_out['num_image_classes']
            sampled_points, similarities = model_out['sampled_points'], model_out['similarities']
            cluster_image = visualize_cluster(image, 
                                                gt=gt_masks[0].cpu(), 
                                                pred=pred_masks[0], 
                                                num_image_classes=num_image_classes,
                                                num_gt_classes=self.num_gt_classes) # H W*3 3
            sim_image = visualize_cos_similarity(image=image,
                                                  sampled_points=sampled_points, similarities=similarities,) #  H W*5 3
            cluster_image = F.pad(cluster_image, pad=(0, 0, 0, 2*W, 0, 0), value=0)
            whole_image = torch.cat([cluster_image, sim_image], dim=0) # 2H W*5 3
            Image.fromarray(whole_image.numpy()).save(image_path)

            gt_masks = gt_masks.flatten()
            pred_masks = pred_masks.flatten()
            
            background = (gt_masks == 255)
            
            metric.update(gt_masks[~background], pred_masks[~background])

        cluster_miou = metric.compute() * 100
        
        return {
            'cluster_miou': cluster_miou
        }

#endregion


# region: stegeo的allimage + hug方法
# class UnsupervisedMetrics(Metric):
#     def __init__(self, 
#                  prefix: str, 
#                  num_queries: int, 
#                  num_gt_classes,
#                  compute_hungarian: bool,
#                  dist_sync_on_step=True):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.compute_hungarian = compute_hungarian
#         self.prefix = prefix
#         self.num_gt_classes = num_gt_classes
#         self.num_queries = num_queries
#         self.stats = torch.zeros(num_queries, num_gt_classes, dtype=torch.int64, device="cuda")
#         assert self.num_queries >= self.num_gt_classes
#         self.extra_clusters = self.num_queries - self.num_gt_classes
        
#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         # target: b h w, int, 0-num_gt_class-1, 255代表背景
#         with torch.no_grad():
#             target = target.reshape(-1)
#             preds = preds.reshape(-1) 
#             truth = target[target != 255]
#             preds = preds[target != 255]
#             assert truth.max() < self.num_gt_classes
#             assert preds.max() < self.num_queries
            
#             self.stats += torch.bincount(
#                 (self.num_queries) * truth + preds,
#                 minlength=self.num_gt_classes * (self.num_queries)) \
#                 .reshape(self.num_gt_classes, self.num_queries).t().to(self.stats.device)

#     def map_clusters(self, clusters):
#         if self.extra_clusters == 0:
#             return torch.tensor(self.assignments[1])[clusters]
#         else:
#             missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
#             cluster_to_class = self.assignments[1]
#             for missing_entry in missing:
#                 if missing_entry == cluster_to_class.shape[0]:
#                     cluster_to_class = np.append(cluster_to_class, -1)
#                 else:
#                     cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
#             cluster_to_class = torch.tensor(cluster_to_class)
#             return cluster_to_class[clusters]

#     def compute(self):
#         self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True) 
#         if self.extra_clusters == 0:
#             # nq10 gt10 -> list[0,1,2], list[3, 5, 2]
#             self.histogram = self.stats[np.argsort(self.assignments[1]), :] # 第i行是和gt_i匹配到的query和各个gt的交集

#         if self.extra_clusters > 0:
#             # gt10, nq20 -> list[0,1,2], list[3 ,4 2]
#             self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
#             histogram = self.stats[self.assignments_t[1], :]
#             # nq20, gt10 -> list[2,1, 20], list[5,2,1]
#             missing = list(set(range(self.num_queries)) - set(self.assignments[0]))
#             new_row = self.stats[missing, :].sum(0, keepdim=True)
#             histogram = torch.cat([histogram, new_row], axis=0)
#             new_col = torch.zeros(self.num_gt_classes + 1, 1, device=histogram.device)
#             self.histogram = torch.cat([histogram, new_col], axis=1)

#         tp = torch.diag(self.histogram)
#         fp = torch.sum(self.histogram, dim=0) - tp
#         fn = torch.sum(self.histogram, dim=1) - tp

#         iou = tp / (tp + fp + fn)
#         prc = tp / (tp + fn)
#         opc = torch.sum(tp) / torch.sum(self.histogram)

#         metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
#                        self.prefix + "Accuracy": opc.item()}
#         return {k: 100 * v for k, v in metric_dict.items()}
# endregion



