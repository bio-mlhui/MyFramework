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

# 最公平的方法, 现在采用的方法:
# hug + allimage + fixed_query

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

from models.utils.visualize_sem_seg import visualize_cluster
from PIL import Image
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
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        visualize_path = os.path.join(evaluator_path, 'visualizev1')
        os.makedirs(visualize_path, exist_ok=True)
        metric = PredsmIoU_DynamicQueries(num_gt_classes=self.num_gt_classes)
        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_id = eval_metas['image_ids'][0]
            image = batch_dict['images'][0] # 3 h w
            gt_masks = batch_dict['masks'] # b h w, 0-27, 255, uint8
            model_out = model.sample(batch_dict) # dict{'pred_masks': b nq h w, logits}   
            pred_masks: torch.Tensor = model_out['pred_masks']  
            pred_masks = pred_masks.softmax(1)
            num_image_classes = pred_masks.shape[1]
            pred_masks = pred_masks.max(dim=1)[1].cpu() # b h w, 0-num_queries
            
            whole_image = visualize_cluster(image=image, gt=gt_masks[0].int(), pred=pred_masks[0], 
                              num_image_classes=num_image_classes, num_gt_classes=self.num_classes)
            Image.fromarray(whole_image.numpy()).save(os.path.join(visualize_path, f'{image_id}.jpg'))
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
class UnsupervisedMetrics(Metric):
    def __init__(self, 
                 prefix: str, 
                 num_queries: int, 
                 num_gt_classes,
                 compute_hungarian: bool,
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.num_gt_classes = num_gt_classes
        self.num_queries = num_queries
        self.stats = torch.zeros(num_queries, num_gt_classes, dtype=torch.int64, device="cuda")
        assert self.num_queries >= self.num_gt_classes
        self.extra_clusters = self.num_queries - self.num_gt_classes
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # target: b h w, int, 0-num_gt_class-1, 255代表背景
        with torch.no_grad():
            target = target.reshape(-1)
            preds = preds.reshape(-1) 
            truth = target[target != 255]
            preds = preds[target != 255]
            assert truth.max() < self.num_gt_classes
            assert preds.max() < self.num_queries
            
            self.stats += torch.bincount(
                (self.num_queries) * truth + preds,
                minlength=self.num_gt_classes * (self.num_queries)) \
                .reshape(self.num_gt_classes, self.num_queries).t().to(self.stats.device)

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
        self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True) 
        if self.extra_clusters == 0:
            # nq10 gt10 -> list[0,1,2], list[3, 5, 2]
            self.histogram = self.stats[np.argsort(self.assignments[1]), :] # 第i行是和gt_i匹配到的query和各个gt的交集

        if self.extra_clusters > 0:
            # gt10, nq20 -> list[0,1,2], list[3 ,4 2]
            self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
            histogram = self.stats[self.assignments_t[1], :]
            # nq20, gt10 -> list[2,1, 20], list[5,2,1]
            missing = list(set(range(self.num_queries)) - set(self.assignments[0]))
            new_row = self.stats[missing, :].sum(0, keepdim=True)
            histogram = torch.cat([histogram, new_row], axis=0)
            new_col = torch.zeros(self.num_gt_classes + 1, 1, device=histogram.device)
            self.histogram = torch.cat([histogram, new_col], axis=1)

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}

@EVALUATOR_REGISTRY.register()
class AllImages_FixedQueries_HugMatching_Evaluator:
    def __init__(self,
                 dataset_name,
                 data_loader,
                 configs) -> None:
        dataset_meta = MetadataCatalog.get(dataset_name)
        self.dataset_name = dataset_name
        self.loader = data_loader
        self.num_classes = dataset_meta.get('num_classes')
        eval_configs: dict = configs['data']['evaluate'][dataset_name]['evaluator']
        self.num_gt_classes = MetadataCatalog.get(dataset_name).get('num_classes')
        
    def visualize_path(self, meta_idxs, visualize, evaluator_path):
        return [os.path.join(evaluator_path, f'meta_{meta_idx}') if vis else None for (meta_idx, vis) in zip(meta_idxs, visualize)]
    
    @torch.no_grad()
    def __call__(self, model, output_dir):
        # 假设: 固定输入输出大小, evaluate也在固定大小上测试
        # b 3 h w -> b nq h w; N_eval nq h w -> N_eval h w, 0-num_nq-1
        from .evaluator_utils import batched_crf, get_metrics
        model.eval()                    
        evaluator_path = os.path.join(output_dir, f'eval_{self.dataset_name}')
        os.makedirs(evaluator_path, exist_ok=True)
        cluster_metrics = UnsupervisedMetrics("Cluster_", num_gt_classes=self.num_gt_classes, num_queries=model.num_queries,
                                              compute_hungarian=True)        
        for batch_dict in tqdm(self.loader):
            eval_metas = batch_dict.pop('metas') # image_ids: list[str]
            image_id = eval_metas['image_ids'][0]
            image = batch_dict['images'][0] # 3 h w
            gt_masks = batch_dict['masks'].to(model.device, non_blocking=True) # b h w, 0-27, 255, uint8
            model_out = model.sample(batch_dict) # dict{'pred_masks': }   
            pred_masks: torch.Tensor = model_out['pred_masks'] # b nq h w, logits

            # if self.crf:
            #     cluster_preds = torch.log_softmax(pred_masks, dim=1)
            #     cluster_preds = batched_crf(image, cluster_preds).argmax(1)
            # else:
            cluster_preds = pred_masks.argmax(1) # b h w
                
            cluster_metrics.update(cluster_preds, gt_masks) 

        eval_metrics = cluster_metrics.compute()
        
        return eval_metrics

# endregion

