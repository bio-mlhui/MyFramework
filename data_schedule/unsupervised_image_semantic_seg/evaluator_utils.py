_metric_entrypoints = {}

def register_metric(fn):
    metric_name = fn.__name__
    if metric_name in _metric_entrypoints:
        raise ValueError(f'metric name {metric_name} has been registered')
    _metric_entrypoints[metric_name] = fn

    return fn

def metric_entrypoint(metric_name):
    try:
        return _metric_entrypoints[metric_name]
    except KeyError as e:
        print(f'metric Name {metric_name} not found')

import numpy as np
_EPS = np.spacing(1)
_TYPE = np.float64

def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


class Smeasure(object):
    def __init__(self, length, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray, idx):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info['weight']
        pred1, pred2, pred3, pred4 = part_info['pred']
        gt1, gt2, gt3, gt4 = part_info['gt']
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x, y) -> dict:
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(gt=(gt_LT, gt_RT, gt_LB, gt_RB),
                    pred=(pred_LT, pred_RT, pred_LB, pred_RB),
                    weight=(w1, w2, w3, w4))

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self):
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(Smeasure=sm)


import torch  
import os
import shutil
from PIL import Image

@register_metric
def mask_dice_iou(frame_pred, dataset_meta, **kwargs):
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')
    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
    gt_mask = gt_mask[0].int() # h w

    inter, union    = (pred_mask*gt_mask).sum(), (pred_mask+gt_mask).sum()
    dice = (2*inter+1)/(union+1)
    iou = (inter+1)/(union-inter+1)

    return {'dice': dice, 'iou': iou}


@register_metric
def mask_dice_iou_sen_mae_smeasure(frame_pred, dataset_meta, **kwargs):
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w
    get_frames_gt_mask_fn = dataset_meta.get('get_frames_gt_mask_fn')
    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    gt_mask, _ = get_frames_gt_mask_fn(video_id=video_id, frames=[frame_name]) # 1 h w
    gt_mask = gt_mask[0].int() # h w

    # tp, tp*2 + fp + fn
    inter, union    = (pred_mask*gt_mask).sum(), (pred_mask+gt_mask).sum()
    dice = (2*inter+1)/(union+1) # 2*tp / tp + tp + fp + fn
    iou = (inter+1)/(union-inter+1) # tp / tp + fp + fn


    tp = (pred_mask * gt_mask).sum().float()
    fp = (pred_mask.sum() - tp).float()
    fn = (gt_mask.sum() - tp).float()
    tn = (pred_mask.shape[0] * pred_mask.shape[1] - (tp + fp + fn)).float()
    their_dice = tp * 2 / (tp + fp + fn + tp)
    their_iou = tp / (tp + fp + fn)
    # their_spe = tn / (tn + fp)
    their_sen = tp / (tp + fn)
    their_mae = (pred_mask.float() - gt_mask.float()).abs().mean()
    
    Np = gt_mask.sum()
    Nn = gt_mask.shape[0] * gt_mask.shape[1] - Np
    
    null = Smeasure(length=1, alpha=0.5)
    null.step(pred=(pred_mask.float() * 255 ).numpy(), gt=(gt_mask.float() * 255).numpy(), idx=None)
    their_smeasure = torch.tensor(null.get_results()['Smeasure']).float()
    return {'dice': dice, 'iou': iou,
            'their_dice': their_dice,
            'their_iou': their_iou,
            'their_sen': their_sen,
            'their_mae_abs': their_mae,
            'their_smeasure': their_smeasure,
            
            'tp': tp, # true  positive
            'fp': fp, # false positive
            'fn': fn, # false negative
            'tn': tn, # true  negative
            'Np': Np, # positive accumulation
            'Nn': Nn} # negative accumulation

# 改成detectron2的形式
@register_metric
def web(frame_pred, output_dir, **kwargs):

    os.makedirs(os.path.join(output_dir, 'web'), exist_ok=True) 
    video_id = frame_pred['video_id']
    frame_name = frame_pred['frame_name']
    masks = frame_pred['masks'] # nq h w

    scores = torch.tensor(frame_pred['classes']) # nq c
    foreground_scores = scores[:, :-1].sum(-1) # nq
    max_idx = foreground_scores.argmax()
    pred_mask = masks[max_idx].int() # h w

    mask = Image.fromarray(255 * pred_mask.int().numpy()).convert('L')
    save_path = os.path.join(output_dir, 'web', video_id)

    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f'{frame_name}.png')
    if os.path.exists(png_path):
        os.remove(png_path)
    mask.save(png_path)
    return {}


class RunningAverage:
    def __init__(self):
        self._avg = 0.0
        self._count = 0

    def append(self, value: float) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._avg = (value + self._count * self._avg) / (self._count + 1)
        self._count += 1

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._avg = 0.0
        self._count = 0

from typing import Any, Dict, List
from typing import Dict, Any
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
from numbers import Number
def all_reduce_scalar(value: Number, op: str = "sum") -> Number:
    """All-reduce single scalar value. NOT torch tensor."""
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils/distributed.py
    if dist.is_initialized() and dist.is_available():
        op = op.lower()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        elif op == "min":
            dist_op = dist.ReduceOp.MIN
        elif op == "max":
            dist_op = dist.ReduceOp.MAX
        elif op == "product":
            dist_op = dist.ReduceOp.PRODUCT
        else:
            raise RuntimeError(f"Invalid all_reduce op: {op}")

        backend = dist.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError(f"Unsupported distributed backend: {backend}")

        tensor = torch.tensor(value, device=device, requires_grad=False)
        dist.all_reduce(tensor, op=dist_op)
        if op == "mean":
            tensor /= dist.get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret
def all_reduce_tensor(tensor: torch.Tensor, op="sum", detach: bool = True) -> torch.Tensor:
    if dist.is_initialized() and dist.is_available():
        ret = tensor.clone()
        if detach:
            ret = ret.detach()
        if (op == "sum") or (op == "mean"):
            dist_op = dist.ReduceOp.SUM
        else:
            raise RuntimeError(f"Invalid all_reduce op: {op}")

        dist.all_reduce(ret, op=dist_op)
        if op == "mean":
            ret /= dist.get_world_size()
    else:
        ret = tensor
    return ret
def all_reduce_dict(result: Dict[str, Any], op="sum") -> Dict[str, Any]:
    new_result = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            new_result[k] = all_reduce_tensor(v, op)
        elif isinstance(v, Number):
            new_result[k] = all_reduce_scalar(v, op)
        else:
            raise RuntimeError(f"Dictionary all_reduce should only have either tensor or scalar, got: {type(v)}")
    return new_result
def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    if dist.is_initialized() and dist.is_available():
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        output = [
            tensor if (i == local_rank) else torch.empty_like(tensor) for i in range(world_size)
        ]
        dist.all_gather(output, tensor, async_op=False)
        return output
    else:
        return [tensor]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)

    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)


import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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


def get_metrics(m1: UnsupervisedMetrics, m2: UnsupervisedMetrics) -> Dict[str, Any]:
    metric_dict_1 = m1.compute()
    metric_dict_2 = m2.compute()
    metrics = all_reduce_dict(metric_dict_1, op="mean")
    tmp = all_reduce_dict(metric_dict_2, op="mean")
    metrics.update(tmp)

    return metrics
