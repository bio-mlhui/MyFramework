
import os
import sys
sys.path.append('../')
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json

from models.UN_IMG_SEM.maskcut.utils.dino import ViTFeat
sys.path.append('../')
sys.path.append('../third_party')

from models.UN_IMG_SEM.tokencut.utils.unsupervised_saliency_detection import utils, metric
from models.UN_IMG_SEM.tokencut.utils.unsupervised_saliency_detection.object_discovery import detect_box
# bilateral_solver codes are modfied based on https://github.com/poolio/bilateral_solver/blob/master/notebooks/bilateral_solver.ipynb
# from TokenCut.unsupervised_saliency_detection.bilateral_solver import BilateralSolver, BilateralGrid
# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from .utils.crf import densecrf

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    """
    Implementation of MaskCut.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs

def maskcut(pil_image, backbone,patch_size, tau, N=1, fixed_size=480, cpu=False) :
    I = pil_image
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0].flatten(1) # c h w -> c hw

    _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, cpu=cpu)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, I_new

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}


from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.data import MetadataCatalog
import cv2
from argparse import Namespace
import os
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
import logging
import torchvision.transforms.functional as Trans_F
import copy
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import math
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from detectron2.structures import Instances, BitMasks
class OptimizeModel(nn.Module):
    """
    optimize_setup:
        optimizer, scheduler都是标准类
        log_lr_idx随着训练不改变
        
    optimize:
        backward, optimzier_step, optimizer_zero_grad, scheduler_step
        
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.log_lr_group_idx: Dict = None

    def optimize_setup(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

class MaskCut(OptimizeModel):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        model_configs = configs['model']
        self.backbone = BACKBONE_REGISTRY.get(model_configs["backbone"]['name'])(model_configs["backbone"])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.backbone_feat_type = model_configs["backbone"]["dino_feat_type"]
        self.feat_dim = self.backbone.embed_dim # n_feats
        self.backbone_patch_size = self.backbone.patch_size # 8 / 16
        self.backbone_nheads = self.backbone.num_heads
        
        self.nb_vis = model_configs['nb_vis']
        self.img_path = ''
        self.dataset_path = ''
        self.pretrain_path = None
        self.tau = model_configs['tau']
        self.fixed_size = model_configs['fixed_size']
        self.N = model_configs['N']

    @torch.no_grad()
    def forward_backbone(self, img):
        """
        img: b 3 h w, normalize之后的
        return: b c h w
        """
        self.backbone.eval()
        B, _, H, W = img.shape
        # b 3 h w -> b c h w
        rets = self.backbone(img)
        # b cls+hw c, 3 b head cls_hw head_dim
        feat, qkv = rets['features'][0], rets['qkvs'][0]
        if self.backbone_feat_type == "feat":
            image_feat = feat[:, 1:, :].reshape(B, H//self.backbone_patch_size, W//self.backbone_patch_size, -1).permute(0, 3, 1, 2)
        elif self.backbone_feat_type == "key":
            image_feat = qkv[1, :, :, 1:, :].reshape(B, self.backbone_nheads, 
                                                     H//self.backbone_patch_size, W//self.backbone_patch_size, -1)
            image_feat = image_feat.permute(0, 1, 4, 2, 3).flatten(1,2) # b c h w
        else:
            raise ValueError("Unknown feat type:{}".format(self.feat_type))
        return image_feat
    @property
    def device(self):
        return self.pixel_mean.device

    @torch.no_grad()
    def sample(self, batch_dict):
        # list[dict]
        assert not self.training
        self.backbone.eval()
        image = batch_dict[0]['image']
        I = Image.fromarray(image.permute(1,2,0).numpy()).convert('RGB')
        # get pseudo-masks for each image using MaskCut
        bipartitions, _, I_new = maskcut(I, self.forward_backbone, self.backbone_patch_size, self.tau, N=self.N, fixed_size=self.fixed_size, cpu=False)
        height, width = batch_dict[0]['height'], batch_dict[0]['width']
        all_masks = []

        for idx, bipartition in enumerate(bipartitions):
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo_mask)
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
            pseudo_mask = torch.from_numpy(np.asarray(pseudo_mask.resize((width, height)))) # h w
            pseudo_mask = pseudo_mask > 128

            all_masks.append(pseudo_mask)
            # if pseudo_mask.int().sum() > 20:
            #     all_masks.append(pseudo_mask)

        pred_masks = torch.stack(all_masks, dim=0) # N h w
        # from models.utils.visualize_instance_seg import cls_ag_visualize_pred
        # import random
        # cls_ag_visualize_pred(transforms.ToTensor()(I), pred=pred_masks, save_dir=f'./test{random.randint(0, 20)}.png')

        # N_orig = pred_masks.shape[0]
        # pred_masks = pred_masks.repeat(100 // N_orig, 1, 1)
        # remaininig = 100 - pred_masks.shape[0]
        # pred_masks = torch.cat([pred_masks, pred_masks[:remaininig]], dim=0) # 100 h 

        N, image_size = pred_masks.shape[0],pred_masks.shape[-2:]
        scores = torch.ones(N).float() # N
        pred_masks = BitMasks(pred_masks)
        pred_boxes = pred_masks.get_bounding_boxes()
        pred_classes = torch.zeros(N).int()
        return [{
            'instances': Instances(image_size=image_size, pred_masks=pred_masks,scores=scores,
                                   pred_boxes=pred_boxes, pred_classes=pred_classes)
        }]     

        
    def optimize_state_dict(self,):
        return {}
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def get_lr_group_dicts(self, ):
        return None

# [10/19 08:21:24 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 0.03 seconds.
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.11 seconds.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.032
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.041
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.031
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.069
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.078
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.065
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.082
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.082
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.064
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.249
# [10/19 08:21:24 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.evaluation.coco_evaluation coco_evaluation.py]: Evaluation results for bbox: 
# |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
# |:-----:|:------:|:------:|:-----:|:-----:|:-----:|
# | 3.162 | 4.077  | 3.145  | 0.000 | 6.931 | 7.779 |
# Loading and preparing results...
# DONE (t=0.00s)
# creating index...
# index created!
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: Evaluate annotation type *segm*
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 0.02 seconds.
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
# [10/19 08:21:24 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.11 seconds.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.023
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.048
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.018
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.034
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.060
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.049
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.064
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.064
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.053
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.189
# [10/19 08:21:24 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.evaluation.coco_evaluation coco_evaluation.py]: Evaluation results for segm: 
# |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
# |:-----:|:------:|:------:|:-----:|:-----:|:-----:|
# | 2.269 | 4.805  | 1.835  | 0.000 | 3.406 | 6.020 |
# [10/19 08:21:24 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.engine.defaults defaults.py]: Evaluation results for cls_agnostic_coco_only20Test_AdaptiveImgIds in csv format:
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: Task: bbox
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: 3.1625,4.0769,3.1450,0.0000,6.9307,7.7788
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: Task: segm
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/19 08:21:24 d2.evaluation.testing]: copypaste: 2.2686,4.8049,1.8346,0.0000,3.4059,6.0200
# [10/19 08:21:24 root evaluator.py]: OrderedDict([('bbox', {'AP': 3.1624927198602215, 'AP50': 4.0768782760629, 'AP75': 3.14502038439138, 'APs': 0.0, 'APm': 6.9306930693069315, 'APl': 7.77881594611074}), ('segm', {'AP': 2.2686250242671324, 'AP50': 4.804892253931275, 'AP75': 1.8345952242283055, 'APs': 0.0, 'APm': 3.405940594059405, 'APl': 6.0200045203459345})])
# [10/19 08:21:24 root Trainer_SingleProcess.py]: AP_cocoval2017_instance_cls_agnostic : 2.268625 AP50_cocoval2017_instance_cls_agnostic : 4.804892 AP75_cocoval2017_instance_cls_agnostic : 1.834595 APs_cocoval2017_instance_cls_agnostic : 0.000000 APm_cocoval2017_instance_cls_agnostic : 3.405941 APl_cocoval2017_instance_cls_agnostic : 6.020005

# [10/19 18:31:53 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
# [10/19 18:31:59 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 5.74 seconds.
# [10/19 18:31:59 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
# [10/19 18:31:59 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.38 seconds.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.019
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.039
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.015
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.028
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.097
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.053
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.078
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.078
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.032
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.279
# [10/19 18:31:59 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.evaluation.coco_evaluation coco_evaluation.py]: Evaluation results for bbox: 
# |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
# |:-----:|:------:|:------:|:-----:|:-----:|:-----:|
# | 1.863 | 3.871  | 1.546  | 0.010 | 2.772 | 9.749 |
# Loading and preparing results...
# DONE (t=0.18s)
# creating index...
# index created!
# [10/19 18:31:59 d2.evaluation.fast_eval_api]: Evaluate annotation type *segm*
# [10/19 18:32:04 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 4.97 seconds.
# [10/19 18:32:04 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
# [10/19 18:32:05 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.37 seconds.
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.015
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.033
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.012
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.016
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.077
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.045
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.065
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.065
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.029
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.231
# [10/19 18:32:05 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.evaluation.coco_evaluation coco_evaluation.py]: Evaluation results for segm: 
# |  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
# |:-----:|:------:|:------:|:-----:|:-----:|:-----:|
# | 1.454 | 3.314  | 1.152  | 0.011 | 1.565 | 7.663 |
# [10/19 18:32:05 data_schedule.unsupervised_image_semantic_seg.eval_cls_agnostic_seg.engine.defaults defaults.py]: Evaluation results for cls_agnostic_coco in csv format:
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: Task: bbox
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: 1.8630,3.8708,1.5465,0.0102,2.7720,9.7486
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: Task: segm
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
# [10/19 18:32:05 d2.evaluation.testing]: copypaste: 1.4537,3.3137,1.1522,0.0108,1.5649,7.6633
# [10/19 18:32:05 root evaluator.py]: OrderedDict([('bbox', {'AP': 1.863046654434621, 'AP50': 3.870752793687976, 'AP75': 1.5464787198200207, 'APs': 0.010192677205231811, 'APm': 2.771990626128849, 'APl': 9.748607245958569}), ('segm', {'AP': 1.4537459049474604, 'AP50': 3.313748801229599, 'AP75': 1.1521712590157636, 'APs': 0.010808118566477794, 'APm': 1.5648502448421908, 'APl': 7.663320016193895})])
# [10/19 18:32:05 root Trainer_SingleProcess.py]: AP_cocoval2017_instance_cls_agnostic : 1.453746 AP50_cocoval2017_instance_cls_agnostic : 3.313749 AP75_cocoval2017_instance_cls_agnostic : 1.152171 APs_cocoval2017_instance_cls_agnostic : 0.010808 APm_cocoval2017_instance_cls_agnostic : 1.564850 APl_cocoval2017_instance_cls_agnostic : 7.663320

@register_model
def maskcut_model(configs, device):
    from data_schedule.unsupervised_image_semantic_seg.mapper import Online_Cutler_EvalCluster_AUXMapper
    from data_schedule import build_singleProcess_schedule
    aux_mapper = Online_Cutler_EvalCluster_AUXMapper()
    train_loader, eval_function = build_singleProcess_schedule(configs, aux_mapper.mapper, partial(aux_mapper.collate))

    model = MaskCut(configs)
    model.to(device)

    return model, train_loader, eval_function
