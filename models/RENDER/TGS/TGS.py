import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from torch.utils.data import DataLoader

import tgs
from tgs.models.image_feature import ImageFeature
from .utils.saving import SaverMixin
from .utils.config import parse_structured
from .utils.ops import points_projection
from .utils.misc import load_module_weights
from .utils.typing import *
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_schedule import build_schedule
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
import torchvision.transforms.functional as Trans_F
import copy
from models.registry import register_model
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler 
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
import math
from detectron2.utils import comm
from torch.nn.parallel import DistributedDataParallel as DDP
from models.backbone.utils import VideoMultiscale_Shape
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import logging
from .image_feature import ImageFeature

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

    def optimize(self,
                loss_weight=None,
                loss_dict_unscaled=None,
                closure=None,
                num_iterations=None,
                **kwargs):
        
        loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  
        self.optimizer.step(closure=closure)
        self.optimizer.zero_grad(set_to_none=True) # delete gradient 
        self.scheduler.step(epoch=num_iterations,)

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict=state_dict['scheduler'])

class Trainer_GSModel:
    def __init__(self, **kwargs) -> None:
        pass

    def train(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def eval(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')
    
    def optimize(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def get_lr_group_dicts(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def optimize_state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')


    def load_state_dict(self, **kwargs):
         raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def load_optimize_state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def __call__(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')
    
    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

class TGS(torch.nn.Module, SaverMixin):
    def __init__(self, configs):
        super().__init__()
        self.image_tokenizer = META_ARCH_REGISTRY.get(configs['model']['image_tokenizer']['name'])(configs['model']['image_tokenizer'])  
        
        assert configs['model']['camera_embedder']['name'] == 'tgs.models.networks.MLP'
        weights = configs['model']['camera_embedder'].pop("weights", None)
        self.camera_embedder = META_ARCH_REGISTRY.get(configs['model']['camera_embedder']['name'])(configs['model']['camera_embedder'])  
        if weights:
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(configs['model']['image_feature'])

        self.tokenizer = META_ARCH_REGISTRY.get(configs['model']['tokenizer']['name'])(configs['model']['tokenizer'])  

        self.backbone = META_ARCH_REGISTRY.get(configs['model']['backbone']['name'])(configs['model']['backbone'])  

        self.post_processor = META_ARCH_REGISTRY.get(configs['model']['post_processor']['name'])(configs['model']['post_processor'])

        self.renderer = META_ARCH_REGISTRY.get(configs['model']['renderer']['name'])(configs['model']['renderer'])

        # pointcloud generator
        self.pointcloud_generator = META_ARCH_REGISTRY.get(configs['model']['pointcloud_generator']['name'])(configs['model']['pointcloud_generator'])

        self.point_encoder =  META_ARCH_REGISTRY.get(configs['model']['point_encoder']['name'])(configs['model']['point_encoder'])

        # weights: Optional[str] = None
        # weights_ignore_modules: Optional[List[str]] = None
        # # load checkpoint
        # if self.cfg.weights is not None:
        #     self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)


    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # generate point cloud
        out = self.pointcloud_generator(batch)
        pointclouds = out["points"]

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)

        camera_feats = self.camera_embedder(camera_feats)

        input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], 'B Nv H W C -> B Nv C H W'),
            modulation_cond=camera_feats,
        )
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)

        # get image features for projection
        image_features = self.image_feature(
            rgb = batch["rgb_cond"],
            mask = batch.get("mask_cond", None),
            feature = input_image_tokens
        )

        # only support number of input view is one
        c2w_cond = batch["c2w_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)
        proj_feats = points_projection(pointclouds, c2w_cond, intrinsic_cond, image_features)

        point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats], dim=-1))
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        rend_out = self.renderer(scene_codes,
                                query_points=pointclouds,
                                additional_features=proj_feats,
                                **batch)

        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))

            for index, render_image in enumerate(out["comp_rgb"][b]):
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
        


@register_model
def TGS(configs, device):
    from ..aux_mapper import Image_3DGS_Optimize_AuxMapper
    model_input_mapper = Image_3DGS_Optimize_AuxMapper(configs['model']['input_aux'])

    model = TGS(configs)
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))
    model.to(device)
    model.optimize_setup(optimize_configs=configs['optim'])

    if comm.is_main_process():
        logging.debug(f'初始化的总参数数量:{sum(p.numel() for p in model.parameters())}')
        logging.debug(f'初始化的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if comm.get_world_size() > 1:
        # broadcast_buffers = False
        model = DDP(model, device_ids=[comm.get_local_rank()], find_unused_parameters=True, broadcast_buffers = False)

    return model, train_samplers, train_loaders, eval_function
