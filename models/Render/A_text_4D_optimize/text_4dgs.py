
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
import os
import json
import logging
import torchvision.transforms.functional as Trans_F
import time
from models.registry import register_model
from models.optimization.scheduler import build_scheduler 
from data_schedule import build_schedule
from detectron2.modeling import META_ARCH_REGISTRY


# 每个都是一个scene
class Align_Your_Gaussians(nn.Module):
    def __init__(
        self,
        configs,):
        super().__init__()
        self.render_model = META_ARCH_REGISTRY.get(configs['model']['render_model']['name'])(configs['model']['render_model'])
        self.sds_model = META_ARCH_REGISTRY.get(configs['model']['sds_model']['name'])(configs['model']['sds_model'])
        self.loss_weight = configs['model']['loss_weight']

        self.training_time = torch.tensor(0.).float()

        # initialization
        # time for initalization
        initialize_time = time.time()

        self.training_time += time.time() - initialize_time


    
    def initialize_render_model(self, configs):
        pass

    @property
    def device(self):
        return self.sds_model.device
    
    def forward(self, batch_dict): 
        assert self.training
        view = torch.randn([3])
        text = batch_dict['text']
        # phi
        rendering = self.render_model(view)

        loss_value_dict = self.sds_model(rendering=rendering,
                                         text=text)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict): # text -> model
        return {
            'scene_representation': self.render_model
        }

@register_model
def align_your_gaussians(configs, device):
    model = Align_Your_Gaussians(configs)
    model.to(device)
    log_lr_group_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation': 5}
    optimizer, log_lr_group_idx, scheduler = model.get_optimizer(configs)

    # assert configs['optim']['scheduler']['name'] == 'static_gs_xyz'
    # scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    from .render_aux_mapper import Text_4DGS_AuxMapper
    model_input_mapper = Text_4DGS_AuxMapper(configs['model']['input_aux'])
    train_samplers, train_loaders, eval_function = build_schedule(configs,  model_input_mapper.mapper, partial(model_input_mapper.collate))

    return model, optimizer, scheduler, train_samplers, train_loaders, log_lr_group_idx, eval_function
