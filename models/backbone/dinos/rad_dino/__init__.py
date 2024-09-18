
from transformers import AutoModel, AutoImageProcessor
import os
import logging
import copy
import os
from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Dict, List, Optional


import torch.nn.functional as F
import torch.nn as nn
import torch
import detectron2.utils.comm as comm
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat

import logging
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

@BACKBONE_REGISTRY.register()
class RAD_DINO(nn.Module):
    def __init__(self, configs,):
        super().__init__()
        PT_PATH = os.environ.get('PT_PATH')
        self.ssl = AutoModel.from_pretrained(os.path.join(PT_PATH, "radino/radino/rad-dino/"))
        processor = AutoImageProcessor.from_pretrained(os.path.join(PT_PATH, "radino/radino/rad-dino/"))

        for p in self.ssl.parameters():
            p.requires_grad_(False)  
        self.embed_dim = 768
        self.patch_size = 14

    def forward(self, x, n=None):
        input = {'pixel_values': x}
        outputs = self.ssl(**input)
        return {
            'features': [outputs.last_hidden_state], # b cls+hw c
        }
  
