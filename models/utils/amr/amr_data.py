import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from packaging import version
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor
from copy import deepcopy as dcopy
from einops import repeat, rearrange
import torch.nn as nn

class AMRData(object):
    def __init__(self, 
                 amr,
                 amr_seg_ids,
                 amr_feats,
                 text_feats,
                 text_pad_masks):
        self.amr = amr
        self.amr_seg_ids = amr_seg_ids
        self.amr_feats = amr_feats
        self.text_feats = text_feats
        self.text_pad_masks = text_pad_masks

    def to(self, device):
        cast_asi = self.amr_seg_ids.to(device)
        cast_af = self.amr_feats.to(device)
        cast_tf = self.text_feats.to(device)
        cast_tpm = self.text_pad_masks.to(device)

        return AMRData(self.amr,cast_asi, cast_af, cast_tf, cast_tpm)

    def repeat(self, repeats):
        amr = self.amr
        amr_seg_ids = self.amr_seg_ids
        amr_feats = self.amr_feats
        text_feats = self.text_feats
        text_pad_masks = self.text_pad_masks

        B = len(self.amr)
        # repeat amr by L times
        repeated_amrs = []
        for _ in range(repeats):
            for idx in range(B): # 为什么要考虑 amr 是none?
                repeated_amrs.append(dcopy(amr[idx]))

        amr_feats = repeat(amr_feats, 'b s c -> (L b) s c', L=repeats)
        text_feats = repeat(text_feats, 'b s c -> (L b) s c',L=repeats) if text_feats is not None else None
        amr_seg_ids = repeat(amr_seg_ids, 'b s -> (L b) s', L=repeats)
        text_pad_masks = repeat(text_pad_masks, 'b s -> (L b) s',L=repeats) if text_pad_masks is not None else None

        return AMRData(repeated_amrs, amr_seg_ids, amr_feats, text_feats, text_pad_masks)

    def decompose(self):
        return self.amr, self.amr_seg_ids, self.amr_feats, self.text_feats, self.text_pad_masks

    
    def clone(self):
        return AMRData(amr=dcopy(self.amr), 
                       amr_seg_ids=self.amr_seg_ids.clone(),
                       amr_feats=self.amr_feats.clone(),
                       text_feats=self.text_feats.clone(),
                       text_pad_masks=self.text_pad_masks.clone())
    