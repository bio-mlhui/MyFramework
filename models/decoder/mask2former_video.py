
# multi-scale features, b c h w -> module -> obj queries, predictions, b nq c
import torch.nn as nn
from models.layers.decoder_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from models.layers.anyc_trans import MLP
import torch.nn.functional as F
import torch
import copy
from models.layers.utils import zero_module, _get_clones
from models.layers.position_encoding import build_position_encoding
from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment
from models.layers.matching import batch_dice_loss, batch_sigmoid_ce_loss, batch_sigmoid_focal_loss, dice_loss, ce_mask_loss
from detectron2.modeling import META_ARCH_REGISTRY
import detectron2.utils.comm as comm
import data_schedule.utils.box_ops as box_ops
from data_schedule.utils.segmentation import small_object_weighting
from models.layers.utils import zero_module
from utils.misc import is_dist_avail_and_initialized
from collections import defaultdict
from detectron2.projects.point_rend.point_features import point_sample
from torch.cuda.amp import autocast

    # def scores_loss(self, layer_gscore_output, matching_indices,  targets):
    #     pass
    #     #     is_valid = targets['isvalid'] # list[ni], batch
    #     #     referent_idx = targets['gt_referent_idx'] # list[int], batch
    #     #     ref_is_valid = torch.tensor([isva[ridx].any() for isva, ridx in zip(is_valid, referent_idx)]).bool() # b
    #     #     num_refs = (ref_is_valid.int().sum())
    #     #     match_as_gt_indices = [] # list[int], bt
    #     #     for ref_idx, (tgt_idx, src_idx) in zip(referent_idx,  matching_indices): # b
    #     #         sel_idx = src_idx.tolist().index(ref_idx)
    #     #         match_as_gt_idx = tgt_idx[sel_idx]
    #     #         match_as_gt_indices.append(match_as_gt_idx.item())
    #     #     match_as_gt_indices = torch.tensor(match_as_gt_indices).long().to(layer_gscore_output.device) # b
    #     #     choose_loss = F.cross_entropy(layer_gscore_output[ref_is_valid], match_as_gt_indices[ref_is_valid], reduction='none') # b
    #     #     return {'objdecoder_reason': choose_loss.sum() / num_refs}

# q thw
class Video_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass