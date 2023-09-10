import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pycocotools.mask as mask_util
from einops import rearrange


class A2DSentencesPostProcess_referdiff(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self):
        super(A2DSentencesPostProcess_referdiff, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size, resized_sample_sizes, orig_sample_sizes):
        """ Perform the computation
        Inputs:
            - outputs: 
                pred_masks: T(t_target*b num_queries mask_H, mask_W)   
                pred_is_referred: T(t_target*b num_queries, 1)
            resized_padded_sample_size: (H, W)
            resized_sample_sizes: list[(Resized_H Resized_W)], t_valid*b
            orig_sample_sizes: list[Original_H, Original_W], t_valid*b
        """
        # T(t_target*b n)
        scores = outputs['pred_is_referred']
        # T(t_target*b n mask_H mask_W)
        pred_masks = outputs['pred_masks']
        pred_masks = F.interpolate(pred_masks, size=resized_padded_sample_size, mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  # remove the samples' padding
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(f_pred_masks_no_pad.float(), size=orig_size, mode="nearest-exact")
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{'scores': s, 'masks': m, 'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions




