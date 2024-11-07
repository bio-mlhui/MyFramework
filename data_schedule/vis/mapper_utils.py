from .vis_aug_utils import VIS_EVAL_AUG_REGISTRY, VIS_TRAIN_AUG_REGISTRY
import torch
from copy import deepcopy as dcopy
import cv2 as cv
from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.registry import Mapper
import copy
from data_schedule.vis.apis import VIS_TrainAPI_clipped_video

class VIS_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.get_frames_fn = dataset_meta.get('get_frames_fn')

class VIS_TrainMapper(VIS_Mapper):

    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')   
        self.clip_global_targets_map_to_local_targets = mapper_config['clip_global_targets_map_to_local_targets'] 
 
        # assert mapper_config['augmentation']['name'] in ['Hflip_RandomResize', 'WeakPolyP_TrainAug', 'Flanet_TrainAug',
        #                                                  'WeakPolyP_TrainAug_RotateImageToClip']
        if 'augmentation' in mapper_config:
            self.augmentation = VIS_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])

    def map_to_frame_targets(self, clip_targets):
        VIS_TrainAPI_clipped_video
        clip_rets = copy.deepcopy(clip_targets)
        masks = clip_rets['masks'].transpose(0, 1).contiguous() # t' N h w
        class_labels = clip_rets['classes'] # [10, 32, 10, 4]
        has_box = 'boxes' in clip_rets
        if has_box:
            boxes = clip_rets['boxes'].transpose(0, 1).contiguous() # t' N 4
            assert len(masks) == len(boxes)
        ret = []
        for idx, frame_mk in enumerate(masks):
            frame_targets = {
                'masks': frame_mk.unsqueeze(1), # N 1 h w
                'classes': class_labels, # N
            }
            if has_box:
                frame_targets.update({'boxes': boxes[idx].unsqueeze(1)}) # N 1 4
            if self.clip_global_targets_map_to_local_targets:
                frame_targets = self.map_global_targets_to_local_targets(frame_targets)
            frame_targets['masks'] = frame_targets['masks'].squeeze(1)
            if has_box:
                frame_targets['boxes'] = frame_targets['boxes'].squeeze(1)
            ret.append(frame_targets)
        return ret

    def map_global_targets_to_local_targets(self, ret):
        VIS_TrainAPI_clipped_video
        masks = ret['masks'] # N t' h w
        # 每个global object是否出现在了这个clip/frame
        global_obj_appear = masks.flatten(1).any(-1) # N [True, False, True, False, False, False, True]
        ret['masks'] = ret['masks'][global_obj_appear]
        ret['classes'] = ret['classes'][global_obj_appear]
        if 'boxes' in ret:
            ret['boxes'] = ret['boxes'][global_obj_appear] # n t' 4
        return ret
    

class VIS_EvalMapper(VIS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.augmentation = VIS_EVAL_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        

def get_frames_from_middle_frame(all_frames, mid_frame_id, step_size):
    # 根据step_size进行选择中间的
    # 5, 7 -> range(2, 9):2345678; 5, 6 -> range(2, 8):234567
    ann_frame_idx = all_frames.index(mid_frame_id)
    all_idxs = list(range(ann_frame_idx - step_size// 2, ann_frame_idx + (step_size+1)//2))
    all_idxs = torch.tensor(all_idxs).clamp(0, max=len(all_frames)-1).int() # annotate_frame_idx = T//2
    frames = [all_frames[idx] for idx in all_idxs]
    return frames

import torch.nn.functional as F
def bilinear_resize_mask(mask, shape):
    # h w, uint8, 0-255, label, bilinear
    H, W = mask.shape
    unique_labels = mask.unique()
    lab_to_mask = []
    for lab in unique_labels:
        binary_mask = (mask == lab).float()
        binary_mask = F.interpolate(binary_mask[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0]
        lab_to_mask.append(binary_mask)
    lab_to_mask = torch.stack(lab_to_mask, dim=-1) # h w num_class
    new_mask = lab_to_mask.max(dim=-1)[1] # h w, indices
    new_label = unique_labels[new_mask.flatten()].reshape(shape)
    return new_label


def bilinear_semantic_resize_mask(mask, shape):
    # k h w, bool, -> k h w
    mask = F.interpolate(mask[None, ...].float(), size=shape, align_corners=False, mode='bilinear')[0] > 0.5 # k h w
    return mask