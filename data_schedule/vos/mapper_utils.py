from .vos_aug import VOS_AUG_REGISTRY
import torch
from copy import deepcopy as dcopy
import cv2 as cv
from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.registry import Mapper

class VOS_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.get_frames_fn = dataset_meta.get('get_frames_fn') # yVOS/a2ds提供一个get_frames_fn

class VOS_TrainMapper(VOS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')    
        assert mapper_config['augmentation']['name'] in ['Hflip_RandomResize'] # 比RVOS多得多
        self.augmentation = VOS_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])

    # vos的目标 每帧可以通过connectedComponents转换成instance-level 
    def map_to_frame_targets(self, clip_targets):
        """ 返回完全新的东西, 而不是更改输入
        'masks': t' h w, True/False

        list of [dict], t'
        'masks': n h w,
        'boxes': n 4,
        """
        clip_rets = dcopy(clip_targets)
        masks = clip_rets['masks'] # t' h w
        ret = []
        for frame_mk in masks: 
            # h w, bool
            num_objs_plus_ground, labeled_mask = cv.connectedComponents(cv.Mat(frame_mk.numpy().astype('uint8') * 255))  # BBDT algorithm
            labeled_mask = torch.from_numpy(labeled_mask) 
            all_obj_ids = list(range(num_objs_plus_ground)[1:]) # [1, 2]
            instance_masks = torch.stack([labeled_mask == obj_id for obj_id in all_obj_ids], dim=0) # N h w
            instance_boxes = torch.stack([bounding_box_from_mask(ins_mask) for ins_mask in instance_masks], dim=0) # N 4
            # x1y1x2y2
            frame_targets = {
                'masks': instance_masks,
                'boxes': instance_boxes
            }
            
            ret.append(frame_targets)
        return ret



class VOS_EvalMapper(VOS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        assert mapper_config['augmentation']['name'] in ['RandomResize']
        self.augmentation = VOS_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        


