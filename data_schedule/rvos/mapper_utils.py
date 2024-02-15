from .rvos_aug_utils import RVOS_EVAL_AUG_REGISTRY, RVOS_TRAIN_AUG_REGISTRY
import copy
import torch

from data_schedule.rvos.apis import RVOS_TrainAPI_ForEachRefer_clipped_video, RVOS_Aug_CallbackAPI
from data_schedule.registry import Mapper

class RVOS_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.category_to_id = dataset_meta.get('category_to_ids')
        self.normalize_text_fn = dataset_meta.get('normalize_text_fn') # yrvos/a2ds提供一个normalize_text_fn
        self.get_frames_fn = dataset_meta.get('get_frames_fn') # yrvos/a2ds提供一个get_frames_fn

class RVOS_TrainMapper(RVOS_Mapper): # ForEachRefer
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')    
    
        assert mapper_config['augmentation']['name'] in ['Hflip_RandomResize', 'RVOS_Train_Default']
        self.augmentation = RVOS_TRAIN_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        self.frame_targets_is_refer = mapper_config['frame_targets_is_refer'] 
        self.clip_global_targets_map_to_local_targets = mapper_config['clip_global_targets_map_to_local_targets'] 

    def map_to_frame_targets(self, clip_targets):
        """
        由全局N targets 得到每一帧的targets, 每一帧targets再map_local_to_global
        如果frame_targets_is_refer的话, 要保证 每一帧+referent_objs是一个legitimate clip
        """
        RVOS_Aug_CallbackAPI
        clip_rets = copy.deepcopy(clip_targets)
        masks = clip_rets['masks'].transpose(0, 1) # t' N h w
        boxes = clip_rets['boxes'].transpose(0, 1) # t' N 4
        class_labels = clip_rets['classes'] # [10, 32, 10, 4]
        referent_objs = clip_rets['referent_objs'] # list[int]

        assert len(masks) == len(boxes)
        ret = []
        for frame_mk, frame_bx in zip(masks, boxes):
            frame_targets = {
                'masks': frame_mk.unsqueeze(1).contiguous(), # N 1 h w
                'boxes': frame_bx.unsqueeze(1).contiguous(), # N 1 4
                'classes': class_labels, # N
            }
            if self.frame_targets_is_refer:
                frame_targets['referent_objs'] = referent_objs # list[int]

            if self.clip_global_targets_map_to_local_targets:
                frame_targets = self.map_global_targets_to_local_targets(frame_targets)

            frame_targets['masks'] = frame_targets['masks'].squeeze(1)
            frame_targets['boxes'] = frame_targets['boxes'].squeeze(1)
            ret.append(frame_targets)
        return ret

    def map_global_targets_to_local_targets(self, ret):
        """
        如果refernet_objs是一个key的话, 输入的必须是legitimate clip
        """
        masks = ret['masks'] # N t' h w
        boxes = ret['boxes'] # N t' 4
        class_labels = ret['classes'] # N
        global_obj_appear = masks.flatten(1).any(-1) # N [True, False, True, False, False, False, True]
        assert global_obj_appear.any(), '这个clip里没有物体了'
        # 如果某一帧没有ref, clip有ref, map_to_frame_targets + per_frame_is_refer 会导致生成没有ref的帧, 
        masks = masks[global_obj_appear] # n t' h w
        boxes = boxes[global_obj_appear] # n t' 4
        class_labels = class_labels[global_obj_appear] # n
        ret['masks'] = masks
        ret['boxes'] = boxes
        ret['classes'] = class_labels

        if 'referent_objs' in ret: 
            appear_global_obj_idxs = torch.arange(len(global_obj_appear))[global_obj_appear].tolist() # [0, 2, 6] / [0, 2, 4, 8, 9, 10, 16]
            referent_obj_idxs = ret['referent_objs'] #  [2, 4, 5] / [4 ,10]
            if self.legimate_clip == 'intersect_not_none':
                # 如果是 至少出现一个referent obj的话, 那么referent_obj_idxs要更新，去掉没有出现的global referent
                referent_obj_idxs = list(set(referent_obj_idxs) & set(appear_global_obj_idxs)) # [2] 
                assert len(referent_obj_idxs) != 0, f'你输入的这个clip不是legitimate的, 出现了: {appear_global_obj_idxs}, refer的是: {referent_obj_idxs}' 

            elif self.legimate_clip == 'all_in':
                # 如果是 referent必须都出现在clip里, 那么referent_obj_idxs不用更改, 因为已经是legitimate clip了
                assert set(referent_obj_idxs).issubset(set(appear_global_obj_idxs)), f'你输入的这个clip不是legitimate的, 出现了: {appear_global_obj_idxs}, refer的是: {referent_obj_idxs}'

            elif self.legimate_clip == 'any':
                # robust referring, referent_obj_idxs 和 appear_global_obj_idxs做交 
                referent_obj_idxs = list(set(referent_obj_idxs) & set(appear_global_obj_idxs))
                # 如果referent_obj_idxs是空的话
            else:
                raise ValueError()

            referent_obj_idxs = [appear_global_obj_idxs.index(ref_obj_id) for ref_obj_id in referent_obj_idxs] # [1] / [2, 5]
            ret['referent_objs'] = referent_obj_idxs

        return ret
    

class RVOS_EvalMapper(RVOS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        assert mapper_config['augmentation']['name'] in ['RVOS_Eval_Default', 'RandomResize']
        self.augmentation = RVOS_EVAL_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        


