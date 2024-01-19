
import json
import os
from typing import List
import copy
from functools import partial
from PIL import Image
import random
import numpy as np
import torch
import logging
from einops import rearrange
import torchvision.transforms.functional as F
import cv2 as cv
from data_schedule.utils.segmentation import bounding_box_from_mask
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import VOS_TrainMapper, VOS_EvalMapper
from .vos_frame_sampler import VOS_FRAMES_SAMPLER_REGISTRY
from data_schedule.registry import VOS_TrainAPI_clipped_video, VOS_EvalAPI_clipped_video_request_ann

# 一个视频, 在某些帧的性能上model当前不好, 然后抽取这些帧, 这个就可以使用(video, text), 
# 如果clip_size是None, 模型的输入也是整个视频, 然后根据model的某些指标去抽取一些frames
# 这个model的API就是 VOS_Train_API_exist_texts_not_clipped, 就是说让model去决定clip sampling的过程
# 注意如果模型用到了exist texts, 那么dataset的每个sample 不用再区分那个text被refer, 这样会重复的

@MAPPER_REGISTRY.register()
class VOS_Video_EvalMapper(VOS_EvalMapper):
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        assert dataset_name in ['fibroid_validate', 
                                'polyp_hard_unseen_validate', 
                                'polyp_hard_seen_validate', 
                                'polyp_easy_unseen_validate',
                                'polyp_easy_seen_validate']
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict):
        VOS_EvalAPI_clipped_video_request_ann
        # 'video_id':
        # 'all_frames'
        # 'meta_idx'
        video_id, all_frames = data_dict['video_id'], data_dict['all_frames']
        # list[Image], t
        video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
        T = len(all_frames)
        # call back 有分割结果
        ret = {
            'video': video_frames,
            'callback_fns': []
        }
        ret = self.augmentation(ret)
        video = ret.pop('video')
        callback_fns = ret.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        ret['video_dict'] = {'video': video}
        ret['meta'] = {
            'video_id': video_id,
            'frames': all_frames,
            'request_ann': torch.ones(T).bool(),
            'callback_fns': callback_fns # 
        }
        return ret

@MAPPER_REGISTRY.register()
class VOS_Video_Clip_TrainMapper(VOS_TrainMapper):
    """一整个video全输入model里进行训练 / 一整个video进行clip sampling进行训练
    video to clip是video2video的genearlization, 就是换了个sampler
    """ 
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, 
                 meta_idx_shift,
                 ):  
        good_dataset_names = ['fibroid_train', 'polyp_train']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        assert dataset_meta.get('step_size') is None
        self.frames_sampler = VOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_config=mapper_config['frames_sampler'],
                                                                                                        dataset_meta=dataset_meta)

    def _call(self, data_dict):
        VOS_TrainAPI_clipped_video
        video_id, all_frames = data_dict['video_id'], data_dict['all_frames']
        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(all_frames=all_frames, 
                                                 video_id=video_id)
            # t' h w, has_ann
            frames_mask, has_ann = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames)
            re_sample = not frames_mask.any()
            sampled_counts += 1
            if sampled_counts > 50:
                logging.error('sampled two much times')
                raise RuntimeError()

        video_frames = self.get_frames_fn(video_id=video_id, 
                                          frames=sampled_frames) # list[PIL.Image]
        width, height = video_frames[0].size
        aug_ret = {
            'video': video_frames,
            'masks': frames_mask, # t' h w
            'has_ann': has_ann,
        }

        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')

        frame_targets = self.map_to_frame_targets(aug_ret)
        
        ret = {}
        ret['video_dict'] = {'video': video}
        ret['targets'] = aug_ret # video mask的第一种表示
        ret['frame_targets'] = frame_targets
        # frame_targets # video mask的第二种表示
        return ret


@MAPPER_REGISTRY.register()
class VOS_Step_Clip_TrainMapper(VOS_TrainMapper):  
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, # 比如train/evaluate的meta格式是不同的
                 meta_idx_shift,
                 ):  
        """
        从video targets转换到frame targets的逻辑:
            clip的mask targets 是N t' h w, 就是把整个video mask targets截对应的几帧; referent_objs是 N的index
            然后进行augmentation, {video, referent_text, masks, class_labels,  has_ann, referent_objs}
                augmentation 和mapper的接口是主要数据, 
                augmentation_callback和model.sample的接口和这个接口一样

            boxes targets由 masks得出, N t' 4 {video, referent_text, masks, class_labels, has_ann, referent_objs, boxes}
            -然后计算每帧的targets, 
                如果frame_targets_is_refer, 那么把referent objs重复t'放到每帧里面; 
                否则每帧只有mask, box是 N 1 h w, N 1 4
                    如果clip_global_targets_map_to_local_targets, 那么对于每帧, 去除掉没有出现的video global objects
            -如果clip_global_targets_map_to_local_targets, 对于这个clip targets, 去除掉没有出现的video global objects
        
        map_global_targets_to_local_targets函数的逻辑:
            从clip/frame mask得出每个global object是否出现, 得出出现object的appear index 
            然后mask, box, class_labels都进行index
            如果输入的dict中有'referent_objs'这个key, 
                那么referent_objs 根据legimate_clip策略, appear index 进行修改
                新的refenrt_objs是 是apper_index.index(ro) for ro in renfert_ojb
        """
        good_dataset_names = ['fibroid_train_step6','fibroid_train_step12','fibroid_train_step18']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        self.step_size = dataset_meta.get('step_size')

        self.frames_sampler = VOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_config=mapper_config['frames_sampler'],
                                                                                                        dataset_meta=dataset_meta)

    def _call(self, old_data_dict):
        data_dict = copy.deepcopy(old_data_dict)
        # 'video_id': 
        # 'exp_id': 
        # 'frame_idx': 
        # 'all_frames': list[str] 
        # 'all_exps': {exp_id: {exp, obj_ids}}, obj_id从1开始, 0是background
        # 'all_objs': {obj_id: {class_label}}
        VOS_TrainAPI_clipped_video
        video_id, all_frames, frame_idx = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['frame_idx']
      
        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(frame_idx=frame_idx, all_frames=all_frames, video_id=video_id,)
            # t' h w, has_ann
            frames_mask, has_ann = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames)
            re_sample = not frames_mask.any()
            sampled_counts += 1
            if sampled_counts > 50:
                logging.error('sampled two much times')
                raise RuntimeError()
        
        video_frames = self.get_frames_fn(video_id=video_id, frames=sampled_frames) # list[PIL.Image]
        width, height = video_frames[0].size

        aug_ret = {
            'video': video_frames,
            'masks': frames_mask, # t' h w
            'has_ann': has_ann,
        }

        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')

        frame_targets = self.map_to_frame_targets(aug_ret)
        
        ret = {}
        ret['video_dict'] = {'video': video}
        ret['targets'] = aug_ret # video mask的第一种表示
        ret['frame_targets'] = frame_targets
        # frame_targets # video mask的第二种表示
        return ret

    def map_to_frame_targets(self, clip_targets):
        """ 返回完全新的东西, 而不是更改输入
        'masks': t' h w,

        list of [dict], t'
        'masks': n h w,
        'boxes': n 4,
        """
        clip_rets = copy.deepcopy(clip_targets)
        masks = clip_rets['masks'] # t' h w

        ret = []
        for frame_mk in masks: 
            num_objs_plus_ground, labeled_mask = cv.connectedComponents(frame_mk) 
            labeled_mask = torch.from_numpy(labeled_mask) 
            all_obj_ids = range(num_objs_plus_ground)[1:] # [1, 2]
            instance_masks = torch.stack([labeled_mask == obj_id for obj_id in all_obj_ids], dim=0) # N h w
            instance_boxes = torch.stack([bounding_box_from_mask(ins_mask) for ins_mask in instance_masks], dim=0) # N 4
            # x1y1x2y2
            frame_targets = {
                'masks': instance_masks,
                'boxes': instance_boxes
            }
            
            ret.append(frame_targets)
        return ret



@MAPPER_REGISTRY.register()
class VOS_Step_EvalMapper(VOS_EvalMapper):
    """一个video被分成相同size的clips进行eval/ 一个video的每帧进行eval, 这个可以弄成online测试
    frame_idx是参考帧, 
    """
    def __init__(self, 
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,) -> None:
        assert dataset_name in ['fibroid_validate_step[1]']
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        step_size = dataset_meta.get('step_size')
        assert step_size is not None 
        self.request_ann = mapper_config['request_ann'] # request_ann是full->整个clip都需要annotation, request_ann是frame->只有参考帧需要annotation

        self.clip_size = mapper_config['sampling']['clip_size']          # clip_size是sample多少帧
        self.clip_position = mapper_config['sampling']['clip_position']  # clip_position是其他帧相对于参考帧的位置, former, latter, center
        self.clip_method = mapper_config['sampling']['clip_method']      # clip_method抽取其他帧的方法: continue / sparse[scales]

    def sample_frames(self, all_frames, frame_idx):
        video_len = len(all_frames)
        if self.clip_position == 'latter':
            # 在参考帧后 抽取一个clip
            sampled_frames = all_frames[frame_idx : min(frame_idx + self.clip_size, video_len)]
            if len(sampled_frames) < self.clip_size:
                pass
        
        if self.clip_position == 'center':
            # 左右抽取
            sample_indx = [frame_idx]
            if self.clip_size != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_idx - sample_id_before), min(frame_idx - 1, frame_idx + sample_id_after)]
                sample_indx.extend(local_indx)
    
                # global sampling
                if self.clip_size > 3:
                    all_inds = list(range(video_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = video_len - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif video_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(video_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(video_len), global_n - video_len) + list(range(video_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

        if self.clip_position == 'former':
            # 在参考帧前 抽取一个clip
            pass


    def _call(self, data_dict):
        # 'video_id'
        # 'exp_id'
        # 'exp': 
        # 'all_frames'
        # 'frame_idx'
        # 'meta_idx'
        VOS_EvalAPI_clipped_video_request_ann
        # 你说整个clip_sampler可不可以让模型去决定clip那些帧?
        data_dict = copy.deepcopy(data_dict)
        video_id, all_frames, frame_idx, exp_id, expression, meta_idx = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['frame_idx'], data_dict['exp_id'],  data_dict['exp'] , data_dict['meta_idx']
        
        sampled_clip_frames = self.sample_frames(all_frames, frame_idx)

        # list[Image], t
        clip_frames = self.get_frames_fn(video_id=video_id, frames=sampled_clip_frames)
        expression = self.normalize_text_fn(expression)
        width, height = clip_frames[0].size
        T = len(clip_frames)
        if self.request_ann == 'full':
            request_ann = torch.ones(T).bool()
        elif self.request_ann == 'frame':
            request_ann = torch.zeros(T).bool()
            request_idx = clip_frames.index(all_frames[frame_idx])
            request_ann[request_idx] = True

        ret = {
            'frames': clip_frames,
            'size': torch.tensor([T, height, width]),
            'orig_size': torch.tensor([T, height, width]),
            'request_ann': request_ann, 
            'video_id': video_id,
            'exp_id': exp_id,
        }
        video_frames, expression, ret = self.augmentation(video_frames, expression, ret)  # T(t 3 h w), list[none]
        ret['video'] = video_frames
        ret['referent_text'] = expression
        return ret









