
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

from data_schedule.utils.segmentation import bounding_box_from_mask
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import RVOS_TrainMapper, RVOS_EvalMapper

from data_schedule.registry import RVOS_TrainAPI_referent_text_clipped_video, \
    RVOS_Train_API_exist_texts_clipped_video, RVOS_Train_API_exist_texts_not_clipped, RVOS_EvalAPI_referent_text_clipped_video_request_ann

# 一个视频, 在某些帧的性能上model当前不好, 然后抽取这些帧, 这个就可以使用(video, text), 
# 如果clip_size是None, 模型的输入也是整个视频, 然后根据model的某些指标去抽取一些frames
# 这个model的API就是 RVOS_Train_API_exist_texts_not_clipped, 就是说让model去决定clip sampling的过程
# 注意如果模型用到了exist texts, 那么dataset的每个sample 不用再区分那个text被refer, 这样会重复的

@MAPPER_REGISTRY.register()
class RVOS_VideoForEachRefer_EvalMapper(RVOS_EvalMapper):
    """
    一整个video全输入model里进行测试, offline测试, 得到整个video的mask prediction
    """
    def __init__(self,
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,
                 ): 
        assert dataset_name in ['yrvos_test_ForEachReferText']
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict):
        RVOS_EvalAPI_referent_text_clipped_video_request_ann
        # 'exp': 
        # 'video_id':
        # 'exp_id':
        # 'all_frames'
        # 'meta_idx'
        video_id, all_frames, exp_id, expression = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['exp_id'],  data_dict['exp']
        # list[Image], t
        video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
        expression = self.normalize_text_fn(expression)
        width, height = video_frames[0].size
        T = len(all_frames)
        # call back 有分割结果
        ret = {
            'video': video,
            'referent_text': expression,
            'callback_fns': []
        }
        ret = self.augmentation(ret)
        video = ret.pop('video')
        referent_text = ret.pop('referent_text')
        callback_fns = ret.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        ret['video_dict'] = {'video': video}
        ret['refer_dict'] = {'text': referent_text}
        ret['meta'] = {
            'video_id': video_id,
            'exp_id': exp_id,
            'frames': all_frames,
            'request_ann': torch.ones(T).bool(),
            'callback_fns': callback_fns # 
        }
        return ret


@MAPPER_REGISTRY.register()
class RVOS_StepForEachRefer_ClipForEachRefer_TrainMapper(RVOS_TrainMapper):  
    def __init__(self,
                 dataset_name,
                 configs,
                 mode, # 比如train/evaluate的meta格式是不同的
                 meta_idx_shift,
                 ):  
        """
        clip_global_targets_map_to_local_targets
            对于clip/frame targets, 是否剔除掉clip/frame中没有出现的 objects
            这个api只有referent text, 没有考虑到exist texts, 所以应该把global targets映射奥local targets
            对于那些使用了exist texts的, 应该加上global targets, 比如要做langauge as queries
        legimate_clip: 
            intersect_not_none: 这个句子refer的objects至少有一个出现在这个clip/里
            all_in: 这个句子refer的objects必须全部出现在这个clip里
        frame_targets_is_refer 
            frame targets是否考虑要加上referent_obj_indexs, 
            如果True, 而且clip_global_targets_map_to_local_targets也是True,
            那么frame_targets也按照legimate_clip 

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
        good_dataset_names = ['yrvos_train_step[6]_ForEachReferText']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        self.step_size = dataset_meta.get('step_size')

        from .rvos_frame_sampler import RVOS_FRAMES_SAMPLER_REGISTRY
        self.frames_sampler = RVOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_config=mapper_config['frames_sampler'],
                                                                                                        dataset_meta=dataset_meta)
        self.legimate_clip = mapper_config['legitimate_clip'] # intersect_not_none
        

        self.frame_targets_is_refer = mapper_config['frame_targets_is_refer'] # 
        self.clip_global_targets_map_to_local_targets = mapper_config['clip_global_targets_map_to_local_targets'] 

    def _call(self, old_data_dict):
        data_dict = copy.deepcopy(old_data_dict)
        # 'video_id': 
        # 'exp_id': 
        # 'frame_idx': 
        # 'all_frames': list[str] 
        # 'all_exps': {exp_id: {exp, obj_ids}}, obj_id从1开始, 0是background
        # 'all_objs': {obj_id: {class_label}}
        RVOS_TrainAPI_referent_text_clipped_video
        video_id, all_frames, frame_idx, exp_id, all_exps, all_objs, = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['frame_idx'], data_dict['exp_id'], \
                data_dict['all_exps'], data_dict['all_objs']
        all_obj_ids = list(all_objs.keys()) # [1, 2, 5, 4]
        referent_objs = all_exps[exp_id]['obj_ids'] # [1, 4] 
        assert len(list(set(all_obj_ids))) == len(all_obj_ids)
        assert len(list(set(referent_objs))) == len(referent_objs) # 保证annotation里, referent_objs没有重复
        assert set(referent_objs).issubset(all_obj_ids) # 保证不refer背景, 因为背景不包含在all_obj_ids

        class_labels = [all_objs[key]['class_label'] for key in all_obj_ids] # [8, 10, 20 34]
        referent_text = self.normalize_text_fn(all_exps[exp_id]['exp'])

        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(frame_idx=frame_idx, all_frames=all_frames, video_id=video_id,)
            # t' h w, has_ann, obj_ids
            frames_mask, has_ann = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames)
            appear_objs = frames_mask.unique() # [0, 1, 2, 5, 4, 10]
            assert set(appear_objs.tolist()).issubset(set([0] + all_obj_ids))
            if self.legimate_clip == 'intersect_not_none':
                re_sample = len(set(appear_objs.tolist()) & set(referent_objs)) == 0 # 直到抽到的几帧里 和 referent objs有交集
            elif self.legimate_clip == 'all_in':
                re_sample = not (set(referent_objs).issubset(set(appear_objs.tolist())))
                # 比如必须 refer的多个物体必须同时出现
            else:
                raise ValueError()
            sampled_counts += 1
            if sampled_counts > 50:
                logging.error('sampled two much times')
                raise RuntimeError()
            
        # 应该把mask, 0,1,2...也看成一个图像, 用pillow.resize
        frames_mask = torch.stack([frames_mask == obj_id for obj_id in all_obj_ids], dim=0) # N t' h w, bool
        video_frames = self.get_frames_fn(video_id=video_id, frames=sampled_frames) # list[PIL.Image]
        width, height = video_frames[0].size

        aug_ret = {
            'video': video_frames,
            'referent_text': referent_text,
            'masks': frames_mask, # N t' h w
            'has_ann': has_ann,
            'class_labels': torch.tensor(class_labels), # N
            'referent_objs': [all_obj_ids.index(ro) for ro in referent_objs], # 去掉了背景
        }

        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')
        referent_text = aug_ret.pop('referent_text')

        # x1y1x2y2
        N, T = aug_ret['masks'].shape[:2]
        boxes = torch.stack([bounding_box_from_mask(mask) for mask in copy.deepcopy(aug_ret['masks']).flatten(0, 1)], dim=0) # Nt 4
        boxes = rearrange(boxes, '(N T) c -> N T c', N=N, T=T)
        boxes[:, :, 0::2].clamp_(min=0, max=width)
        boxes[:, :, 1::2].clamp_(min=0, max=height)
        aug_ret['boxes'] = boxes

        frame_targets = self.map_to_frame_targets(aug_ret)
        
        if self.clip_global_targets_map_to_local_targets:
            aug_ret = self.map_global_targets_to_local_targets(aug_ret)

        ret = {}
        ret['video_dict'] = {'video': video}
        ret['refer_dict'] = {'text': referent_text}
        ret['targets'] = aug_ret # video mask的第一种表示
        ret['frame_targets'] = frame_targets
        # frame_targets # video mask的第二种表示
        return ret

    def map_to_frame_targets(self, clip_targets):
        """ 返回完全新的东西, 而不是更改输入
        'masks': N t' h w,
        'boxes': N t' 4
        'class_labels': N
        'referent_objs': list[int], 按照N来的, index

        list of [dict], t'
        'masks': N/n h w,
        'boxes': N/n 4,
        'class_labels': N/n,
        'referent_objs': list[int], 按照N/n来的
        """
        clip_rets = copy.deepcopy(clip_targets)
        masks = clip_rets['masks'].transpose(0, 1) # t' N h w
        boxes = clip_rets['boxes'].transpose(0, 1) # t' N 4
        class_labels = clip_rets['class_labels'] # [10, 32, 10, 4]
        referent_objs = clip_rets['referent_objs'] # [1, 0]

        assert len(masks) == len(boxes)
        ret = []
        for frame_mk, frame_bx in zip(masks, boxes):
            frame_targets = {
                'masks': frame_mk.unsqueeze(1), # N 1 h w
                'boxes': frame_bx.unsqueeze(1), # N 1 4
                'class_labels': class_labels, # N
            }

            if self.frame_targets_is_refer:
                frame_targets['referent_objs'] = referent_objs # list[int]
            
            if self.clip_global_targets_map_to_local_targets:
                frame_targets = self.map_global_targets_to_local_targets(frame_targets)

            frame_targets['masks'] = frame_targets['masks'].squeeze(1)
            frame_targets['boxes'] = frame_targets['boxes'].squeeze(1)
            ret.append(frame_targets)
        return ret

    def map_global_targets_to_local_targets(self, ret_with_global_targets):
        """ 去掉clip/frame中没有出现的global object的target
        'masks': N t' h w,
        'boxes': N t' 4
        'class_labels': N
        'referent_objs': list[int], 按照N来的, index

        'masks': n t' h w,
        'boxes': n t' 4,
        'class_labels': n,
        'referent_objs': list[int], 按照n来的, index
        """
        ret = copy.deepcopy(ret_with_global_targets)
        masks = ret['masks'] # N t' h w
        boxes = ret['boxes'] # N t' 4
        class_labels = ret['class_labels'] # N
        # 每个global object是否出现在了这个clip/frame
        global_obj_appear = masks.flatten(1).any(-1) # N [True, False, True, False, False, False, True]
        masks = masks[global_obj_appear] # n t' h w
        boxes = boxes[global_obj_appear] # n t' 4
        class_labels = class_labels[global_obj_appear] # n
        ret['masks'] = masks
        ret['boxes'] = boxes
        ret['class_labels'] = class_labels

        if 'referent_objs' in ret: 
            appear_global_obj_idxs = torch.where(global_obj_appear)[0].tolist() # [0, 2, 6] / [0, 2, 4, 8, 9, 10, 16]
            referent_obj_idxs = ret['referent_objs'] #  [2, 4, 5] / [4 ,10]
            if self.legimate_clip == 'intersect_not_none':
                referent_obj_idxs = list(set(referent_obj_idxs) & set(appear_global_obj_idxs)) # [2] 
                assert len(referent_obj_idxs) != 0, '保证至少有一个referent obj出现在这个clip里' 

            elif self.legimate_clip == 'all_in':
                for refer_idx in referent_obj_idxs:  # [4, 10]
                    assert refer_idx in appear_global_obj_idxs, '保证每个referent obj都在这个clip里'
            else:
                raise ValueError()

            referent_obj_idxs = [appear_global_obj_idxs.index(ref_obj_id) for ref_obj_id in referent_obj_idxs] # [1] / [2, 5]
            ret['referent_objs'] = referent_obj_idxs
        return ret
    

# 'yrvos_test_step[6]_ForEachReferText'
# online eval
@MAPPER_REGISTRY.register()
class RVOS_StepForEachRefer_EvalMapper(RVOS_EvalMapper):
    """一个video被分成相同size的clips进行eval/ 一个video的每帧进行eval, 这个可以弄成online测试
    frame_idx是参考帧, 
    """
    def __init__(self, 
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,) -> None:
        assert dataset_name in ['yrvos_test_step[1]_ForEachReferText']
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
        RVOS_EvalAPI_referent_text_clipped_video_request_ann
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

# sampling加在model中
@MAPPER_REGISTRY.register()
class RVOS_VideoForEachRefer_VideoForEachRefer_TrainMapper(RVOS_TrainMapper):
    """一整个video全输入model里进行训练 / 一整个video进行clip sampling进行训练
    """ 
    def __init__(self,
                 configs,
                 dataset_name,
                 mode, # 比如train/evaluate的meta格式是不同的
                 meta_idx_shift,
                 
                 # train, 训练的时候怎么抽一个sample
                 train_clip_size=None,
                 train_temporal_scales=None,
                 # 围绕obj 主要出现的时刻附近，按照不同的temporal scale抽取, 也是Naive的, 没有模型和其他数据的加成
                 # 随机抽取一个window, 直到有这个物体
                 ):  
        good_dataset_names = ['yrvos_train', 'yrvos_train_valSplit[300][2024]']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('clip_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=MetadataCatalog.get(dataset_name),
                         mapper_config=mapper_config)
        self.train_clip_size = mapper_config['train_clip_size']
        self.train_temporal_scales = mapper_config['train_temporal_scales']

    def _call(self, data_dict):
        # video_id
        # all_frames: list[str]
        # all_exps: {exp_id: {exp, obj_ids}}, 这个video的所有expressions, 
        # all_objs: {obj_id: {class_label: 0, frames,}}, 
        # exp_id: 0
        # meta_idx
        # N 整个video里的物体数量
        # n_r 某个句子refer的物体的数量
        # n_t 这个clip中出现的物体的数量
        # T 整个video的长度
        # t 抽到的clip的长度
        # t' 抽到的clip中有annotation的长度
        assert exp_id in all_exps
        video_id, all_frames, all_exps, all_objs, exp_id = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['all_exps'], data_dict['all_objs'], data_dict['exp_id']
        # list[int], n_r
        referred_obj_ids = all_exps[exp_id]['obj_ids'] 
        # list[Image], t; N t' h w, t(boo)
        video_frames, all_obj_masks, has_ann = self.sample_training_valid_frames(all_frames=all_frames,
                                                                            referred_obj_ids=referred_obj_ids)
        all_obj_class_labels = [all_objs[key]['class_label'] for key in all_objs]
        # list[str], N
        exist_texts = [self.normalize_text(all_exps[key]['exp']) for key in all_exps.keys()]
        exist_texts_referents = [all_exps[key]['obj_ids'] for key in all_exps.keys()]
        all_exp_ids = list(all_exps.keys())
        referent_text_idx = all_exp_ids.index(exp_id)

        width, height = video_frames[0].size
        targets = {
            'has_ann': has_ann, # t (bool)
            'masks': all_obj_masks, # N t' h w (bool) 
            'class_labels': all_obj_class_labels, # N
            'referent_text_idx': referent_text_idx, # int
            'exist_texts': exist_texts, # list[str], N
            'exist_texts_referents': exist_texts_referents, # list[ list[int], ]
            'orig_size': torch.tensor([len(video_frames), height, width]), # T h w
            'size': torch.tensor([len(video_frames),  height, width]), # T h w
            # validate
        } # 训练不需要知道video_id, 
        video_frames, targets = self.augmentation(video_frames, targets)
        return video_frames, targets
           
    # 抽取出一个视频里包含referred_obj_ids的几帧
    # list[Image], T
    def sample_all_frames_to_frames(self, all_frames, referred_obj_ids):
        # 这个exp对应的所有objects, 四个鸟，3个鸟在clip内, 1个在clip外, 算不算?
        # 算
        appear_obj_ids = []
        while len(set(appear_obj_ids) & set(referred_obj_ids)) == 0:
            sampled_frames = self.sample_clip(all_frames)
            # list[t h w], list[id]
            frame_masks, appear_obj_ids = self.get_all_objs_masks(video_id=vid, all_frames=sampled_frames)
            # 如果数据集中出错了, 一个(video, text)可能没有对应, 则会陷入循环        
        pass
  
  
@MAPPER_REGISTRY.register()
class RVOS_VideoForEachRefer_ClipForEachRefer_TrainMapper(RVOS_TrainMapper):
    """一整个video全输入model里进行训练 / 一整个video进行clip sampling进行训练
    """ 
    def __init__(self,
                 configs,
                 dataset_name,
                 mode, # 比如train/evaluate的meta格式是不同的
                 meta_idx_shift,
                 
                 # train, 训练的时候怎么抽一个sample
                 train_clip_size=None,
                 train_temporal_scales=None,
                 # 围绕obj 主要出现的时刻附近，按照不同的temporal scale抽取, 也是Naive的, 没有模型和其他数据的加成
                 # 随机抽取一个window, 直到有这个物体
                 ):  
        good_dataset_names = ['yrvos_train', 'yrvos_train_valSplit[300][2024]']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('clip_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=MetadataCatalog.get(dataset_name),
                         mapper_config=mapper_config)
        self.train_clip_size = mapper_config['train_clip_size']
        self.train_temporal_scales = mapper_config['train_temporal_scales']

    def _call(self, data_dict):
        # video_id
        # all_frames: list[str]
        # all_exps: {exp_id: {exp, obj_ids}}, 这个video的所有expressions, 
        # all_objs: {obj_id: {class_label: 0, frames,}}, 
        # exp_id: 0
        # meta_idx
        # N 整个video里的物体数量
        # n_r 某个句子refer的物体的数量
        # n_t 这个clip中出现的物体的数量
        # T 整个video的长度
        # t 抽到的clip的长度
        # t' 抽到的clip中有annotation的长度
        RVOS_TrainAPI_referent_text_clipped_video
        assert exp_id in all_exps
        video_id, all_frames, all_exps, all_objs, exp_id = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['all_exps'], data_dict['all_objs'], data_dict['exp_id']
        # list[int], n_r
        referred_obj_ids = all_exps[exp_id]['obj_ids'] 
        # list[Image], t; N t' h w, t(boo)
        video_frames, all_obj_masks, has_ann = self.sample_training_valid_frames(all_frames=all_frames,
                                                                            referred_obj_ids=referred_obj_ids)
        all_obj_class_labels = [all_objs[key]['class_label'] for key in all_objs]
        # list[str], N
        exist_texts = [self.normalize_text(all_exps[key]['exp']) for key in all_exps.keys()]
        exist_texts_referents = [all_exps[key]['obj_ids'] for key in all_exps.keys()]
        all_exp_ids = list(all_exps.keys())
        referent_text_idx = all_exp_ids.index(exp_id)

        width, height = video_frames[0].size
        targets = {
            'has_ann': has_ann, # t (bool)
            'masks': all_obj_masks, # N t' h w (bool) 
            'class_labels': all_obj_class_labels, # N
            'referent_text_idx': referent_text_idx, # int
            'exist_texts': exist_texts, # list[str], N
            'exist_texts_referents': exist_texts_referents, # list[ list[int], ]
            'orig_size': torch.tensor([len(video_frames), height, width]), # T h w
            'size': torch.tensor([len(video_frames),  height, width]), # T h w
            # validate
        } # 训练不需要知道video_id, 
        video_frames, targets = self.augmentation(video_frames, targets)
        return video_frames, targets
           
    # 抽取出一个视频里包含referred_obj_ids的几帧
    # list[Image], T
    def sample_all_frames_to_frames(self, all_frames, referred_obj_ids):
        # 这个exp对应的所有objects, 四个鸟，3个鸟在clip内, 1个在clip外, 算不算?
        # 算
        appear_obj_ids = []
        while len(set(appear_obj_ids) & set(referred_obj_ids)) == 0:
            sampled_frames = self.sample_clip(all_frames)
            # list[t h w], list[id]
            frame_masks, appear_obj_ids = self.get_all_objs_masks(video_id=vid, all_frames=sampled_frames)
            # 如果数据集中出错了, 一个(video, text)可能没有对应, 则会陷入循环        
        pass
  
    def map_global_targets_to_local_targets(self, dict):
        # 整个视频的N个物体, 映射为这个clip的n个物体
        # N t h w -> n_t t h w
        appear_obj_ids = sorted(appear_obj_ids) # 1，2，3, 5
        annotated_exps_by_object = []
        masks_by_object = [] 
        obj_classes_by_object = []
        for obj_id in appear_obj_ids:
            masks_by_object.append(all_objects_masks == obj_id) # t h w, uint8
            obj_classes_by_object.append(self.catname_to_id[all_objs_dict[str(obj_id)]["category"]])
            obj_exps = [value['exp'] for key, value in all_exps_dict.items() if int(value['obj_id']) == obj_id]
            if len(obj_exps) == 0:
                logging.debug('there are some objects that in the video has no expressions')
            annotated_exps_by_object.append(obj_exps)
        masks = torch.stack(masks_by_object, dim=0) # n t h w, bool
        class_labels = torch.tensor(obj_classes_by_object).long() # n
        referent_idx = appear_obj_ids.index(int(all_exps_dict[exp_id]['obj_id'])) # 在1, 2, 3, 5中的下标  


@MAPPER_REGISTRY.register()
class RVOS_Video_VideoExistTexts_TrainMapper():
    pass
@MAPPER_REGISTRY.register()
class RVOS_Video_ClipExistTexts_TrainMapper():
    pass
    # exist_texts

# exist_texts = [self.normalize_text_fn(all_exps[key]['exp']) for key in all_exps.keys()]
# exist_texts_referents = [all_exps[key]['obj_ids'] for key in all_exps.keys()]
@MAPPER_REGISTRY.register()
class RVOS_Step_ClipExistTexts_TrainMapper():
    pass






