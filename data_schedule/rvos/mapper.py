
import json
import os
from typing import List
from functools import partial
from PIL import Image

import torch
import logging
from einops import rearrange
import torchvision.transforms.functional as F

from data_schedule.utils.segmentation import bounding_box_from_mask
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import RVOS_TrainMapper, RVOS_EvalMapper

from data_schedule.rvos.apis import RVOS_Dataset, RVOS_TrainAPI_ForEachRefer_clipped_video, RVOS_Aug_CallbackAPI,\
      RVOS_FrameSampler_InputOutput_API, RVOS_EvalAPI_referent_text_clipped_video_request_ann
from .rvos_frame_sampler import RVOS_FRAMES_SAMPLER_REGISTRY

# 一个视频, 在某些帧的性能上model当前不好, 然后抽取这些帧, 这个就可以使用(video, text), 
# 如果clip_size是None, 模型的输入也是整个视频, 然后根据model的某些指标去抽取一些frames
# 这个model的API就是 RVOS_Train_API_exist_texts_not_clipped, 就是说让model去决定clip sampling的过程
# 注意如果模型用到了exist texts, 那么dataset的每个sample 不用再区分那个text被refer, 这样会重复的

# request_ann: all
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
        assert dataset_name in ['yrvos_test_ForEachRefer', 'a2ds_test_ForEachRefer']
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('step_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict):
        RVOS_Dataset
        video_id, all_frames, exp_id, expression = data_dict['video_id'], \
            data_dict['all_frames'], data_dict['exp_id'],  data_dict['referent_text']
        # list[Image], t
        video_frames = self.get_frames_fn(video_id=video_id, frames=all_frames)
        expression = self.normalize_text_fn(expression)
        width, height = video_frames[0].size
        T = len(all_frames)
        # call back 有分割结果
        ret = {
            'video': video_frames,
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


# request_ann: frame_idx
# clip_for_each_refer的评估方式: step1 + 中间帧dense左右抽取
@MAPPER_REGISTRY.register()
class RVOS_StepForEachRefer_EvalMapper(RVOS_EvalMapper):
    def __init__(self, 
                 configs,
                 dataset_name,
                 mode,
                 meta_idx_shift,) -> None:
        assert dataset_name in ['yrvos_test_step[1]_ForEachRefer', 'a2ds_test_step[1]_ForEachRefer']
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        assert dataset_meta.get('step_size') is not None 
        self.all_request = mapper_config['all_request']
        self.frames_sampler = RVOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_configs=mapper_config['frames_sampler'],
                                                                                                       dataset_meta=dataset_meta)

    def _call(self, data_dict):
        RVOS_Dataset
        
        video_id, all_frames, frame_idx, exp_id, referent_text = \
            data_dict['video_id'], data_dict['all_frames'], data_dict['frame_idx'], data_dict['exp_id'],  data_dict['referent_text']
        
        referent_text = self.normalize_text_fn(referent_text)        
        RVOS_FrameSampler_InputOutput_API
        video_frames_paths = self.frames_sampler(frame_idx=frame_idx, 
                                            all_frames=all_frames, 
                                            video_id=video_id,) # 各种frame_sampler需要的所有数据
        if self.all_request:
            request_ann = torch.ones(len(video_frames_paths)).bool() # t
        else:
            request_ann = torch.zeros(len(video_frames_paths)).bool() # t
            request_ann[video_frames_paths.index(all_frames[frame_idx])] = True
        video_frames = self.get_frames_fn(video_id=video_id, frames=video_frames_paths)
        aug_ret = {
            'video': video_frames,
            'request_ann': request_ann,
            'referent_text': referent_text,
            'callback_fns': []
        }
        RVOS_Aug_CallbackAPI
        aug_ret = self.augmentation(aug_ret)
        video = aug_ret.pop('video')
        referent_text= aug_ret.pop('referent_text')
        callback_fns = aug_ret.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        RVOS_EvalAPI_referent_text_clipped_video_request_ann
        return {
            'video_dict': {'video': video},
            'refer_dict': {'text': referent_text},
            'meta': {
                'video_id': video_id,
                'exp_id': exp_id,
                'frames': [fm for idx, fm in enumerate(video_frames_paths) if request_ann[idx]], # t'
                'request_ann': request_ann,
                'callback_fns': callback_fns             
            }
        }


@MAPPER_REGISTRY.register()
class RVOS_Video_Or_Step_ForEachRefer_ClipForEachRefer_TrainMapper(RVOS_TrainMapper):  
    def __init__(self,
                 dataset_name,
                 configs,
                 mode,
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
        good_dataset_names = ['yrvos_train_step[6]_ForEachRefer',  'yrvos_train_step[12]_ForEachRefer', 'yrvos_train_ForEachRefer',
                              'a2ds_train_step[6]_ForEachRefer',]
        assert (dataset_name in good_dataset_names) and (mode == 'train')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        self.frames_sampler = RVOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_configs=mapper_config['frames_sampler'],
                                                                                                        dataset_meta=dataset_meta)
        self.legimate_clip = mapper_config['legitimate_clip'] # intersect_not_none / all_in

    def _call(self, data_dict):
        RVOS_Dataset
        video_id, referent_text, referent_objs, all_objs, all_frames  = \
            data_dict['video_id'], self.normalize_text_fn(data_dict['referent_text']), data_dict['referent_objs'], \
                 data_dict['all_objs'], data_dict['all_frames'], 
        frame_idx = data_dict['frame_idx'] if 'frame_idx' in data_dict else None

        all_obj_ids = list(all_objs.keys()) # [1, 2, 5, 4]
        assert len(list(set(all_obj_ids))) == len(all_obj_ids)
        assert len(list(set(referent_objs))) == len(referent_objs)
        assert set(referent_objs).issubset(set(all_obj_ids)), f'referents: {referent_objs}, all_objs: {all_obj_ids}'
        assert 0 not in referent_objs, '还没考虑过text refer了背景'

        class_labels = [all_objs[key]['class_label'] for key in all_obj_ids] # [8, 10, 20 34]

        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(frame_idx=frame_idx, 
                                                 all_frames=all_frames, 
                                                 video_id=video_id,
                                                 referent_objs=referent_objs, 
                                                 referent_text=referent_text,
                                                 all_obj_ids=all_obj_ids) # 各种frame_sampler需要的所有数据
            # t' h w, int, 0,1,2,3; has_ann t
            return_format = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames, all_obj_ids=all_obj_ids)
            if len(return_format) == 2:
                frames_mask, has_ann = return_format
                appear_objs = frames_mask.unique() # [0, 1, 2, 5, 4, 10]
                assert set(appear_objs.tolist()).issubset(set([0] + all_obj_ids))
                if self.legimate_clip == 'intersect_not_none':
                    re_sample = len(set(appear_objs.tolist()) & set(referent_objs)) == 0 # 出现了至少一个referent obj
                elif self.legimate_clip == 'all_in':
                    re_sample = not (set(referent_objs).issubset(set(appear_objs.tolist()))) # # 比如必须 refer的多个物体必须同时出现
                else:
                    raise ValueError()
                frames_mask = torch.stack([frames_mask == obj_id for obj_id in all_obj_ids], dim=0) # N t' h w, bool
            elif len(return_format) == 3:
                # N t' h w
                obj_masks, has_ann, _ = return_format
                ref_appear = torch.tensor([obj_masks[all_obj_ids.index(ref_obj_id)].any() for ref_obj_id in referent_objs])
                if self.legimate_clip == 'intersect_not_none':
                    re_sample = not (ref_appear.any())
                elif self.legimate_clip == 'all_in':
                    re_sample = not (ref_appear.all()) # # 比如必须 refer的多个物体必须同时出现
                else:
                    raise ValueError()
                frames_mask = obj_masks
            else:
                raise ValueError()
                
            sampled_counts += 1
            if sampled_counts > 50:
                logging.error('sampled two much times')
                return None # another index
            
        video_frames = self.get_frames_fn(video_id=video_id, frames=sampled_frames) # list[PIL.Image]
        width, height = video_frames[0].size

        aug_ret = {
            'video': video_frames,
            'referent_text': referent_text,
            'referent_objs': [all_obj_ids.index(ro) for ro in referent_objs], 
            'masks': frames_mask, # N t' h w, bool
            'has_ann': has_ann, # t, 可以确定这个referent有没有出现在那个帧, 没有就是没有
            'classes': torch.tensor(class_labels), # N
        }
        RVOS_Aug_CallbackAPI
        aug_ret = self.augmentation(aug_ret)

        video = aug_ret.pop('video')
        referent_text = aug_ret.pop('referent_text')
        frame_targets = self.map_to_frame_targets(aug_ret)
        if self.clip_global_targets_map_to_local_targets:
            aug_ret = self.map_global_targets_to_local_targets(aug_ret)

        RVOS_TrainAPI_ForEachRefer_clipped_video
        ret = {}
        ret['video_dict'] = {'video': video}
        ret['refer_dict'] = {'text': referent_text}
        ret['targets'] = aug_ret
        ret['frame_targets'] = frame_targets
        return ret


# 每个sample:
# clip_allexists == clip
# clip_for_each_refer != clip
@MAPPER_REGISTRY.register()
class RVOS_Video_Or_Step_AllExists_ClipAllExists_TrainMapper(RVOS_TrainMapper): 
    def __init__(self,
                 configs,
                 dataset_name,
                 mode, # 比如train/evaluate的meta格式是不同的
                 meta_idx_shift,
                 ):  
        good_dataset_names = ['yrvos_train', 'yrvos_train_valSplit[300][2024]']
        assert dataset_name in good_dataset_names
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        assert dataset_meta.get('clip_size') == None
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        raise NotImplementedError()
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=MetadataCatalog.get(dataset_name),
                         mapper_config=mapper_config)
        self.step_size = dataset_meta.get('step_size')
        self.frames_sampler = RVOS_FRAMES_SAMPLER_REGISTRY.get(mapper_config['frames_sampler']['name'])(sampler_config=mapper_config['frames_sampler'],
                                                                                                        dataset_meta=dataset_meta)
        self.legimate_clip = mapper_config['legitimate_clip'] # intersect_not_none / all_in

    def _call(self, data_dict):
        RVOS_Dataset
        video_id, all_frames, all_exps, all_objs= \
            data_dict['video_id'], data_dict['all_frames'], data_dict['all_exps'], data_dict['all_objs']
        
        # exist_texts = [self.normalize_text_fn(all_exps[key]['exp']) for key in all_exps.keys()]
        # exist_texts_referents = [all_exps[key]['obj_ids'] for key in all_exps.keys()]
        # 'all_objs': {obj_id(int): {'class_label': 0,}},
        # all_exps: {exp_id: {exp, obj_ids}}
        frame_idx = data_dict['frame_idx'] if 'frame_idx' in data_dict else None
        all_obj_ids = list(all_objs.keys()) # [1, 2, 5, 4]

        re_sample = True
        sampled_counts = 0
        while re_sample:
            sampled_frames = self.frames_sampler(frame_idx=frame_idx, 
                                                 all_frames=all_frames, 
                                                 video_id=video_id,
                                                 referent_objs=referent_objs,
                                                 referent_text=referent_text,
                                                 all_objs=all_objs) # 各种frame_sampler需要的所有数据
            # t' h w, int, obj_ids; has_ann t
            frames_mask, has_ann = self.get_frames_mask_fn(video_id=video_id, frames=sampled_frames)
            appear_objs = frames_mask.unique() # [0, 1, 2, 5, 4, 10]
            assert set(appear_objs.tolist()).issubset(set([0] + all_obj_ids))
            if self.legimate_clip == 'intersect_not_none':
                re_sample = len(set(appear_objs.tolist()) & set(referent_objs)) == 0 # 出现了至少一个referent obj
            elif self.legimate_clip == 'all_in':
                re_sample = not (set(referent_objs).issubset(set(appear_objs.tolist()))) # # 比如必须 refer的多个物体必须同时出现
            else:
                raise ValueError()
            sampled_counts += 1
            if sampled_counts > 50:
                logging.error('sampled two much times')
                raise RuntimeError()
            
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

