

from detectron2.utils.registry import Registry
from detectron2.utils import comm
import logging
import random
import numpy as np
# 你说整个clip_sampler可不可以让模型去决定clip那些帧? 可以的, 就不叫naive sampler了, 那个代码需要写到model里

"""
合法的定义,
    比如必须 refer的多个物体必须同时出现
    比如只要有交集就行

    naive sampler是指不借助任何model, data 选择其他帧, 没有任何其他知识, 
    比如要用到optical flow确定动作的快慢, 动作快的抽的scale大一些, 动作小的丑的scale小一些
    比如要用到其他model, 或者model内部的一些知识, 则需要把sample放到model里, model的接口也要变

和rvos_aug一样, 和model-data api无关, 只关注重要的东西
"""
RVOS_FRAMES_SAMPLER_REGISTRY = Registry('RVOS_FRAMES_SAMPLER')
import torch
# TODO: 让model的当前状态/model参数决定抽取哪些帧, 每个model的sampling方式不一样
# RVOS训练集 -> model训练利用了多个text, model可以控制如何进行clip sampling
@RVOS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_ReferenceFrame_FrameSampler:
    # naive: 纯纯的就是抽帧, 没有使用外部模型或者数据, # 没有考虑每一帧的情况, 只是按照下标进行抽样
    def __init__(self, sampler_configs, dataset_meta, **kwargs):
        assert dataset_meta.get('name') in ['yrvos_test_step[1]_ForEachRefer',
                                            'a2ds_test_step[1]_ForEachRefer',  
                                            'yrvos_train_step[6]_ForEachRefer',
                                            'yrvos_train_step[12]_ForEachRefer',
                                            'a2ds_train_step[6]_ForEachRefer']
        self.reference_frame_step_size = dataset_meta.get('step_size')

        self.clip_sizes = list(sampler_configs['clip_sizes']) # list[int]
        self.clip_distribute = sampler_configs['clip_distribute'] # dense, sparse, local_global
        self.clip_position = sampler_configs['clip_position'] # former, center, latter

        if (max(self.clip_sizes) > self.reference_frame_step_size):
            if dataset_meta.get('mode') == 'train':
                if comm.is_main_process():
                    logging.warning('训练的clip大小大于数据集的step size,可能会造成训练时每个sample之间帧重复')
            elif dataset_meta.get('mode') == 'evaluate':
                if comm.is_main_process():
                    logging.warning('测试的clip大小大于数据集的step size, 注意要控制request ann')               
            else:
                raise ValueError()
    # 如果是test_step[1]_foreachrefer
    # clip_distribute 是local_global, clip_position是former, 那么可以做online setting
    def __call__(self, 
                 frame_idx=None,
                 all_frames=None, # list[str]
                 **kwargs):
        random_clip_size = random.choice(self.clip_sizes)
        video_len = len(all_frames)
        sample_indx = [frame_idx]
        if (self.clip_position == 'center') and (self.clip_distribute == 'local_global'):
            if random_clip_size != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_idx - sample_id_before), min(video_len - 1, frame_idx + sample_id_after)]
                sample_indx.extend(local_indx)
                # global sampling
                if random_clip_size > 3:
                    all_inds = list(range(video_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = random_clip_size - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif video_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(video_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(video_len), global_n - video_len) + list(range(video_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])
        elif (self.clip_position == 'center') and (self.clip_distribute == 'dense'):
            half_size = (random_clip_size - 1) // 2
            # 把负的换成0, 大于最大的换成最后一帧
            sample_indx += list(range(frame_idx - half_size, frame_idx))
            sample_indx += list(range(frame_idx+1, half_size + frame_idx + 1))

            if len(sample_indx) < random_clip_size: # 向前补一个
                sample_indx = [min(sample_indx)] + sample_indx
            assert len(sample_indx) == random_clip_size
            sample_indx = torch.tensor(sample_indx)
            sample_indx = sample_indx.clamp_(min=0, max=video_len-1)
            sample_indx = sample_indx.tolist()
        else:
            raise ValueError()
                        
        sample_indx.sort()
        sampled_frames = [all_frames[idx] for idx in sample_indx]
        return sampled_frames


@RVOS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_Hybrid_Temporal_Scales:
    # naive: 没有使用外部模型或者数据
    # hybrid temporal scale: 对于一个video, 随机选一帧, 然后随机选一帧, 进行local-global
    def __init__(self, 
                 sampler_config,
                 dataset_meta,
                 **kwargs):
        super().__init__()
        self.clip_sizes = np.array(sampler_config['clip_sizes'])
        self.no_global = sampler_config['no_global']

    def __call__(self,
                 all_frames = None, # list[str]
                 **kwargs):
        video_len = len(all_frames)
        rand_clip_size = np.random.choice(self.clip_sizes, 1)[0]
        if self.no_global:
            frame_idx = random.randint(0, video_len - rand_clip_size) # 之后的几帧
            sample_index = list(range(frame_idx, frame_idx + rand_clip_size))
            assert len(sample_index) == rand_clip_size
            
        else:
            frame_idx = random.randint(0, video_len-1) # center_frame
            sample_indx = [frame_idx]
            if rand_clip_size != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_idx - sample_id_before), min(video_len - 1, frame_idx + sample_id_after)]
                sample_indx.extend(local_indx)
                # global sampling
                if rand_clip_size > 3:
                    all_inds = list(range(video_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = rand_clip_size - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif video_len >= global_n:  # sample long range global frames
                        select_id = random.sample(range(video_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(video_len), global_n - video_len) + list(range(video_len))           
                        for s_id in select_id:                                                                   
                            sample_indx.append(all_inds[s_id])

        sample_indx.sort()
        sampled_frames = [all_frames[idx] for idx in sample_indx]
        return sampled_frames

# Naive_Bulk_FrameSampler()
@RVOS_FRAMES_SAMPLER_REGISTRY.register()
class Referent_Centric_FrameSampler:
    # 要求模型能够输入 变长clip
    # 围绕obj/refernt 主要出现的时刻附近，按照不同的temporal scale抽取, frame_sampler要用到get_frame_masks_fn, refernet_objs
    # 把referent_obj 出现 的地方想象成一个bulk, 然后对这个bulk进行抽取
    def __init__(self, sampler_configs, dataset_meta, **kwargs):
        assert dataset_meta.get('name') in ['yrvos_train_ForEachRefer'] # video_for_each_refer, 没有step
    
        self.clip_sizes = list(sampler_configs['clip_sizes']) # list[int]
        self.clip_distribute = sampler_configs['clip_distribute'] # dense, sparse, local_global
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')

        self.get_each_obj_appear_frame_idxs = dataset_meta.get('get_each_obj_appear_frame_idxs')()
        # [vid_id][frmae-Idx] = list[int]

        self.legitimate_clip = sampler_configs['legitimate_clip'] # 

    def __call__(self,
                 all_frames, # list[str]
                 referent_objs,
                 video_id,
                 **kwargs):
        # 所有referent 出现的帧的交集
        # legitimate clip == 'all_in':
        if self.legitimate_clip == 'intersect_not_none':
            # 所有referent出现的帧的并集
            refer_obj_appear_frame_idxs = [self.get_each_obj_appear_frame_idxs[video_id][refer_obj_id] \
                                        for refer_obj_id in referent_objs] # list[list[int]]
            all_refer_objs_appear_frame_idxs = set(refer_obj_appear_frame_idxs[0])
            for taylor in refer_obj_appear_frame_idxs[1:]:
                all_refer_objs_appear_frame_idxs = all_refer_objs_appear_frame_idxs | set(taylor)
                        
        elif self.legitimate_clip == 'all_in':
            refer_obj_appear_frame_idxs = [self.get_each_obj_appear_frame_idxs[video_id][refer_obj_id] \
                                        for refer_obj_id in referent_objs] # list[list[int]]
            all_refer_objs_appear_frame_idxs = set(refer_obj_appear_frame_idxs[0])
            for taylor in refer_obj_appear_frame_idxs[1:]:
                all_refer_objs_appear_frame_idxs = all_refer_objs_appear_frame_idxs & set(taylor)

        else:
            raise ValueError()
        
        random_clip_size = random.choice(self.clip_sizes)
        if (self.clip_position == 'center') and (self.clip_distribute == 'local_global'):
            sample_idxs = [all_refer_objs_appear_frame_idxs[len(all_refer_objs_appear_frame_idxs) // 2]]
        if random_clip_size <= len(all_refer_objs_appear_frame_idxs):
            
            pass


        # 那你说还能第一帧obj1, 第二帧obj2, 第三帧obj3, 然后expression refer这三个物体?
        pass