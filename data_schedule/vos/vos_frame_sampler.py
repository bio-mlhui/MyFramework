

from detectron2.utils.registry import Registry
import random
import numpy as np
import torch
""" 
frame_idx, all_frames -> frames
包含怎么抽其他帧, 
    method, position,
合法的定义,
    naive sampler是指不借助任何model, data 选择其他帧, 没有任何其他知识, 
    比如要用到optical flow确定动作的快慢, 动作快的抽的scale大一些, 动作小的丑的scale小一些
    比如要用到其他model, 或者model内部的一些知识, 则需要把sample放到model里, model的接口也要变

和VOS_aug一样, 和model-data api无关, 只关注重要的东西
"""
VOS_FRAMES_SAMPLER_REGISTRY = Registry('VOS_FRAMES_SAMPLER')

import random
class Frames_Sampler:
    def __init__(self) -> None:
        pass

class Frames_Sampler_Eval(Frames_Sampler):
    def __init__(self) -> None:
        super().__init__()

class Frames_Sampler_Train(Frames_Sampler):
    def __init__(self) -> None:
        super().__init__()


@VOS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_Center_Local_Global_Sparse_Train(Frames_Sampler_Train):
    # naive: 没有使用外部模型或者数据
    # center: 参考帧在中间
    def __init__(self, 
                 sampler_config,
                 dataset_meta,
                 **kwargs):
        super().__init__()
        self.clip_size = sampler_config['clip_size']
        # self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')
        # # clip method['name'] clip_method['config]
        # self.clip_position = mapper_config['sampling']['clip_position']  # clip_position是其他帧相对于参考帧的位置, former, latter, center
        # self.clip_method = mapper_config['sampling']['clip_method']      # clip_method抽取其他帧的方法: continue / sparse[scales]
    def __call__(self, 
                 frame_idx=None,
                 all_frames=None, # list[str]
                 **kwargs):
        video_len = len(all_frames)
        sample_indx = [frame_idx]
        if self.clip_size != 1:
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_idx - sample_id_before), min(video_len - 1, frame_idx + sample_id_after)]
            sample_indx.extend(local_indx)
            # global sampling
            if self.clip_size > 3:
                all_inds = list(range(video_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = self.clip_size - len(sample_indx)
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


@VOS_FRAMES_SAMPLER_REGISTRY.register()
class Naive_Hybrid_Temporal_Scales(Frames_Sampler_Train):
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