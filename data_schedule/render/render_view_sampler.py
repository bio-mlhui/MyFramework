
from detectron2.utils.registry import Registry
RENDER_VIEWS_SAMPLER_REGISTRY = Registry('RENDER_VIEWS_SAMPLER')  # 多个帧

@RENDER_VIEWS_SAMPLER_REGISTRY.register()
class Naive_MultiView_Sampler:
    # naive: 没有使用外部模型或者数据, # 没有考虑每一帧的情况, 只是按照下标进行抽样
    def __init__(self, sampler_configs, dataset_meta, **kwargs):            
        self.reference_frame_step_size = dataset_meta.get('step_size')

        self.multiview_sizes = list(sampler_configs['multiview_sizes']) # list[int]
        self.multiview_distribute = sampler_configs['multiview_distribute'] # dense, sparse, local_global

        self.num_input_views = configs['num_input_views']
        
    def __call__(self, 
                 all_cameras=None, # list[str]
                 **kwargs):
        """
        return: input_views, output_views
        """
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

