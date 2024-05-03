
from detectron2.utils.registry import Registry
RENDER_VIEWS_SAMPLER_REGISTRY = Registry('RENDER_VIEWS_SAMPLER')  # 多个帧
import random
import torch


@RENDER_VIEWS_SAMPLER_REGISTRY.register()
class Naive_MultiView3D_InOut_Sampler:
    # naive: 没有使用外部模型或者数据, # 没有考虑每一帧的情况, 只是按照下标进行抽样
    def __init__(self, 
                 sampler_configs=None, 
                 dataset_meta=None, 
                 **kwargs):      
        self.in_out_sizes =  list(sampler_configs['in_out_sizes']) # [(4, 8), (3, 6)]
        self.multiview_distribute = sampler_configs['multiview_distribute'] # dense, sparse
        # sparse_both:
        #   in_out一块抽, 然后选in个， 剩下的是out
        # in的几个是global, out的几个是另外的global

    def __call__(self, 
                 all_cameras=None, # list[str]
                 **kwargs):
        in_size, out_size = random.choice(self.in_out_sizes)
        if self.multiview_distribute == 'sparse_both':
            chosen_idxs = torch.randperm(len(all_cameras))[:(in_size + out_size)]
            if len(chosen_idxs) < (in_size + out_size):
                chosen_idxs = chosen_idxs + [chosen_idxs[-1]] * (in_size + out_size - len(chosen_idxs))
            chosen_cameras = [all_cameras[haosen] for haosen in chosen_idxs]
            return chosen_cameras[:in_size], chosen_cameras[-out_size:]
        else:
            raise NotImplementedError()


