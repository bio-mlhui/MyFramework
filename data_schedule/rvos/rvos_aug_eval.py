# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import torch
import torchvision.transforms.functional as F
from einops import rearrange
from copy import deepcopy as dcopy
from .rvos_aug_utils import get_size_with_aspect_ratio, get_tgt_size, RVOS_EVAL_AUG_REGISTRY
from data_schedule.rvos.apis import RVOS_Aug_CallbackAPI
from data_schedule.utils.segmentation import bounding_box_from_mask
import copy
"""
所有naive augmentation的公共代码, 没有考虑api的形式, 所以每个mapper内部需要对其 aug的公共接口,
对于api_specific augmentation, 也可以加到这里, 只不过要保证别的代码不能使用这个augmentation,
对于model-aware augmentation, 需要加到aux_mapper里, 要注意aux_mapper只能添加dict, 不能改变video_dict/refer_dict 

rvos_aug公共接口
{'video', 'referent_text'/'exist_texts', 'masks', 'boxes, 'callback_fns', 'has_ann', 'class_labels', 'referent_objs'}

假设train_mapper 不考虑 box (虽然实现了), 会自动从mask得到box
eval_mapper 的callback会输入同样的接口, 所以box也要是x1y1x2y2的形式

future:
    考虑在temporal维度上进行resize
    要实现callback, 
    aug(meta) -> input
    model(input) -> prediction
    aug_callback(prediction) -> meta_prediction,

    model_aware augmentation不用考虑实现aug_callback, 因为没有改变video_dict, text_dict中的主要东西

    model.sample 需要返回一个dict, 这个dict的api和augmentation的接口一样

"""
# class RandomHorizontalFlip:
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, ret):
#         if random.random() < self.p:
#             video = ret['video']
#             w, h = video[0].size
#             flipped_video = [F.hflip(frame) for frame in video]
#             ret['video'] = flipped_video

#             if 'all_refer_exps' in ret:
#                 all_refer_exps = ret['all_refer_exps']
#                 all_refer_exps = [q.replace('left', '@').replace('right', 'left').replace('@', 'right') for q in all_refer_exps]
#                 ret['all_refer_exps'] = all_refer_exps

#             if 'referent_text' in ret:
#                 referent_text = ret['referent_text']
#                 referent_text = referent_text.replace('left', '@').replace('right', 'left').replace('@', 'right')
#                 ret['referent_text'] = referent_text                

#             if 'pred_boxes' in ret:
#                 boxes = ret["pred_boxes"] # n t 4, x1y1x2y2
#                 valid_box_idxs = boxes.any(-1, keepdim=True) # n t 1
#                 # n tt/t (-x2+w y1 -x1+w y2)
#                 boxes = boxes[:, :, [2, 1, 0, 3]] * (torch.tensor([-1, 1, -1, 1])[None, None, :]) + torch.tensor([w, 0, w, 0])[None, None, :] 
#                 boxes = torch.where(valid_box_idxs.repeat(1, 1, 4), boxes, torch.zeros_like(boxes))
#                 ret['pred_boxes'] = boxes

#             if "pred_masks" in ret:
#                 # N t'(train)/t(eval_callback) h w, bool
#                 ret['pred_masks'] = ret['pred_masks'].flip(-1)

#             if 'callback_fns' in ret:
#                 ret['callback_fns'].append(RandomHorizontalFlip(1.))
         
#         return ret


class _RandomResize:
    def __init__(self, sizes, max_size=None):
        """
        Input:  
            - sizes: 
                list of (w_final, h_final): 
                    你就是想resize到这些大小, 此时max_size不起作用
                list of (size of shorter side):
                    保持ratio进行resize
                    给出较短边的目标长度,
                        如果max_size没有给出, 则就是将较短边resize到这个长度, 较长边resize到对应大小
                        如果max_size给出, 则是将短边resize到该长度, 如果此时较长边超过了max_size,则按照 较长边放大大max_size 进行resize
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, ret):
        video = ret['video']
        orig_size = video[0].size # w h
        tgt_size = get_tgt_size(video[0].size, random.choice(self.sizes), self.max_size) # h w

        resized_video = [F.resize(frame, tgt_size) for frame in video]
        ratio_width, ratio_height = tuple(float(s) / float(s_orig) for s, s_orig in zip(tgt_size[::-1], orig_size))
        ret['video'] = resized_video

        if 'callback_fns' in ret:
            RVOS_Aug_CallbackAPI
            ret['callback_fns'].append(_RandomResize(sizes=[orig_size], max_size=None))

        if "pred_masks" in ret:
            assert (len(self.sizes) == 1) and (self.max_size == None)
            RVOS_Aug_CallbackAPI
            pred_masks = ret['pred_masks'] # list[nt h w], t
            pred_masks = [torch.nn.functional.interpolate(mk.unsqueeze(0).float(), tgt_size, mode='nearest')[0].bool()
                          for mk in pred_masks]
            ret['pred_masks'] = pred_masks # list[nt h w], t

        if "pred_boxes" in ret:
            RVOS_Aug_CallbackAPI
            pred_boxes = ret["pred_boxes"] # list[nt 4], t
            scaled_boxes = [bx * (torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])[None, :])
                            for bx in pred_boxes]
            ret["pred_boxes"] = scaled_boxes

        return ret
    
    
  
class VideoToPIL:
    def __call__(self, ret):
        video = ret['video'] # t 3 h w ->
        assert video.dtype == torch.float and (video.max() <= 1) and (video.min() >=0)  
        pil_video = [F.to_pil_image(frame, mode='RGB') for frame in video] # 3 h w, float, 0-1
        ret['video'] = pil_video
        assert 'callback_fns' not in ret
        return ret


class VideoToTensor:
    def __call__(self, ret):
        video = ret['video']
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0) # t 3 h w, float, 0-1
        ret['video'] = tensor_video

        if 'callback_fns' in ret:
            RVOS_Aug_CallbackAPI
            ret['callback_fns'].append(VideoToPIL())

        return ret


@RVOS_EVAL_AUG_REGISTRY.register()
class RVOS_Eval_Default:
    def __init__(self, configs):
        # self.hflip = RandomHorizontalFlip(0.5)
        self.resize = _RandomResize(sizes=[360], max_size=640)
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret

@RVOS_EVAL_AUG_REGISTRY.register()
class RandomResize:
    def __init__(self, configs):
        # self.hflip = RandomHorizontalFlip(0.5)
        self.resize = _RandomResize(sizes=configs['sizes'], max_size=configs['max_size'])
        self.tensor_video = VideoToTensor()

    def __call__(self, ret):
        ret = self.resize(ret)
        ret = self.tensor_video(ret)        
        return ret
    


# @RVOS_AUG_REGISTRY.register()
# class Fix_Size(Aug):
#     def __init__(self, configs) -> None:
#         super().__init__()
#         fixsize = configs['fixsize'] # [360, 584]
#         assert isinstance(fixsize, (list, tuple))
#         assert len(fixsize) == 2 # w, h
#         self.aug = Compose([
#             _RandomResize([fixsize]),
#             ToTensor(),
#         ])
    

# @RVOS_AUG_REGISTRY.register()
# class Hflip_Fix_Size(Aug):
#     def __init__(self, configs):
#         super().__init__()
#         fixsize = configs['fixsize']
#         assert isinstance(fixsize, (list, tuple))
#         assert len(fixsize) == 2 # w, h
#         self.aug =  Compose([
#             RandomHorizontalFlip(0.5),
#             _RandomResize([fixsize]),
#             ToTensor(),
#         ])

# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
 
 # class Check:
#     def __init__(self,):
#         pass
#     def __call__(self, video, texts, target):
#         if "boxes" in target or "masks" in target:
#             if "masks" in target:
#                 # n t h w -> n t hw
#                 keep = target['masks'].flatten(2).any(-1) # n t
#             else:
#                 # n t 4 -> n t 2 2
#                 boxes = rearrange(target['boxes'], 'n t (s xy) -> n t s xy', s=2, xy=2)
#                 # n t 2 > n t 2 -> n t
#                 keep = torch.all(boxes[:, :, 1, :] > boxes[:, :, 0, :], dim=-1) 
#             target['valid'] = keep.bool()
#         return  video, texts, target


# class RandomSelect:
#     """
#     Randomly selects between transforms1 and transforms2,
#     with probability p for transforms1 and (1 - p) for transforms2
#     """
#     def __init__(self, transforms1, transforms2, p=0.5):
#         self.transforms1 = transforms1
#         self.transforms2 = transforms2
#         self.p = p

#     def __call__(self, ret):
#         if random.random() < self.p:
#             return self.transforms1(ret)
#         return self.transforms2(ret)

# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, video, texts, target):
#         assert type(video) is torch.Tensor
#         normalized_images = F.normalize(video, mean=self.mean, std=self.std)
#         return normalized_images, texts, target
