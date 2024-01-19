# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random

import torch
import torchvision.transforms.functional as F
from einops import rearrange
from copy import deepcopy as dcopy
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
from detectron2.utils.registry import Registry
RVOS_AUG_REGISTRY = Registry('RVOS_AUG')

# size can be min_size (scalar) or (w, h) tuple
def get_size_with_aspect_ratio(image_size, size, max_size=None):
    """
    Input:
        - image_size: 图片的原先大小
        - size: 较短边的目标长度
        - max_size: 如果 放大较短边 导致 较长边 大于max_size
    """
    # 保持ratio不变，
    # 让较短边resize到size，如果较长边大于了max_size, 则依照较长边到max_size进行resize
    # 返回最终的大小(h_target, w_target)
    w, h = image_size
    # 确定较短边的最终长度, 防止较长边大于max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)

def get_tgt_size(image_size, size, max_size=None):
    # if size is like [w, h], then just scale the images to that(会改变长短比)
    if isinstance(size, (list, tuple)):
        return size[::-1]
    # else if size is a number (短边的目标长度), then we need to determine the final size（固定长短比)
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)


class RandomHorizontalFlip:
    """
    将video, texts, masks 进行左右转换, 不考虑box 
    texts可以是exist_texts, 也可以是referent_text
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, old_ret):
        ret = dcopy(old_ret)
        if random.random() < self.p:
            video = ret['video']
            w, h = video[0].size
            flipped_video = [F.hflip(frame) for frame in video]
            ret['video'] = flipped_video

            if 'exist_texts' in ret:
                exist_texts = ret['exist_texts']
                flipped_texts = [q.replace('left', '@').replace('right', 'left').replace('@', 'right') for q in exist_texts]
                ret['exist_texts'] = flipped_texts

            if "masks" in ret:
                # N t h w, bool
                ret['masks'] = ret['masks'].flip(-1)

            if 'referent_text' in ret:
                refer = ret['referent_text']
                hfipped_refer = refer.replace('left', '@').replace('right', 'left').replace('@', 'right')
                ret['referent_text'] = hfipped_refer                


            if 'pred_boxes' in ret:
                boxes = ret["pred_boxes"] # n t 4, x1y1x2y2
                valid_box_idxs = boxes.any(-1, keepdim=True) # n t 1
                # n tt/t (-x2+w y1 -x1+w y2)
                boxes = boxes[:, :, [2, 1, 0, 3]] * (torch.tensor([-1, 1, -1, 1])[None, None, :]) + torch.tensor([w, 0, w, 0])[None, None, :] 
                boxes = torch.where(valid_box_idxs.repeat(1, 1, 4), boxes, torch.zeros_like(boxes))
                ret['pred_boxes'] = boxes

            if "pred_masks" in ret:
                # N t'(train)/t(eval_callback) h w, bool
                ret['pred_masks'] = ret['pred_masks'].flip(-1)

            if 'callback_fns' in ret:
                ret['callback_fns'].append(RandomHorizontalFlip(1.))
         
        return ret


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

    def __call__(self, old_ret):
        ret = dcopy(old_ret)
        video = ret['video']
        orig_size = video[0].size # w h
        tgt_size = get_tgt_size(video[0].size, random.choice(self.sizes), self.max_size) # h w

        resized_video = [F.resize(frame, tgt_size) for frame in video]
        ratio_width, ratio_height = tuple(float(s) / float(s_orig) for s, s_orig in zip(tgt_size[::-1], orig_size))
        ret['video'] = resized_video

        if "masks" in ret:
            # N t' h w, bool
            masks = ret['masks']
            # nearest 不会改变0-1值
            masks = torch.nn.functional.interpolate(masks.float(), tgt_size, mode='nearest', align_corners=None).bool()
            ret['masks'] = masks
        
        if "pred_masks" in ret:
            # N t h w, float
            pred_masks = ret['pred_masks']
            pred_masks = torch.nn.functional.interpolate(pred_masks, tgt_size, mode='bilinear', align_corners=True)
            ret['pred_masks'] = pred_masks

        if "pred_boxes" in ret:
            boxes = ret["pred_boxes"] # n t x1y1x2y2
            # n t (x1*rw y1*rh x2*rw y2*rh)
            scaled_boxes = boxes * (torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])[None, None, :])
            ret["pred_boxes"] = scaled_boxes


        if 'callback_fns' in ret:
            ret['callback_fns'].append(_RandomResize(sizes=[orig_size], max_size=None))
        
        return ret
    
class Aug:
    def __call__(self, ret):
        return self.aug(ret)    
    
class ToTensor:
    @staticmethod
    def to_pil_image(old_ret):
        ret = dcopy(old_ret)
        video = ret['video'] # t 3 h w, 必须是float
        assert video.dtype == torch.float and (video.max() <= 1) and (video.min() >=0)
        frames = [F.to_pil_image(tensor) for tensor in video]
        ret['video'] = frames
        return ret

    def __call__(self, old_ret):
        ret = dcopy(old_ret)
        # list[pil] -> t 3 h w
        video = ret['video']
        tensor_video = torch.stack([F.to_tensor(frame) for frame in video], dim=0) # 3 h w, float, 0-1
        ret['video'] = tensor_video

        if 'callback_fns' in ret:
            ret['callback_fns'].append(ToTensor.to_pil_image)
        return ret


@RVOS_AUG_REGISTRY.register()
class No_Aug(Aug):
    def __init__(self, configs):
        super().__init__()
        self.aug = Compose([
            ToTensor(),
        ])



@RVOS_AUG_REGISTRY.register()
class Hflip_RandomResize(Aug):
    def __init__(self, configs):
        super().__init__()
        sizes = configs['sizes'] # list[(w_final, h_final)] / list[短边的目标长度, 保持ratio]
        max_size = configs['max_size']
        self.aug =  Compose([
            RandomHorizontalFlip(0.5),
            _RandomResize(sizes, max_size),
            ToTensor(),
        ])

@RVOS_AUG_REGISTRY.register()
class RandomResize(Aug):
    def __init__(self, configs):
        super().__init__()
        sizes = configs['sizes'] # list[(w_final, h_final)] / list[短边的目标长度, 保持ratio]
        max_size = configs['max_size']
        self.aug =  Compose([
            _RandomResize(sizes, max_size),
            ToTensor(),
        ])

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ret):
        for t in self.transforms:
            ret = t(ret)
        return ret

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

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
