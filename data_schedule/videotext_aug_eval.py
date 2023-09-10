
from typing import Any
import torch
import torchvision.transforms.functional as F
from util.misc import interpolate
from functools import partial
import copy

_videotext_evalaug_entrypoints = {}

def register_videotext_evalaug(fn):
    aug_name = fn.__name__
    _videotext_evalaug_entrypoints[aug_name] = fn

    return fn


def videotext_evalaug_entrypoints(aug_name):
    try:
        return _videotext_evalaug_entrypoints[aug_name]
    except KeyError as e:
        print(f'VideoText Eval Augmentation {aug_name} not found')



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


def normalize_callback(tensor_video, texts, preds, mean, std):
    # t 3 h w [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tensor_video = F.normalize(tensor_video, mean = [ 0., 0., 0. ], std = [ 1/std[0], 1/std[1], 1/std[2] ])
    tensor_video = F.normalize(tensor_video, mean = [ -mean[0], -mean[1], -mean[2] ], std = [ 1., 1., 1. ])
    return tensor_video, texts, preds

def to_tensor_callback(tensor_video, texts, preds):
    # tensor(t 3 h w)
    return [F.to_pil_image(f) for f in tensor_video], texts, preds

def hflip_callback(hfliped_video, hfliped_texts, preds):
    video = [F.hflip(frame) for frame in hfliped_video]
    w, h = video[0].size
    texts =[q.replace('left', '@').replace('right', 'left').replace('@', 'right') for q in hfliped_texts]
    if 'boxes' in preds:
        # n t (x1 y1 x2 y2)
        boxes = preds["boxes"]
        # n t (-x2+w y1 -x1+w y2)
        preds["boxes"] = boxes[:, :, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        
    if "masks" in preds:
        # n t h w
        preds['masks'] = preds['masks'].flip(-1)

    return video, texts, preds

def resize_callback(rescaled_video, texts, preds, size):
    video = [F.resize(frame, size) for frame in rescaled_video]
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(video[0].size, video[0].size))
    ratio_width, ratio_height = ratios

    if "boxes" in preds:
        boxes = preds["boxes"] 
        # n t (x1*rw y1*rh x2*rw y2*rh)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        preds["boxes"] = scaled_boxes

    if "masks" in preds:
        # n t h w
        preds['masks'] = interpolate(preds['masks'].float(), size, mode="nearest") > 0.5
        # TODO: 如果interpolate把小物体给整没了

    return rescaled_video, texts, preds
        
    

def resize(video, size):
    # 'has_ann', 'image_id'
    rescaled_video = [F.resize(frame, size) for frame in video]
    return rescaled_video

def hflip(video, texts):
    """
    水平翻转每帧图像, 并且将对应所有object的text query进行翻转
    """
    hfliped_video = [F.hflip(frame) for frame in video]
    hfliped_texts =[q.replace('left', '@').replace('right', 'left').replace('@', 'right') for q in texts]
    return hfliped_video, hfliped_texts

def to_tensor(video):
    # list[pil]
    return torch.stack([F.to_tensor(frame) for frame in video], dim=0)

def normalize(video, mean, std):
    # t 3 h w
    return F.normalize(video, mean, std)


       
class RandomResize_HFlip_Asemble(object):
    def __init__(self, sizes, 
                 max_size,
                 use_hflip,
                 normalize_mean,
                 normalize_std):
        """
        Input:  
            - sizes: 
                list of (w_final, h_final):
            - use_hflip:
                bool
        """
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.use_hflip = use_hflip
        
        self.mean = normalize_mean
        self.std = normalize_std

    def __call__(self, video, texts, meta):
        """
        Input:  
            - video: list[pillow image]
            - text: list[str]
            - meta: 'has_ann': t
        """
        asembles = []
        
        asembles.append([
            normalize(to_tensor(video), mean=self.mean, std=self.std),
            texts,
            meta,
            [partial(normalize_callback, mean=self.mean, std=self.std),
                     to_tensor_callback,]
        ])
        
        orig_w, orig_h = video[0].size
        for tgt_size in self.sizes:
            tgt_size = get_tgt_size(video[0].size, tgt_size, max_size=self.max_size)
            resized_video = resize(copy.deepcopy(video), tgt_size)
            resized_video = normalize(to_tensor(resized_video), mean=self.mean, std=self.std)
            asembles.append([resized_video, texts, meta, [partial(normalize_callback, mean=self.mean, std=self.std),
                                                    to_tensor_callback,
                                                    partial(resize_callback, size=[orig_h, orig_w])]])
            
        if self.use_hflip:
            hfliped_video, hfliped_texts =  hflip(copy.deepcopy(video), copy.deepcopy(texts))
            hfliped_video = normalize(to_tensor(hfliped_video), mean=self.mean, std=self.std)
            asembles.append([hfliped_video, hfliped_texts, meta, [partial(normalize_callback, mean=self.mean, std=self.std),
                                                            to_tensor_callback,
                                                            hflip_callback]
                             ])
        return asembles

@register_videotext_evalaug
def randomresize_hflip_asemble(configs):
    return RandomResize_HFlip_Asemble(sizes=configs['sizes'],
                                      use_hflip=configs['use_hflip'],
                                      max_size=configs['max_size'],
                                      normalize_mean=configs['normalize_mean'],
                                      normalize_std=configs['normalize_std'])

class JustNormalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, video, texts, meta):
        asembles = []
        
        asembles.append([
            normalize(to_tensor(video), mean=self.mean, std=self.std),
            texts,
            meta,
            [partial(normalize_callback, mean=self.mean, std=self.std),
                     to_tensor_callback,]
        ])

        return asembles
    
@register_videotext_evalaug
def justnormalize(configs):
    return 