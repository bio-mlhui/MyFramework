
class VIDenoise_Meta:
    """
    'video_id': str,
    'frames' : list[str]
    """

class VIDenoise_OptimizeMapper:
    """
    'video': list[pil Image, rgb], t
    'masks': n t' h w, bool
    'has_ann': t, bool
    'classes': n,
    'class_names': list[str], n
    'boxes': n t' 4, x1y1x2y2绝对值, 如果是instance的话, semantic segmentation没有box

    模型测试的输出api: 每一帧有多个mask/box预测, 每个预测都有类别的概率, 最后一个类别是背景类
    'video': t 3 h w, 0-1
    'pred_masks': list[no h w], t bool
    'pred_boxes': list[no 4], t, x1y1x2y2绝对值
    'pred_class': list[no c] t, 概率值
    'callback_fns': list[fn]
    """

class VIDenoise_LearnMapper:
    """每一帧的预测
    'video_id': str,
    'frame_name': str,
    'masks': list[rle], no
    'boxes': no 4, x1y1x2y2绝对值
    'classes': no c, softmax之后, 最后一个类别是背景类
    """