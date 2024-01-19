
class VOS_EvalAPI_output:
    """每一帧有多个mask预测, 每个预测都有一个概率值
    'video': t 3 h w, test_aug之后的video
    'pred_masks': list[nt h w], t 
        logits, float
    
    'pred_boxes': list[nt 4], t,
         x1y1x2y2, 绝对值

    'pred_obj': list[nt] t, 
        每一帧上每个预测的可信度
    """

class VOS_TrainAPI_clipped_video:
    """model的输入是一个clip
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'targets': {
            'has_ann': b t (bool)
            'masks': list[N t' h w] (bool)    
        }
        'frame_targets':{
            'masks': list[n h w], bt'
            'boxes': list[n 4], bt'
        }
        'meta_idx': int
        'visualize': True/False
    """

class VOS_EvalAPI_clipped_video_request_ann:
    """输入整个video, 按照request ann输出对应帧的mask
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'meta': {
            'video_id': str,
            'frames': list[str],
            'request_ann': t, bool 
            'callback_fns': list[__call__]     
        }
        'meta_idx': int
        'visualize': True/False
    """

