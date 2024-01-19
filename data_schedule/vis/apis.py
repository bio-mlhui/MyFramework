
class VIS_Dataset:
    """
    'video_id': str,
    'all_frames' : list[str],
    训练集:
    'all_objs': {obj_id: {'class': 0,}},

    训练集/测试集
    'frame_idx': 抽clip的时候的参考帧下标, 相对于all_frames, 具体怎么抽要传到frame_sampler里
    如果没有的话, (train)就从整个video抽取clip, (eval)或者对整个video进行测试
    """

class VIS_Aug_CallbackAPI:
    """
    'video': list[pil Image, rgb], t
    'masks': n t' h w, bool
    'has_ann': t, bool
    'classes': n,
    'class_names': list[str], n
    'boxes': n t' 4, x1y1x2y2绝对值

    模型测试的输出api: 每一帧有多个mask/box预测, 每个预测都有类别的概率, 最后一个类别是背景类
    'video': t 3 h w, 0-1
    'pred_masks': list[no h w], t bool
    'pred_boxes': list[no 4], t, x1y1x2y2绝对值
    'pred_class': list[no c] t, 概率值
    'callback_fns': list[fn]
    """

class VIS_Evaluator_OutAPI_EvalFn_API:
    """每一帧的预测
    'video_id': str,
    'frame_name': str,
    'masks': list[rle], no
    'boxes': no 4, x1y1x2y2绝对值
    'classes': no c, softmax之后, 最后一个类别是背景类
    """

class VIS_TrainAPI_clipped_video:
    """has_ann代表哪些帧有ann, 进入aux_mapper train collate的API
        'video_dict': {
            'video': t 3 h w, 0-1,
        }
        'targets': {
            'has_ann': t, (bool)
            'masks': N/n t' h w, bool 
            'boxes': N/n t' 4, x1y1x2y2绝对值
            'classes': N/n
        }
        'frame_targets':{
            'masks': N/n h w, t'
            'boxes': N/n 4, t' 
            'classes': N/n             
        }
    """

class VIS_EvalAPI_clipped_video_request_ann:
    """request ann测试哪些帧, 进入aux_mapper eval collate的API
        'video_dict': {
            'video': t 3 h w, 0-1,
        }
        'meta': {
            'video_id': str, 
            'frames': list[str], t'
            'request_ann': t, bool
            'callback_fns': list[fn]   
        }
    """

class VIS_FrameSampler_InputOutput_API:
    """
    训练时抽帧
        all_frames: list[str]
        frame_idx: 参考帧, 根据参考帧和sampler的策略

        video_id
        
        output:
            frames: list[str], t
            has_ann: t, bool
    
    测试时抽帧:
        all_frames: list[str]
        frame_idx: 参考帧
        video_id
        output:
            frames: list[str], t
            request_ann: t, bool
    
    """

class GetFrames:
    """
    Input:
        video_id
        frames:
    Output:
        t' h w, int 每一个值是obj_id, 0是背景
    """