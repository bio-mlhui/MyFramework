

# foreachrefer, 
# step_size
class RVOS_Dataset:
    """
    foreachrefer训练集
        'video_id': str,
        'all_frames' : list[str],
        'referent_text': str
        'referent_objs': list[obj_id(int)]
        'all_objs': {obj_id(int): {'class_label': 0,}},
    
    测试集(for each refer)
        'video_id': str,
        'exp_id': str,
        'referent_text': str
        'all_frames': list[str]
    
    allexists训练集:
        'video_id': str,
        'all_frames' : list[str],
        'all_objs': {obj_id: {'class_label': 0,}},
        'all_exps': {exp_id: {'expression': 'obj_ids': list[obj_id]}}
    
        
    训练集/测试集
    'frame_idx': 抽clip的时候的参考帧下标, 相对于all_frames, 具体怎么抽要传到frame_sampler里
    如果没有的话, (train)就从整个video抽取clip, (eval)或者对整个video进行测试
    """

class RVOS_Aug_CallbackAPI:
    """
    'video': list[pil Image, rgb], t
    'has_ann': t, bool
    'masks': N t' h w, bool
    'classes': N,
    'boxes': N t' 4, x1y1x2y2绝对值

    
    'referent_text'/'all_refer_exps',  如果是instance的话, semantic segmentation没有box
    'referent_objs'/'all_referent_objs': list[int], N 的下标

    模型测试的输出api: 每一帧有多个mask/box预测, 每个预测都有类别的概率, 最后一个类别是背景类
    'video': t 3 h w, 0-1
    'pred_masks': list[no h w], t bool
    'pred_boxes': list[no 4], t, x1y1x2y2绝对值
    'pred_class': list[no c] t, 概率值
    'callback_fns': list[fn]
"""


class RVOS_FrameSampler_InputOutput_API:
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

class RVOS_EvalAPI_referent_text_clipped_video_request_ann:
    """
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'refer_dict': {
            'text': str,
            'aux': None
        }
        'meta': {
            'video_id': str,
            'frames': list[str], t'
            'exp_id': str,
            'request_ann': t, bool 
            'callback_fns': list[fn]     
        }
    """


class RVOS_TrainAPI_ForEachRefer_clipped_video:
    """
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux': None
        }
        'refer_dict': {
            'text': str,
            'aux': None
        }
        'video_refer_dict':{
            'aux': None
        }
        'targets': {
            'has_ann': t (bool)
            'boxes': N/n t' 4, x1y1x2y2绝对值
            'masks': N/n t' h w, bool
            'classes': N/n
            'referent_objs': list[int], N/n的下标
        }
        'frame_targets':{
            'masks': list[N/n h w], t'
            'boxes': list[N/n 4], t'
            'classes': list[N/n], t'
            'referent_objs': list[list[int]], t'
        }
    """


class RVOS_TrainAPI_exist_texts_clipped_video:
    """
        'video_dict': {
            'video': t 3 h w, 0-1,
            'aux'
        }
        'exists_text_dict': {
            'texts': list[str],
            'amr': list[amr]
        }
        'targets': {
            'has_ann': t (bool)
            'boxes': N/n t' 4, x1y1x2y2绝对值
            'masks': N/n t' h w, bool 
            'classes': N/n      
            'referent_objs': list[ list[int],],  文本的数量 
        }
        'frame_targets':{
            'masks': N/n h w, bt'
            'boxes': N/n 4, bt' 
            'classes': N/n             
        }
    """
