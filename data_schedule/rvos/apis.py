

# foreachrefer, 
# step_size
class RVOS_Dataset:
    """
    class_label从0开始, 第K+1个类代表background 
    instance_id从1开始, 0代表所有没有被标注的区域, 
    exp_id(str)
    get_frame_mask_fn能够根据video_id, frames给出 n t' h w(bool)), list[instance_id] / t' h w(int) 的形式
    
    'all_frames': list[str],
    'all_objs': {obj_id: {'class_label': 0,}}
    'all_exps': {exp_id: {'expression': 'obj_ids': list[obj_id]}}
    'frame_idx': index_int, 参考帧下标, 相对于all_frames, 传到frame_sampler抽取clip和request_ann/has_ann
    
    训练集foreachrefer
        'video_id': str,
        'exp_id': str
        'referent_text': str
        'referent_objs': list[obj_id, int]

    训练集allexists
        'video_id': str,
        'all_frames' : list[str],
        'all_objs': {obj_id: {'class_label': 0,}}
    
    测试集for each refer
        'video_id': str,
        'exp_id': str,
        'referent_text': str
        
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
