

# foreachrefer, 
# step_size
class RVOS_Dataset:
    """
    foreachrefer/测试集
        'video_id': str,
        'all_frames' : list[str],
        'all_objs': {obj_id: {'class_label': 0,}},
        'all_exps': {exp_id: {'expression': 'obj_ids': list[int]}}
        exp_id
    
    allexists:
        'video_id': str,
        'all_frames' : list[str],
        'all_objs': {obj_id: {'class_label': 0,}},
        'all_exps': {exp_id: {'expression': 'obj_ids': list[int]}}

    训练集/测试集
    'frame_idx': 抽clip的时候的参考帧下标, 相对于all_frames, 具体怎么抽要传到frame_sampler里
    如果没有的话, (train)就从整个video抽取clip, (eval)或者对整个video进行测试
    """

class RVOS_Aug_CallbackAPI:
    """单个样本 不是batch
    'video': list[Image], t
    'masks': n t' h w, bool
    'boxes': n t' 4, x1y1x2y2绝对值
    'has_ann': t, bool
    'class_labels': n,
    'callback_fns':
    'referent_text'/'exist_texts'

    模型测试的输出api: 每一帧有多个mask/box预测, 每个预测都有类别的概率, 最后一个类别是背景类
    'video': t 3 h w, 0-1
    'pred_masks': list[no h w], t bool
    'pred_boxes': list[no 4], t, x1y1x2y2绝对值
    'pred_class': list[no c] t, 概率值
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


class RVOS_TrainAPI_referent_text_clipped_video:
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
