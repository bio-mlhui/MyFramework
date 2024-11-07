
from . import mapper
from . import evaluator
from . import augmentations
from . import sample_dataset
from . import cocostuff27
from . import vessel
# ssl_few_shot
# from . import abd_ct

# 每个数据集
# 有些数据集需要运行5-fold

# dataset: list[meta]


# ssl_few_shot 和 unsupervised  在相同的测试集上测试

# 哪一种可以实现 模态间互助？
"""
对于ssl_few_shot,
每个Sample是一个episode 
训练集没有label, 测试集上有label, 


对于unsupervised,
训练集没有Label, 测试集上没有label
每个sample是一个图像

对于 few-shot-supervised,
训练集有label, 测试集上没有label

"""


class Online_UNSEG_Train_API:
    """

    """
    pass

class UNSEG_Eval_GenPseudoMask_API:
    """
    输入:
        'images': b 3 h w, 0-1, 没有normalize  #TODO: coco有很多小物体, 放大resolution
        'metas':
            image_ids:  list[str], batch
                
    输出:
        'pred_masks': list[nqi h w], batch, bool
        'pred_scores': list[nqi], batch, 0-1的分数
        'total_time': 这个batch除了抽特征之外所用的时间

    """
    pass

class UNSEG_Eval_Cls_Ag_Instance_API:
    """
    输入:
        'images': b 3 h w, 0-1, 没有normalize  #TODO: coco有很多小物体, 放大resolution
        'metas':
            image_ids:  list[str], batch
                
    输出:
        'pred_masks': list[nqi h w], batch, bool
        'pred_scores': list[nqi], batch, 0-1的分数
        'total_time': 这个batch除了抽特征之外所用的时间

    """
    pass