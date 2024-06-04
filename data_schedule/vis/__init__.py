# 每个数据集通过evaluator_utils注册自己的 et_name_metric,
# 多个数据集公用的metric可以写到evaluator_utils里
from . import polyp
# from . import fibroid
# from . import visha

from . import mapper # 注册vis mappers
from . import evaluator # 注册vis evaluator
from . import evaluator_fast

from . import vis_aug_eval  # 注册vis aug
from . import vis_aug_train
from . import vis_frame_sampler # 注册vis frame sampler





