import os

from . import scene_utils
from . import mapper 
from . import evaluator 

from . import render_aug_utils
from . import render_aug_train
from . import render_aug_eval
from . import render_view_sampler

# 注册所有scene, 包括deepblend, mesh.ply, ..各种各样形式的scene
from . import register_scenes

# 注册不同任务的需要的东西
from . import A_multiview_3D






