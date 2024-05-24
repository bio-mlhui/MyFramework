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

if os.getenv('RENDER_TASK') == 'Multiview3D_Learn':
    from . import A_multiview_3D_learn
elif os.getenv('RENDER_TASK') == 'SingleView3D_Optimize':
    from . import A_multiview_3D_optimize
elif os.getenv('RENDER_TASK') == 'Text3D_Optimize' or os.getenv('RENDER_TASK') == 'Text3D_Learn':
    from . import A_text_3D
else:
    raise ValueError()






