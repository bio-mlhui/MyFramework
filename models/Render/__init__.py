

import os

from .. import modality_input_mappers
from .. import backbone
from .. import decoder
from .. import encoder
from . import representation
if os.getenv('RENDER_TASK') == 'Multiview3D_Learn':
    from . import A_multiview_3D_learn
elif os.getenv('RENDER_TASK') == 'SingleView3D_Optimize':
    from . import A_multiview_3D_optimize
elif os.getenv('RENDER_TASK') == 'Text3D_Optimize':
    from . import A_text_3D_optimize
else:
    raise ValueError()
