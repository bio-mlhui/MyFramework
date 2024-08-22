import os
from .registry import model_entrypoint

if os.getenv('CURRENT_TASK') == 'RVOS':
    from . import RVOS
elif os.getenv('CURRENT_TASK') == 'VIS':
    from . import VIS
elif os.getenv('CURRENT_TASK') == 'RENDER':
    from . import RENDER
elif os.getenv('CURRENT_TASK') == 'VIDVID':
    from . import VIDVID
elif os.getenv('CURRENT_TASK') == 'UN_IMG_SEM':
    from . import UN_IMG_SEM
else:
    raise ValueError()










