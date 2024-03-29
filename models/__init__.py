import os
from .registry import model_entrypoint

if os.getenv('CURRENT_TASK') == 'RVOS':
    from . import RVOS
elif os.getenv('CURRENT_TASK') == 'VIS':
    from . import VIS
elif os.getenv('CURRENT_TASK') == 'Render':
    from . import Render
else:
    raise ValueError()









