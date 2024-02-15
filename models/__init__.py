import os
from .registry import model_entrypoint

if os.getenv('CURRENT_TASK') == 'RVOS':
    from . import RVOS
elif os.getenv('CURRENT_TASK') == 'RIOS':
    from . import RIOS
elif os.getenv('CURRENT_TASK') == 'VOS':
    from . import VOS
elif os.getenv('CURRENT_TASK') == 'VIS':
    from . import VIS

elif os.getenv('CURRENT_TASK') == 'DYNAMIC_RENDER':
    from . import DYNAMIC_RENDER
else:
    raise ValueError()









