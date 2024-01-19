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
else:
    raise ValueError()









