

import os

if os.getenv('VIDENOISE_SUBTASK') == 'learn':
    from . import A_learn
elif os.getenv('VIDENOISE_SUBTASK') == 'optimize':
    from .. import distiller
    from . import A_optimize
else:
    raise ValueError()
