from . import register_videos_selfDataset
from . import mapper
from . import evaluator
import os

if os.getenv('VIDENOISE_SUBTASK') == 'learn':
    from . import aug_train
    from . import aug_eval
elif os.getenv('VIDENOISE_SUBTASK') == 'optimize':
    # 直接在model init的时候通过global dataset的metalog获取
    pass
else:
    raise ValueError()
