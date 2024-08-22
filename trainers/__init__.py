
from .Trainer import Trainer  # RVOS, RIOS, Generation
# from .Trainer_Render import Trainer_Render
task_to_trainer = {
    'RVOS': Trainer,
    'VIS': Trainer,
    'RENDER': Trainer,
    'VIDVID': Trainer,
    'UN_IMG_SEM': Trainer
}

# optimizer, schedule都交给model构建
# 3D情况下, 每个3d model就是一个数据集


