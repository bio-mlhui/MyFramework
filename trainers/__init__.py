
from .Trainer import Trainer
from .Trainer_SingleProcess import Trainer_SingleProcess
task_to_trainer = {
    'RVOS': Trainer,
    'VIS': Trainer,
    'RENDER': Trainer,
    'VIDVID': Trainer,
    'UN_IMG_SEM': Trainer_SingleProcess
}

# optimizer, schedule都交给model构建
# 3D情况下, 每个3d model就是一个数据集


