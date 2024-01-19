
from .Trainer import Trainer  # RVOS, RIOS, Generation

task_to_trainer = {
    'RVOS': Trainer,
    'RIOS': Trainer,
    'RPCS': Trainer, # point cloud
    'VOS': Trainer, 
    'VIS': Trainer
}

# optimizer, schedule都交给model构建
# 3D情况下, 每个3d model就是一个数据集


