
from .render_aug_utils import RENDER_TRAIN_AUG_REGISTRY
from data_schedule.render.apis import Scene_Meta
import torchvision.transforms.functional as F
import torch
class RenderingToTensor:
    def __call__(self, ret):
        rendering = ret['rendering']
        ret['rendering'] = F.to_tensor(rendering)
        return ret


@RENDER_TRAIN_AUG_REGISTRY.register()
class GS_TrainAug:
    def __init__(self, configs) -> None:
        self.tensor_rendering = RenderingToTensor()

    def __call__(self, ret):
        Scene_Meta
        ret = self.tensor_rendering(ret)
        return ret





