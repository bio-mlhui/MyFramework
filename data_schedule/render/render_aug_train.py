
from .render_aug_utils import RENDER_TRAIN_AUG_REGISTRY
from data_schedule.render.apis import Scene_Meta
import torchvision.transforms.functional as F
import torch
import random
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
    

@RENDER_TRAIN_AUG_REGISTRY.register()
class SingleView_3D_LGM_TrainAug:
    def __init__(self, configs) -> None:
        self.tensor_rendering = RenderingToTensor()
        self.prob_grid_distortion = configs['prob_grid_distortion']
        self.prob_cam_jitter = configs['prob_cam_jitter']

    def __call__(self, ret):
        images_input = ret['images_input']
        cam_poses_input = ret['cam_poses_input']

        # apply random grid distortion to simulate 3D inconsistency
        if random.random() < self.prob_grid_distortion:
            images_input[1:] = grid_distortion(images_input[1:])
        # apply camera jittering (only to input!)
        if random.random() < self.prob_cam_jitter:
            cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        ret['images_input'] = images_input
        ret['cam_poses_input'] = cam_poses_input
        return ret
    





