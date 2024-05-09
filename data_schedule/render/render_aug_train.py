
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
    

import roma
import numpy as np
import torchvision.transforms.functional as TF

def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T
    
    return new_poses

def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        x_steps = (x_steps * W).long() # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0) # [W]

        y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
        y_steps = (y_steps * H).long() # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0) # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

        grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(images.device) # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@RENDER_TRAIN_AUG_REGISTRY.register()
class SingleView_3D_LGM_TrainAug:
    def __init__(self, configs) -> None:
        self.tensor_rendering = RenderingToTensor()
        self.prob_grid_distortion = configs['prob_grid_distortion']
        self.prob_cam_jitter = configs['prob_cam_jitter']

    def __call__(self, ret):
        images_input = ret['rendering_rgbs']
        cam_poses_input = ret['extrin']

        # apply random grid distortion to simulate 3D inconsistency
        if random.random() < self.prob_grid_distortion:
            images_input[1:] = grid_distortion(images_input[1:])
        # apply camera jittering (only to input!)
        if random.random() < self.prob_cam_jitter:
            cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        ret['rendering_rgbs'] = images_input
        ret['extrin'] = cam_poses_input
        return ret
    





