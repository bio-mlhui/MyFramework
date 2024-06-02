import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from .core.unet import UNet
from models.Render.representation.gaussian_renderer import GaussianRender_SameIntrin
from detectron2.modeling import META_ARCH_REGISTRY
from models.registry import register_model
from data_schedule import build_schedule
from functools import partial
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math
import torch.optim as optim
import itertools
class OptimizeModel(nn.Module):
    """
    optimize_setup:
        optimizer, scheduler都是标准类
        log_lr_idx随着训练不改变
        
    optimize:
        backward, optimzier_step, optimizer_zero_grad, scheduler_step
        
    """
    def __init__(self, ) -> None:
        super().__init__()
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.log_lr_group_idx = None

    def optimize_setup(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def optimize(self,
                loss_weight=None,
                loss_dict_unscaled=None,
                closure=None,
                num_iterations=None,
                **kwargs):
        
        loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  
        self.optimizer.step(closure=closure)
        self.optimizer.zero_grad(set_to_none=True) # delete gradient 
        self.scheduler.step(epoch=num_iterations,)

    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')        

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict=state_dict['scheduler'])

class LGM(OptimizeModel):
    def __init__(self, configs):
        super().__init__()

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=configs['model']['unet']['down_channels'],
            down_attention=configs['model']['unet']['down_attention'],
            mid_attention=configs['model']['unet']['mid_attention'],
            up_channels=configs['model']['unet']['up_channels'],
            up_attention=configs['model']['unet']['up_attention'],
        )
        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRender_SameIntrin(configs['model']['render'])

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        self.loss_weight = configs['model']['loss_weight'] # 1/lambda_lpips
        # LPIPS loss
        if self.loss_weight['lambda_lpips'] > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians

    def model_preds(self, data):
        from data_schedule.render.apis import SingleView_3D_Mapper
        step_ratio=1
        rendering_rgbs = data['inviews_dict']['rendering_rgbs'] # b V 3 h w  [B, 4, 9, h, W], input features
        rendering_alphas = data['inviews_dict']['rendering_alphas'] # b V h w
        ray_embeddings = data['inviews_dict']['ray_embeddings'] # b V 5 h w
        images = torch.cat([rendering_rgbs, rendering_alphas[:, :, None], ray_embeddings], dim=2)

        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images) # [B, N, 14]

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=self.device)
        
        outviews_dict = data['outviews_dict']
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, outviews_dict['cam_view'], outviews_dict['cam_view_proj'],
                                  outviews_dict['cam_pos'], bg_color=bg_color)

        return gaussians, results

    def forward(self, data, ):
        # data: output of the dataloader
        # return: loss
        gaussians, results = self.model_preds(data)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        gt_images = data['outviews_dict']['rendering_rgbs'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['outviews_dict']['rendering_alphas'] # [B, V, 1, output_size, output_size], ground-truth masks

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=self.device)
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)

        loss_lpips = 0
        if self.loss_weight['lambda_lpips'] > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            
        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))

        return {'loss_mse': loss_mse, 'loss_lpips': loss_lpips}, self.loss_weight, {'gaussians': gaussians,
                                                                                    'pred_images': pred_images,
                                                                                    'pred_alphas': pred_alphas,
                                                                                    'psnr': psnr}


    def sample(self, data, **kwargs):
        gaussians, results = self.model_preds(data)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        return {
            'outviews_preds': pred_images,
        }

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def optimize_setup(self, configs):
        self.optimizer = get_optimizer(self.parameters(), configs=configs)
        # scheduler (per-iteration)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
        total_steps = configs['optim']['total_steps']
        pct_start = 3000 / total_steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=configs['optim']['base_lr'], total_steps=total_steps, pct_start=pct_start)
        self.log_lr_group_idx = {'base': 0}

@register_model
def lgm(configs, device):
    from ..aux_mapper import AuxMapper
    model_input_mapper = AuxMapper(configs['model']['input_aux'])

    model = LGM(configs)
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))
    # 通过metalog 获得数据相关的东西
    model.to(device)
    model.optimize_setup(configs)

    if comm.is_main_process():
        logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
        logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if comm.get_world_size() > 1:
        # broadcast_buffers = False
        model = DDP(model, device_ids=[comm.get_local_rank()], find_unused_parameters=True, broadcast_buffers = False)
    return model, train_samplers, train_loaders, eval_function
