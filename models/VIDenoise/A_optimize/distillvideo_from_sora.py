import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
from models.registry import register_model
from data_schedule import build_schedule
from functools import partial
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
import logging
import math
from detectron2.utils import comm
import logging
import cv2 
import os
from argparse import Namespace
from detectron2.modeling import META_ARCH_REGISTRY


class DistillVideo_From_SORA:
    def __init__(
        self,
        configs,):
        self.loss_weight = configs['model']['loss_weight']
        video_list = MetadataCatalog.get('global_dataset').get('subset_list')
        assert len(video_list) == 1, '基于optimize的必须是一个video'
        video_meta_id = video_list[0]

        # t 3 h w
        self.input_video = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('tensor_frames')()
        self.input_video = self.input_video.to(self.device)
        self.prompt = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('input_text', "")
        self.negative_prompt = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('input_negative_text', "")

        distiller_config = configs['model']['distiller']
        optimize_configs = configs['optim']
        self.distiller = META_ARCH_REGISTRY.get('OpenSora')(distiller_config)
        self.distiller.to(self.device)
        self.distiller.get_text_embeds([self.prompt], [self.negative_prompt])

    @property
    def device(self,):
        return torch.device('cuda')

    def __call__(self, data,):    
        video = self.input_video.unsqueeze(0)  # b t 3 h w
        self.distiller.train_step(images, poses, step_ratio=step_ratio if self.anneal_timestep else None)
        return {}, self.loss_weight, out

    def sample(self, data, **kwargs):
        # ignore if no need to update
        if not self.need_update:
            return
        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniMiniMiniCam(
                self.cam.pose,
                self.cam.W,
                self.cam.W,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.render_fn(cur_cam, scaling_modifier=self.gaussain_scale_factor,
                                 bg_color=self.bg_color)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.cam.H, self.cam.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

    
    def optimize(self, 
                 loss_weight=None,
                 loss_dict_unscaled=None,
                 closure=None,
                 num_iterations=None,
                 optimize_dict=None,
                 **kwargs):
        num_iterations += 1
        self.update_learning_rate(num_iterations)  
        step_ratio = min(1,  float(num_iterations) / self.total_steps)

        loss = 0
        loss += 10000 * (step_ratio if self.warmup_rgb_loss else 1) * loss_dict_unscaled['loss_rgb']
        loss += 1000 * (step_ratio if self.warmup_rgb_loss else 1) * loss_dict_unscaled['loss_mask']
        loss += loss_dict_unscaled['loss_sd'] * loss_weight['loss_sd']
        loss += loss_dict_unscaled['loss_zero123'] * loss_weight['loss_zero123']

        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        if num_iterations >= self.densify_from_iter and num_iterations <= self.densify_until_iter:
            viewspace_point_tensor, visibility_filter, radii = optimize_dict['viewspace_points'], optimize_dict['visibility_filter'], optimize_dict['radii']
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if num_iterations % self.densification_interval == 0:
                self.densify_and_prune(self.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
            if num_iterations % self.opacity_reset_interval == 0:
                self.reset_opacity()

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
        }
    
    def load_optimize_state_dict(self, state_dict):
        pass

    def load_state_dict(self, ckpt, strict=True):
        pass     

    def state_dict(self,):
        return ()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)
        pass


@register_model
def distillvideo_from_sora(configs, device):
    from .aux_mapper import AuxMapper
    model_input_mapper = AuxMapper(configs['model']['input_aux'])
    
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))

    model = DistillVideo_From_SORA(configs)
    assert comm.get_world_size() == 1
    return model, train_samplers, train_loaders, eval_function