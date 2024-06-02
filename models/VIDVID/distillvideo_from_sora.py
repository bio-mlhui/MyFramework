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
from einops import rearrange

class DistillVideo_From_SORA:
    def __init__(
        self,
        configs,):
        self.loss_weight = configs['model']['loss_weight']
        video_list = MetadataCatalog.get('global_dataset').get('subset_list')
        assert len(video_list) == 1, '基于optimize的必须是一个video'
        video_meta_id = video_list[0]

        # t 3 h w
        image_size = configs['model']['distiller']['image_size']
        self.input_video = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('tensor_frames')()
        self.original_image_size = self.input_video.shape[2:]

        self.input_video = F.interpolate(self.input_video, size=image_size, mode='bilinear', align_corners=False)
        self.input_video = self.input_video.bfloat16().to(self.device) # t 3 h w
        self.image_mean = torch.tensor([0.5, 0.5, 0.5]).bfloat16().to(self.device)[:, None, None] # 3 1 1
        self.image_var = torch.tensor([0.5, 0.5, 0.5]).bfloat16().to(self.device)[:, None, None] # 3 1 1
        self.input_video = (self.input_video - self.image_mean) / self.image_var

        self.prompt = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('input_text', "")
        self.negative_prompt = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('input_negative_text', "")

        self.frame_interval = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('frame_interval')
        self.fps = MetadataCatalog.get(video_meta_id).get('videnoise_optimize').get('fps')

        
        distiller_config = configs['model']['distiller']
        num_frames = self.input_video.shape[0]
        
        distiller_config['num_frames'] = num_frames 
        distiller_config['image_size'] = image_size
        distiller_config['pos_text'] = self.prompt
        distiller_config['neg_text'] = self.negative_prompt
        distiller_config['fps'] = self.fps
        distiller_config['frame_interval'] = self.frame_interval
        distiller_config['optim'] = configs['optim']
        from .. import distiller
        # sora可以输入任意大小的的video
        self.distiller = META_ARCH_REGISTRY.get(distiller_config['name'])(distiller_config)
        for p in self.distiller.parameters():
            p.requires_grad = False
        self.distiller.to(self.device).eval()
                
        self.latents = self.distiller.vae.encode(rearrange(self.input_video, 't c h w -> 1 c t h w'))

        optimize_configs = configs['optim']

        self.total_steps = optimize_configs['total_steps']
        self.denoise_from_step = optimize_configs['denoise_from_step']
        self.log_lr_group_idx = {}


    @property
    def device(self,):
        return torch.device('cuda')
     

    def __call__(self, data,):    
        num_iterations = data['num_iterations'] + 1
        step_ratio = min(1,  float(num_iterations) / self.total_steps)
        self.latents = self.distiller(latents=self.latents,
                                      step_ratio=step_ratio,).bfloat16() # b t 3 h w
        # self.input_video = F.interpolate(denoised_video, size=self.original_image_size, mode='bilinear', align_corners=False)
        return {}, self.loss_weight, {}

    def save_video_frames(self, frames, video_id, output_dir):
        denoised_video = self.distiller.vae.decode(self.latents).float().permute(0, 2, 1, 3, 4).squeeze(0) # b 3 t h w -> t 3 h w
        original_image_size = self.original_image_size
        input_video = F.interpolate(denoised_video, size=original_image_size, mode='bilinear', align_corners=False)
        assert input_video.shape[0] == len(frames)
        from torchvision.utils import save_image
        for haosen, frame in zip(input_video[:1], frames[:1]):
            save_image([haosen], os.path.join(output_dir, 'web_video', video_id, f'{frame}.png'), 
                    normalize=True, value_range=(-1, 1))


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
        pass

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': None if value is not None else 0 for key, value in self.log_lr_group_idx.items()}

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

    def train(self, ):
        pass

    def eval(self, ):
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