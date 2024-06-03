import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
from models.registry import register_model
from data_schedule import build_schedule
from functools import partial
from detectron2.data import MetadataCatalog, DatasetCatalog
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
import torchvision.transforms.functional as torchv_F
from PIL import Image
from utils.misc import generate_unique_key, to_torch_dtype
from torchvision.io import write_video
class DenoiseVideo_Sora:
    def __init__(
        self,
        configs,):
        self.loss_weight = configs['model']['loss_weight']

        self.dtype = to_torch_dtype(configs['model']['dtype'])
        self.image_size = configs['model']['image_size']
        self.num_frames = configs['model']['num_frames']
        self.fps = configs['model']['fps']
        self.frame_interval = configs['model']['frame_interval']
        input_video = configs['model']['input_video']
        self.video_id = configs['model']['video_id']
        video_key = generate_unique_key(self.image_size, self.num_frames, self.fps, self.frame_interval, input_video, self.video_id)
        video_save_file = os.path.join(configs['out_dir'], f'{video_key}.pth')
        from .. import distiller
        sora_config = configs['model']['distiller']
        self.vae = META_ARCH_REGISTRY.get(sora_config['vae']['name'])(sora_config['vae']).to(self.device).eval()
        if os.path.exists(video_save_file):
            saved_pth = torch.load(video_save_file)
            self.input_latents = saved_pth['input_latents'].to(self.device)
            self.original_image_size = saved_pth['original_image_size']
            self.latent_size = saved_pth['latent_size']
            self.frame_strs = saved_pth['frame_strs']
        else:
            assert os.path.exists(input_video)
            frames = sorted(os.listdir(input_video))
            assert len(frames) >= self.num_frames
            frames = frames[:self.num_frames]
            self.frame_strs =sorted([png[:-4] for png in frames if png.endswith('.png') or png.endswith('.jpg')])
            input_video = [Image.open(os.path.join(input_video, f)).convert('RGB') for f in frames]
            input_video = torch.stack([torchv_F.to_tensor(haosen) for haosen in input_video], dim=0) # t 3 h w, 0-1, float
            self.original_image_size = input_video.shape[2:]

            input_video = F.interpolate(input_video, size=self.image_size, mode='bilinear', align_corners=False)
            input_video = (input_video - (torch.tensor([0.5, 0.5, 0.5]))[:, None, None]) / (torch.tensor([0.5, 0.5, 0.5])[:, None, None])
            input_video = input_video.to(self.device) # t 3 h w
            input_video = rearrange(input_video, 't c h w -> 1 c t h w')
            with torch.no_grad():
                self.input_latents = self.vae.encode(input_video)
            self.latent_size = self.vae.get_latent_size((self.num_frames, *self.image_size))
            torch.save({
                'input_latents': self.input_latents,
                'latent_size': self.latent_size,
                'frame_strs': self.frame_strs,
                'original_image_size': self.original_image_size,
            }, video_save_file)

        self.input_latents = self.input_latents.to(self.dtype)
        self.prompt = configs['model']['prompt']
        self.negative_prompt = configs['model']['negative_prompt']
        self.prompt_key = generate_unique_key(self.prompt, self.negative_prompt)

        
        if sora_config['multi_resolution'] == "PixArtMS":
            image_size = image_size
            # 1 2
            hw = torch.tensor([image_size], dtype=torch.float).to(self.device)
            # 1 1
            ar = torch.tensor([[image_size[0] / image_size[1]]], dtype=torch.float).to(self.device)
            model_args = {
                'data_info': {'ar': ar, 'hw': hw}
            }

        elif sora_config['multi_resolution'] == "STDiT2":
            self.model_args = {
                'height': torch.tensor([self.image_size[0]], dtype=self.dtype).to(self.device), # 1
                'width': torch.tensor([self.image_size[1]], dtype=self.dtype).to(self.device), # 1
                'num_frames': torch.tensor([self.num_frames], dtype=self.dtype).to(self.device), # 1
                'ar': torch.tensor([self.image_size[0] / self.image_size[1]], dtype=self.dtype).to(self.device), # 1
                'fps': torch.tensor([self.fps], dtype=self.dtype).to(self.device) # 1
            }
        sora_config['dit']['input_size'] = self.latent_size
        sora_config['dit']['in_channels'] = self.vae.out_channels 
        sora_config['dit']['caption_channels'] = 4096
        sora_config['dit']['model_max_length'] = 200
        sora_config['dit']['enable_sequence_parallelism'] = False
        self.dit =  META_ARCH_REGISTRY.get(sora_config['dit']['name'])(sora_config['dit']).to(self.device).eval()
        self.get_text_embed(self.prompt, self.negative_prompt, out_dir=configs['out_dir'], sora_config=sora_config)

        self.dit.to(self.dtype)
        self.vae.to(self.dtype)
        self.scheduler = META_ARCH_REGISTRY.get(sora_config['scheduler']['name'])(sora_config['scheduler'])
        self.total_iterations = configs['optim']['total_iterations']
        
        t_range = configs['optim']['t_range']
        total_steps = self.scheduler.num_timesteps
        self.min_step, self.max_step = int(total_steps * t_range[0]), int(total_steps * t_range[1])
        self.total_steps = self.max_step - self.min_step
        self.log_lr_group_idx = {}

    def get_text_embed(self, pos_text, neg_text, out_dir, sora_config):
        text_embed_file = os.path.join(out_dir, f'{self.prompt_key}.pth')
        if os.path.exists(text_embed_file):
            text_embed = torch.load(text_embed_file)
            self.model_args['y'] = text_embed['y'].to(self.dtype).to(self.device)
            self.model_args['mask'] = text_embed['mask'].to(self.device)
        else:
            with torch.no_grad():
                text_encoder = META_ARCH_REGISTRY.get(sora_config['text_encoder']['name'])(sora_config['text_encoder'])
                text_encoder.y_embedder = self.dit.y_embedder  # hack for classifier-free guidance
                pos_text_embed = text_encoder.encode([pos_text])
                y = pos_text_embed['y']
                self.model_args['mask'] = pos_text_embed['mask']
                y_null = text_encoder.null(1).to(self.device) # classifier-free guidance
                y = torch.cat([y, y_null], 0)
                del text_encoder
                torch.cuda.empty_cache()
                torch.save({'y': y, 'mask': self.model_args['mask'], 'pos_text': pos_text, 'neg_text': neg_text},
                            text_embed_file)
                self.model_args['y'] = y.to(self.dtype)


    def get_video_meta(self):
        return self.video_id, self.frame_strs
    @property
    def device(self,):
        return torch.device('cuda')
     
    def __call__(self, data,):    
        num_iterations = data['num_iterations'] + 1
        step_ratio = min(1,  float(num_iterations) / self.total_iterations)
        batch_size, _, nf, _, _ = self.input_latents.shape

        img = self.input_latents 
        img = torch.cat([img, img], 0)
        forward = partial(DenoiseVideo_Sora.forward_with_cfg, 
                          self.dit, cfg_scale=self.scheduler.cfg_scale, cfg_channel=self.scheduler.cfg_channel)
        with torch.no_grad():
            if step_ratio is not None:
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.total_steps + self.min_step).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                # t = torch.full((batch_size,), 1, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            t = torch.cat([t, t]) # 2b
            samples = self.scheduler.p_sample(forward, 
                                              img, t,                    
                                            clip_denoised=False,
                                            denoised_fn=None,
                                            cond_fn=None,
                                            model_kwargs=self.model_args,
                                            mask=None,)['sample']
            self.input_latents, _ = samples.chunk(2, dim=0)
            self.input_latents = self.input_latents.to(self.dtype)
        return {}, self.loss_weight, {}


    @staticmethod
    def forward_with_cfg(model, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if "x_mask" in kwargs and kwargs["x_mask"] is not None:
            if len(kwargs["x_mask"]) != len(x):
                kwargs["x_mask"] = torch.cat([kwargs["x_mask"], kwargs["x_mask"]], dim=0)
        model_out = model.forward(combined, timestep, y, **kwargs)
        model_out = model_out["x"] if isinstance(model_out, dict) else model_out
        if cfg_channel is None:
            cfg_channel = model_out.shape[1] // 2
        eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    def save_video_frames(self, frames, video_id, output_dir):
        x = self.vae.decode(self.input_latents).float().permute(0, 2, 1, 3, 4).squeeze(0) # b 3 t h w -> t 3 h w
        x = F.interpolate(x, size=self.original_image_size, mode='bilinear', align_corners=False)
        save_path = os.path.join(output_dir, f'{self.get_video_meta()[0]}.mp4')
        low, high = (-1, 1)
        x.clamp_(min=low, max=high)
        x.sub_(low).div_(max(high - low, 1e-5))
        x = x.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).permute(0, 2, 3, 1)
        write_video(save_path, x, fps=self.fps // self.frame_interval, video_codec="h264")

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
def sora_denoise_video(configs, device):
    from .aux_mapper import Optimize_AuxMapper
    model_input_mapper = Optimize_AuxMapper(configs['model']['input_aux'])
    
    train_samplers, train_loaders, eval_function = build_schedule(configs, model_input_mapper.mapper, partial(model_input_mapper.collate,))
    model = DenoiseVideo_Sora(configs)
    assert comm.get_world_size() == 1
    return model, train_samplers, train_loaders, eval_function