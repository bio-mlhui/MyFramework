
import matplotlib.pyplot as plt
from typing import Any, Optional, List, Dict, Set, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_schedule import build_schedule
from torch import Tensor
from models.optimization.utils import get_total_grad_norm
from einops import repeat, rearrange, reduce
from functools import partial
import numpy as np
from detectron2.modeling import META_ARCH_REGISTRY

class Sora_VideoDenoise(nn.Module):
    def __init__(self, distiller_configs) -> None:
        super().__init__()
        sora_config = distiller_configs
        num_frames = sora_config['num_frames']
        image_size = sora_config['image_size']
        input_size = (num_frames, *image_size) # t h w

        self.vae = META_ARCH_REGISTRY.get(sora_config['vae']['name'])(sora_config['vae'])

        self.latent_size = self.vae.get_latent_size(input_size)

        self.text_encoder = META_ARCH_REGISTRY.get(sora_config['text_encoder']['name'])(sora_config['text_encoder'])

        sora_config['dit']['input_size'] = self.latent_size
        sora_config['dit']['in_channels'] = self.vae.out_channels
        sora_config['dit']['caption_channels'] = self.text_encoder.output_dim
        sora_config['dit']['model_max_length'] = self.text_encoder.model_max_length
        sora_config['dit']['enable_sequence_parallelism'] = False
        self.dit =  META_ARCH_REGISTRY.get(sora_config['dit']['name'])(sora_config['dit'])

        self.text_encoder.y_embedder = self.dit.y_embedder  # hack for classifier-free guidance

        self.vae.to(self.device, torch.float).eval()
        self.dit.to(self.device, torch.float).eval()

        
        self.scheduler = META_ARCH_REGISTRY.get(sora_config['scheduler']['name'])(sora_config['scheduler'])

        self.multi_resolution = sora_config['multi_resolution']

        if self.multi_resolution == "PixArtMS":
            image_size = image_size
            # 1 2
            hw = torch.tensor([image_size], dtype=torch.float).to(self.device)
            # 1 1
            ar = torch.tensor([[image_size[0] / image_size[1]]], dtype=torch.float).to(self.device)
            model_args = {
                'data_info': {'ar': ar, 'hw': hw}
            }

        elif self.multi_resolution == "STDiT2":
            image_size = image_size
            height = torch.tensor([image_size[0]], dtype=torch.float) # 1
            width = torch.tensor([image_size[1]], dtype=torch.float) # 1
            num_frames = torch.tensor([num_frames], dtype=torch.float) # 1
            ar = torch.tensor([image_size[0] / image_size[1]], dtype=torch.float) # 1
            if num_frames == 1:
                fps = 120
            fps = torch.tensor([fps], dtype=torch.float) # 1
            model_args = {
                'height': height,
                'width': width,
                'num_frames': num_frames,
                'ar': ar,
                'fps': fps
            }

        self.image_mean = torch.tensor([0.5, 0.5, 0.5]).float().to(self.device)[:, None, None] # 3 1 1
        self.image_var = torch.tensor([0.5, 0.5, 0.5]).float().to(self.device)[:, None, None] # 3 1 1

        t_range=[0.02, 0.98]
        self.num_train_timesteps = self.scheduler.num_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.embeddings = None

        # vae之前的augmentation

        # vaue之后的augmentation
    
    def get_text_embeds(self, text):
        # self.distiller.get_text_embeds([self.prompt], [self.negative_prompt])
        pass

    @property
    def device(self,):
        return torch.device('cuda') # current device

    # 输入有噪声的video, 输出没有噪声的video
    def forward(
        self,
        pred_rgb, # b t 3 h w, float, 0-1
        guidance_scale=5,
        step_ratio=None
    ):  
        batch_size, nf, _, H, W = pred_rgb.shape

        # 假设t是符合sora的, 但是h w不是符合的
        assert nf == self.latent_size[0]
        if (pred_rgb.shape[3] != self.latent_size[1]) or (pred_rgb.shape[34] != self.latent_size[2]):
            pred_rgb = F.interpolate(pred_rgb.flatten(0, 1), size=self.latent_size[-2:], mode='bilinear', align_corners=False)
        pred_rgb = (pred_rgb - self.image_mean) / self.image_var
        pred_rgb = rearrange(pred_rgb, '(b t) c h w -> b c t h w',b=batch_size, t=nf)
        latents = self.vae.encode(pred_rgb) # b c t h w

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.scheduler.alphas_cumprod[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), 
                                    self.embeddings['neg'].expand(batch_size, -1, -1)])
   
            noise_pred = self.dit(latent_model_input, tt, encoder_hidden_states=embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        denoised_video = self.vae.decode(target) # b 3 t h w

        denoised_video = F.interpolate(rearrange(denoised_video, 'b c t h w -> (b t) c h w'), size=(H, W), mode='blinear', align_corners=False)
        denoised_video = rearrange(denoised_video, '(b t) c h w -> b t c h w', b=batch_size, t=nf)

        return denoised_video


class Sora_3DReconstruction(nn.Module):
    pass