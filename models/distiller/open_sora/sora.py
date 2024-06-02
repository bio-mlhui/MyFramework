
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
from detectron2.data import MetadataCatalog

@META_ARCH_REGISTRY.register()
class Sora_VideoDenoise(nn.Module):
    def __init__(self, distiller_configs) -> None:
        super().__init__()
        sora_config = distiller_configs
        self.vae = META_ARCH_REGISTRY.get(sora_config['vae']['name'])(sora_config['vae'])
        num_frames, image_size, fps, frame_interval = sora_config['num_frames'], sora_config['image_size'], \
            sora_config['fps'],sora_config['frame_interval']
        input_size = (num_frames, *image_size) # t h w
        # t/patch_t, h/patch_h, w/patch_w
        self.latent_size = self.vae.get_latent_size(input_size)
        
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
            image_size = image_size
            height = torch.tensor([image_size[0]], dtype=torch.bfloat16).to(self.device) # 1
            width = torch.tensor([image_size[1]], dtype=torch.bfloat16).to(self.device) # 1
            num_frames = torch.tensor([num_frames], dtype=torch.bfloat16).to(self.device) # 1
            ar = torch.tensor([image_size[0] / image_size[1]], dtype=torch.bfloat16).to(self.device) # 1
            if num_frames == 1:
                fps = 120
            fps = torch.tensor([fps], dtype=torch.bfloat16).to(self.device) # 1
            model_args = {
                'height': height,
                'width': width,
                'num_frames': num_frames,
                'ar': ar,
                'fps': fps
            }


        # text encoder只用来获取text embedding, 获得text embedding之后就删除
        text_encoder = META_ARCH_REGISTRY.get(sora_config['text_encoder']['name'])(sora_config['text_encoder'])
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length

        sora_config['dit']['input_size'] = self.latent_size
        sora_config['dit']['in_channels'] = self.vae.out_channels
        sora_config['dit']['caption_channels'] = text_encoder_output_dim
        sora_config['dit']['model_max_length'] = text_encoder_model_max_length
        sora_config['dit']['enable_sequence_parallelism'] = False
        self.dit =  META_ARCH_REGISTRY.get(sora_config['dit']['name'])(sora_config['dit'])

        text_encoder.y_embedder = self.dit.y_embedder  # hack for classifier-free guidance
        with torch.no_grad():
            # 1 1 max_length 4096; 1 max_length
            # pos_text, neg_text = distiller_configs['pos_text'], distiller_configs['neg_text']
            pos_text = distiller_configs['pos_text']
            pos_text_embed = text_encoder.encode([pos_text])
            model_args['y'] = pos_text_embed['y'].to(torch.bfloat16)
            model_args['mask'] = pos_text_embed['mask']
            y_null = text_encoder.null(1).bfloat16().to(self.device) # classifier-free guidance
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)
            del text_encoder
            torch.cuda.empty_cache()
        self.model_args = model_args
        self.vae.to(self.device, torch.bfloat16).eval()
        self.dit.to(self.device, torch.bfloat16).eval()

        self.scheduler = META_ARCH_REGISTRY.get(sora_config['scheduler']['name'])(sora_config['scheduler'])
        
        self.t_range = sora_config['optim']['t_range']
        self.num_train_timesteps = self.scheduler.num_timesteps
        self.min_step = int(self.num_train_timesteps * self.t_range[0])
        self.max_step = int(self.num_train_timesteps * self.t_range[1])

    @property
    def device(self,):
        return torch.device('cuda') # current device


    def forward(
        self,
        latents, # b 4 t h w
        guidance_scale=5,
        step_ratio=None
    ):  
        batch_size, _, nf, _, _ = latents.shape
        img = latents 
        img = torch.cat([img, img], 0)
        forward = partial(forward_with_cfg, self.dit, cfg_scale=self.scheduler.cfg_scale, cfg_channel=self.scheduler.cfg_channel)
        with torch.no_grad():
            if step_ratio is not None:
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
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
            samples, _ = samples.chunk(2, dim=0)
        return samples


    # # 输入有噪声的video, 输出没有噪声的video
    # def forward(
    #     self,
    #     pred_rgb, # b t 3 h w, float, 0-1
    #     guidance_scale=5,
    #     step_ratio=None
    # ):  
    #     batch_size, nf, _, H, W = pred_rgb.shape

        
    #     # assert nf == self.latent_size[0]
    #     # if (pred_rgb.shape[3] != self.latent_size[1]) or (pred_rgb.shape[4] != self.latent_size[2]):
    #     #     pred_rgb = F.interpolate(pred_rgb.flatten(0, 1), size=self.latent_size[-2:], mode='bilinear', align_corners=False)
    #     pred_rgb = (pred_rgb - self.image_mean) / self.image_var
    #     pred_rgb = rearrange(pred_rgb, 'b t c h w -> b c t h w',b=batch_size, t=nf)
    #     latents = self.vae.encode(pred_rgb) # b 4 t h/8 w/8

    #     with torch.no_grad():
    #         if step_ratio is not None:
    #             # dreamtime-like
    #             # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
    #             t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
    #             t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
    #         else:
    #             t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

    #         # w(t), sigma_t^2
    #         w = (1 - self.scheduler.alphas_cumprod[t]).view(batch_size, 1, 1, 1)

    #         # predict the noise residual with unet, NO grad!
    #         # add noise
    #         noise = torch.randn_like(latents)
    #         latents_noisy = self.scheduler.q_sample(latents, t, noise)
    #         # pred noise
    #         latent_model_input = torch.cat([latents_noisy] * 2)
    #         tt = torch.cat([t] * 2)

    #         embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), 
    #                                 self.embeddings['neg'].expand(batch_size, -1, -1)])
   
    #         noise_pred = self.dit(latent_model_input, tt, encoder_hidden_states=embeddings).sample

    #         # perform guidance (high scale from paper!)
    #         noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    #         grad = w * (noise_pred - noise)
    #         grad = torch.nan_to_num(grad)

    #         # seems important to avoid NaN...
    #         # grad = grad.clamp(-1, 1)

    #     target = (latents - grad).detach()
    #     denoised_video = self.vae.decode(target) # b 3 t h w

    #     denoised_video = F.interpolate(rearrange(denoised_video, 'b c t h w -> (b t) c h w'), size=(H, W), mode='blinear', align_corners=False)
    #     denoised_video = rearrange(denoised_video, '(b t) c h w -> b t c h w', b=batch_size, t=nf)

    #     return denoised_video



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



class Sora_3DReconstruction(nn.Module):
    pass