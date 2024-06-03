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
from utils.misc import generate_unique_key, to_torch_dtype
from torchvision.io import write_video

class Sora_GenerateVideo:
    def __init__(
        self,
        configs,):
        self.loss_weight = configs['model']['loss_weight']

        self.prompt = configs['model']['prompt']
        self.negative_prompt = configs['model']['negative_prompt']
        self.prompt_key = generate_unique_key(self.prompt, self.negative_prompt)

        self.dtype = configs['model']['dtype']
        self.image_size = configs['model']['image_size']
        self.num_frames = configs['model']['num_frames']
        self.fps = configs['model']['fps']
        self.frame_interval = configs['model']['frame_interval']
        self.video_key = generate_unique_key(self.image_size, self.num_frames, self.fps, self.frame_interval, self.dtype)
        
        self.dtype = to_torch_dtype(self.dtype)
        self.video_id = configs['model']['video_id']

        sora_config = configs['model']['distiller']
        from .. import distiller
        self.vae = META_ARCH_REGISTRY.get(sora_config['vae']['name'])(sora_config['vae'])
        input_size = (self.num_frames, *self.image_size) # t h w
        # t/patch_t, h/patch_h, w/patch_w
  
        
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
        self.dit =  META_ARCH_REGISTRY.get(sora_config['dit']['name'])(sora_config['dit'])

        self.vae.to(self.device, self.dtype).eval()
        self.dit.to(self.device, self.dtype).eval()

        self.scheduler = META_ARCH_REGISTRY.get(sora_config['scheduler']['name'])(sora_config['scheduler'])

        self.get_text_embed(self.prompt, self.negative_prompt, out_dir=configs['out_dir'], sora_config=sora_config)
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
        return self.video_id, list(range(1))
    @property
    def device(self,):
        return torch.device('cuda')
   
    def __call__(self, data,): 
        z = torch.randn(1, self.vae.out_channels, *self.latent_size, device=self.device, dtype=self.dtype) # 1 4 t h/8 w/8
        self.samples = self.scheduler.sample(
            self.dit,
            text_encoder=None,
            z=z,
            prompts=None,
            device=self.device,
            additional_args=self.model_args,
        ).to(self.dtype)
        self.samples = self.vae.decode(self.samples) # 1 3 t h w

        return {}, self.loss_weight, {}

    def save_video_frames(self, frames, video_id, output_dir):
        x = self.samples.permute(0, 2, 1, 3, 4).squeeze(0) # b 3 t h w -> t 3 h w
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
def sora_generate_video(configs, device):
    from .aux_mapper import Optimize_AuxMapper
    model_input_mapper = Optimize_AuxMapper(configs['model']['input_aux'])
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))

    model = Sora_GenerateVideo(configs)
    assert comm.get_world_size() == 1
    return model, train_samplers, train_loaders, eval_function