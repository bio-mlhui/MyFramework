import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange
import os
from detectron2.modeling import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class VideoAutoencoderKL(nn.Module):
    def __init__(self, vae_configs):
        # from_pretrained=None, micro_batch_size=None, cache_dir=None, local_files_only=False
        from_pretrained=vae_configs.pop('from_pretrained', None)
        micro_batch_size=vae_configs.pop('micro_batch_size', None)
        cache_dir=vae_configs.pop('cache_dir', None)
        local_files_only=vae_configs.pop('local_files_only', False)
        from_pretrained = os.path.join(os.getenv('PT_PATH'), from_pretrained)
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@META_ARCH_REGISTRY.register()
class VideoAutoencoderKLTemporalDecoder(nn.Module):
    def __init__(self, vae_configs, ):
        # from_pretrained=None, cache_dir=None, local_files_only=False
        from_pretrained=vae_configs['from_pretrained']
        cache_dir=vae_configs['cache_dir']
        local_files_only=vae_configs['local_files_only']
        super().__init__()
        self.module = AutoencoderKLTemporalDecoder.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        B, _, T = x.shape[:3]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.module.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
