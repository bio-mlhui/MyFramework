import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from sgm.inference.helpers import embed_watermark
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
from glob import glob
import os
from argparse import Namespace
from detectron2.modeling import META_ARCH_REGISTRY
from einops import rearrange
import torchvision.transforms.functional as torchv_F
from PIL import Image
from utils.misc import generate_unique_key, to_torch_dtype
from torchvision.io import write_video
from sgm.util import default, instantiate_from_config
from omegaconf import OmegaConf
from einops import rearrange, repeat

from models.distiller.stable_dms.scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    pt_path=None,
    sampler_num_steps = None,
    sampler_sigma_minmax = None
):
    config = OmegaConf.load(config)
    transform_local_pt_path(config, pt_path)
    transform_sampler(config, sigma_minmax=sampler_sigma_minmax, num_steps=sampler_num_steps)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = ( # bt
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id": # bt
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(  # bt
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise": # t 3 h w
            batch[key] = repeat(value_dict[key], "t ... -> (b t) ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad": # b
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc
def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def transform_local_pt_path(config, pt_path):
    ckpt_path = config.model.params.ckpt_path
    splits = ckpt_path.split('/')
    splits[0] = os.path.join(os.getenv('PT_PATH'), pt_path)
    ckpt_path = '/'.join(splits)
    config.model.params.ckpt_path = ckpt_path

from sgm.util import get_obj_from_str

import models.VIDVID.svd_denoise_video
import sgm.modules.diffusionmodules.sampling
def transform_sampler(config, sigma_minmax, num_steps):
    sampler_target = config.model.params.sampler_config.target
    obj_cls = get_obj_from_str(sampler_target)

    class Distill_Sampler_Class(obj_cls):  # 0 到 最大
        def __init__(self,  **kwargs):
            super().__init__(**kwargs)
            self.discretization.sigma_min = sigma_minmax[0]
            self.discretization.sigma_max = sigma_minmax[1]
            self.num_steps = num_steps
            self.sigmas = self.discretization(
                self.num_steps if num_steps is None else num_steps, device=self.device
            )
            # let x times the initial sigma
            pass
            
        # def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        #     sigmas = self.discretization(
        #         self.num_steps if num_steps is None else num_steps, device=self.device
        #     )
        #     uc = default(uc, cond)

        #     x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        #     num_sigmas = len(sigmas)

        #     s_in = x.new_ones([x.shape[0]])

        #     return x, s_in, sigmas, num_sigmas, cond, uc

        def __call__(self, denoiser, x, cond, uc=None, num_steps=None, step_ratio=None,):
            uc = default(uc, cond)
            s_in = x.new_ones([x.shape[0]])
            if step_ratio is not None:
                sigma_idx = int(np.round(step_ratio * self.num_steps).clip(0, len(self.sigmas) - 2))
            else:
                sigma_idx = int(torch.randint(0, len(self.sigmas) - 1))
            step_sigma, next_sigma = self.sigmas[sigma_idx], self.sigmas[sigma_idx + 1]
            gamma = (
                min(self.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
                if self.s_tmin <= step_sigma <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * step_sigma,
                s_in * next_sigma,
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )
            return x
    setattr(models.VIDVID.svd_denoise_video, 'Distill_Sampler_Class', Distill_Sampler_Class)
    config.model.params.sampler_config.target = 'models.VIDVID.svd_denoise_video.Distill_Sampler_Class'

class DenoiseVideo_SVD_NoText_ReferenceImage:
    def __init__(
        self,
        configs,):
        self.loss_weight = configs['model']['loss_weight']
        self.output_folder = configs['out_dir']
        # 自己定义的
        self.image_size = configs['model']['image_size']

        num_frames = configs['model']['num_frames'] # 
        num_steps = configs['model']['num_steps']
        version = configs['model']['version'] # svd
        self.fps_id = configs['model']['fps_id'] # 6
        
        self.motion_bucket_id = configs['model']['motion_bucket_id']
        self.cond_aug = configs['model']['cond_aug']
        self.decoding_t = configs['model']['decoding_t']
        self.encoding_t = configs['model']['encoding_t']
        self.elevations_deg = configs['model']['elevations_deg']
        self.azimuths_deg = configs['model']['elevations_deg']
        self.image_frame_ratio = configs['model']['image_frame_ratio']

        if version == "svd":
            self.num_frames = default(num_frames, 14)
            self.num_steps = default(num_steps, 25)
            model_config = "models/distiller/stable_dms/scripts/sampling/configs/svd.yaml"
            pt_path = 'stabilityai_stable-video-diffusion-img2vid'
        elif version == "svd_xt":
            self.num_frames = default(num_frames, 25)
            self.num_steps = default(num_steps, 30)
            model_config = "models/distiller/stable_dms/scripts/sampling/configs/svd_xt.yaml"
            pt_path = 'stabilityai_stable-video-diffusion-img2vid-xt'
        elif version == "svd_image_decoder":
            self.num_frames = default(num_frames, 14)
            self.num_steps = default(num_steps, 25)
            model_config = "models/distiller/stable_dms/scripts/sampling/configs/svd_image_decoder.yaml"
            pt_path = 'stabilityai_stable-video-diffusion-img2vid'
        elif version == "svd_xt_image_decoder":
            self.num_frames = default(num_frames, 25)
            self.num_steps = default(num_steps, 30)
            model_config = "models/distiller/stable_dms/scripts/sampling/configs/svd_xt_image_decoder.yaml"
            pt_path = 'stabilityai_stable-video-diffusion-img2vid'
        else:
            raise ValueError(f"Version {version} does not exist.")

        self.model, self.filter = load_model(
            model_config,
            self.device,
            self.num_frames,
            self.num_steps,
            True,
            pt_path=pt_path,
            sampler_num_steps = configs['optim']['sampler_num_steps'],
            sampler_sigma_minmax = configs['optim']['sampler_sigma_minmax']
        )
        
        input_video = configs['model']['input_video']
        self.video_id = configs['model']['video_id']
        video_key = generate_unique_key(input_video, self.video_id)
        video_save_file = os.path.join(configs['out_dir'], f'{video_key}.pth')
        if os.path.exists(video_save_file):
            saved_pth = torch.load(video_save_file)
            input_video = saved_pth['input_video']
            frame_strs = saved_pth['frame_strs']
            original_image_size = saved_pth['original_image_size']
        else:
            assert os.path.exists(input_video)
            frames = sorted(os.listdir(input_video))
            frame_strs =sorted([png[:-4] for png in frames if png.endswith('.png') or png.endswith('.jpg')])
            input_video = [Image.open(os.path.join(input_video, f)).convert('RGB') for f in frames]
            input_video = torch.stack([torchv_F.to_tensor(haosen) for haosen in input_video], dim=0) # t 3 h w, 0-1, float
            original_image_size = input_video.shape[2:]
            torch.save({'input_video': input_video, 'original_image_size': original_image_size, 'frame_strs': frame_strs}, video_save_file)
        
        assert input_video.shape[0] >= self.num_frames
        self.input_video = input_video[:self.num_frames].to(self.device) # t 3 h w
        self.frame_strs = frame_strs[:self.num_frames]
        self.original_image_size = original_image_size
        H, W = self.original_image_size
        if H != self.image_size[0] or W != self.image_size[1]:
            self.input_video = F.interpolate(self.input_video, size=self.image_size, mode='bilinear', align_corners=False)
            H, W = self.image_size
        _F = 8
        C = 4
        self.input_video = self.input_video * 2 - 1 # t 3 h w

        latent_save_file = os.path.join(configs['out_dir'], generate_unique_key('latent', self.image_size, video_key, self.num_frames, self.encoding_t))
        if os.path.exists(latent_save_file):
            saved_pth = torch.load(latent_save_file)
            noisy_latents = saved_pth['noisy_latents']
        else:
            with torch.no_grad():
                self.model.to('cpu')
                torch.cuda.empty_cache()
                self.model.first_stage_model.to(self.device)
                en_and_decode_n_samples_a_time = self.model.en_and_decode_n_samples_a_time
                self.model.en_and_decode_n_samples_a_time = self.encoding_t
                noisy_latents = self.model.encode_first_stage(self.input_video)
                self.model.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
                self.model.first_stage_model.to('cpu')
                torch.cuda.empty_cache()
                self.model.to(self.device)
            torch.save({'noisy_latents': noisy_latents, }, latent_save_file)
        
        
        self.noisy_latents = noisy_latents   
        # times the initial sigma       

        reference_video = self.input_video # t 3 h w TODO: 第一帧是reference frame
        self.shape = (num_frames, C, H // _F, W // _F)
        # self.noisy_latents = torch.randn(self.shape, device=self.device)
        self.noisy_latents *= torch.sqrt(1.0 + self.model.sampler.sigmas[0] ** 2.0) 
        
        value_dict = {}
        value_dict["cond_frames_without_noise"] = reference_video # 1 t 3 h w
        value_dict["motion_bucket_id"] = self.motion_bucket_id
        value_dict["fps_id"] = self.fps_id
        value_dict["cond_aug"] = self.cond_aug
        value_dict["cond_frames"] = reference_video + self.cond_aug * torch.randn_like(reference_video)
        self.value_dict = value_dict
        self.log_lr_group_idx = {}
        # self.prompt = configs['model']['prompt']
        # self.negative_prompt = configs['model']['negative_prompt']
        # self.prompt_key = generate_unique_key(self.prompt, self.negative_prompt)
        # self.get_text_embed(self.prompt, self.negative_prompt, out_dir=configs['out_dir'], sora_config=sora_config)
        
        # ['cond_frames_without_noise', 'fps_id', 'motion_bucket_id', 'cond_frames', 'cond_aug']
        
        
        with torch.no_grad():
            with torch.autocast('cuda'):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                    self.value_dict,
                    [1, self.num_frames],
                    T=self.num_frames,
                    device=self.device,
                )  # frames_without_noise: bt 3 h w, frames: bt 3 h w(cond_aug), fps_id: 1t, motion_bucket_id:1t, cond_aug:1t
                # cond_frames = batch.pop('cond_frames') # bt 3 h w
                # cond_frames_without_noise = batch.pop('cond_frames_without_noise') # bt 3 h w                
                self.c, self.uc = self.model.conditioner.get_unconditional_conditioning( # CLIP encode
                    batch,
                    None,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )    
        


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
    
    @torch.no_grad()
    def __call__(self, data,):    
        num_iterations = data['num_iterations'] + 1
        step_ratio = min(1, math.floor(float(num_iterations) / self.model.sampler.num_steps))

        with torch.autocast('cuda'):
            # randn = torch.randn(self.shape, device=self.device)
            randn = self.noisy_latents
            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, self.num_frames
            ).to(self.device)
            additional_model_inputs["num_video_frames"] = self.num_frames

            def denoiser(input, sigma, c):
                return self.model.denoiser(
                    self.model.model, input, sigma, c, **additional_model_inputs
                )

            self.noisy_latents = self.model.sampler(denoiser, randn, cond=self.c, uc=self.uc, step_ratio=step_ratio)

        return {}, self.loss_weight, {}

    @torch.no_grad()
    def save_video_frames(self, frames, video_id, output_dir):
        # only uses the first_stage_model in gpu
        with torch.autocast('cuda'):
            self.model.to('cpu')
            torch.cuda.empty_cache()
            self.model.first_stage_model.to(self.device)
            en_and_decode_n_samples_a_time = self.model.en_and_decode_n_samples_a_time
            self.model.en_and_decode_n_samples_a_time = self.decoding_t
            samples_x = self.model.decode_first_stage(self.noisy_latents)
            self.model.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

            self.model.first_stage_model.to('cpu')
            torch.cuda.empty_cache()
            self.model.to(self.device)

        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        os.makedirs(output_dir, exist_ok=True)
        samples = embed_watermark(samples)
        samples = self.filter(samples)
        vid = (
            (rearrange(samples, "t c h w -> t h w c") * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        video_path = os.path.join(output_dir, f"{self.video_id}.mp4")
        imageio.mimwrite(video_path, vid)


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
def svd_notext_referenceImage_denoise_video(configs, device):
    from .aux_mapper import Optimize_AuxMapper
    model_input_mapper = Optimize_AuxMapper(configs['model']['input_aux'])
    
    train_samplers, train_loaders, eval_function = build_schedule(configs, model_input_mapper.mapper, partial(model_input_mapper.collate,))
    model = DenoiseVideo_SVD_NoText_ReferenceImage(configs)
    assert comm.get_world_size() == 1
    return model, train_samplers, train_loaders, eval_function

