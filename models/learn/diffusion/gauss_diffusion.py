
from typing import Any, Optional, List, Dict, Set
import sys
import os
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision.transforms.functional as Trans_F
from einops import repeat, reduce, rearrange
from utils.misc import NestedTensor
from copy import deepcopy as dcopy
import logging
from tqdm import tqdm
from functools import partial
from utils.misc import to_device
from models.utils.visualize_amr import save_model_output
from models.registry import register_model
from data_schedule.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from models.optimization.scheduler import build_scheduler 
from detectron2.config import configurable
from models.registry import register_model
import detectron2.utils.comm as comm
import copy
from models.optimization.utils import get_total_grad_norm

from .gauss_diffusion_utils import GaussDiffusion_Beta_Schedule_Registry, extract, exists, cycle, identity, default

class GaussianDiffusion(nn.Module):
    def __init__(self, 
                 configs,
                ) -> None:
        super().__init__()

        # region
        # objective
            # pred_x0 / pred_e
        # input_scaling
            # 0.1
        # loss_type
        # sop

        # timesteps=1000,
        # beta_schedule='linear',
        # schedule_fn_kwargs={'beta_start':1e-4, 'beta_end':0.02},
        
        # ddim_sampling_eta=0.,
        # sampling_timesteps=20,
        # fig_directory=None,
        
        # objective='pred_x0',
        # input_scaling=0.1,
        # diffuse_resolution='stride4',     
        # loss_type='ce',
        # small_object_weight_p=0.,
        # num_instances=1,
        # num_noise=1,
        
        # p2_loss_weight_gamma=0.,
        # p2_loss_weight_k=1,
        # schedule
            # name
            # configs
        # sampling:
            # num_steps
            # ddim_eta
        #endregion
        betas = GaussDiffusion_Beta_Schedule_Registry.get(configs['beta_schedule']['name'])(configs['beta_schedule'])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.num_timesteps = len(betas)
        # sampling
        self.sample_num_timesteps = configs['sample_num_timesteps']
        self.is_ddim_sampling = self.sample_num_timesteps < self.num_timesteps
        self.ddim_sampling_eta = configs['ddim_sampling_eta']

        p2_loss_weight_k = configs['p2_loss_weight_k']
        p2_loss_weight_gamma = configs['p2_loss_weight_gamma']
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        # calculate p2 reweighting
        register_buffer('p2_loss_weight', 
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def q_posterior(self, x_start, x_t, t):
        """Given xt and x0, return the forward posterior mean
        Input:
            - x_start:  
                T(b 1 h w)
            - x_t:
                T(b 1 h w)
            - t:
                T(b, )
        Output:
            - mu(xt, t)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
     
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def model_predictions(self, 
                          xt, 
                          time_steps, 
                          conditionals):
        pass

    # 输入xt, t, condition, 输出e/x0/
    def p_mean_variance(self, xt, batch_times, conditions, clip_denoised=False):
        """ p(x_t_1|x_t) 的 mean, variance
        Input:
            xt: b ...
            batch_times: b
            conditions: dict
        """
        _, pred_x0 = self.model_predictions(xt, time_steps, conditionals, clip_x_start=clip_denoised)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_x0, xt, time_steps)
        return model_mean, posterior_variance, posterior_log_variance, pred_x0
    
    # -> x_{t-1} = posterior_mean + noise * posterior_variance
    def p_sample(self, 
                 xt, 
                 batch_times, 
                 conditions,
                 
                 cond_fn):
        """ 从 p(x_{t-1}|x_t) = N(model(xt, t), beta*I) 抽样
        Input:
            - xt: b ...
            - batch_times: b
            - conditions: dict
        """
        # b ...
        nonzero_mask = (batch_times != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))

        out = self.p_mean_variance(xt=xt, 
                                   batch_times=batch_times,
                                   conditions=conditions,)
        noise = torch.randn_like(xt)
        
        pred_x0 = out['pred_x0']
        log_variance = out['log_variance']
        if cond_fn is not None:
            pass
        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * log_variance) * noise
        return {'sample': sample, 'pred_x0': pred_x0,}
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
     # for each timestep     
    
    def ddpm_sample(self, shape, conditions, return_all=False):
        """
        - shape: b ...
        - conditions: dict
        - return_all: b T ...
        """
        xt = torch.randn(shape, device=self.device)
        samples_over_time = [xt]
        for time in tqdm(reversed(range(self.num_timesteps))):
            batch_times = torch.tensor([time] * shape[0]).to(self.device)
            out = self.p_sample(xt=xt, 
                                batch_times=batch_times, 
                                conditions=conditions)
            xt = out['sample']
            samples_over_time.append(xt)
            
        return xt if not return_all else torch.stack(samples_over_time, dim = 1)   
    
    def ddim_sample(self, shape, conditionals, return_all=False):
        """
        Input:
            - shape:
                TBN 1 h w
            - conditionals:
                same API with mask decoder
        Output:
            - T(TBN 1 h w)
            - T(TBN 2) / None
        """
        assert return_all == False
        TBN, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        TB = TBN // self.num_instances
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((TB,), time, device = device, dtype = torch.long)
            time_cond = repeat(time_cond, 'TB -> (TB N)', N=self.num_instances)
            pred_noise, x_start, pred_is_referred = self.model_predictions(img, time_cond, conditionals, clip_x_start = False)

            imgs.append(img)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise


        ret = img if not return_all else torch.stack(imgs, dim = 1)
        return ret, pred_is_referred

    @property
    def device(self):
        return self.p2_loss_weight.device

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out
