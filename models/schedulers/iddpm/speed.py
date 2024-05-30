import numpy as np
import torch
import torch.nn.functional as F


from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


from detectron2.modeling import META_ARCH_REGISTRY
@META_ARCH_REGISTRY.register()
class SpeeDiffusion(SpacedDiffusion):
    def __init__(
        self,
        scheduler_configs,
    ):
        num_sampling_steps = scheduler_configs.pop('num_sampling_steps', None)
        timestep_respacing = scheduler_configs.pop('timestep_respacing', None)
        noise_schedule = scheduler_configs.pop('noise_schedule', "linear")
        use_kl = scheduler_configs.pop('use_kl', False)
        sigma_small = scheduler_configs.pop('sigma_small', False)
        predict_xstart = scheduler_configs.pop('predict_xstart', False)
        learn_sigma = scheduler_configs.pop('learn_sigma', True)
        rescale_learned_sigmas = scheduler_configs.pop('rescale_learned_sigmas', False)
        diffusion_steps = scheduler_configs.pop('diffusion_steps', 1000)
        cfg_scale = scheduler_configs.pop('cfg_scale', 4.0)

        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
        )

        self.cfg_scale = cfg_scale
        # we fallback to numpy here as argmax_cuda is not implemented for Bool
        grad = np.gradient(self.sqrt_one_minus_alphas_cumprod.cpu())
        self.meaningful_steps = np.argmax(grad < 5e-5) + 1

        # p2 weighting from: Perception Prioritized Training of Diffusion Models
        self.p2_gamma = 1
        self.p2_k = 1
        self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_cumprod
        p = torch.tanh(1e6 * (torch.gradient(sqrt_one_minus_alphas_bar)[0] - 1e-4)) + 1.5
        self.p = F.normalize(p, p=1, dim=0)
        self.weights = 1 / (self.p2_k + self.snr) ** self.p2_gamma

    def t_sample(self, n, device):
        t = torch.multinomial(self.p, n // 2 + 1, replacement=True).to(device)
        dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
        t = torch.cat([t, dual_t], dim=0)[:n]
        return t

    def training_losses(self, model, x, t, *args, **kwargs):  # pylint: disable=signature-differs
        t = self.t_sample(x.shape[0], x.device)
        return super().training_losses(model, x, t, weights=self.weights, *args, **kwargs)

    def sample(self, *args, **kwargs):
        raise NotImplementedError("SpeeDiffusion is only for training")
