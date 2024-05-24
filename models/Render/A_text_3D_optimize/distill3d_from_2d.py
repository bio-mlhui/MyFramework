import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from models.Render.representation.gaussian_renderer import GaussianRender_SameIntrin
from detectron2.modeling import META_ARCH_REGISTRY
from models.registry import register_model
from data_schedule import build_schedule
from functools import partial
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from models.optimization.optimizer import get_optimizer
from models.optimization.scheduler import build_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import math
import torch.optim as optim
import itertools
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from detectron2.utils import comm
import logging
import cv2 
import rembg
import os
from data_schedule.render.scene_utils.graphics_utils import BasicPointCloud
from data_schedule.render.scene_utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from data_schedule.render.scene_utils.cameras import orbit_camera, OrbitCamera, MiniMiniMiniCam
from plyfile import PlyData
from models.Render.representation.GS3D import GaussianModel
from argparse import Namespace
from models.optimization.gs_optimizer import get_expon_lr_func
from models.Render.representation.gaussian_renderer import render_with_depth_alpha

class Distill3DGS_From_2DDM(GaussianModel):
    def __init__(
        self,
        configs,):
        super().__init__(configs=configs)
        from . import distiller_2ddm
        render_config = configs['model']['render']
        distiller_config = configs['model']['distiller']
        zero123_config = configs['model']['zero123']
        self.loss_weight = configs['model']['loss_weight']
        self.cam = OrbitCamera(render_config['W'], render_config['H'], r=render_config['radius'], fovy=render_config['fovy'])

        # sample_setup
        self.pipe = Namespace(convert_SHs_python=configs['model']['sample']['convert_SHs_python'],
                         compute_cov3D_python=configs['model']['sample']['compute_cov3D_python'],
                         debug=False)
    
        self.gaussain_scale_factor = 1

        self.guidance_sd = META_ARCH_REGISTRY.get(distiller_config['name'])(distiller_config)
 
        # input image
        self.input_img = None
        self.input_mask = None # alpha
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        scene_list = MetadataCatalog.get('global_dataset').get('subset_list')
        assert len(scene_list) == 1, '基于optimize的必须是一个scene'
        scene_meta_id = scene_list[0]

        input_view_file = MetadataCatalog.get(scene_meta_id).get('text_3d').get('image_prompt', None)
        if input_view_file is not None:
            self.load_input(input_view_file)

        self.prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_text', None)
        self.negative_prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_negative_text', None)

        self.initialize_render(scene_meta_id, num_pts=render_config['initial_num_pts'], radius=render_config['initial_radius'])

        self.optimize_setup(optimize_configs=configs)
        self.active_sh_degree = self.max_sh_degree

    
    def initialize_render(self, scene_meta_id, num_pts, radius=0.5):
        # initialization
        gs_initialize = MetadataCatalog.get(scene_meta_id).get('gs_initialize', None)
        if isinstance(gs_initialize, BasicPointCloud):
            self.create_from_pcd(pcd=gs_initialize, spatial_lr_scale=1)
        elif isinstance(gs_initialize, str):
            self.load_ply(input)
        else:
            # init from random point cloud
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.create_from_pcd(pcd, 10)            

    def load_input(self, file):
        # load image
        logging.debug(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

    @property
    def device(self,):
        return torch.device('cuda')


    def optimize_setup(self, configs,):
        render_config = configs['model']['render']
        distiller_config = configs['model']['distiller']
        zero123_config = configs['model']['zero123']
        optimize_configs = configs['optimize']
        """
        模型已经根据场景初始化了, 并且放到了gpu里
        """
        self.percent_dense = optimize_configs['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        
        l = [
            {'params': [self._xyz], 'lr': optimize_configs['position_lr_init'] * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': optimize_configs['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest], 'lr': optimize_configs['feature_lr'] / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': optimize_configs['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': optimize_configs['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': optimize_configs['rotation_lr'], "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init= optimize_configs['position_lr_init']* self.spatial_lr_scale,
                                    lr_final= optimize_configs['position_lr_final']* self.spatial_lr_scale,
                                    lr_delay_mult=optimize_configs['position_lr_delay_mult'],
                                    max_steps=optimize_configs['position_lr_max_steps'])   

        self.log_lr_group_idx = {'xyz': 0, 'f_dc': 1, 'f_rest': 2, 'opacity': 3, 'scaling': 4, 'rotation':5}
        # self.scene_white_background = MetadataCatalog.get(scene_features['scene_name']).get('white_background')
        # self.scene_cameras_extent = MetadataCatalog.get(scene_features['scene_name']).get('cameras_extent')
        
        
        self.position_lr_init = optimize_configs['position_lr_init']
        self.position_lr_final = optimize_configs['position_lr_final']
        self.position_lr_delay_mult = optimize_configs['position_lr_delay_mult']
        self.position_lr_max_steps = optimize_configs['position_lr_max_steps']
        
        self.densify_from_iter = optimize_configs['densify_from_iter']
        self.densification_interval = optimize_configs['densification_interval']
        self.opacity_reset_interval = optimize_configs['opacity_reset_interval']
        self.densify_grad_threshold = optimize_configs['densify_grad_threshold']
        self.densify_until_iter = optimize_configs['densify_until_iter']

        # default camera
        if render_config['mvdream'] or render_config['imagedream']:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(render_config['elevation'], 90, render_config['radius'])
        else:
            pose = orbit_camera(render_config['elevation'], 0, render_config['radius'])
        
        
        self.fixed_cam = MiniMiniMiniCam(
            pose,
            render_config['ref_size'],
            render_config['ref_size'],
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )        

        self.enable_sd = self.loss_weight['loss_sd'] > 0 and self.prompt != ""
        self.enable_zero123 = self.loss_weight['loss_zero123'] > 0 and self.input_img is not None

        if self.enable_zero123:
            from .distiller_2ddm import Zero123
            if zero123_config['is_stable']:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')

        ref_size = render_config['ref_size']
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():
            if self.enable_sd:
                if distiller_config['imagedream']:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

        self.bg_color = torch.tensor(
            [1, 1, 1],
            dtype=torch.float32,
            device="cuda",
        )
        self.imagedream = distiller_config['imagedream']

    # 返回loss
    def forward(self, data,):
        ### known view
        if self.input_img_torch is not None and not self.imagedream:
            cur_cam = self.fixed_cam
            out = render_with_depth_alpha(cur_cam,
                                          pc=self,
                                          pipe=self.pipe,
                                          bg_color=self.bg_color)
            # rgb loss
            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]            
            rbb_loss = F.mse_loss(image, self.input_img_torch)
            # mask loss
            mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
            mask_loss =  F.mse_loss(mask, self.input_mask_torch)

        ### novel view (manual batch)
        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images = []
        poses = []
        vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

        for _ in range(self.opt.batch_size):

            # render random view
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            radius = 0

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
            poses.append(pose)

            cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            images.append(image)

            # enable mvdream training
            if self.opt.mvdream or self.opt.imagedream:
                for view_i in range(1, 4):
                    pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                    poses.append(pose_i)

                    cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                    # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                    image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                
        images = torch.cat(images, dim=0)
        poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

        # import kiui
        # print(hor, ver)
        # kiui.vis.plot_image(images)

        # guidance loss
        if self.enable_sd:
            if self.opt.mvdream or self.opt.imagedream:
                loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
            else:
                loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

        if self.enable_zero123:
            loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
        
        # optimize step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # densify and prune
        if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.step % self.opt.densification_interval == 0:
                self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
            if self.step % self.opt.opacity_reset_interval == 0:
                self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps
        
        return {'loss_sds': sds_loss}, self.loss_weight, {'gaussians': gaussians,
                                                                                    'pred_images': pred_images,
                                                                                    'pred_alphas': pred_alphas,
                                                                                    'psnr': psnr}


    def sample(self, data, **kwargs):
        gaussians, results = self.model_preds(data)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        return {
            'outviews_preds': pred_images,
        }

    def state_dict(self, **kwargs):
        return self.renderer.state_dict()

    def load_state_dict(self, ckpt, strict=True):
        self.renderer.load_state_dict(ckpt)     



    def optimize(self, 
                 loss_weight=None,
                 loss_dict_unscaled=None,
                 closure=None,
                 num_iterations=None,
                 optimize_dict=None,
                 **kwargs):
        num_iterations += 1
        self.update_learning_rate(num_iterations)  
        if num_iterations % 1000 == 0:
            self.oneupSHdegree()

        step_ratio = min(1, self.step / self.opt.iters)


        loss = 0
        loss = loss + 10000 * (step_ratio if self.loss_weight['warmup_rgb_loss'] else 1) * rbb_loss
        loss = loss + 1000 * (step_ratio if self.loss_weight['warmup_rgb_loss'] else 1) * mask_loss


        # loss backward了嘛?
        loss = sum([loss_dict_unscaled[k] * loss_weight[k] for k in loss_weight.keys()])
        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  

        viewspace_point_tensor = optimize_dict['viewspace_points']
        visibility_filter = optimize_dict['visibility_filter']
        radii = optimize_dict['radii']
        cameras_extent = optimize_dict['cameras_extent']
        white_background = optimize_dict['white_background']

        if num_iterations < self.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if num_iterations > self.densify_from_iter and num_iterations % self.densification_interval == 0:
                size_threshold = 20 if num_iterations > self.opacity_reset_interval else None
                self.densify_and_prune(self.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
            
            if num_iterations % self.opacity_reset_interval == 0 or (white_background and num_iterations == self.densify_from_iter):
                self.reset_opacity()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict=state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict=state_dict['scheduler'])


@register_model
def distill3d_from_2ddm(configs, device):
    from .aux_mapper import AuxMapper
    model_input_mapper = AuxMapper(configs['model']['input_aux'])

    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))

    model = Distill3D_From_2DDM(configs)

    # 通过metalog 获得数据相关的东西
    from detectron2.data import MetadataCatalog
    model.load_input()


    model.to(device)
    model.optimize_setup(configs)

    if comm.is_main_process():
        logging.debug(f'模型的总参数数量:{sum(p.numel() for p in model.parameters())}')
        logging.debug(f'模型的可训练参数数量:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if comm.get_world_size() > 1:
        # broadcast_buffers = False
        model = DDP(model, device_ids=[comm.get_local_rank()], find_unused_parameters=True, broadcast_buffers = False)
    return model, train_samplers, train_loaders, eval_function
