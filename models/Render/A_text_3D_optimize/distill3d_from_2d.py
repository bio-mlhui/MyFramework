import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS
from collections import defaultdict
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
from detectron2.utils import comm
import logging
import cv2 
import rembg
from data_schedule.render.scene_utils.mesh import Mesh, safe_normalize
import os
from data_schedule.render.scene_utils.graphics_utils import BasicPointCloud
from data_schedule.render.scene_utils.sh_utils import eval_sh, SH2RGB, RGB2SH
from data_schedule.render.scene_utils.cameras import orbit_camera, OrbitCamera, MiniMiniMiniCam
from plyfile import PlyData
from models.Render.representation.GS3D import GaussianModel_meshfy
from argparse import Namespace
from models.optimization.gs_optimizer import get_expon_lr_func
from models.Render.representation.gaussian_renderer import render_with_depth_alpha
from data_schedule.render.scene_utils.grid_put import mipmap_linear_grid_put_2d


class Distill3DGS_From_2DDM(GaussianModel_meshfy):
    def __init__(
        self,
        configs,):
        super().__init__(configs=configs)
        render_config = configs['model']['render']
        self.loss_weight = configs['model']['loss_weight']
        self.cam = OrbitCamera(render_config['W'], render_config['H'], r=render_config['radius'], fovy=render_config['fovy'])
        self.mode = "image"
    
        self.gaussain_scale_factor = 1
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

        self.prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_text', "")
        self.negative_prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_negative_text', "")

        self.initialize_render(scene_meta_id, num_pts=render_config['initial_num_pts'], radius=render_config['initial_radius'])

        self.optimize_setup(configs=configs)
        

    
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
        self.bg_remover = None
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.cam.W, self.cam.H), interpolation=cv2.INTER_AREA)
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
        optimize_configs = configs['optim']
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
        self.active_sh_degree = self.max_sh_degree

        # default camera
        if distiller_config['mvdream'] or distiller_config['imagedream']:
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

        self.imagedream = distiller_config['imagedream']
        self.mvdream = distiller_config['mvdream']

        self.guidance_sd = None
        self.guidance_zero123 = None
        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.mvdream:
                from .distiller_2ddm.mvdream import MVDream
                self.guidance_sd = MVDream(self.device)
            elif self.imagedream:
                from .distiller_2ddm.imagedream import ImageDream
                self.guidance_sd = ImageDream(self.device)
            else:
                from .distiller_2ddm.stable_diffusion import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)

        if self.guidance_zero123 is None and self.enable_zero123:
            from .distiller_2ddm.zero123 import Zero123
            if zero123_config['is_stable']:
                self.guidance_zero123 = Zero123(self.device, model_key=os.path.join(os.getenv('PT_PATH'), 'zero123_stable'))
            else:
                self.guidance_zero123 = Zero123(self.device, model_key=os.path.join(os.getenv('PT_PATH'), 'zero123'))

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
        self.total_steps = configs['optim']['iters']
        self.min_ver = render_config['min_ver']
        self.max_ver = render_config['max_ver']
        self.batch_size = configs['optim']['batch_size']
        self.radius = render_config['radius']
        self.invert_bg_prob = configs['optim']['invert_bg_prob']
        self.render_fn = partial(render_with_depth_alpha, pc=self, pipe=Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False))
        self.anneal_timestep = configs['optim']['anneal_timestep']
        self.elevation = render_config['elevation']
        self.warmup_rgb_loss = configs['optim']['warmup_rgb_loss']

    # 返回loss
    def __call__(self, data,):
        num_iterations = data['num_iterations'] + 1
        step_ratio = min(1,  float(num_iterations) / self.total_steps)
        losses = defaultdict(lambda : torch.tensor(0., device=self.device))

        if self.input_img_torch is not None and not self.imagedream:
            cur_cam = self.fixed_cam
            out = self.render_fn(cur_cam, bg_color=self.bg_color)
            # rgb loss
            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]            
            losses['loss_rgb'] = F.mse_loss(image, self.input_img_torch)
            # mask loss
            mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
            losses['loss_mask'] = F.mse_loss(mask, self.input_mask_torch)

        ### novel view (manual batch)
        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images = []
        poses = []
        vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.min_ver, self.min_ver - self.elevation), -80 - self.elevation)
        max_ver = min(max(self.max_ver, self.max_ver - self.elevation), 80 - self.elevation)

        for _ in range(self.batch_size):

            # render random view
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            radius = 0

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            pose = orbit_camera(self.elevation + ver, hor, self.radius + radius)
            poses.append(pose)

            cur_cam = MiniMiniMiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

            bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            out = self.render_fn(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            images.append(image)

            # enable mvdream training
            if self.mvdream or self.imagedream:
                for view_i in range(1, 4):
                    pose_i = orbit_camera(self.elevation + ver, hor + 90 * view_i, self.radius + radius)
                    poses.append(pose_i)

                    cur_cam_i = MiniMiniMiniCam(pose_i, 
                                                render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                    out_i = self.render_fn(cur_cam_i, bg_color=bg_color)

                    image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                
        images = torch.cat(images, dim=0)
        poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

        # import kiui
        # print(hor, ver)
        # kiui.vis.plot_image(images)

        # guidance loss
        if self.enable_sd:
            if self.mvdream or self.imagedream:
                losses['loss_sd'] = self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.anneal_timestep else None)
            else:
                losses['loss_sd'] = self.guidance_sd.train_step(images, step_ratio=step_ratio if self.anneal_timestep else None)

        if self.enable_zero123:
            losses['loss_zero123'] = self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.anneal_timestep else None, default_elevation=self.elevation)

        self.need_update = True
        

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps
        
        return losses, self.loss_weight, out

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
        num_iterations += 1
        self.update_learning_rate(num_iterations)  
        step_ratio = min(1,  float(num_iterations) / self.total_steps)

        loss = 0
        loss += 10000 * (step_ratio if self.warmup_rgb_loss else 1) * loss_dict_unscaled['loss_rgb']
        loss += 1000 * (step_ratio if self.warmup_rgb_loss else 1) * loss_dict_unscaled['loss_mask']
        loss += loss_dict_unscaled['loss_sd'] * loss_weight['loss_sd']
        loss += loss_dict_unscaled['loss_zero123'] * loss_weight['loss_zero123']

        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        if num_iterations >= self.densify_from_iter and num_iterations <= self.densify_until_iter:
            viewspace_point_tensor, visibility_filter, radii = optimize_dict['viewspace_points'], optimize_dict['visibility_filter'], optimize_dict['radii']
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if num_iterations % self.densification_interval == 0:
                self.densify_and_prune(self.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
            if num_iterations % self.opacity_reset_interval == 0:
                self.reset_opacity()

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'spatial_lr_scale': self.spatial_lr_scale,

            'defaults':{
                'percent_dense': self.percent_dense,
                'position_lr_init': self.position_lr_init,
                'feature_lr': self.optimizer.param_groups[1]['lr'],
                'opacity_lr': self.optimizer.param_groups[3]['lr'],
                'scaling_lr': self.optimizer.param_groups[4]['lr'],
                'rotation_lr': self.optimizer.param_groups[5]['lr'],

                'position_lr_final': self.position_lr_final,
                'position_lr_delay_mult': self.position_lr_delay_mult,
                'position_lr_max_steps': self.position_lr_max_steps, 

                
                'densify_from_iter': self.densify_from_iter,
                'densification_interval': self.densification_interval,
                'opacity_reset_interval': self.opacity_reset_interval,
                'densify_grad_threshold': self.densify_grad_threshold,
                'densify_until_iter': self.densify_until_iter,                

                }
        },
    
    def load_optimize_state_dict(self, state_dict):
        # 先init当前状态对应的默认optimize状态
        self.optimize_setup(state_dict['defaults']) # 应该是config的形式

        self.xyz_gradient_accum = state_dict['xyz_gradient_accum']  # tensor
        self.denom = state_dict['denom']  # tensor
        self.max_radii2D = state_dict['max_radii2D'] # tensor
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.spatial_lr_scale = state_dict['spatial_lr_scale']

    def load_state_dict(self, ckpt, strict=True):
        # 只load model
        from models.trainer_model_api import Trainer_Model_API
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,) = ckpt       

    def state_dict(self,):
        from models.trainer_model_api import Trainer_Model_API
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,          
        )

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(output_dir, 'pred_mesh.ply')
            mesh = self.extract_mesh(path, 1)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(output_dir, 'pred_mesh.obj')
            mesh = self.extract_mesh(path, 1)

            # perform texture extraction
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr
 
            glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniMiniMiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.render_fn(cur_cam,bg_color=self.bg_color)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(output_dir, 'pred_model.ply')
            self.save_ply(path)

        logging.debug(f"[INFO] save model to {path}.")


from .mesh_renderer import Renderer
class Distill3DGS_From_2DDM_mesh:
    def __init__(
        self,
        configs,):
        render_config = configs['model']['render']
        self.loss_weight = configs['model']['loss_weight']
        self.loss_weight.pop('loss_mask')
        self.cam = OrbitCamera(render_config['W'], render_config['H'], r=render_config['radius'], fovy=render_config['fovy'])
        self.mode = "image"
        self.need_update = True
        self.renderer = Renderer(configs).to(self.device)
    
        self.gaussain_scale_factor = 1
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

        self.prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_text', "")
        self.negative_prompt = MetadataCatalog.get(scene_meta_id).get('text_3d').get('input_negative_text', "")

        self.optimize_setup(configs=configs)
        self.mesh_format = 'obj'
        
    def load_input(self, file):
        # load image
        logging.debug(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        self.bg_remover = None
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.cam.W, self.cam.H), interpolation=cv2.INTER_AREA)
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
        optimize_configs = configs['optim']
        self.optimizer = torch.optim.Adam(self.renderer.get_params()) 
        self.log_lr_group_idx = {'base': 0}

        # default camera
        if distiller_config['mvdream'] or distiller_config['imagedream']:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(render_config['elevation'], 90, render_config['radius'])
        else:
            pose = orbit_camera(render_config['elevation'], 0, render_config['radius'])
        
        self.fixed_cam = (pose, self.cam.perspective)

        self.enable_sd = self.loss_weight['loss_sd'] > 0 and self.prompt != ""
        self.enable_zero123 = self.loss_weight['loss_zero123'] > 0 and self.input_img is not None

        self.imagedream = distiller_config['imagedream']
        self.mvdream = distiller_config['mvdream']

        self.guidance_sd = None
        self.guidance_zero123 = None
        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.mvdream:
                from .distiller_2ddm.mvdream import MVDream
                self.guidance_sd = MVDream(self.device)
            elif self.imagedream:
                from .distiller_2ddm.imagedream import ImageDream
                self.guidance_sd = ImageDream(self.device)
            else:
                from .distiller_2ddm.stable_diffusion import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)

        if self.guidance_zero123 is None and self.enable_zero123:
            from .distiller_2ddm.zero123 import Zero123
            if zero123_config['is_stable']:
                self.guidance_zero123 = Zero123(self.device, model_key=os.path.join(os.getenv('PT_PATH'), 'zero123_stable'))
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')

        ref_size = render_config['ref_size']
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (ref_size, ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

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
        self.total_steps = configs['optim']['iters_refine']
        self.min_ver = render_config['min_ver']
        self.max_ver = render_config['max_ver']
        self.batch_size = configs['optim']['batch_size']
        self.radius = render_config['radius']
        self.invert_bg_prob = configs['optim']['invert_bg_prob']
        self.anneal_timestep = configs['optim']['anneal_timestep']
        self.elevation = render_config['elevation']
        self.warmup_rgb_loss = configs['optim']['warmup_rgb_loss']
        self.ref_size = render_config['ref_size']

    # 返回loss
    def __call__(self, data,):
        num_iterations = data['num_iterations'] + 1
        step_ratio = min(1,  float(num_iterations) / self.total_steps)
        losses = defaultdict(lambda : torch.tensor(0., device=self.device))

        if self.input_img_torch is not None and not self.imagedream:
            ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            out = self.renderer.render(*self.fixed_cam, self.ref_size, self.ref_size, ssaa=ssaa)
            # rgb loss
            image = out["image"] # [H, W, 3] in [0, 1]
            valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()           
            losses['loss_rgb'] = F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

        ### novel view (manual batch)
        render_resolution = 512
        images = []
        poses = []
        vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.min_ver, self.min_ver - self.elevation), -80 - self.elevation)
        max_ver = min(max(self.max_ver, self.max_ver - self.elevation), 80 - self.elevation)

        for _ in range(self.batch_size):

            # render random view
            ver = np.random.randint(min_ver, max_ver)
            hor = np.random.randint(-180, 180)
            radius = 0

            vers.append(ver)
            hors.append(hor)
            radii.append(radius)

            pose = orbit_camera(self.elevation + ver, hor, self.radius + radius)
            poses.append(pose)

            ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

            image = out["image"] # [H, W, 3] in [0, 1]
            image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

            images.append(image)

            # enable mvdream training
            if self.mvdream or self.imagedream:
                for view_i in range(1, 4):
                    pose_i = orbit_camera(self.elevation + ver, hor + 90 * view_i, self.radius + radius)
                    poses.append(pose_i)

                    out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                    image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                
        images = torch.cat(images, dim=0)
        poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

        # import kiui
        # print(hor, ver)
        # kiui.vis.plot_image(images)

        # guidance loss
        strength = step_ratio * 0.15 + 0.8
        if self.enable_sd:
            if self.mvdream or self.imagedream:
                refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                losses['loss_sd'] = F.mse_loss(images, refined_images)
            else:
                refined_images = self.guidance_sd.refine(images, strength=strength).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                losses['loss_sd'] = F.mse_loss(images, refined_images)

        if self.enable_zero123:
            # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
            refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.elevation).float()
            refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
            # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)
            losses['loss_zero123'] = F.mse_loss(images, refined_images)
        
        
        return losses, self.loss_weight, out

    def sample(self, data, **kwargs):
        # ignore if no need to update
        if not self.need_update:
            return

        if self.need_update:
            # render image
            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.cam.H, self.cam.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()
            
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
        num_iterations += 1
        step_ratio = min(1,  float(num_iterations) / self.total_steps)

        loss = 0
        loss += loss_dict_unscaled['loss_rgb']
        loss += loss_dict_unscaled['loss_sd'] * loss_weight['loss_sd']
        loss += loss_dict_unscaled['loss_zero123'] * loss_weight['loss_zero123']

        assert math.isfinite(loss.item()), f"Loss is {loss.item()}, stopping training"
        loss.backward()  
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

    def get_lr_group_dicts(self, ):
        return  {f'lr_group_{key}': self.optimizer.param_groups[value]["lr"] if value is not None else 0 \
                 for key, value in self.log_lr_group_idx.items()}

    def optimize_state_dict(self,):
        return {
            'optimizer': self.optimizer.state_dict(),
        },
    
    def load_optimize_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def load_state_dict(self, ckpt, strict=True):
        self.renderer.load_state_dict(ckpt)      

    def state_dict(self,):
        return self.renderer.state_dict()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, output_dir=None):
        os.makedirs(output_dir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(output_dir, 'pred_mesh.ply')
            mesh = self.extract_mesh(path, 1)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(output_dir, 'pred_mesh.obj')
            mesh = self.extract_mesh(path, 1)

            # perform texture extraction
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr
 
            glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniMiniMiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.render_fn(cur_cam,bg_color=self.bg_color)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(output_dir, 'pred_model.ply')
            self.save_ply(path)

        logging.debug(f"[INFO] save model to {path}.")

    def save_model(self, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
    
        path = os.path.join(output_dir, 'pred.' + self.mesh_format)
        self.renderer.export_mesh(path)

        logging.debug(f"[INFO] save model to {path}.")

    def train(self):
        pass

    def eval(self):
        pass

@register_model
def distill3dgs_from_2ddm(configs, device):
    from .aux_mapper import AuxMapper
    model_input_mapper = AuxMapper(configs['model']['input_aux'])
    
    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                  model_input_mapper.mapper, 
                                                                  partial(model_input_mapper.collate,))

    if configs['model']['mesh'] is not None:
        model = Distill3DGS_From_2DDM_mesh(configs)
    else:
        model = Distill3DGS_From_2DDM(configs)
    assert comm.get_world_size() == 1
    return model, train_samplers, train_loaders, eval_function