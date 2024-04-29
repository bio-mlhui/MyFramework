#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from data_schedule.render.scene_utils.sh_utils import eval_sh
import numpy as np
import kiui

class GaussianRender:
    def __init__(self, configs):
        # 相机的内参
        self.configs = configs

        self.camera_tan_half_fov = configs['camera']['tan_half_fov']
        self.camera_zfar = configs['camera']['zfar']
        self.camera_znear = configs['camera']['znear']
        self.camera_height = configs['camera']['output_size']
        self.camera_width = configs['camera']['outptu_width']

        wbcg = configs.pop('wbcg', True)
        self.bg_color = torch.tensor([1, 1, 1]).float().cuda() if wbcg else torch.tensor([0, 0, 0]).float().cuda()
    
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        """
        gaussians: [B, N, 14]
        cam_view, cam_view_proj: [B, V, 4, 4]
        cam_pos: [B, V, 3]
        """
        # TODO: camera_view_proj 不是由内参决定的嘛?
        
        # V视角的个数
        B, V = cam_view.shape[:2]
        images = []
        alphas = []
        for b in range(B):
            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()  
                raster_settings = GaussianRasterizationSettings(
                    image_height=self.camera_height,
                    image_width=self.camera_width,
                    tanfovx=self.camera_tan_half_fov,
                    tanfovy=self.camera_tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )       
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32).cuda(),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.camera_height, self.camera_width)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.camera_height, self.camera_width)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }                                                   



    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians


class GaussianRender:
    def __init__(self, configs):
        # 相机的内参
        self.configs = configs
    
    def render(self, gaussians, camera, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        """
        gaussians: [B, N, 14]
        cam_view, cam_view_proj: [B, V, 4, 4]
        cam_pos: [B, V, 3]
        """
        # TODO: camera_view_proj 不是由内参决定的嘛?

        camera_height = camera.camera_height
        camera_width = camera.camera_width
        camera_tan_half_fov = camera.camera_tan_half_fov


        # V视角的个数
        B, V = cam_view.shape[:2]
        images = []
        alphas = []
        for b in range(B):
            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()  
                raster_settings = GaussianRasterizationSettings(
                    image_height=camera_height,
                    image_width=camera_width,
                    tanfovx=camera_tan_half_fov,
                    tanfovy=camera_tan_half_fov,
                    bg=bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )       
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32).cuda(),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, camera_height, camera_width)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, camera_height, camera_width)

        return {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }                                                   



    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians


def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
