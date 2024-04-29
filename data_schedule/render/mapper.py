
import json
import os
from typing import List
import copy
from functools import partial
import random
import numpy as np
import torch
import logging
from einops import rearrange
from data_schedule.utils.segmentation import bounding_box_from_mask
from detectron2.data import MetadataCatalog

from data_schedule.registry import MAPPER_REGISTRY
from .mapper_utils import Render_TrainMapper, Render_EvalMapper
from .render_view_sampler import RENDER_VIEWS_SAMPLER_REGISTRY
from data_schedule.render.apis import Multiview3D_Optimize_Mapper, Scene_Meta, SingleView_3D_Mapper
import torch.nn.functional as F

@MAPPER_REGISTRY.register()
class Video4D_TrainMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        assert mode == 'train'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        rendering = self.get_rendering_fn(scene_id=data_dict['scene_id'], 
                                           scene_video_id=data_dict['scene_video_id'],
                                           view_camera=data_dict['view_camera'])
        
        data_dict['rendering'] = rendering

        data_dict = self.augmentation(data_dict)

        return {
            'view_dict': {'viewcamera': data_dict['view_camera'] },
            'targets': {
                'rendering': data_dict['rendering']
            }
        }


@MAPPER_REGISTRY.register()
class Video4D_EvalMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        assert mode == 'evaluate'
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        rendering = self.get_rendering_fn(scene_id=data_dict['scene_id'], 
                                           scene_video_id=data_dict['scene_video_id'],
                                           view_camera=data_dict['view_camera'])
        
        data_dict['rendering'] = rendering

        data_dict = self.augmentation(data_dict)
        callback_fns = data_dict.pop('callback_fns')[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        return {
            'view_dict': {'viewcamera': data_dict['view_camera'] },
            'meta': {
                'scene_id': data_dict['scene_id'],
                'scene_video_id': data_dict['scene_video_id'],
                'callback_fns': callback_fns
            }
        }




@MAPPER_REGISTRY.register()
class Image3D_TrainMapper(Render_TrainMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'train':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        scene_id, view_camera, metalog_name = data_dict['scene_id'], data_dict['view_camera'], data_dict['metalog_name']

        rendering = self.get_rendering_fn(scene_id=scene_id, 
                                          view_camera=view_camera)
        data_dict['rendering'] = rendering
        data_dict = self.augmentation(data_dict)
        Multiview3D_Optimize_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name']
            },
            'view_dict': {
                'view_camera': data_dict['view_camera']
            },
            'rendering_dict': {
                'rendering': data_dict['rendering']
            }
        }


@MAPPER_REGISTRY.register()
class Image3D_EvalMapper(Render_EvalMapper):
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'evaluate':
            logging.warning('this mapper should be used on evaluate datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)

    def _call(self, data_dict): 
        Scene_Meta
        data_dict = self.augmentation(data_dict)
        callback_fns = data_dict.pop('callback_fns', [])[::-1] # A -> B -> C; C_callback -> B_callback -> A_callback
        Multiview3D_Optimize_Mapper
        return {
            'scene_dict':{
                'scene_id': data_dict['scene_id'],
                'metalog_name': data_dict['metalog_name'],
                'callback_fns': callback_fns
            },
            'view_dict': {
                'view_camera': data_dict['view_camera'],
            },
        }


@MAPPER_REGISTRY.register()
class SingleView_3D_TrainerMapper(Render_TrainMapper):
    """
    同一个sample, camera的内参完全一样
    """ 
    def __init__(self,
                 configs=None,
                 dataset_name=None,
                 mode=None,
                 meta_idx_shift=None,
                 ): 
        if mode != 'train':
            logging.warning('this mapper should be used on training datasets')
        dataset_meta = MetadataCatalog.get(dataset_name)
        mapper_config = configs['data'][mode][dataset_name]['mapper']
        super().__init__(meta_idx_shift=meta_idx_shift,
                         dataset_meta=dataset_meta,
                         mapper_config=mapper_config)
        
        self.multiview_sampler = RENDER_VIEWS_SAMPLER_REGISTRY.get(\
            mapper_config['multiview_sampler']['name'])(sampler_configs=mapper_config['multiview_sampler'],
                                                        dataset_meta=dataset_meta)
        
        self.input_view_size = configs['input_view_size']
        self.output_view_size = configs['output_view_size']


    def transform_camera(self, camera):
        c2w = camera.c2w
        # TODO: you may have a different camera system
        # blender world + opencv cam --> opengl world & cam
        c2w[1] *= -1
        c2w[[1, 2]] = c2w[[2, 1]]
        c2w[:3, 1:3] *= -1 # invert up and forward direction

        # scale up radius to fully use the [-1, 1]^3 space!
        c2w[:3, 3] *= camera.radius / 1.5 # 1.5 is the default scale
        camera.c2w = c2w
        return camera

    def _call(self, data_dict): 
        # 同一个sample, camera的内参完全一样
        Scene_Meta
        # scene, view_cameras
        scene_id, view_cameras, metalog_name = data_dict['scene_id'], data_dict['view_cameras'], data_dict['metalog_name']
        
        # V_in + V_out
        input_views, output_views = self.multiview_sampler(all_cameras=view_cameras)
        sampled_views = input_views + output_views
        num_input_views = len(input_views)
        renderings, masks = list(zip(*[self.get_rendering_fn(haosen) for haosen in sampled_views]))
        cameras = [self.transform_camera(self.get_camera_fn(haosen)) for haosen in sampled_views]

        cam_poses = [haosen.c2w for haosen in cameras]
        renderings = torch.stack(renderings, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4], extrinstic 
        
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cameras[0].radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # [V_in, C, H, W]
        images_input = F.interpolate(renderings[:num_input_views].clone(), size=(self.input_view_size, self.input_view_size), mode='bilinear', align_corners=False)
        cam_poses_input = cam_poses[:num_input_views].clone()

        results = {
            'images_input': images_input,
            'cam_poses_input': cam_poses_input
        }
        results = self.augmentation(results)

        images_input = results['images_input']
        cam_poses_input = results['cam_poses_input']

        # build rays for input views
        rays_embeddings = []
        for i in range(num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.input_view_size, self.input_view_size, cameras[i].fovY) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ cameras[0].proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        SingleView_3D_Mapper
        return {
            'scene_dict':{
                'scene_id': scene_id,
                'metalog_name': metalog_name,
            },
            # 输入的views
            'inviews_dict':{
                'input_views': torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            },
            # 输出的views
            'outviews_dict':{
                'cam_view': cam_view,
                'cam_view_proj': cam_view_proj,
                'cam_pos': cam_pos,
                # resize render ground-truth images, range still in [0, 1]
                'rendering_rgbs': F.interpolate(renderings, size=(self.output_view_size, self.output_view_size), mode='bilinear', align_corners=False), # [V, C, output_size, output_size],
                'rendering_alphas': F.interpolate(masks.unsqueeze(1), size=(self.output_view_size, self.output_view_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
            }
        }
                  

        

# class Text4D_Condense_TrainMapper
# class Text4D_TrainMapper


