import os
import networkx as nx
from torch_geometric.data import Data
import torch
import json
from tqdm import tqdm
import logging
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
# models.aux_data 没有__init__函数, 对于每一个task, 都要在对应的data_schedule里进行实例化, 比如RAMR -> RVOS_RAMR
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from einops import rearrange, repeat, reduce
from skimage import color
import cv2
import numpy as np

@MODELITY_INPUT_MAPPER_REGISTRY.register()
class Ray_Embeddings:
    def __init__(self,
                 configs,
                 ) -> None:
        pass

    def mapper(self, views_dict):
        # ray embeddings:
        inviews_extrin = views_dict['extrin'] # V 4 4
        inviews_rendering_rgbs =  views_dict['rendering_rgbs'] # V 3 H W
        inviews_intrin = views_dict['intrin'] # list[obj] / obj
        V, _, H, W = inviews_rendering_rgbs.shape
        # build rays for input views
        rays_embeddings = []
        for i in range(V):
            fovY = inviews_intrin[i].fovY if isinstance(inviews_intrin, list) else inviews_intrin.fovY
            rays_o, rays_d = get_rays(inviews_extrin[i], H, W, fovY) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V_in, 6, h, w]
        views_dict['ray_embedding'] = rays_embeddings

        return views_dict
        
    def collate(self, list_of_haosen, batch_videos):    
        return { 
            'images_lab_sim': images_lab_sim, # b t h w 8 每个pixel和它周围8个Pixel的相似度
            'post_similarity': post_similarity, # b t h w 9
            'patch_kernel_size': self.patch_kernel_size,
            'patch_dilation': self.patch_dilation
        }

           