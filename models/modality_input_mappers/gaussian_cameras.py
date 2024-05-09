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
class Gaussian_Cameras:
    def __init__(self,
                 configs,
                 ) -> None:
        pass

    def mapper(self, views_dict):
        # gaussian colmap camera trainsform
        cam_poses = views_dict['extrin'] # V 4 4
        outviews_intrin = views_dict['intrin'] # list[obj] / obj
        if not isinstance(outviews_intrin):
            intrin_matrix = outviews_intrin.proj_matrix
        else:
            raise ValueError()
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        # c2w -> w2c^T
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ intrin_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        views_dict['gaussian_cam'] = {
            'cam_view': cam_view, # w2c^T
            'cam_view_proj': cam_view_proj, # w2c^T * proj^T
            'cam_pos': cam_pos # camera_center
        }
        return views_dict
        
    def collate(self, list_of_haosen, batch_videos):    
        return { 
            'images_lab_sim': images_lab_sim, # b t h w 8 每个pixel和它周围8个Pixel的相似度
            'post_similarity': post_similarity, # b t h w 9
            'patch_kernel_size': self.patch_kernel_size,
            'patch_dilation': self.patch_dilation
        }

           