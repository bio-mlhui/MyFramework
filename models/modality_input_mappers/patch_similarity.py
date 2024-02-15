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

def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4 # b c h w
    assert kernel_size % 2 == 1


    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    ) # bt 3k^2 hw
    
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    ) # bt 3 k^2 hw

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2) # bt 3 k^2-1 hw

    return unfolded_x

def get_images_color_similarity(images, kernel_size, dilation):  # k = 3, dialation=2
    # b 3 h w
    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    ) # b 3 8 h w

    diff = images[:, :, None] - unfolded_images # b 3 1 h w - b 3 k^2-1 h w
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5) # b k^2-1 h w
    return similarity

def unfold_w_center(x, kernel_size, dilation):
    # b 3 h w
    # kernel_size: 3
    assert kernel_size % 2 == 1

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    ) # b 3 k^2 hw

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )
    return unfolded_x

# images的每一个patch和images_neighbor每一个像素的similairty
def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation):
    # b c h w, b c h w
    unfolded_images = unfold_w_center(
        images, kernel_size=kernel_size, dilation=dilation
    ) 
    diff = images_neighbor[:, :, None] - unfolded_images # b c 9 h w
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5) # b 9 h w
    return similarity # b 9 h w

# images的每一个patch 到 images_neighbor每一个像素的similairty
def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation):
    unfolded_images = unfold_w_center( # b 3 9 h w
        images, kernel_size=kernel_size, dilation = 1 
    ) 
    unfolded_images_neighbor = unfold_w_center( # b 3 9 h w
        images_neighbor, kernel_size=kernel_size, dilation= 1 #dilation
    ) 
    unfolded_images = unfolded_images.flatten(1,2) 
    unfolded_images_neighbor = unfolded_images_neighbor.flatten(1,2) # b 27 h w
    similarity = get_neighbor_images_color_similarity(unfolded_images, unfolded_images_neighbor, 3, 3)  
    return similarity


@MODELITY_INPUT_MAPPER_REGISTRY.register()
class Video_PatchSimilarity:
    def __init__(self,
                 configs,
                 ) -> None:
        k_size = configs['k_size']
        self.k_size = k_size
            
    def mapper(self, video):
        return {
            'taylor_swift': None,
        }
        
    def collate(self, list_of_taylor_swifts, batch_videos):
        batch_size, T = batch_videos.shape[:2] 
        downsampled_images = F.avg_pool2d(batch_videos.flatten(0, 1), kernel_size=4, stride=4, padding=0) # bt 3 h/4 w/4
        images_lab = [torch.as_tensor(color.rgb2lab(ds_image.byte().permute(1, 2, 0).cpu().numpy()),\
                      device=ds_image.device, dtype=torch.float32).permute(2, 0, 1) for ds_image in downsampled_images] # list[3 h w], bt
        images_lab = torch.stack(images_lab, dim=0) # bt 3 h w

        # bt k^2-1 h w: 每个patch和中心像素的similarity
        images_lab_sim = get_images_color_similarity(images_lab, kernel_size=self.k_size, dilation=2)
        images_lab_sim = rearrange(images_lab_sim, '(b t) c h w -> b t c h w',b=batch_size, t=T) # b t 8 h/4 w/4

        images_lab = rearrange(images_lab, '(b t) c h w -> b t c h w',b=batch_size, t=T)
        past_similarity = [] # list[b k^2 h w], T
        post_similarity = [] # list[b k^2 h w], T
        
        for temporal_idx in range(T):
            past_idx = temporal_idx - 1
            post_idx = (temporal_idx + 1) % T

            past_frames = images_lab[:, past_idx] 
            current_frames = images_lab[:, temporal_idx] # b 3 h w
            post_frames = images_lab[:, post_idx] 

            past_similarity.append(get_neighbor_images_patch_color_similarity(past_frames, current_frames, kernel_size=self.k_size, dilation=None))
            post_similarity.append(get_neighbor_images_patch_color_similarity(post_frames, current_frames, kernel_size=self.k_size, dilation=None))

        bidirection_similarity = \
            torch.stack([torch.stack(past_similarity, dim=1), torch.stack(post_similarity, dim=1)], dim=2) # b T 2 k^2 h w
        
        return {
            'images_lab_sim': images_lab_sim,
            'bidirection_similairy': bidirection_similarity
        }

                