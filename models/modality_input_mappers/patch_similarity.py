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
    return similarity # b 8 h w

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

# images_neighbor每一个像素和 image对应的patch的similarity
def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation):
    # b c h w, b c h w
    unfolded_images = unfold_w_center(
        images, kernel_size=kernel_size, dilation=dilation
    ) 
    diff = images_neighbor[:, :, None] - unfolded_images # b c 9 h w
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5) # b 9 h w
    return similarity # b 9 h w

# images_neighbor每一个像素和 image对应的patch的similarity
def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation):
    unfolded_images = unfold_w_center( 
        images, kernel_size=kernel_size, dilation = 1 
    ) # b 3 9 h w
    unfolded_images_neighbor = unfold_w_center( 
        images_neighbor, kernel_size=kernel_size, dilation= 1 
    ) 
    unfolded_images = unfolded_images.flatten(1,2) 
    unfolded_images_neighbor = unfolded_images_neighbor.flatten(1,2) # b 27 h w
    # b 9 h w
    similarity = get_neighbor_images_color_similarity(unfolded_images, unfolded_images_neighbor, kernel_size, dilation)   
    return similarity

# def calculate_deformation_coordinates(k, d):
#     # Calculate effective kernel size
#     k_prime = d * (k - 1) + 1
#     # Calculate the range for the deformation coordinates
#     offset = (k_prime - 1) // 2
#     # Generate deformation coordinates
#     deformation_coordinates = [(y - offset, x - offset) for y in range(0, k_prime, d) for x in range(0, k_prime, d)]
#     return deformation_coordinates


@MODELITY_INPUT_MAPPER_REGISTRY.register()
class Video_PatchSimilarity:
    def __init__(self,
                 configs,
                 ) -> None:

        # self.k_size = 3

        self.patch_kernel_size = configs['patch_kernel_size'] if 'patch_kernel_size' in configs else 3
        self.patch_dilation = configs['patch_dilation'] if 'patch_dilation' in configs else 3

    def topk_mask(self, images_lab_sim, k):
        # b t h w 2 k^2
        batch_size, nf, H, W = images_lab_sim.shape[:-2]
        images_lab_sim_mask = torch.zeros_like(images_lab_sim)
        topk, indices = torch.topk(images_lab_sim, k, dim =-1) # value, idx
        # # b t h w 2(y,x 绝对)
        # y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij') # in the same order as the cardinality of the inputs
        # abs_coords = torch.stack([y_coords, x_coords], dim=2)[None, None, ...].repeat(batch_size, nf, 1, 1, 1) # b t h w 2

        images_lab_sim_mask = images_lab_sim_mask.scatter(5, indices, topk)
        return images_lab_sim_mask  
              
    def mapper(self, video):
        return {
            'haosen': None,
        }
        
    def collate(self, list_of_haosen, batch_videos):
        batch_size, T = batch_videos.shape[:2] 
        downsampled_images = F.avg_pool2d(batch_videos.flatten(0, 1), kernel_size=4, stride=4, padding=0) # bt 3 h/4 w/4
        images_lab = [torch.as_tensor(color.rgb2lab(ds_image.byte().permute(1, 2, 0).cpu().numpy()),\
                      device=ds_image.device, dtype=torch.float32).permute(2, 0, 1) for ds_image in downsampled_images] # list[3 h w], bt
        images_lab = torch.stack(images_lab, dim=0) # bt 3 h w

        # bt 8 h w: 每个像素和dilation2 kernel3的8个pixel的color similarity
        images_lab_sim = get_images_color_similarity(images_lab, kernel_size=3, dilation=2)
        images_lab_sim = rearrange(images_lab_sim, '(b t) c h w -> b t c h w',b=batch_size, t=T) # b t 8 h/4 w/4
        images_lab_sim = images_lab_sim.permute(0, 1, 3, 4, 2) # b t h w c # 每个像素和当前帧的patch的similarity
        images_lab = rearrange(images_lab, '(b t) c h w -> b t c h w',b=batch_size, t=T)
        past_similarity = [] # list[b k^2 h w], T
        post_similarity = [] # list[b k^2 h w], T
        
        for temporal_idx in range(T):
            past_idx = temporal_idx - 1
            post_idx = (temporal_idx + 1) % T

            past_frames = images_lab[:, past_idx] 
            current_frames = images_lab[:, temporal_idx] # b 3 h w
            post_frames = images_lab[:, post_idx] 

            past_similarity.append(get_neighbor_images_patch_color_similarity(past_frames, current_frames, 
                                                                              kernel_size=self.patch_kernel_size, 
                                                                              dilation=self.patch_dilation)) # b 9 h w
            post_similarity.append(get_neighbor_images_patch_color_similarity(post_frames, current_frames, 
                                                                              kernel_size=self.patch_kernel_size, 
                                                                              dilation=self.patch_dilation))

        bidirection_similarity = torch.stack([torch.stack(past_similarity, dim=1), torch.stack(post_similarity, dim=1)], dim=2) # b T 2 k^2 h w
        bidirection_similarity = bidirection_similarity.permute(0, 1, 4, 5, 2, 3) # b t h w 2 k^2

        bidirection_similarity = self.topk_mask(bidirection_similarity, k=5) # # b t h w 2 9
        
        return { 
            'images_lab_sim': images_lab_sim, # b t h w 8 每个pixel和它周围8个Pixel的相似度
            'bidirection_similairy': bidirection_similarity, # b t h w 2 9
            'patch_kernel_size': self.patch_kernel_size,
            'patch_dilation': self.patch_dilation
        }

           