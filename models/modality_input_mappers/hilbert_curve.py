
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
# models.aux_data 没有__init__函数, 对于每一个task, 都要在对应的data_schedule里进行实例化, 比如RAMR -> RVOS_RAMR
import logging

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from models.layers.gilbert.gilbert2d import gilbert2d, gilbert2d_widthBigger
from models.layers.gilbert.gilbert3d import gilbert3d
@MODELITY_INPUT_MAPPER_REGISTRY.register()
class HilbertCurve_FrameQuery:
    def __init__(self,
                 configs,
                 ) -> None:
        
        self.frame_query_number = configs['frame_query_number']   
              
    def mapper(self, video):
        return {
            'haosen': None,
        }
        
    def collate(self, list_of_haosen, batch_videos):
        batch_size, T = batch_videos.shape[:2] 
        batch_size, T, _, H, W = batch_videos.shape
        # t nq, tnq的坐标
        hilbert_curve = list(gilbert2d_widthBigger(width=self.frame_query_number, height=T)) # list[(x(width), y(height))]
        hilbert_curve = torch.tensor(hilbert_curve).long()
        hilbert_curve = hilbert_curve[:, 1] * self.frame_query_number + hilbert_curve[:, 0]
        
        return { 
            'hilbert_curve': hilbert_curve,
        }


@MODELITY_INPUT_MAPPER_REGISTRY.register()
class HilbertCurve3D_FrameQuery:
    def __init__(self,
                 configs,
                 ) -> None:
        pass

    def mapper(self, video):
        return {
            'haosen': None,
        }

    def collate(self, list_of_haosen, batch_videos):
        batch_size, T, _, H, W = batch_videos.shape

        hilbert_curve_3d = list(gilbert3d(width=W, height=H, depth=T)) # list[(x,y,z)]
        hilbert_curve_3d = torch.tensor(hilbert_curve_3d).long() # N 3

        # t h w -> thw
        hilbert_curve_3d = hilbert_curve_3d[:, 2] * (H * W) + hilbert_curve_3d[:, 1] * W + hilbert_curve_3d[:, 0]
        return {    
            'hilbert_curve_3d': hilbert_curve_3d
        }
    
        

     