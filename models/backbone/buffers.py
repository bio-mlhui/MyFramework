from detectron2.modeling import BACKBONE_REGISTRY
import torch.nn as nn
from einops import rearrange

@BACKBONE_REGISTRY.register()
class VideoMultiscale_Text_Split(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        self.video_backbone = BACKBONE_REGISTRY.get(configs['video_backbone']['name'])(configs['video_backbone'])
        # video -> multiscale
        self.max_stride = self.video_backbone.max_stride
        self.multiscale_shapes = self.video_backbone.multiscale_shapes

        self.text_backbone = BACKBONE_REGISTRY.get(configs['text_backbone']['name'])(configs['text_backbone'])
        self.text_dim = self.text_backbone.text_dim
    
    def forward(self, videos, text_dict):
        videos = rearrange(videos, 'b t c h w -> b c t h w')
        multiscales = self.video_backbone(videos)  # b c t h w
        text_dict = self.text_backbone(text_dict)
        return multiscales, text_dict
    

# @BACKBONE_REGISTRY.register()
# class VideoText_Combine(nn.Module):
#     def __init__(self, configs) -> None:
#         super().__init__()

#         self.video_backbone = BACKBONE_REGISTRY.get(configs['video_backbone']['name'])(configs['video_backbone'])
#         # video -> multiscale
#         self.max_spatial_stride = self.video_backbone.max_spatial_stride

#         self.text_backbone = BACKBONE_REGISTRY.get(configs['text_backbone'])(configs['text_backbone'])
#         self.text_dim = self.text_backbone.text_dim
    
#     def forward(self, videos, text_dict):
#         multiscales = self.video_backbone(videos)  # b t c h w
#         text_dict = self.text_backbone(text_dict)
#         return multiscales, text_dict
        
