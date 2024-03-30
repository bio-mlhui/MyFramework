from detectron2.modeling import BACKBONE_REGISTRY
from einops import rearrange, reduce, repeat
from .utils import VideoMultiscale_Shape, ImageMultiscale_Shape
import os
import time
import torch.nn as nn
import torch
from transformers import SegformerForSemanticSegmentation

@BACKBONE_REGISTRY.register()
class Segformer(nn.Module):
    def __init__(self, configs) -> None:
        pt_path = os.getenv('PT_PATH') # nvidia/segformer-b3-finetuned-ade-512-512
        super().__init__()
        pretrained_model = SegformerForSemanticSegmentation.from_pretrained(os.path.join(pt_path, "segformer"), 
                                                                                 num_labels=1, ignore_mismatched_sizes=True)
        self.config = pretrained_model.config

        self.segformer_backbone = pretrained_model.segformer

        freeze = configs['freeze']
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

        self.multiscale_shapes = {}
        for name, spatial_stride, dim  in zip(['res2', 'res3', 'res4', 'res5'], 
                                              [4, 8, 16, 32],
                                              [64, 128, 320, 512]):
            self.multiscale_shapes[name] =  ImageMultiscale_Shape(spatial_stride=spatial_stride, dim=dim)
        self.max_stride = 32

    def forward(self, x):
        #bt c h w
        if not self.training:
            batch_feats = []
            for haosen in x:
                feats =  self.segformer_backbone(haosen.unsqueeze(0), # 1 c h w
                                        output_attentions=True,
                                        output_hidden_states=True,  
                                        return_dict=False,)[1]
                batch_feats.append(feats)
            batch_feats = list(zip(*batch_feats)) # 4
            batch_feats = [torch.cat(haosen, dim=0) for haosen in batch_feats] # list[bt c h w]
            encoder_hidden_states = batch_feats
        else:
            outputs = self.segformer_backbone(x,
                                            output_attentions=True,
                                            output_hidden_states=True,  # we need the intermediate hidden states
                                            return_dict=False,)
            encoder_hidden_states = outputs[1]
        # attended_hidden_states = [encoder_hidden_states[0]]
        # logits = self.decode_head(encoder_hidden_states)
        ret = {}
        names = ['res2', 'res3', 'res4', 'res5']
        for name, feat in zip(names, encoder_hidden_states):
            ret[name] = feat
        return ret
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



@BACKBONE_REGISTRY.register()
class Video2D_Segformer(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.image_homo = Segformer(configs=configs,)

        self.multiscale_shapes = {}
        for name, temporal_stride, spatial_stride, dim  in zip(['res2', 'res3', 'res4', 'res5'],  
                                                               [1, 1, 1, 1], 
                                                               [4, 8, 16, 32],
                                                               [64, 128, 320, 512]):
            self.multiscale_shapes[name] =  VideoMultiscale_Shape(temporal_stride=temporal_stride, 
                                                                  spatial_stride=spatial_stride, dim=dim)
        self.max_stride = [1, 32]

    
    def forward(self, x):
        # b c t h w
        batch_size, _, T = x.shape[:3]
        x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        layer_outputs = self.image_homo(x)

        layer_outputs = {key: rearrange(value.contiguous(), '(b t) c h w -> b c t h w',b=batch_size, t=T).contiguous() \
                         for key, value in layer_outputs.items()}
        return layer_outputs
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)