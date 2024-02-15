"""
This file contains a wrapper for Video-Swin-Transformer so it can be properly used as a temporal encoder for MTTR.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter
from utils.misc import NestedTensor, is_main_process
from detectron2.modeling import BACKBONE_REGISTRY

class FrozenBatchNorm2d(torch.nn.Module):
    """
    Modified from DETR https://github.com/facebookresearch/detr
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


@BACKBONE_REGISTRY.register()
class ResNet(nn.Module):
    """
    Modified from DETR https://github.com/facebookresearch/detr
    ResNet backbone with frozen BatchNorm.
    """
    def __init__(self, backbone_name: str = 'resnet50',
                 train_backbone: bool = True,
                 dilation: bool = True,
                 **kwargs):
        super(ResNetBackbone, self).__init__()
        backbone = getattr(torchvision.models, backbone_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        output_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
        self.layer_output_channels = [output_channels // 8, output_channels // 4, output_channels // 2, output_channels]

    def forward(self, tensor_list: NestedTensor):
        t, b, _, _, _ = tensor_list.tensors.shape
        video_frames = rearrange(tensor_list.tensors, 't b c h w -> (t b) c h w')
        padding_masks = rearrange(tensor_list.mask, 't b h w -> (t b) h w')
        features_list = self.body(video_frames)
        out = []
        for _, f in features_list.items():
            resized_padding_masks = F.interpolate(padding_masks[None].float(), size=f.shape[-2:]).to(torch.bool)[0]
            f = rearrange(f, '(t b) c h w -> t b c h w', t=t, b=b)
            resized_padding_masks = rearrange(resized_padding_masks, '(t b) h w -> t b h w', t=t, b=b)
            out.append(NestedTensor(f, resized_padding_masks))
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

