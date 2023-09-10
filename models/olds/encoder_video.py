
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange
from torchvision.models._utils import IntermediateLayerGetter
from .video_swin import SwinTransformer3D
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
#rom natten import NeighborhoodAttention3D
import copy
        

class VideoSwinTransformer(nn.Module):
    """
    A wrapper which allows using Video-Swin Transformer as a temporal encoder for MTTR.
    Check out video-swin's original paper at: https://arxiv.org/abs/2106.13230 for more info about this architecture.
    Only the 'tiny' version of video swin was tested and is currently supported in our project.
    Additionally, we slightly modify video-swin to make it output per-frame embeddings as required by MTTR (check our
    paper's supplementary for more details), and completely discard of its 4th block.
    """
    def __init__(self, backbone_pretrained, backbone_pretrained_path, running_mode):
        super(VideoSwinTransformer, self).__init__()
        # patch_size is (1, 4, 4) instead of the original (2, 4, 4).
        # this prevents swinT's original temporal downsampling so we can get per-frame features.
        swin_backbone = SwinTransformer3D(patch_size=(1, 4, 4), embed_dim=96, depths=(2, 2, 6, 2),
                                          num_heads=(3, 6, 12, 24), window_size=(8, 7, 7), drop_path_rate=0.1,
                                          patch_norm=True)
        if backbone_pretrained and running_mode == 'train':
            state_dict = torch.load(backbone_pretrained_path)['state_dict']
            # extract swinT's kinetics-400 pretrained weights and ignore the rest (prediction head etc.)
            state_dict = {k[9:]: v for k, v in state_dict.items() if 'backbone.' in k}

            # sum over the patch embedding weight temporal dim  [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
            patch_embed_weight = state_dict['patch_embed.proj.weight']
            patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
            state_dict['patch_embed.proj.weight'] = patch_embed_weight
            swin_backbone.load_state_dict(state_dict)

        self.patch_embed = swin_backbone.patch_embed
        self.pos_drop = swin_backbone.pos_drop
        self.layers = swin_backbone.layers
        self.downsamples = nn.ModuleList()
        for layer in self.layers:
            self.downsamples.append(layer.downsample)
            layer.downsample = None
        self.downsamples[-1] = None  # downsampling after the last layer is not necessary

        self.layer_output_channels = [swin_backbone.embed_dim * 2 ** i for i in range(len(self.layers))]
    
        self.scale_strides = [[1,4],[1,8],[1,16],[1,32]] 
        assert len(self.scale_strides) == len(self.layer_output_channels)

    def get_desc(self):
        return self.layer_output_channels, self.scale_strides
    
    def forward(self, samples: NestedTensor):
        vid_frames = rearrange(samples.tensors, 't b c h w -> b c t h w')

        vid_embeds = self.patch_embed(vid_frames)
        vid_embeds = self.pos_drop(vid_embeds)
        layer_outputs = []  # layer outputs before downsampling
        for layer, downsample in zip(self.layers, self.downsamples):
            vid_embeds = layer(vid_embeds.contiguous())
            layer_outputs.append(vid_embeds)
            if downsample:
                vid_embeds = rearrange(vid_embeds, 'b c t h w -> b t h w c')
                vid_embeds = downsample(vid_embeds)
                vid_embeds = rearrange(vid_embeds, 'b t h w c -> b c t h w')
        layer_outputs = [rearrange(o, 'b c t h w -> t b c h w') for o in layer_outputs]

        outputs = []
        orig_pad_mask = samples.mask
        for l_out in layer_outputs:
            pad_mask = F.interpolate(orig_pad_mask.float(), size=l_out.shape[-2:]).to(torch.bool)
            outputs.append(NestedTensor(l_out, pad_mask))
        return outputs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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


class VideoResNet(nn.Module):
    """
    Modified from DETR https://github.com/facebookresearch/detr
    ResNet backbone with frozen BatchNorm.
    """
    def __init__(self, 
                 backbone_name: str = 'resnet50',
                 train_backbone: bool = True,
                 dilation: bool = True,):
        super(VideoResNet, self).__init__()
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

class VideoMAE(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel
        self.vae_model =  VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    
    def forward(self, tensor_list: NestedTensor):
        # b t 3 h w; b t h w
        videos, video_pad_masks = tensor_list.decompose()
        
        video_feats = self.vae_model.forward(
            pixel_values = videos, # b t c h w
            bool_masked_pos = video_pad_masks.flatten(1), # b thw
            head_mask = None,
            output_attentions= None,
            output_hidden_states = True,
            return_dict = False,
        )


class VideoClip_VideoEncoder(nn.Module):
    pass

_video_encoder_entrypoints = {}

def register_video_encoder(fn):
    video_encoder_name = fn.__name__
    _video_encoder_entrypoints[video_encoder_name] = fn

    return fn

def video_encoder_entrypoints(video_encoder_name):
    try:
        return _video_encoder_entrypoints[video_encoder_name]
    except KeyError as e:
        print(f'video_encoder {video_encoder_name} not found')

def get_norm(norm, dim):
    if norm == 'group_32':
        return nn.GroupNorm(32, dim)
    elif norm == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError()

# 改backbone(build_backbone)
# 必须是4 8 16 32的输出
class VideoEncoder_MVM(nn.Module):
    def __init__(self, 
                 backbone_configs, 
                 proj_configs,
                 d_model, 
                 mvmhead_configs,
                 scale_before_fuse_configs) -> None:
        super().__init__()
        self.build_backbone(backbone_configs) 

        self.build_feat_proj(proj_configs, d_model=d_model)

        self.build_mvmhead(mvmhead_configs)
        
        self.vid_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d')

    def build_feat_proj(self, proj_configs, d_model):
        configs = vars(proj_configs)
        self.proj_name = configs['name']
        
        if self.proj_name != 'no_proj':
            bb_out_channels, bb_scalestrides = self.vid_backbone.get_desc()
            num_bb_lvls = len(bb_out_channels)

            out_scale_strides = configs['out_scale_strides']
            self.out_scale_strides = out_scale_strides
            proj_types = configs['each_proj_types']
            assert len(proj_types) == len(out_scale_strides)
            assert len(out_scale_strides) >= num_bb_lvls
            # 假设:
            # scale_strides 是 out_scale_strides的子集:  out_scale_strides = [scale_strides, ...]
            #
            # 对scale_strides都做projection
            # 不对temporal做downsample
            # local的kernel: 3;  linear的kernel: 1

            self.vid_proj = nn.ModuleList()

            for idx, ((out_temporal_stride, out_spatial_stride), tp) in enumerate(zip(out_scale_strides, proj_types)):
                if idx < num_bb_lvls:
                    in_temporal_stride, in_spatial_stride, in_channel = *bb_scalestrides[idx], bb_out_channels[idx]
                else:
                    in_temporal_stride, in_spatial_stride, in_channel = *bb_scalestrides[-1], bb_out_channels[-1]

                spatial_stride = out_spatial_stride // in_spatial_stride
                temporal_stride = out_temporal_stride // in_temporal_stride
                if self.proj_name == 'conv2d':
                    kp_dict = {3:1, 5:2}
                    lks = configs['local_kernel_size']
                    lkp = kp_dict[lks]
                    assert temporal_stride == 1, 'conv2d does not downsample in the temporal dimension'
                    if tp == 'local':
                        self.vid_proj.append(nn.Sequential(nn.Conv2d(in_channel, d_model, 
                                                                    kernel_size=lks, padding=lkp,
                                                                    stride=spatial_stride, 
                                                                    bias=configs['bias']), get_norm(configs['norm'], dim=d_model) ))    
                    elif tp == 'linear':   
                        self.vid_proj.append(nn.Sequential(nn.Conv2d(in_channel, d_model, 
                                                                    kernel_size=1, 
                                                                    stride=spatial_stride,
                                                                    bias=configs['bias']), get_norm(configs['norm'], dim=d_model)))
                    else:
                        raise ValueError()
                    
                elif self.proj_name == 'conv3d':
                    kp_dict = {3:1, 5:2}
                    lks = configs['local_kernel_size']
                    lkp = kp_dict[lks]
                    if tp == 'local':
                        self.vid_proj.append(nn.Sequential(nn.Conv3d(in_channel, d_model, 
                                                                    kernel_size=lks, padding=lkp,
                                                                    stride=[temporal_stride, spatial_stride, spatial_stride], 
                                                                    bias=configs['bias']), get_norm(configs['norm'], dim=d_model) ))    
                    elif tp == 'linear':   
                        self.vid_proj.append(nn.Sequential(nn.Conv3d(in_channel, d_model, 
                                                                    kernel_size=1, 
                                                                    stride=[temporal_stride, spatial_stride, spatial_stride],
                                                                    bias=configs['bias']), get_norm(configs['norm'], dim=d_model)))
                    else:
                        raise ValueError()            
                else:
                    raise NotImplementedError() # neighborhood
                
            for proj in self.vid_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
        else:
            self.out_scale_strides = self.vid_backbone.get_desc()[1]
            pass
        
    def build_backbone(self, backbone_configs):
        configs = vars(backbone_configs)
        if backbone_configs.name == 'video_swin_t':
            self.vid_backbone = VideoSwinTransformer(backbone_pretrained=configs['pretrained'],
                                                    backbone_pretrained_path=configs['pretrained_path'],
                                                    train_backbone=configs['train'],
                                                    running_mode=configs['running_mode'],)
            
        elif backbone_configs.name == 'video_resnet':
            self.vid_backbone = VideoResNet(backbone_name=configs['resnet_name'],
                                            train_backbone=configs['train'],
                                            dilation=configs['dilation'])
        elif backbone_configs.name == 'unmasked_teacher':
            pass
        elif backbone_configs.name == 'videoclip':
            self.vid_backbone = VideoCLIP_VideoEncoder()
        else:
            raise ValueError()
        
    def build_mvmhead(self, configs):
        pass
    
    def proj_bakcbone_out(self, backbone_out, pad_mask):
        """
        pad_mask: b t h w,  t: valid_t
        """
        # all the padding masks must be in bool type,因为会作为attention mask使用
        batch_size = backbone_out[0].tensors.shape[1] #  t b c h w
        srcs = [] # b t c h w
        masks = []
        poses = []  
        if self.proj_name == 'conv2d':
            nf = backbone_out[0].tensors.shape[0]
            for layer_out in backbone_out:
                layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
                layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
            for lvl, feat in enumerate(backbone_out): 
                src, mask = feat.decompose() 
                src_proj_l = self.vid_proj[lvl](src)
                src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                pos = self.vid_pos_embed(src_proj_l, None) # b t c h w
                srcs.append(src_proj_l)
                masks.append(mask)
                poses.append(pos)
                if lvl == (len(backbone_out) - 1):
                    for idx in range(lvl+1, len(self.vid_proj)):
                        src_proj_l = self.vid_proj[idx](src.clone())
                        src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                        mask = F.interpolate(pad_mask.float(), size=src_proj_l.shape[-2:],mode='nearest-exact').bool()
                        pos = self.vid_pos_embed(src_proj_l, None)
                        srcs.append(src_proj_l)
                        masks.append(mask)
                        poses.append(pos)

        elif self.proj_name == 'conv3d':
            for layer_out in backbone_out:
                layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> b c t h w')
                layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
            for lvl, feat in enumerate(backbone_out): 
                src, mask = feat.decompose() 

                src_proj_l = self.vid_proj[lvl](src)
                src_proj_l = rearrange(src_proj_l, 'b c t h w -> b t c h w')
                pos = self.vid_pos_embed(src_proj_l, None) # b t c h w
                srcs.append(src_proj_l)
                masks.append(mask)
                poses.append(pos)
                if lvl == (len(backbone_out) - 1):
                    for idx in range(len(self.vid_proj[(lvl+1):])):
                        src_proj_l = self.vid_proj[idx](src.clone())
                        src_proj_l = rearrange(src_proj_l, 'b c t h w -> b t c h w')
                        mask = F.interpolate(pad_mask.float(), size=src_proj_l.shape[-2:],mode='nearest-exact').bool()
                        pos = self.vid_pos_embed(src_proj_l, None)
                        srcs.append(src_proj_l)
                        masks.append(mask)
                        poses.append(pos)           
        
        elif self.proj_name == 'no_proj':
            nf = backbone_out[0].tensors.shape[0]
            for layer_out in backbone_out:
                layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> b t c h w')
                layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
            for lvl, feat in enumerate(backbone_out): 
                src, mask = feat.decompose() 
                pos = self.vid_pos_embed(src, None) # b t c h w
                srcs.append(src)
                masks.append(mask)
                poses.append(pos)           
        else:
            raise ValueError()
        
        return srcs, masks, poses


    def forward(self, videos, video_pad_mask, valid_indices, mask_video, masked_indices):
        """
        videos: t b c h w
        video_pad_mask: t b h w
        valid_indices: tensor(num_valid, ) int
        mask_video: True/False
        masked_indices: #TODO
        """
        vids = NestedTensor(videos, video_pad_mask)
        # t b 3 h w
        batch_size, device = videos.shape[1], videos.device
        nf = len(valid_indices)
        
        # list[NT(t b ci hi wi)], 4
        backbone_out = self.vid_backbone(vids)
        # 4 -> 8 -> 16 -> 32; t b c h w
        if valid_indices is not None:
            for layer_out in backbone_out:
                layer_out.tensors = layer_out.tensors.index_select(0, valid_indices) # t b c h w
                layer_out.mask = layer_out.mask.index_select(0, valid_indices)            

        pad_mask = video_pad_mask.index_select(0, valid_indices).permute(1, 0, 2, 3) # b t h w
        # 4 8 16 32 64
        srcs, masks, poses = self.proj_bakcbone_out(backbone_out, pad_mask=pad_mask)

        if mask_video:
            raise NotImplementedError()
        
        return {
            'multiscales': srcs,
            'multiscale_pad_masks': masks,
            'multiscale_poses': poses,
            'multiscale_des': copy.deepcopy(self.out_scale_strides),
            'mvm_video_gt': None
        }
        
    def mask_tokens(self, masked_indices):
        pass
    
    def forward_mvm(self, video_feats, masked_video_gt):   
        # output, loss_dict
        pass

    
@register_video_encoder
def video_encoder_mvm(configs, d_model):
    return VideoEncoder_MVM(
        backbone_configs=configs.video_backbone,
        proj_configs=configs.proj,
        d_model=d_model,
        mvmhead_configs=configs.mvmhead_configs,
        scale_before_fuse_configs=configs.scale_before_fuse_configs
    )

# with no projection
class VideoEncoderDecoder_MVM(nn.Module):
    def __init__(self, 
                 backbone_configs, 
                 proj_configs,
                 d_model, 
                 mvmhead_configs,
                 scale_encoder_configs) -> None:
        super().__init__()
        self.build_backbone(backbone_configs) 

        self.build_feat_proj(proj_configs, d_model=d_model)

        self.build_mvmhead(mvmhead_configs)
        
        self.vid_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d')

    def build_feat_proj(self, proj_configs, d_model):
        configs = vars(proj_configs)
        self.proj_name = configs['name']

        bb_out_channels, bb_scalestrides = self.vid_backbone.get_desc()
        num_bb_lvls = len(bb_out_channels)

        out_scale_strides = configs['out_scale_strides']
        self.out_scale_strides = out_scale_strides
        proj_types = configs['each_proj_types']
        assert len(proj_types) == len(out_scale_strides)
        assert len(out_scale_strides) >= num_bb_lvls
        # 假设:
        # scale_strides 是 out_scale_strides的子集:  out_scale_strides = [scale_strides, ...]
        #
        # 对scale_strides都做projection
        # 不对temporal做downsample
        # local的kernel: 3;  linear的kernel: 1

        self.vid_proj = nn.ModuleList()

        for idx, ((out_temporal_stride, out_spatial_stride), tp) in enumerate(zip(out_scale_strides, proj_types)):
            if idx < num_bb_lvls:
                in_temporal_stride, in_spatial_stride, in_channel = *bb_scalestrides[idx], bb_out_channels[idx]
            else:
                raise ValueError('the number of scales should be equal to the number of backbone output')

            spatial_stride = out_spatial_stride // in_spatial_stride
            temporal_stride = out_temporal_stride // in_temporal_stride
            if self.proj_name == 'conv2d':
                kp_dict = {3:1, 5:2}
                lks = configs['local_kernel_size']
                lkp = kp_dict[lks]
                assert temporal_stride == 1, 'conv2d does not downsample in the temporal dimension'
                if tp == 'local':
                    self.vid_proj.append(nn.Sequential(nn.Conv2d(in_channel, d_model, 
                                                                kernel_size=lks, padding=lkp,
                                                                stride=spatial_stride, 
                                                                bias=configs['bias']), get_norm(configs['norm'], dim=d_model) ))    
                elif tp == 'linear':   
                    self.vid_proj.append(nn.Sequential(nn.Conv2d(in_channel, d_model, 
                                                                kernel_size=1, 
                                                                stride=spatial_stride,
                                                                bias=configs['bias']), get_norm(configs['norm'], dim=d_model)))
                else:
                    raise ValueError()
                
            elif self.proj_name == 'conv3d':
                kp_dict = {3:1, 5:2}
                lks = configs['local_kernel_size']
                lkp = kp_dict[lks]
                if tp == 'local':
                    self.vid_proj.append(nn.Sequential(nn.Conv3d(in_channel, d_model, 
                                                                kernel_size=lks, padding=lkp,
                                                                stride=[temporal_stride, spatial_stride, spatial_stride], 
                                                                bias=configs['bias']), get_norm(configs['norm'], dim=d_model) ))    
                elif tp == 'linear':   
                    self.vid_proj.append(nn.Sequential(nn.Conv3d(in_channel, d_model, 
                                                                kernel_size=1, 
                                                                stride=[temporal_stride, spatial_stride, spatial_stride],
                                                                bias=configs['bias']), get_norm(configs['norm'], dim=d_model)))
                else:
                    raise ValueError()            
            else:
                raise NotImplementedError() # neighborhood
            
        for proj in self.vid_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def build_backbone(self, backbone_configs):
        configs = vars(backbone_configs)
        if backbone_configs.name == 'video_swin_t':
            self.vid_backbone = VideoSwinTransformer(backbone_pretrained=configs['pretrained'],
                                                    backbone_pretrained_path=configs['pretrained_path'],
                                                    train_backbone=configs['train'],
                                                    running_mode=configs['running_mode'],)
            
        elif backbone_configs.name == 'video_resnet':
            self.vid_backbone = VideoResNet(backbone_name=configs['resnet_name'],
                                            train_backbone=configs['train'],
                                            dilation=configs['dilation'])
        else:
            raise ValueError()
        
    def build_mvmhead(self, configs):
        pass
    
    def proj_multiscales(self, multiscales):
        """
        pad_mask: b t h w,  t: valid_t
        """
        # all the padding masks must be in bool type,因为会作为attention mask使用
        batch_size = multiscales[0].shape[0] #  b t c h w
        srcs = [] # b t c h w 
        if self.proj_name == 'conv2d':
            nf = multiscales[0].shape[1]
            for lvl, feat in enumerate(multiscales): 
                feat = rearrange(feat, 'b t c h w -> (b t) c h w')
                src_proj_l = self.vid_proj[lvl](feat)
                src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                srcs.append(src_proj_l)

        elif self.proj_name == 'conv3d':
            for lvl, feat in enumerate(multiscales): 
                feat = rearrange(feat, 'b t c h w -> b c t h w')
                src_proj_l = self.vid_proj[lvl](feat)
                src_proj_l = rearrange(src_proj_l, 'b c t h w -> b t c h w')
                srcs.append(src_proj_l)
        return srcs


    def forward(self, videos, video_pad_mask, valid_indices, mask_video, masked_indices):
        """
        videos: t b c h w
        video_pad_mask: t b h w
        valid_indices: tensor(num_valid, ) int
        mask_video: True/False
        masked_indices: #TODO
        """
        vids = NestedTensor(videos, video_pad_mask)
        # t b 3 h w
        batch_size, device = videos.shape[1], videos.device
        nf = len(valid_indices)
        
        # list[NT(t b ci hi wi)], 4
        backbone_out = self.vid_backbone(vids)
        # 4 -> 8 -> 16 -> 32; t b c h w
        srcs = []
        masks = []
        poses = []
        if valid_indices is not None:
            for layer_out in backbone_out:
                layer_out.tensors = layer_out.tensors.index_select(0, valid_indices) # t b c h w
                layer_out.mask = layer_out.mask.index_select(0, valid_indices)  # t b h w

        for layer_out in backbone_out:
            feat = layer_out.tensors.permute(1,0,2,3,4)
            pos = self.vid_pos_embed(feat, None) # b t c h w
            mask = layer_out.mask.permute(1,0,2,3) # b t h w
            srcs.append(feat)
            poses.append(pos)
            masks.append(mask)

        if mask_video:
            raise NotImplementedError()
        
        return (srcs, masks, poses, copy.deepcopy(self.out_scale_strides)), None  # place holder for the masked ground truth
    
    def mask_tokens(self, masked_indices):
        pass
    
    def forward_mvm(self, video_feats, masked_video_gt):   
        # output, loss_dict
        pass

    
@register_video_encoder
def video_encoder_mvm(configs, d_model):
    return VideoEncoder_MVM(
        backbone_configs=configs.video_backbone,
        proj_configs=configs.proj,
        d_model=d_model,
        mvmhead_configs=configs.mvmhead_configs,
        scale_before_fuse_configs=configs.scale_before_fuse_configs
    )
    

