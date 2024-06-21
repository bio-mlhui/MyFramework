import logging
import copy
import os
from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable, Dict, List, Optional


import torch.nn.functional as F
import torch.nn as nn
import torch
import detectron2.utils.comm as comm
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange, repeat


# metaforer
from models.backbone.metaformer_build_tool import SepConv, Attention,  Attention_REG, \
    DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG, LayerNormWithoutBias, LayerNormGeneral, MetaFormerBlock,\
        DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG, DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG, StarReLU
from models.backbone.metaformer_build_tool import Mlp as Meta_MLP
from models.backbone.metaformer_build_tool import MlpHead 
# dino
from models.backbone.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from kan import KANLayer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, LoraModel
# multiscale
from models.encoder.ops.modules import MSDeformAttn_with_GlobalRegisters
from models.layers.position_encoding import build_position_encoding
from models.backbone.utils import ImageMultiscale_Shape

from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY

# from timm.models.registry import register_model

# # metaforer
# from models.layers.build_tool import SepConv, Attention,  Attention_REG, \
#     DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG, LayerNormWithoutBias, LayerNormGeneral, MetaFormerBlock,\
#         DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG, DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG, StarReLU
# from models.layers.build_tool import Mlp as Meta_MLP
# from models.layers.build_tool import MlpHead 
# # dino
# from models.layers.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

# from models.layers.encoder.ops.modules import MSDeformAttn_with_GlobalRegisters
# from models.layers.position_encoding import build_position_encoding
# import importlib
# @register_model
# def maskdino_dinoBReg_pt100_in1k(pretrained=False, **kwargs):
#     config_file = '.'.join(['mask_dino_configs', 'dinoBReg_pt100'])
#     configs = importlib.import_module(config_file).trainer_configs
#     configs['model']['video_backbone']['num_classes'] = 1000
#     model = Dinov2_LORA_REG(configs['model']['video_backbone'])
#     return model

# @register_model
# def maskdino_dinoSReg_pt100_in1k(pretrained=False, **kwargs):
#     config_file = '.'.join(['mask_dino_configs', 'dinoSReg_pt100'])
#     configs = importlib.import_module(config_file).trainer_configs
#     configs['model']['video_backbone']['num_classes'] = 1000
#     model = Dinov2_LORA_REG(configs['model']['video_backbone'])
#     return model

# @register_model
# def maskdino_dinoBReg_pt300_in1k(pretrained=False, **kwargs):
#     config_file = '.'.join(['mask_dino_configs', 'dinoBReg_pt300'])
#     configs = importlib.import_module(config_file).trainer_configs
#     model = Dinov2_LORA_REG(configs['model']['video_backbone'])
#     return model



DINO_NAME_TO_CONFIGS = {
    'dinov2_vits14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),
    },
    'dinov2_vitb14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),              
    },
    'dinov2_vitl14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'mlp',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),              
    },
    'dinov2_vitg14_reg': {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': 'swiglufused',
        'block_chunks': 0,
        'num_register_tokens': 4, 
        'interpolate_antialias': True,
        'interpolate_offset': 0.0,
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'mlp_ratio': 4,
        'block_fn': partial(Block, attn_class=MemEffAttention),             
    },
}

@BACKBONE_REGISTRY.register()
class Dinov2_LORA_REG(nn.Module):
    def valueable_rubbish(self,):
        pass
        # 假设: 在 8/阶段
        # def _load_from_state_dict(
        #     self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        # ):
        #     """
        #     partially load the token embeddings
        #     """
        #     with torch.no_grad():
        #         # state_dict
        #         reg_keys = [key for key in state_dict.keys() if 'register_tokens' in key]
        #         assert len(reg_keys) == 1
        #         reg_keys = reg_keys[0]
        #         reg_token_weights = state_dict[reg_keys] # 1 4 dim
        #         original_number = reg_token_weights.shape[1]
        #         # repeat
        #         if self.num_register_tokens != original_number:
        #             repeats = self.num_register_tokens // original_number
        #             reg_token_weights = reg_token_weights.repeat(1, repeats, 1) # 1 

        #         state_dict[reg_keys] = reg_token_weights  
            
        #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
        # 
        # self.task_to_scales = {haosen: [] for haosen in self.tasks}
        # for task_name in self.tasks:
        #     for scale_name, scale_tasks in self.scale_to_tasks.items():
        #         if task_name in scale_tasks:
        #             self.task_to_scales[task_name].append(scale_name)           

        # transformed_srcs = {}
        # for idx, scale_name in enumerate(self.fpn_not_deform):
        #     transformed_srcs[scale_name] = {'hw': fpns[idx]} # b h w c

        # task_names = [haosen[0] for haosen in reg_split]
        # split_output = output.split([haosen[1] for haosen in reg_split])
        # split_output = {task_name : haosen for haosen, task_name in zip(split_output, task_names)}

        # # hw split
        # scale_hw_length = [haosen[0] * haosen[1] for haosen in scale_to_hw_length]
        # hw_by_scale = split_output['hw'].split(scale_hw_length, dim=1)

        # # task split
        # split_output = {task_name: split_output[task_name].split([ [self.task_to_num_regs[task_name]] * len(self.task_to_scales[task_name])], dim=1) for task_name in split_output.keys()}

        # for idx, scale_name in enumerate(self.deform_scales):
        #     trans_scale = {'hw': hw_by_scale[idx]}
        #     scale_tasks = self.scale_to_tasks[scale_name]
        #     for task_name in scale_tasks:
        #         trans_scale[task_name] = split_output['task_name']
            
            
        # for spli_name, split_length 



        # class FPNLayer(nn.Module):
        #     def __init__(self, 
        #                 d_model, 
        #                 num_fpn_levels,
        #                 fpn_configs=None,
        #                 drop_path=None,) -> None:
        #         super().__init__()
        #         self.d_model = d_model
        #         self.num_fpn_levels = num_fpn_levels
                
        #         self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
                
        #         lateral_linears = [] 
        #         output_convs = []
        #         for idx in range(self.num_fpn_levels):
        #             lateral_linear = nn.Linear(d_model, d_model,)
        #             output_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1,)
        #             weight_init.c2_xavier_fill(lateral_linear)
        #             weight_init.c2_xavier_fill(output_conv)
        #             lateral_linears.append(lateral_linear)
        #             output_convs.append(output_conv)
        #         self.lateral_linears = nn.ModuleList(lateral_linears)
        #         self.output_convs = nn.ModuleList(output_convs)

        #     def forward(self, 
        #                 fpn_srcs=None,
        #                 output=None):
        #         pass

        # ssl_finetune
        # peft_config = LoraConfig(
        #     r=8,
        #     target_modules=None, # list[str]
        #     lora_alpha = 8,
        #     lora_dropout = 0,
        #     fan_in_fan_out = True, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        #     bias="none", # If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation
        #     use_rslora= False, # scaling factor设置成 alpha/r 还是 alpha/sqrt(r)
        #     modules_to_save = None, # List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint
        #     init_lora_weights='pissa', # How to initialize the weights of the adapter layers
        #     layers_to_transform= None, # list[int] If a list of ints is passed, it will apply the adapter to the layer indices that are specified in this list. 
        #     layers_pattern= None, #  The layer pattern name, used only if layers_to_transform is different from None.
        #     rank_pattern = None, # the mapping from layer names or regexp expression to ranks which are different from the default rank specified by r.
        #     alpha_pattern = None, # The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha.
        #     megatron_config = None, #  The TransformerConfig arguments for Megatron.
        #     megatron_core="megatron.core", #  The core module from Megatron to use
        #     loftq_config = dict, #  The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights and initialize Lora layers. Also pass init_lora_weights='loftq'. Note that you should not pass a quantized model in this case, as LoftQ will quantize the model itself.
        #     use_dora= False, # Weight-Decomposed Low-Rank Adaptation
        #     layer_replication= None #  Build a new stack of layers by stacking the original model layers according to the ranges specified. This allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will all have separate LoRA adapters attached to them.
        # )
        # self.ssl = LoraModel(self.ssl, config=peft_config, adapter_name='default')


        # # reg {'cls': b scale_s c, 'sem_seg': b scale_s c}
        # task_to_reg_feats = {}
        # for task_name in self.tasks:
        #     reg_feats = []
        #     for feat in deform_scales:
        #         if task_name in feat:
        #             reg_feats.append(feat[task_name])
        #     task_to_reg_feats[task_name] = torch.cat(reg_feats, dim=1)
        # reg_feats = torch.cat([task_to_reg_feats[task_name] for task_name in self.tasks], dim=1) # b s c
        # reg_split = [(task_name, task_to_reg_feats[task_name].shape[1]) for task_name in self.tasks]
        # reg_poses = torch.zeros_like(reg_feats)
        # reg_reference_points = reg_feats.new_zeros([reg_feats.shape[0], reg_feats.shape[1], hw_reference_points.shape[-2], hw_reference_points.shape[-1]])
        # reg_masks = torch.zeros_like(reg_feats[..., 0]).bool()
        
        # # concate
        # reg_split.append(('hw', hw_feats.shape[1]))
        
        # output = torch.cat([reg_feats, hw_feats], dim=1)  # b reg_sigma+hw_sigma c
        # reference_points = torch.cat([reg_reference_points, hw_reference_points], dim=1)
        # output_poses = torch.cat([reg_poses, hw_pos_embed], dim=1)
        # padding_masks = torch.cat([reg_masks, hw_masks], dim=1)
        
        
        # predictions_by_layer = [] 
        # for idx, deform_layer in enumerate(self.deform_layers):
        #     output, fpns = deform_layer(src=output, pos=output_poses, fpns=fpns, reg_split=reg_split, scale_to_hw_length=scale_to_hw_length,
        #                                 reference_points=reference_points, spatial_shapes=hw_spatial_shapes, level_start_index=hw_level_start_index, padding_mask=padding_masks)
        #     if self.training or ((not self.training) and (idx == len(self.deform_layers) - 1)):
        #         output_by_task = output.split([haosen[1] for haosen in reg_split], dim=1) # by
        #         task_names = [haosen[0] for haosen in reg_split]
        #         # {'cls': b cls, 'sem_seg': {'masks': b nq h w, 'cls': b nq cls}, 'ins_det': {'masks': b nq h w, 'cls': b nq cls, 'boxes': b nq 4}
        #         layer_predictions = {task_name: self.task_heads[task_name](task_output) for task_name, task_output in zip(task_names, output_by_task)}                
        #         predictions_by_layer.append(layer_predictions)
    
    def build_peft_dino(self, dino_configs, meta_configs):
        dino_name, freeze, lora_configs = meta_configs['dino_name'], meta_configs['dino_freeze'], meta_configs['dino_lora']
        self.ssl = DinoVisionTransformer(**dino_configs)
        name_to_pt_path = {'dinov2_vitb14_reg': 'dinov2/dinov2_vitb14_reg4_pretrain.pth',
                           'dinov2_vits14_reg': 'dinov2/dinov2_vits14_reg4_pretrain.pth'}
        state_dict = torch.load(os.path.join(os.getenv('PT_PATH'), name_to_pt_path[dino_name]), map_location='cpu')
        self.ssl.load_state_dict(state_dict, strict=True) 
        if freeze:
            for p in self.ssl.parameters():
                p.requires_grad_(False)
        # lora_finetune
        
    def has_metaformer(self, 
                       configs,
                       mlps=Meta_MLP,
                       norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                       layer_scale_init_values=None,               
                 ):
        meta_configs= configs['meta_configs']
        dino_configs = DINO_NAME_TO_CONFIGS[meta_configs['dino_name']]
        
        self.first_attn_stage_idx = meta_configs['first_attn_stage_idx']
        depths, dims, drop_path_rate = meta_configs['depths'], meta_configs['dims'], meta_configs['drop_path_rate']
        res_scale_init_values = meta_configs.pop('res_scale_init_values')
        self.dino_stage_idx, self.num_stages = dims.index(None), len(depths)
        dims[self.dino_stage_idx] = dino_configs['embed_dim']
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.backbone_dims = dims
        
        token_mixers, downsample_layer_name  = meta_configs['token_mixers'], meta_configs['downsample_layer_name'] 
        
        if downsample_layer_name == 'four_stage_last_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FOUR_STAGES_LAST_REG
        elif downsample_layer_name == 'four_stage_lasttwo_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FOUR_STAGES_LASTTWO_REG
        elif downsample_layer_name == 'five_stage_lasttwo_reg':
            downsample_layers = DOWNSAMPLE_LAYERS_FIVE_STAGES_LASTTWO_REG
        else:
            raise ValueError()
        assert len(token_mixers) == self.num_stages

        down_dims = [3] + dims
        self.downsample_layers = nn.ModuleList([downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(self.num_stages)])
        
        # very hacky
        name_to_token_mixers = {'conv': SepConv, 'attn_32': Attention, 'first_attn_32': partial(Attention_REG, reg_cls=self),\
                                'iden': nn.Identity, 'attn_64': partial(Attention, head_dim=64), 
                                'first_attn_64': partial(Attention_REG, reg_cls=self, head_dim=64),}        
        depth_token_mixers = []
        for stage_idx, (haosen_dep, haosen_mixer) in enumerate(zip(depths, token_mixers)):
            if isinstance(haosen_mixer, str):
                depth_token_mixers.extend([name_to_token_mixers[haosen_mixer]] * haosen_dep)
            elif isinstance(haosen_mixer, list):
                for hhsen_mixer, hhsen_depth in haosen_mixer:
                    if hhsen_mixer is None:
                        assert hhsen_depth == dino_configs['depth']
                        depth_token_mixers.extend([None] * hhsen_depth)
                    else:
                        assert isinstance(hhsen_mixer, str) and isinstance(hhsen_depth, int)
                        depth_token_mixers.extend([name_to_token_mixers[hhsen_mixer]] * hhsen_depth)
            else:
                raise ValueError()
        assert len(depth_token_mixers) == sum(depths)
        
        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * self.num_stages
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * self.num_stages
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * self.num_stages
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * self.num_stages
        
        cur = 0
        self.before_stages = nn.ModuleList()
        self.after_stages = nn.ModuleList()
        for i in range(self.num_stages):
            if i != self.dino_stage_idx:
                stage = nn.Sequential(
                    *[MetaFormerBlock(dim=dims[i],
                    token_mixer=depth_token_mixers[cur+j],
                    mlp=mlps[i],
                    norm_layer=norm_layers[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_values[i],
                    res_scale_init_value=res_scale_init_values[i],
                    ) for j in range(depths[i])]
                )
                if i < self.dino_stage_idx:
                    self.before_stages.append(stage)
                elif i > self.dino_stage_idx:
                    self.after_stages.append(stage)   
            else:
                dino_configs['drop_path_rate_list'] = dp_rates[cur: (cur+dino_configs['depth'])]
                self.build_peft_dino(dino_configs=dino_configs, meta_configs=meta_configs)
                dino_same_stage_depth = depths[i] - len(self.ssl.blocks)
                self.dino_same_stage = nn.Sequential(                    
                    *[MetaFormerBlock(dim=dims[i],
                        token_mixer=depth_token_mixers[cur+len(self.ssl.blocks)+j],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i],
                        drop_path=dp_rates[cur+len(self.ssl.blocks) + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    ) for j in range(dino_same_stage_depth)]
                )
            cur += depths[i]
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.before_stages.apply(_init_weights)
        self.after_stages.apply(_init_weights)
        self.downsample_layers.apply(_init_weights)
        self.dino_same_stage.apply(_init_weights)
        
        stage_to_level_embed = [None] * self.first_attn_stage_idx
        for haosen in range(self.first_attn_stage_idx, len(depths)):
            level_embed = nn.Embedding(1, self.backbone_dims[haosen])
            torch.nn.init.normal_(level_embed.weight)
            stage_to_level_embed.append(level_embed)
            
        self.stage_to_level_embed = nn.ModuleList(stage_to_level_embed)
    
    @property
    def device(self):
        return self.stage_to_level_embed[-1].weight.device
    
    def has_multiscale(self, configs,):  
        ms_d_model, proj_add_norm, proj_add_star_relu, proj_bias, proj_dropout = configs['d_model'], configs['proj_add_norm'], \
            configs['proj_add_star_relu'], configs['proj_bias'], configs['proj_dropout']
        input_proj_list = {}
        for huihui in self.multiscale_shapes.keys():
            input_proj_list[huihui]= nn.Sequential(
                partial(LayerNormGeneral, bias=False, eps=1e-6)(self.multiscale_shapes[huihui].dim) if proj_add_norm else nn.Identity(),
                nn.Linear(self.multiscale_shapes[huihui].dim, ms_d_model, bias=proj_bias),
                StarReLU() if proj_add_star_relu else nn.Identity(),
                nn.Dropout(proj_dropout) if proj_dropout != 0 else nn.Identity(),
            )
        self.input_projs = nn.ModuleDict(input_proj_list)

        num_encoder_layers = configs['num_encoder_layers']
        deform_scales, fpn_scales = configs['deform_scales'], configs['fpn_scales']
        drop_path_rate=configs['drop_path_rate']
        deform_configs, mlp_configs, fpn_configs = configs['deform_configs'], configs['mlp_configs'], configs['fpn_configs']
        self.ms_d_model = ms_d_model
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        
        deform_configs['reg_sequence_length'] = (self.dino_stage_idx - self.first_attn_stage_idx) * self.task_num_regs[0] + \
                                                (self.num_stages - self.dino_stage_idx) * (self.task_num_regs[0] + self.task_num_regs[1] + self.task_num_regs[2])
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        self.deform_layers = nn.ModuleList([MSDeformAttnTransformerEncoderLayer(d_model=ms_d_model,
                                                                                multiscale_shapes=self.multiscale_shapes, deform_scales=deform_scales, fpn_scales=fpn_scales, 
                                                                                fpn_configs=fpn_configs,
                                                                                
                                                                                num_feature_levels=len(self.deform_scales),
                                                                                deform_configs=deform_configs,
                                                                                mlp_configs=mlp_configs,
                                                                                drop_path=dp_rates[j]) for j in range(num_encoder_layers)])
        self.pos_2d = build_position_encoding('2d')
    
    def __init__(self, configs,):
        super().__init__()
        self.has_metaformer(configs=configs)    
        multiscale_shapes = {}
        for idx, _ in enumerate(self.before_stages):
            multiscale_shapes[f'res{idx+2}'] = ImageMultiscale_Shape(2**(idx+2), self.backbone_dims[idx])
        multiscale_shapes[f'res{len(self.before_stages)+2}'] = ImageMultiscale_Shape(self.ssl.patch_size, self.ssl.embed_dim) 
        for idx, _ in enumerate(self.after_stages):
            multiscale_shapes[f'res{len(self.before_stages)+1+idx+2}'] = ImageMultiscale_Shape(self.ssl.patch_size * (2**(idx+1)), self.backbone_dims[len(self.before_stages)+idx+1])        
        self.multiscale_shapes = multiscale_shapes
        scale_names = list(self.multiscale_shapes.keys())
        self.multiscale_shapes = multiscale_shapes
        self.max_stride = self.ssl.patch_size * (2**(len(self.after_stages)))
        # interal tasks, 这些是forward的时候必须用的，不管是finetune还是pretrain
        num_pt_registers = configs['num_pt_registers']
        self.is_finetuning = configs['is_finetuning']
        # global_registers
        pass
        # local_registers, in dino_dim
        self.pt_task_regs = nn.Parameter(torch.zeros(1, num_pt_registers, self.ssl.embed_dim))
        pass
        assert (self.dino_stage_idx - self.first_attn_stage_idx) == 1

        self.register_projs = [None] * self.first_attn_stage_idx
        for stage_idx in range(self.first_attn_stage_idx, self.num_stages):
            reg_proj = nn.Linear(self.ssl.embed_dim, self.backbone_dims[stage_idx], bias=False)
            nn.init.orthogonal_(reg_proj.weight)
            self.register_projs.append(reg_proj)
        self.register_projs = nn.ModuleList(self.register_projs)
        self.num_pt_registers = num_pt_registers
        self.task_num_regs = [num_pt_registers, self.ssl.cls_token.shape[1], self.ssl.register_tokens.shape[1]]
        self.num_registers = sum(self.task_num_regs)
        self.forward_ms = False
        if not self.is_finetuning: 
            # 在预训练
            pretrain_configs = configs['pretrain_configs']
            if pretrain_configs['pretrain_ms']:
                self.has_multiscale(configs.pop('ms_configs'))
                class_in_dim = self.ms_d_model
                logging.debug(f'MS的总参数数量:{sum(p.numel() for p in self.deform_layers.parameters())}')
                self.forward_ms = True
                # b reg_sigma_hw_sigma
                self.chosen_cls_index = self.task_num_regs[0] * (self.dino_stage_idx - self.first_attn_stage_idx) +\
                    (self.task_num_regs[0] + self.task_num_regs[1] + self.task_num_regs[2]) * (self.num_stages - self.dino_stage_idx - 1) +\
                        self.task_num_regs[0]
            else:
                class_in_dim = self.backbone_dims[-1]
            self.num_classes = configs['pretrain_configs']['num_classes']
            self.class_head = MlpHead(class_in_dim, self.num_classes, head_dropout=pretrain_configs['cls']['head_dropout'])

        else:
            self.has_multiscale(configs.pop('ms_configs'))
            logging.debug(f'MS的总参数数量:{sum(p.numel() for p in self.deform_layers.parameters())}') 
            finetune_configs = configs.pop('finetune_configs') # task_name: {'num_registers': }
            self.finetune_num_regs = [(task_name, task_cf['num_registers']) for task_name, task_cf in finetune_configs.items()]
            assert sum([haosen[1] for haosen in self.finetune_num_regs]) == num_pt_registers
            # TODO: build head for each finetune task
            self.forward_ms = True
        
        if comm.is_main_process():
            logging.debug(f'before的总参数数量:{sum(p.numel() for p in self.before_stages.parameters())}')
            logging.debug(f'DINO的总参数数量:{sum(p.numel() for p in self.ssl.parameters())}')
            logging.debug(f'after的总参数数量:{sum(p.numel() for p in self.after_stages.parameters())}')
            logging.debug(f'downsample的总参数数量:{sum(p.numel() for p in self.downsample_layers.parameters())}')
            logging.debug(f'dino_same_stage的总参数数量:{sum(p.numel() for p in self.dino_same_stage.parameters())}')
            logging.debug(f'task_registers的总参数数量:{self.pt_task_regs.numel()}')
    
        assert self.first_attn_stage_idx < len(self.before_stages)

    def first_attn_get_registers(self):
        clsssl_regs = self.get_clsssl_registers()
        pt_regs = self.pt_task_regs

        regs = torch.cat([pt_regs, clsssl_regs], dim=1)
        return self.register_projs[self.first_attn_stage_idx](regs)

    def first_attn_level_embed(self):
        return self.stage_to_level_embed[self.first_attn_stage_idx].weight

    def get_clsssl_registers(self, ):
        cls_toks = self.ssl.cls_token
        ssl_toks = self.ssl.register_tokens
        cls_pos = self.ssl.pos_embed[:, 0].unsqueeze(0)
        cls_toks += cls_pos
        return torch.cat([cls_toks, ssl_toks], dim=1)

    def get_registers(self):
        clsssl_regs = self.get_clsssl_registers()
        pt_regs = self.pt_task_regs

        regs = torch.cat([pt_regs, clsssl_regs], dim=1)
        return regs

    def forward_before(self, x):
        _, _, H, W = x.shape
        ret = []
        for i in range(len(self.before_stages)):
            x = self.downsample_layers[i](x)
            x = self.before_stages[i](x)
            ret.append((None, x.contiguous())) # b h w c
        return ret
    
    def forward_after(self, registers, x):
        ret = []
        for i in range(len(self.after_stages)):
            registers, x = self.downsample_layers[i+self.dino_stage_idx+1](registers, x)
            registers += self.register_projs[i+self.dino_stage_idx+1](self.get_registers())

            _, H, W, _ = x.shape
            input = torch.cat([registers, x.flatten(1,2)], dim=1) # b reg_hw c
            input = input + self.stage_to_level_embed[i+self.dino_stage_idx+1].weight
            
            input = self.after_stages[i](input)
            registers, x = input.split([registers.shape[1], H*W], dim=1)
            x = x.view(x.shape[0], H, W, -1).contiguous()
            ret.append((registers, x))
        return ret

    # NestTensor?
    def forward_dino_ssl(self, x, last_regs, last_hw, masks=None,): # b c h w -> ms: {'res2', 'res3', 'res4, 'res5}, reg: {'reg2', 'reg3', 'reg4', 'reg5'}
        batch_size, _, H, W = x.shape # b c h w
        patch_size = [H // self.ssl.patch_size, W // self.ssl.patch_size]
        x = self.ssl.patch_embed(x) # b c h w -> b hw c
        x = x.view(x.shape[0], patch_size[0], patch_size[1], x.shape[-1]).contiguous() # b h w c

        x = (x + last_hw).flatten(1, 2) # b hw c
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.ssl.mask_token.to(x.dtype).unsqueeze(0), x)    
        x_poses = self.ssl.interpolate_pos_encoding_hw(x, W, H) # b hw c
        x = x + x_poses
                
        x = torch.cat([last_regs, x], dim=1) # b reg_hw c

        x = x + self.stage_to_level_embed[self.dino_stage_idx].weight
        for blk in self.ssl.blocks:
            x = blk(x)
        x = self.ssl.norm(x)
        x = self.dino_same_stage(x)

        reg_feats, hw_feats = x.split([sum(self.task_num_regs), x.shape[1] - sum(self.task_num_regs)], dim=1)
        hw_feats = rearrange(hw_feats, 'b (h w) c -> b h w c', h=patch_size[0], w=patch_size[1])
            
        return (reg_feats, hw_feats)

    def forward_multiscale(self, multiscales=None,): # list[]
        srcs = {}
        for idx, feat in enumerate(multiscales):
            scale_name = f'res{idx+2}'
            input_proj = self.input_projs[f'res{idx+2}']
            reg_feat, hw_feat = feat
            reg_feat = input_proj(reg_feat) if reg_feat is not None else reg_feat
            hw_feat = input_proj(hw_feat)
            srcs[scale_name] = (reg_feat, hw_feat)
        
        # list[b h w c]            
        fpns = [srcs[haosen][-1] for haosen in self.fpn_not_deform]
        
        deform_scales = [srcs[haosen] for haosen in self.deform_scales] # list[(b r c, b h w c)]
    
        scale_to_hw_length = [(haosen[1].shape[1], haosen[1].shape[2]) for haosen in deform_scales]
        hw_feats = [haosen[1].flatten(1, 2) for haosen in deform_scales] # list[b hw c]
        hw_feats = torch.cat(hw_feats, dim=1)
        hw_masks = [torch.zeros_like(haosen[1][..., 0]).bool() for haosen in deform_scales] # list[b h w]
        hw_pos_embed = torch.cat([self.pos_2d(m, hidden_dim=self.ms_d_model).permute(0, 2, 3, 1).flatten(1, 2) for m in hw_masks], dim=1) # b hw_sigma c
        hw_spatial_shapes = torch.as_tensor([(m.shape[1], m.shape[2]) for m in hw_masks], dtype=torch.long, device=self.device,) # L 2
        hw_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in hw_masks], 1) # b L 2
        hw_masks = torch.cat([m.flatten(1, 2) for m in hw_masks], dim=1) # b hw_sigma 
        hw_level_start_index = torch.cat((hw_spatial_shapes.new_zeros((1, )), hw_spatial_shapes.prod(1).cumsum(0)[:-1]))
        hw_reference_points = self.get_reference_points(hw_spatial_shapes, hw_valid_ratios, device=self.device,)
        
        # reg
        reg_length = [haosen[0].shape[1] for haosen in deform_scales]
        reg_feats = torch.cat([haosen[0] for haosen in deform_scales], dim=1) # b s c
        reg_poses = torch.zeros_like(reg_feats)
        reg_reference_points = reg_feats.new_zeros([reg_feats.shape[0], reg_feats.shape[1], hw_reference_points.shape[-2], hw_reference_points.shape[-1]])
        reg_masks = torch.zeros_like(reg_feats[..., 0]).bool()
                
        output = torch.cat([reg_feats, hw_feats], dim=1)  
        reference_points = torch.cat([reg_reference_points, hw_reference_points], dim=1)
        output_poses = torch.cat([reg_poses, hw_pos_embed], dim=1)
        padding_masks = torch.cat([reg_masks, hw_masks], dim=1)
        
        
        predictions_by_layer = [] 
        for idx, deform_layer in enumerate(self.deform_layers):
            output, fpns = deform_layer(src=output, pos=output_poses, fpns=fpns, scale_to_hw_length=scale_to_hw_length,
                                        reference_points=reference_points, spatial_shapes=hw_spatial_shapes, level_start_index=hw_level_start_index, padding_mask=padding_masks)
            if self.training or ((not self.training) and (idx == len(self.deform_layers) - 1)):
                if not self.is_finetuning:
                    layer_predictions = {'cls': self.class_head(output[:, self.chosen_cls_index])}
                else:
                    raise NotImplementedError()
                predictions_by_layer.append(layer_predictions)
        return predictions_by_layer
    
   
    def forward(self, x, masks=None):
        
        masks = None # b t h w
        x = x # b c t h w
        # masks = masks.squeeze(1).flatten(1) # b hw
        x = x.squeeze(2) # b c h w
        
        #  (None/b reg c; b h w c)
        ret = self.forward_before(x)
        
        last_regs, last_hw = ret[-1]
        last_regs, last_hw = self.downsample_layers[len(self.before_stages)](last_regs, last_hw) 
        last_regs = last_regs + self.register_projs[len(self.before_stages)](self.get_registers()) # b s c + b s c

        ret.append(self.forward_dino_ssl(x, last_regs=last_regs, last_hw=last_hw, masks=masks))
        
        last_regs, last_hw = ret[-1]
        ret.extend(self.forward_after(last_regs, last_hw))

        if self.forward_ms:
            predictions_by_layer = self.forward_multiscale(ret)
            if not self.is_finetuning:
                return predictions_by_layer[-1]['cls'] # b class
            else:
                raise NotImplementedError()
        else:
            assert not self.is_finetuning
            return self.class_head(ret[-1][0][:, self.num_pt_registers])


    # @staticmethod
    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        # lsit[h w], L; b L 2
        assert (valid_ratios == 1).all(), '都是1'
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)) # h w
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 1 hw / b 1 (h*h_ratio = h_valid_max)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_) # 绝对坐标 / 最大长宽 = 相对坐标
            ref = torch.stack((ref_x, ref_y), -1) # b hw 2
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) # b
        valid_W = torch.sum(~mask[:, 0, :], 1) # b
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # b 2, w的有效ratio, x的有效ratio
        return valid_ratio


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x
    
class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        drop_path_rate_list=None,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if drop_path_rate_list is not None:
            assert depth == len(drop_path_rate_list)
            dpr = drop_path_rate_list
        
        if ffn_layer == "mlp":
            logging.debug("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logging.debug("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logging.debug("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        elif ffn_layer == 'kan':
            logging.debug('using Kan layer as FFN')
            ffn_layer = KANLayer
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        # b 1_hw c
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def interpolate_pos_encoding_hw(self, x, w, h):
        # b h/14 w/14 c
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim) # b hw c
        return patch_pos_embed.to(previous_dtype)


    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x, masks=None):
        # b s c -> b s c
        for blk in self.blocks:
            x = blk(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)







class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 multiscale_shapes=None, fpn_scales=None, deform_scales=None, fpn_configs=None,
                 num_feature_levels=None,
                 mlp_configs=None,
                 deform_configs=None,
                 drop_path=None,):
        super().__init__()
        self.multiscale_shapes = multiscale_shapes
        def sort_scale_name(scale_names):
            return sorted(scale_names, key=lambda x: self.multiscale_shapes[x].spatial_stride)
        self.deform_scales, self.fpn_scales = sort_scale_name(deform_scales), sort_scale_name(fpn_scales)
        self.fpn_not_deform, self.fpn_and_deform = sort_scale_name(list(set(fpn_scales) - set(deform_scales))), sort_scale_name(list(set(fpn_scales) & set(deform_scales)))
        self.fpn_and_deform_idxs = [self.deform_scales.index(haosen) for haosen in self.fpn_and_deform]
        
        # token mixer
        self.norm1 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        self.self_attn = MSDeformAttn_with_GlobalRegisters(d_model=d_model,
                                                           num_feature_levels=num_feature_levels,
                                                           deform_configs=deform_configs)

        self.alias_convs = nn.ModuleList([nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1) for _ in range(len(self.fpn_scales)-1)])
        
        self.dropout1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # feature mixer
        self.norm2 = partial(LayerNormWithoutBias, eps=1e-6)(d_model)
        mlp_type = mlp_configs['mlp_type']
        if mlp_type == "mlp":
            self.mlp = Mlp(in_features=d_model, hidden_features=int(d_model *  mlp_configs['mlp_ratio']), act_layer=StarReLU, drop=0, bias=False)
        elif mlp_type == "swiglu" or mlp_type == 'kan':
            raise ValueError()
            # SwiGLUFFNFused
        else:
            raise NotImplementedError  
        self.dropout2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward_fpn(self, fpns, fpn_and_deform_feats):
        # bigger -> small
        assert len(fpns) + len(fpn_and_deform_feats) == len(self.fpn_scales)
        ret = fpns + fpn_and_deform_feats
        ret = ret[::-1]
        output = [ret[0].permute(0, 3, 1, 2).contiguous()] # smallest
        for idx, f in enumerate(ret[1:]): # 
            f = f.permute(0, 3, 1, 2)
            f = f + F.interpolate(output[-1], size=(f.shape[2], f.shape[3]), mode="bilinear", align_corners=False)
            f = self.alias_convs[idx](f)
            output.append(f)
        output = output[-len(fpns):][::-1]
        output = [haosen.permute(0, 2, 3, 1) for haosen in output]
        return output
        

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, fpns, scale_to_hw_length, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # b reg_hw_sigma c： fpns: b h w c
        scale_hw_length = [haosen[0] * haosen[1] for haosen in scale_to_hw_length]
        
        src2 = self.norm1(src)
        fpns2 = [self.norm1(haosen) for haosen in fpns]
        
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reference_points, src2, spatial_shapes, level_start_index, padding_mask, sum(scale_hw_length), )
        
        hw_feats = src2[:, -sum(scale_hw_length):].contiguous().split(scale_hw_length, dim=1)
        fpn_and_deform_feats = [rearrange(hw_feats[haosen], 'b (h w) c -> b h w c',h=scale_to_hw_length[haosen][0], w=scale_to_hw_length[haosen][1])\
            for haosen in self.fpn_and_deform_idxs]
        fpns2 = self.forward_fpn(fpns2, fpn_and_deform_feats)
        
        src = src + self.dropout1(src2)
        fpns = [fpns[idx]+self.dropout1(fpns2[idx]) for idx in range(len(fpns))]
        
        src = src + self.dropout2(self.mlp(self.norm2(src)))
        
        fpns = [fpns[idx]+self.dropout2(self.mlp(self.norm2(fpns[idx]))) for idx in range(len(fpns))]

        return src, fpns



class Instance_Detection_Head(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,):
        pass


class SemanticSegmentationHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,):
        pass
