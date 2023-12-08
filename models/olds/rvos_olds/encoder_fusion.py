import torch
from einops import rearrange, repeat, reduce

from torch.nn import functional as F

from .layers_unimodal_attention import *

from .layers_multimodal_attention import multimodal_attention_encoder_entrypoints

import fvcore.nn.weight_init as weight_init
from .transformer import _get_clones

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .encoder_multiscale import multiscale_encoder_entrypoints
from .decoder_refer import refer_decoder_entrypoints
import torch.nn as nn
from util.misc import find_scale_from_multiscales
from .transformer import get_norm

_fusion_entrypoints = {}
def register_fusion(fn):
    fusion_name = fn.__name__
    _fusion_entrypoints[fusion_name] = fn
    return fn

def fusion_entrypoints(fusion_name):
    try:
        return _fusion_entrypoints[fusion_name]
    except KeyError as e:
        print(f'Fusion {fusion_name} not found')


def pad_1d_feats(feat_list):
    # list[ni c] -> b nmax c
    feat_len = [len(feat) for feat in feat_list]
    n_max = max(feat_len) 
    batch_size = len(feat_list)
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
    for i in range(batch_size):
        feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
        pad_mask[i, :feat_len[i]] = False
    feat_list = torch.stack(feat_list, dim=0) # b nmax c
    return feat_list, pad_mask

# 所有fusion encoder的基本类别
# 对于使用单模态encoder的情况, 
# 1. object query poses 在decoder中
# 2. object query poses 在融合encoder中
# 为了统一, query pos放到fusion encoder中, 然后再传入decoder中

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt

class FusionEncoder(nn.Module):
    def __init__(self, 
        fused_scale, 

        d_model: int,
        mmattn_encoder_configs,
        scale_encoder_configs,
        task_conditioning_form=None, # 做mlm的时候, 如何指定instance
    ):
        super().__init__()        
        # 定义 mlm conditioning form
        if task_conditioning_form == 'pos_emb':
            self.front_back_emb = nn.Embedding(1, embedding_dim=d_model, scale_grad_by_freq=True)
        elif (task_conditioning_form == 'none') or (task_conditioning_form == 'attn_mask'):
            pass
        else:
            assert task_conditioning_form == None # 不做mlm任务
        self.task_conditioning_form = task_conditioning_form
        
        # 定义video哪个scale和文本进行了融合
        self.fused_scale = fused_scale # list[[1,32],[1,16]...]
        create_selfattn_encoder = multimodal_attention_encoder_entrypoints(mmattn_encoder_configs.name)
        self.mmattn_encoder = create_selfattn_encoder(mmattn_encoder_configs, d_model=d_model)
        # 定义multi-scale 融合
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)

    
    def forward(self, video_args, text_args, referent_mask_condition=None, mlm=False, mvm=False, refer=False):
        """
        根据mlm/mvm/refer三个任务的不同, 得到相应的融合features
        video_args: dict
        text_args: dict
        """
        pass

class NoFusion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,video_args, text_args,refer=True):
        return None

@register_fusion
def nofusion(configs, d_model):
    return NoFusion()

"""
text那边是sequence
"""
class VideoTextFeats(FusionEncoder):
    def __init__(self, fused_scale, d_model: int, mmattn_encoder_configs, scale_encoder_configs, task_conditioning_form=None):
        super().__init__(fused_scale, d_model, mmattn_encoder_configs, scale_encoder_configs, task_conditioning_form)
        
    def forward(self, video_args, text_args, referent_mask_condition=None, mlm=False, mvm=False, refer=False):
        """
        mask_conditions: 
        如果是mlm, 并且是随机mask token, 句子的fill mask需要知道在fill哪个对象
        如果是mvm,  由于句子只是描述其中一个物体, 无论是哪种masking, 都会让模型根据句子补全这个对象, 如果没有条件加入的话, 
        就成了一个ill-conditioned问题, 也能训练起来, 但是可以加上一些位置信息, 

        mask_conditions: b t h w, 即这个句子的对象对应的referent mask
        """
        # 4 8 16 32 64
        multiscales, multiscales_pad_masks, multiscales_poses, descs \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        idx = find_scale_from_multiscales(descs, self.fused_scale)  # text 只在[1,32]上融合

        token_feats, token_pad_mask = text_args['token_feats'], text_args['token_pad_masks']
        token_max = token_feats.shape[1] # b s c
        assert (mlm and (not mvm) and (not refer)) or (mvm and (not mlm) and (not refer)) or (refer and (not mlm) and (not mvm))
        if mlm:
            if self.task_conditioning_form == 'attn_mask':
                vid_pos = multiscales_poses[idx]
                referent_mask_condition = F.interpolate(referent_mask_condition.float(), size=multiscales[idx].shape[-2:], mode='area').bool()
                attn_mask = (~referent_mask_condition) # b t h w
                attn_mask = repeat(attn_mask, 'b t h w -> b s (t h w)', s=token_max)
            elif self.task_conditioning_form == 'pos':
                # b t c h w 
                # 1 c -> 1 1 c 1 1
                # 1 1 c 1 1 * b t 1 h w
                vid_pos = multiscales_poses[idx] + self.front_back_emb.weight[None,:, :, None, None] * (referent_mask_condition.float().unsqueeze(2))
                attn_mask = None

            fused_text_feat = self.mmattn_encoder(video_feat=multiscales[idx],
                                                    video_pad_mask=multiscales_pad_masks[idx],
                                                    video_pos=vid_pos,
                                                    text_feat=token_feats,
                                                    text_pad_mask=token_pad_mask,
                                                    attention_mask=attn_mask,
                                                    mlm=True)
            return {'fused_text_feats': fused_text_feat}
        
        if mvm:
            raise NotImplementedError()
    
        if refer:
            # self attion on the 32x and tokens
            assert referent_mask_condition == None
            attn_mask = None
            fused_video_feat, fused_text_feat = self.mmattn_encoder(video_feat=multiscales[idx],
                                                video_pad_mask=multiscales_pad_masks[idx],
                                                video_pos=multiscales_poses[idx],
                                                text_feat=token_feats,
                                                text_pad_mask=token_pad_mask,
                                                attention_mask=attn_mask,
                                                refer=True)
            multiscales[idx] = fused_video_feat

            multiscales = self.scale_encoder((multiscales, multiscales_pad_masks, multiscales_poses, descs))

            return {'fused_video_feats': multiscales,
                    'fused_text_feats': fused_text_feat}

        raise ValueError()


@register_fusion
def videotextfeats(fusion_configs, d_model):
    configs = vars(fusion_configs)
    return VideoTextFeats(task_conditioning_form=configs['task_conditioning_form'],
                          fused_scale=configs['fused_scale'],
                          d_model=d_model,
                          mmattn_encoder_configs=fusion_configs.mmattn_encoder,
                          scale_encoder_configs=fusion_configs.scale_encoder,)

"""
text那边是 linearized graph
fusion的解释有两种:
1. 将两个空间对齐
2. 让video那边先highlight 和文本中的objects相关的特征, 然后visual这边通过一堆scale encoder之后进一步highlight对饮的特征,
decoder的时候文本的token就能得到更好的object
3. 如果text 和 video一块变, 可以取消text 的ambuguity, 比如 bird和pigeon, text那边已经知道pigeon和bird相似, video里有pigeon(bird)
"""
class VideoMultiscale_TextLinearized(nn.Module):  
    @classmethod
    def pad_1d_feats(cls, feat_list):
        # list[ni c] -> b nmax c
        feat_len = [len(feat) for feat in feat_list]
        n_max = max(feat_len) 
        batch_size = len(feat_list)
        pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
        for i in range(batch_size):
            feat_list[i] = F.pad(feat_list[i], pad=[0, 0, 0, n_max-feat_len[i]])
            pad_mask[i, :feat_len[i]] = False
        feat_list = torch.stack(feat_list, dim=0) # b nmax c
        return feat_list, pad_mask
        
    def __init__(self,
                 d_model: int, 
                 fused_args, # multimodal attention encoder接受一个scale的video feats和一个text_linearized_graph_seq
                 scale_encoder_configs,):
        """
        fused_args.layer_strategy: 
            video向text靠
            video和text一块变
        """
        super().__init__()
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
        
        if fused_args.layer_strategy == 'video向text靠: 多个video scale  cross  text; text不变; cross attention每层共享; 一般只有一层':
            from .layers_unimodal_attention import CrossAttentionLayer, FFNLayer
            cross_attn_layer = CrossAttentionLayer(d_model=d_model,
                                                   nhead=fused_args.nhead,
                                                   dropout=fused_args.dropout,
                                                   activation='relu',
                                                   normalize_before=False)
            ffn_layer = FFNLayer(d_model=d_model,
                                      dim_feedforward=fused_args.dim_ffd,
                                      dropout=fused_args.dropout,
                                      activation='relu',
                                      normalize_before=False)
            self.cross_attn_layers = _get_clones(cross_attn_layer, N=fused_args.num_self_layers)
            self.ffn_layers = _get_clones(ffn_layer,  N=fused_args.num_self_layers)
            self.video_pos = nn.Embedding(1, d_model)
            self.text_pos = nn.Embedding(1, d_model)
            
            self.fused_scales = fused_args.fused_scales
            self.layer_strategy = fused_args.layer_strategy
            self.graph_which_to_fuse = fused_args.graph_which_to_fuse
            
        elif fused_args.layer_strategy == 'video和text一块变: video只有几个小scale  和text concate  做self; text在变; 一般是只有一个scale':
            # 一块变是否可以取消 text 的 ambiguity
            from .layers_unimodal_attention import SelfAttentionLayer, FFNLayer
            self_attn_layer = SelfAttentionLayer(d_model=d_model,
                                                   nhead=fused_args.nhead,
                                                   dropout=fused_args.dropout,
                                                   activation='relu',
                                                   normalize_before=False)
            ffn_layer = FFNLayer(d_model=d_model,
                                      dim_feedforward=fused_args.dim_ffd,
                                      dropout=fused_args.dropout,
                                      activation='relu',
                                      normalize_before=False)
            self.self_attn_layers = _get_clones(self_attn_layer, N=fused_args.num_self_layers)
            self.ffn_layers = _get_clones(ffn_layer,  N=fused_args.num_self_layers)
            self.video_pos = nn.Embedding(1, d_model)
            self.text_pos = nn.Embedding(1, d_model)
            self.fused_scales = fused_args.fused_scales
            self.layer_strategy = fused_args.layer_strategy
            self.graph_which_to_fuse = fused_args.graph_which_to_fuse
            
        elif fused_args.layer_strategy == '每个scale和text做neighborhood attention':
            pass
        elif fused_args.layer_strategy == '每个scale和text做motion':
            pass
        elif fused_args.layer_strategy == 'contrastive':
            # 需要多一个 fusion loss
            pass
        else:
            raise ValueError(0)

    def forward(self, 
                video_args, 
                text_args,refer=True):
        multiscales, multiscales_pad_masks, multiscales_poses, descs \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
            
        # b s c, b s, 
        # list[list[int], 第i个sequence的原先每个token被tokenize成几个token],b
        # list[list[int], 第i个sequence的原先token那些是node], b
        # list[list[int], 第i个sequence的原先token哪些是edge], b
        # list[dict]
        lin_amr_tree_strings = text_args['amr_tree_string_linearized']
        linearized_graph_feats, linearized_graph_pad_masks, each_token_lengths, node_token_indexs, edge_token_indexs,\
            = text_args['token_feats'], text_args['token_pad_masks'], text_args['each_token_lengths'],\
                text_args['node_token_indexs'], text_args['edge_token_indexs']
        batch_size  = multiscales[0].shape[0]
        
        # 拆开
        if self.graph_which_to_fuse == '整个linearized graph都融合':
            memory = linearized_graph_feats.permute(1, 0, 2) # b s c -> s b c
            memory_key_padding_mask = linearized_graph_pad_masks # b s
        else:
            memory = [] # list[nj c] -> b nmax c
            extracted_indexs_by_batch = []
            for batch_idx, (graph_feat, pad_mask, token_lengths, node_indexs, edge_indexs, amr_tree_string_linearized) in \
                    enumerate(zip(linearized_graph_feats, linearized_graph_pad_masks, \
                        each_token_lengths, node_token_indexs, edge_token_indexs, lin_amr_tree_strings)):
                # graph_feat: max c[max] -> si_after c
                # token_lengths: list[int] 加起来等于 si_after
                # node_indexs/edge_indexs: list[int]
                graph_feat = graph_feat[~pad_mask]
                graph_feat = torch.split(graph_feat, token_lengths) # list[T(ni c)], 原先的长度
                if self.graph_which_to_fuse == '只有node和edge':
                    assert (len(node_indexs) + len(edge_indexs)) == len(set(node_indexs + edge_indexs))
                    extracted_indexs = node_indexs + edge_indexs
                elif self.graph_which_to_fuse == '只有node':
                    extracted_indexs = node_indexs
                else:
                    raise ValueError()
                extracted_indexs_by_batch.append(extracted_indexs)
                # list[T(ni c)] -> ns c
                ect_feats = torch.cat([graph_feat[ect_idx] for ect_idx in extracted_indexs], dim=0)
                memory.append(ect_feats)
            # list[nsi, c] -> b nmax c
            memory, memory_key_padding_mask = VideoMultiscale_TextLinearized.pad_1d_feats(memory)
            memory = memory.permute(1, 0, 2)
        
        
        token_max = memory.shape[0]             
        pos = repeat(self.text_pos.weight, '1 c -> s b c', b=batch_size, s=token_max)            
        # 每个scale共享attention layers
        
        if self.layer_strategy == 'video向text靠: 多个video scale  cross  text; text不变; cross attention每层共享; 一般只有一层':
            for fused_scale in self.fused_scales:
                # 准备cross
                idx = find_scale_from_multiscales(descs, fused_scale)  
                output = rearrange(multiscales[idx], 'b t c h w -> (t h w) b c')
                _, nf, _, h, w = multiscales[idx].shape
                query_pos = multiscales_poses[idx] + repeat(self.video_pos.weight, '1 c -> thw b c', 
                                                            b=batch_size,
                                                            thw=output.shape[0])
                for attn_layer, ffn_layer in zip(self.cross_attn_layers, self.ffn_layers):
                    output = attn_layer(tgt=output,  # n b c
                                memory=memory, # thw b c
                                memory_mask=None, # bh n thw
                                memory_key_padding_mask=memory_key_padding_mask,  # here we do not apply masking on padded region
                                pos=pos,  # thw b c
                                query_pos=query_pos,) # n b c
                    output = ffn_layer(output)
                    
                fused_video_feat = rearrange(fused_video_feat, '(t h w) b c -> b t c h w', t=nf,h=h,w=w)                           
                multiscales[idx] = fused_video_feat

            fused_text_feat = memory.permute(1, 0, 2) # b s c

        elif self.layer_strategy == 'video和text一块变: video只有几个小scale  和text concate  做self; text在变; 一般是只有一个scale':          
            fused_text_feat_by_scale = []
            for fused_scale in self.fused_scales:
                # text 和 video concate
                idx = find_scale_from_multiscales(descs, fused_scale) 
                _, nf, _, h, w = multiscales[idx].shape 
                output = rearrange(multiscales[idx], 'b t c h w -> (t h w) b c')
                tgt_key_padding_mask = rearrange(multiscales_pad_masks[idx], 'b t h w -> b (t h w)')
                query_pos = rearrange(multiscales_poses[idx], 'b t c h w -> (t h w) b c')\
                    + repeat(self.video_pos.weight, '1 c -> thw b c', b=batch_size,thw=output.shape[0])
                
                output = torch.cat([output, memory], dim=0)
                tgt_key_padding_mask = torch.cat([tgt_key_padding_mask, memory_key_padding_mask], dim=1)
                query_pos = torch.cat([query_pos, pos], dim=0)
                
                for attn_layer, ffn_layer in zip(self.self_attn_layers, self.ffn_layers):
                    output = attn_layer(output,
                                    tgt_mask=None,
                                    tgt_key_padding_mask=tgt_key_padding_mask, # 1是padding
                                    query_pos=query_pos) # n b c
                    output = ffn_layer(output)
                    
                fused_video_feat = output[:(nf*h*w), ...]
                fused_video_feat = rearrange(fused_video_feat, '(t h w) b c -> b t c h w', t=nf,h=h,w=w)                           
                multiscales[idx] = fused_video_feat
                
                fused_text_feat = output[(nf*h*w):, ...]
                fused_text_feat = fused_text_feat.permute(1, 0, 2) # b s c
                fused_text_feat_by_scale.append(fused_text_feat)
            fused_text_feat = torch.stack(fused_text_feat_by_scale, dim=0).mean(0) # num b s c -> b s c
        else:
            raise ValueError()
        
        # 对号入座
        if self.graph_which_to_fuse == '整个linearized graph都融合':
            pass
        else:
            for batch_idx, (fused_feat, pad_mask, extracted_indexs, each_token_length) in enumerate(zip(fused_text_feat, memory_key_padding_mask,
                                                                                     extracted_indexs_by_batch, each_token_lengths)):
                fused_feat = fused_feat[~pad_mask]
                left_len = 0
                for idx, ext_idx in enumerate(extracted_indexs):
                    split_len = each_token_length[ext_idx]
                    
                    start_idx = sum(each_token_length[:ext_idx])
                    end_idx = start_idx + split_len
                    
                    linearized_graph_feats[batch_idx, start_idx:end_idx] = fused_feat[left_len: (left_len+split_len)]
                    left_len += split_len
                assert left_len == len(fused_feat)
            fused_text_feat = linearized_graph_feats
            
        multiscales = self.scale_encoder((multiscales, multiscales_pad_masks, multiscales_poses, descs))

        return {'fused_video_feats': multiscales,
                'fused_text_feats': fused_text_feat}

@register_fusion
def videomultiscale_textlinearized(configs, d_model):
    return VideoMultiscale_TextLinearized(d_model=d_model,
                                          fused_args=configs.fused_args,
                                          scale_encoder_configs=configs.scale_encoder)


"""
text那边是graph
将video 和 concept进行cross attention,
1. 所有scale cross text, 只有video变
2. share同一个fusion module
"""
class Video_TextGraph(nn.Module):
    def __init__(self, 
                 video_feats_proj,
                 d_model: int, 
                 scale_encoder_configs,
                 fusion_strategy):
        super().__init__()
        self.build_feat_proj(video_feats_proj, d_model=d_model)
        self.fusion_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
        create_scale_encoder = multiscale_encoder_entrypoints(scale_encoder_configs.name)
        self.scale_encoder = create_scale_encoder(scale_encoder_configs, d_model=d_model)
        self.fusion_strategy = fusion_strategy
        
        self.video_pos = nn.Embedding(1, d_model)
        self.text_pos = nn.Embedding(1, d_model)

    def build_feat_proj(self, proj_configs, d_model):
        configs = vars(proj_configs)
        self.video_feat_proj_name = configs['name']
        if self.video_feat_proj_name != 'no_proj':
            bb_out_channels, bb_scalestrides = proj_configs.bb_out_channels, proj_configs.bb_out_scales
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
                if self.video_feat_proj_name == 'conv2d':
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
                    
                elif self.video_feat_proj_name == 'conv3d':
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
            pass
        
    def proj_bakcbone_out(self, multiscale_feats):
        """
        pad_mask: b t h w,  t: valid_t
        """
        # all the padding masks must be in bool type,因为会作为attention mask使用
        batch_size = multiscale_feats[0].shape[0] #  b t c h w
        srcs = [] # b t c h w
        if self.video_feat_proj_name == 'conv2d':
            for lvl, src in enumerate(multiscale_feats): 
                nf = src.shape[1]
                src = src.flatten(0, 1)  # b t c h w -> (b t) c h w
                src = self.vid_proj[lvl](src)
                src = rearrange(src, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                srcs.append(src)
            return srcs

        elif self.video_feat_proj_name == 'conv3d':
            for lvl, src in enumerate(multiscale_feats): 
                src = src.permute(0, 2, 1, 3, 4) # b t c h w -> b c t h w
                src = self.vid_proj[lvl](src)
                src = rearrange(src, 'b c t h w -> b t c h w')
                srcs.append(src) 
            return srcs
           
        elif self.video_feat_proj_name == 'no_proj':
            return multiscale_feats         
        else:
            raise ValueError()


    def get_cross_attention_index(self, all_seg_ids):
        """
        # 0:variable, 1:concept_val, 2: constant
        # 0: /    1: regular edge 2: constant edge
        Input:
            node_identifier: tensor, num_node
            edge_identifider: tensor, num_edge
        Output:
            list[bool], list[bool]
        """
        # b max

        if self.fusion_strategy == 0:
            # 只对concept/constant node进行融合
            return torch.logical_or(all_seg_ids==2, all_seg_ids==3)
        raise ValueError()

    # 如果不对哪个模态的特征做改变, 则用clone
    def forward(self, video_args, text_args, refer=True):
        
        # b max c
        all_feats, all_seg_ids = text_args['all_feats'].clone(), text_args['all_seg_ids'].clone()
        multiscales = [scale_feat.clone() for scale_feat in video_args['multiscales']]
        multiscales_pad_masks = [pad_mask.clone() for pad_mask in video_args['multiscale_pad_masks']]
        multiscales_poses = [pos.clone() for pos in video_args['multiscale_poses']]
        descs = copy.deepcopy(video_args['multiscale_des'])
        
        multiscales = self.proj_bakcbone_out(multiscales,)
        
        batch_size, *_, device = *multiscales[0].shape, multiscales[0].device
        
        # b V+E_max
        who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids)
        crossed_text_feats = [bt_feat[who_cross] for bt_feat, who_cross in zip(all_feats, who_does_cross_attention_mask)]
        
        # b max c
        crossed_text_feats, crossed_text_pad_mask = pad_1d_feats(crossed_text_feats)
        crossed_text_feats = crossed_text_feats.permute(1, 0, 2)
        crossed_text_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=crossed_text_feats.shape[0],
                                  b=batch_size)

        srcs = []
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat = self.fusion_module(tgt=feat,
                                    memory=crossed_text_feats,
                                    memory_key_padding_mask=crossed_text_pad_mask,
                                    pos=crossed_text_pos,
                                    query_pos=poses)
            feat = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            srcs.append(feat)
        
        multiscales = self.scale_encoder((srcs, multiscales_pad_masks, multiscales_poses, descs))
        return {
            'fused_video_feats': multiscales
        }

@register_fusion
def video_textgraph(configs, d_model):
    return Video_TextGraph(d_model=d_model,
                           video_feats_proj=configs.video_feat_proj,
                        scale_encoder_configs=configs.scale_encoder,
                        fusion_strategy=configs.fusion_strategy)




# text_args是一堆graph
# cross attention 用的是普通的
# self attention 用的是graph的
class VideoGraphFeats(FusionEncoder):
    def __init__(self, num_queries: int, query_feat: str, fused_scale, d_model: int, mmattn_encoder_configs, scale_encoder_configs, decoder_configs, task_conditioning_form=None):
        super().__init__(num_queries, query_feat, fused_scale, d_model, mmattn_encoder_configs, scale_encoder_configs, decoder_configs, task_conditioning_form)
    def forward(self, video_args, text_args, referent_mask_condition=None, mlm=False, mvm=False, refer=False):
        """
        mask_conditions: 
        如果是mlm, 并且是随机mask token, 句子的fill mask需要知道在fill哪个对象
        如果是mvm,  由于句子只是描述其中一个物体, 无论是哪种masking, 都会让模型根据句子补全这个对象, 如果没有条件加入的话, 
        就成了一个ill-conditioned问题, 也能训练起来, 但是可以加上一些位置信息, 

        mask_conditions: b t h w, 即这个句子的对象对应的referent mask
        """
        # 4 8 16 32 64
        multiscales, multiscales_pad_masks, multiscales_poses, descs \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        idx = find_scale_from_multiscales(descs, self.fused_scale)  # text 只在[1,32]上融合

        graphs = text_args['graphs'] # list[Graph]
        token_max = token_feats.shape[1] # b s c
        assert (mlm and (not mvm) and (not refer)) or (mvm and (not mlm) and (not refer)) or (refer and (not mlm) and (not mvm))
        if mlm:
            if self.task_conditioning_form == 'attn_mask':
                vid_pos = multiscales_poses[idx]
                referent_mask_condition = F.interpolate(referent_mask_condition.float(), size=multiscales[idx].shape[-2:], mode='area').bool()
                attn_mask = (~referent_mask_condition) # b t h w
                attn_mask = repeat(attn_mask, 'b t h w -> b s (t h w)', s=token_max)
            elif self.task_conditioning_form == 'pos':
                # b t c h w 
                # 1 c -> 1 1 c 1 1
                # 1 1 c 1 1 * b t 1 h w
                vid_pos = multiscales_poses[idx] + self.front_back_emb.weight[None,:, :, None, None] * (referent_mask_condition.float().unsqueeze(2))
                attn_mask = None

            fused_text_feat = self.mmattn_encoder(video_feat=multiscales[idx],
                                                    video_pad_mask=multiscales_pad_masks[idx],
                                                    video_pos=vid_pos,
                                                    text_feat=token_feats,
                                                    text_pad_mask=token_pad_mask,
                                                    attention_mask=attn_mask,
                                                    mlm=True)
            return fused_text_feat
        
        if mvm:
            raise NotImplementedError()
    
        if refer:
            # self attion on the 32x and tokens
            assert referent_mask_condition == None
            attn_mask = None
            fused_video_feat, fused_text_feat = self.mmattn_encoder(video_feat=multiscales[idx],
                                                video_pad_mask=multiscales_pad_masks[idx],
                                                video_pos=multiscales_poses[idx],
                                                text_feat=token_feats,
                                                text_pad_mask=token_pad_mask,
                                                attention_mask=attn_mask,
                                                refer=True)
            multiscales[idx] = fused_video_feat

            multiscales = self.scale_encoder((multiscales, multiscales_pad_masks, multiscales_poses, descs))

            return multiscales, None, fused_text_feat # placeholder for the query feats

        raise ValueError()
    
    def forward_refer(self, video_args, query_feats, text_args, return_loss=False, targets=None):
        """
        video_args: multiscale, _, _, stride4; b t c h w
        text_args: feat, pad_mask; b s c
        """
        assert query_feats is None
        # 8 16 32 64

        batch_size = text_args['token_feats'].shape[0]

        query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        refer_output, refer_loss = self.decoder(query_feats=query_feats,
                                                query_pos=query_pos,
                                                video_args=video_args,
                                                text_args=text_args,

                                                return_loss=return_loss,
                                                targets=targets)
        
        return refer_output, refer_loss 









   
class QueryVideoTextFeats(FusionEncoder):
    def __init__(self, num_queries: int, query_feat: str, task_conditioning_form, fused_scale, d_model: int, mmattn_encoder_configs, scale_encoder_configs, decoder_configs):
        super().__init__(num_queries, query_feat, task_conditioning_form, fused_scale, d_model, mmattn_encoder_configs, scale_encoder_configs, decoder_configs)

    def forward(self, video_args, text_args, referent_mask_condition=None, mlm=False, mvm=False, refer=False):
        """
        mask_conditions: 
        如果是mlm, 并且是随机mask token, 句子的fill mask需要知道在fill哪个对象
        如果是mvm,  由于句子只是描述其中一个物体, 无论是哪种masking, 都会让模型根据句子补全这个对象, 如果没有条件加入的话, 
        就成了一个ill-conditioned问题, 也能训练起来, 但是可以加上一些位置信息, 

        mask_conditions: b t h w, 即这个句子的对象对应的referent mask
        """
        # 4 8 16 32 64
        multiscales, multiscales_pad_masks, multiscales_poses, descs = video_args
        idx = find_scale_from_multiscales(descs, self.fused_scale)
        batch_size = multiscales[0].shape[0] # b t c h w
        token_feats, token_pad_mask, _ = text_args
        assert (mlm and (not mvm) and (not refer)) or (mvm and (not mlm) and (not refer)) or (refer and (not mlm) and (not mvm))
        if mlm:
            if self.task_conditioning_form == 'attn_mask':
                vid_pos = multiscales_poses[idx]
                attn_mask = (~ referent_mask_condition) # b t h w
            elif self.task_conditioning_form == 'pos':
                # b t c h w 
                # 1 c -> 1 1 c 1 1
                # 1 1 c 1 1 * b t 1 h w
                vid_pos = multiscales_poses[idx] + self.front_back_emb.weight[None,:, :, None, None] * (referent_mask_condition.float().unsqueeze(2))
                attn_mask = None

            query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1) # n b c
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # n b c
            fused_text_feat = self.mmattn_encoder(vid_feat=multiscales[idx],
                                                    vid_pad_mask=multiscales_pad_masks[idx],
                                                    vid_pos=vid_pos,
                                                    text_feat=token_feats,
                                                    text_pad_mask=token_pad_mask,
                                                    query_feats=query_feats,
                                                    query_pos=query_pos,
                                                    attention_mask=attn_mask,
                                                    mlm=True)
            return fused_text_feat
        
        if mvm:
            raise NotImplementedError()
    
        if refer:
            # self attion on the 32x and tokens
            assert referent_mask_condition == None
            query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1) # n b c
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # n b c
            fused_video_feat, fused_text_feat = self.mmattn_encoder(vid_feat=multiscales[idx],
                                                vid_pad_mask=multiscales_pad_masks[idx],
                                                vid_pos=vid_pos,
                                                text_feat=token_feats,
                                                text_pad_mask=token_pad_mask,
                                                query_feats=query_feats,
                                                query_pos=query_pos,
                                                attention_mask=attn_mask,
                                                refer=True)
            multiscales[idx] = fused_video_feat
            # only the last three
            multiscales, multiscales_pad_masks, multiscales_poses, video_stride4 = self.scale_encoder(multiscales, multiscales_pad_masks,
                                                                                                    multiscales_poses, video_stride4)
            multiscales = multiscales[:-1][::-1]
            multiscales_pad_masks = multiscales_pad_masks[:-1][::-1]
            multiscales_poses = multiscales_poses[:-1][::-1]

            return (multiscales, multiscales_pad_masks, multiscales_poses, video_stride4), None, fused_text_feat

        raise ValueError()
    
    def forward_refer(self, video_args, query_feats, text_args, return_loss=False, targets=None):
        """
        video_args: multiscale, _, _, stride4; b t c h w
        text_args: feat, pad_mask; b s c
        """
        # 8 16 32 64

        batch_size = text_args[0].shape[0]

        query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        refer_loss, refer_output = self.decoder(query_feat=query_feats,
                                                    query_pos=query_pos,
                                                    video_args=video_args,
                                                    text_args=text_args,

                                                    return_loss=return_loss,
                                                    targets=targets)
        
        return refer_loss, refer_output  

@register_fusion
def query_videotextfeats(configs, d_model):   
    return QueryVideoTextFeats(num_queries=main_configs['nqueries'],
                          query_feat=main_configs['query_feat'],
                          d_model=d_model,
                          encoder_configs=configs.encoder,
                          scale_encoder_configs=configs.scale_encoder,
                          decoder_configs=configs.decoder)


class QueryVideofeatsBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
class VideoFeatsBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward_text(self, texts, text_auxiliary, device, 
                        mask_sentence=False, masked_indices=None):
        tokens = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        token_padding_mask = tokens['attention_mask'].ne(1).bool() # b max, 1是padding的位置, 0是没有padding的位置
        if mask_sentence:
            tokens['input_ids'], token_gt_labels, token_masked_bool = self._torch_mask_tokens(inputs=tokens['input_ids'],
                                                                                              masked_indices=masked_indices)
            
        tokenized_feats = self.vocab(input_ids=tokens['input_ids']) # b max d
        
        # self_attn_mask = token_padding_mask.float()[:, None, None, :] # b 1 1 max
        # self_attn_mask = self_attn_mask * torch.finfo(tokenized_feats.dtype).min
        # encoder_outputs = self.text_backbone_encoder(
        #     tokenized_feats,
        #     attention_mask=self_attn_mask
        # )
        # sequence_output = encoder_outputs[0]
        # pooled_output = self.text_backbone_pooler(sequence_output) if self.text_backbone_pooler is not None else None
        # text_features = self.text_proj(sequence_output) 
        # text_sentence_features = self.text_proj(pooled_output) 
        
        # b token_max c, b token_max, b c
        if mask_sentence:
            return (tokenized_feats, token_padding_mask), (token_gt_labels, token_masked_bool, tokens['input_ids'])
        else:
            return (tokenized_feats, token_padding_mask), None


from util.misc import find_scale_from_multiscales, find_scales_from_multiscales
from .layers_unimodal_attention import CrossSelfFFN_Module
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class VideoTextCompiler(nn.Module):
    def __init__(
        self, # decoder         
        d_model: int,
        mask_dim,
        
        # attention layers
        nheads: int,

        # important
        used_scales,
        conved_scale,
        nqueries_per_scale, #  list[int], each scale
        nlayers_per_scale, # list[int], each scale
    
        matching_configs,
        aux_loss,
        
        learn_query_feats=False
    ):
        super().__init__()
        # concept vocabulary
        with open('/home/xhh/datasets/knowledge/concept_classes.txt') as f:
            self.concept_names = f.read().splitlines()
        self.num_concepts = len(self.concept_names)
        self.concepts_vocab = nn.Embedding(num_embeddings=self.num_concepts, embedding_dim=d_model)
        self.concept_encoder = {concept: i for i, concept in enumerate(self.concept_names)}

        # relation vocabulary
        with open('/home/xhh/datasets/knowledge/relation_classes.txt') as f:
            self.relation_names = f.read().splitlines()
        self.num_relations = len(self.relation_names)        
        self.relations_vocab = nn.Embedding(num_embeddings=self.num_relations, embedding_dim=d_model)
        self.relation_encoder = {relation: i for i, relation in enumerate(self.relation_names)} 
        
        # 
        assert len(nqueries_per_scale) == len(used_scales)
        assert len(nlayers_per_scale) == len(used_scales)
        self.used_scales = used_scales # 
        self.conved_scale = conved_scale # 
        self.num_feature_scales = len(used_scales)
        self.scale_embedding = nn.Embedding(self.num_feature_scales, d_model) # 3

        # scale_specific queries
        self.concept_queries_by_scale = nn.ModuleList(
            [nn.Embedding(nquery, d_model) for nquery in nqueries_per_scale]
        )
        self.learn_query_feats = learn_query_feats
        if learn_query_feats:
            self.concept_queries_feat_by_scale = nn.ModuleList(
                [nn.Embedding(nquery, d_model) for nquery in nqueries_per_scale]
            )
        
        self.nqueries_per_scale = nqueries_per_scale
        self.decoder_norm_by_scale = _get_clones(nn.LayerNorm(d_model), self.num_feature_scales)

        # sclae_specific layers
        layers = []
        for nlayer in nlayers_per_scale:
            layers.append(_get_clones(CrossSelfFFN_Module(d_model=d_model,
                                                           nhead=nheads,
                                                           dropout=0.), nlayer))
        self.attn_layers_by_scale = nn.ModuleList(layers)


        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
    
    def forward(self, 
                video_args,
                text_args,
                targets=None, return_loss=False):
        """
        video_args: b t c h w
        text_args: 
        """
        # make sure that the video features are fused with the text features before
        
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec = video_args
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]

        graph = text_args['parsed graph']
        # b t c h w
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_scales):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            # scale_feats = self.input_proj[i](scale_feats) # b t c h w, TODO: 每个scale有不同的维度
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.scale_embedding.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
        
        # n c -> n b c -> 3n b c
        output = torch.cat([self.concept_queries_by_scale[i].weight[None, :, :].repeat(batch_size, 1, 1)], dim=0)
        for lvl, (queries, layers, norm, memory, memory_pos) in enumerate(zip(self.concept_queries_by_scale, self.attn_layers_by_scale,
                                                     self.decoder_norm_by_scale, srcs, poses)):
            
            lvl_output = queries.weight[:, None, :].repeat(1, batch_size, 1) # n c -> n b c
            for nlayers in len(layers):
                lvl_output = layer(lvl_output,
                                memory,
                                memory_pos,
                                cross_attn_mask=None)
                cross_attn_mask=None
            
        
        predictions_class = [] # list[b nq 2], init -> 32x -> 16x -> 8x
        predictions_mask = [] # list[b t nq H/4 W/4], 
        attn_mask_size = size_list[0]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output[(-1 * self.nqueries_per_scale[-1]):], 
                                                                                       mask_features, attn_mask_target_size=attn_mask_size)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # b*h n thw
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            if self.concate_text:
                text_am = torch.zeros([attn_mask.shape[0], attn_mask.shape[1], token_feats.shape[0]],
                                       device=attn_mask.device, dtype=attn_mask.dtype) # bh n s
                attn_mask = torch.cat([attn_mask ,text_am], dim=-1) # bh n thw+s

                cross_memory = torch.cat([srcs[level_index], token_feats],dim=0)
                cross_pos = torch.cat([poses[level_index], torch.zeros_like(token_feats)], dim=0)
            else:
                cross_memory = srcs[level_index]
                cross_pos = poses[level_index]

            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # n b c
                memory=cross_memory, # thw b c
                memory_mask=attn_mask, # bh n thw
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=cross_pos,  # thw b c
                query_pos=query_pos, # n b c
            )

            output = self.transformer_self_attention_layers[i](
                output, # n b c
                tgt_mask=None,
                tgt_key_padding_mask=None, # b n 
                query_pos=query_pos, # n b c
            )
            output = self.transformer_ffn_layers[i](
                output # n b c
            )
            
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
            # (b nq 2, real), (b t nq H W, real), bh n thw
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        
        outputs = {
            'pred_logits': predictions_class[-1], # b nq 2
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
        }

        if return_loss:
            assert targets is not None
            losses = self.forward_refer_loss(outputs, targets)
            return outputs, losses
        else:
            assert targets is None
            return outputs, None
    
    def forward_refer_loss(self, out, targets):
        """
        Params:
            targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
        """
        losses = {}
        
        outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
        
        indices = self.criterion.matching(outputs_without_aux, targets)
        
        losses = self.criterion(out, targets, indices)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                indices_i = self.criterion.matching(aux_outputs, targets)
                l_dict_i = self.criterion(aux_outputs, targets, indices_i)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """
        Input:
            - output_class:
                list[T(tb n classes)]
            - outputs_seg_masks:
                list[T(tb n H W)]
            - outputs_boxes:
                list[T(tb n 4)]
        """
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_cls=True, return_attn_mask=True, return_box=False):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        outputs_class = None
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        if return_cls:
            outputs_class = self.class_embed(decoder_output)  # b n 2
            
        return outputs_class, outputs_mask, attn_mask


# Encoders that use Cross Attention
class SameVanCross_TextVanSelf_VideoDivSelf_SameLLN(nn.Module):
    def __init__(self, configs, d_model) -> None:
        super().__init__()
        configs=vars(configs)
        self.fusion_text_self_attention_layers = nn.ModuleList()
        self.fusion_video_self_attention_layers = nn.ModuleList()
        self.fusion_cross_attention_layers = nn.ModuleList()
        self.fusion_ffn_layers = nn.ModuleList()
        self.num_fusion_layers = configs['nlayers']
        for _ in range(self.num_fusion_layers):
            self.fusion_text_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=d_model,
                    nhead=configs['nheads'],
                    dropout=0.0,
                    normalize_before=configs['pre_norm'],
                )
            )
            self.fusion_video_self_attention_layers.append(
                VideoDivSelfAttentionLayer(
                    d_model=d_model,
                    nhead=configs['nheads'],
                    dropout=0.0,
                    normalize_before=configs['pre_norm']
                )
            )
            self.fusion_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=d_model,
                    nhead=configs['nheads'],
                    dropout=0.0,
                    normalize_before=configs['pre_norm']
                )
            )
            self.fusion_ffn_layers.append(
                FFNLayer(
                    d_model=d_model,
                    dim_feedforward=configs['dff'],
                    dropout=0.0,
                    normalize_before=configs['pre_norm'],
                )
            )
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self,
        tgt, tgt_mask, tgt_key_padding_mask, query_pos,
        memory, memory_mask, memory_key_padding_mask, pos, query='text', nf=None, h=None,w=None):
        
        for i in range(self.num_fusion_layers): 
            tgt = self.fusion_cross_attention_layers[i](
                tgt=tgt,  # n b c
                memory=memory, # thw b c
                memory_mask=memory_mask, # bh n thw
                memory_key_padding_mask=memory_key_padding_mask,  # here we do not apply masking on padded region
                pos=pos,  # thw b c
                query_pos=query_pos, # n b c
            )
            # self
            if query == 'text':
                tgt = self.fusion_text_self_attention_layers[i](
                    tgt, # n b c
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, # b n 
                    query_pos=query_pos, # n b c
                )
            elif query == 'visual':
                tgt = self.fusion_video_self_attention_layers[i](
                    tgt, # thw b c
                    tgt_mask=tgt_mask, #None
                    tgt_key_padding_mask=tgt_key_padding_mask, # None  
                    query_pos=query_pos, # thw b c
                    nf=nf,h=h,w=w
                ) 
            else:
                raise ValueError()   
            
            tgt = self.fusion_ffn_layers[i](
                tgt # n b c
            )
        tgt = self.fusion_norm(tgt)
        return tgt

class ConditionAsPositionEmbedding_SameVanCross_TextVanSelf_VideoDivSelf_SameLLN_cls(nn.Module):
    def __init__(self, 
                 configs, d_model) -> None:
        super().__init__()
        
        # If given, this will scale gradients by the inverse of frequency of the words in the mini-batch.
        # max_norm, norm_type, padding_idx
        self.fusion_frontback_embeds = nn.Embedding(2, d_model, scale_grad_by_freq=True)
        self.decoder = SameVanCross_TextVanSelf_VideoDivSelf_SameLLN(configs, d_model)
        
    def forward(self, visual_feat, visual_pos,
                    text_feat, text_pad_mask,
                    query, front_mask=None):
        bs, nf, _, h, w = visual_feat.shape
        text_feat = rearrange(text_feat, 'b s c -> s b c')
        visual_feat = rearrange(visual_feat, 'b t c h w -> (t h w) b c')
        visual_pos = rearrange(visual_pos, 'b t c h w -> (t h w) b c')
        if query =='visual_fuse_text':
            output= self.decoder(
                tgt=visual_feat, 
                tgt_key_padding_mask=None, 
                tgt_mask=None, query_pos=visual_pos,
                
                memory=text_feat, memory_mask=None, 
                memory_key_padding_mask=text_pad_mask, pos=None, 
                
                query='visual', nf=nf,h=h,w=w
            )
            output = rearrange(output, '(t h w) b c -> b t c h w',t=nf,h=h,w=w)

        elif query=='text_ask_visual':
            assert front_mask is not None
            scaled_front_mask = F.interpolate(front_mask.float(), size=(h,w), mode='nearest') # b t h w
            scaled_front_mask = rearrange(scaled_front_mask, 'b t h w -> (t h w) b')
            scaled_front_mask = 1 - (scaled_front_mask.int())
            visual_pos += self.fusion_frontback_embeds(scaled_front_mask) # (t h w) b c
            output = self.decoder(
                tgt=text_feat, tgt_key_padding_mask=text_pad_mask, tgt_mask=None, query_pos=None,
                memory=visual_feat, memory_mask=None, memory_key_padding_mask=None, pos=visual_pos, query='text',
            )
            output = rearrange(output, 's b c -> b s c')
        elif query == 'visual_ask_text':
            pass
        else:
            raise ValueError()
        
        return output

@register_fusion
def ConditionAsPositionEmbedding_SameVanCross_TextVanSelf_VideoDivSelf_SameLLN(configs, d_model):
    return ConditionAsPositionEmbedding_SameVanCross_TextVanSelf_VideoDivSelf_SameLLN_cls(configs, d_model)


class SingleCrossAttention_cls(nn.Module):
    def __init__(self, configs, d_model):
        super().__init__()
        self.decoder = VisionLanguageFusionModule(d_model=d_model, nhead=configs.nhead)
        self.fusion_frontback_embeds = nn.Embedding(2, d_model, scale_grad_by_freq=True)


    def forward(self, visual_feat, visual_pos,
                    text_feat, text_pad_mask,
                    query, front_mask=None):
        
        bs, nf, _, h, w = visual_feat.shape
        text_feat = rearrange(text_feat, 'b s c -> s b c')
        visual_feat = rearrange(visual_feat, 'b t c h w -> (t h w) b c')
        visual_pos = rearrange(visual_pos, 'b t c h w -> (t h w) b c')
        if query =='visual_fuse_text':
            output= self.decoder(
                tgt=visual_feat,  query_pos=visual_pos,
                memory=text_feat,  
                memory_key_padding_mask=text_pad_mask, pos=None, 
            )
            output = rearrange(output, '(t h w) b c -> b t c h w',t=nf,h=h,w=w)

        elif query=='text_ask_visual':
            assert front_mask is not None
            scaled_front_mask = F.interpolate(front_mask.float(), size=(h,w), mode='nearest') # b t h w
            scaled_front_mask = rearrange(scaled_front_mask, 'b t h w -> (t h w) b')
            scaled_front_mask = 1 - (scaled_front_mask.int())
            visual_pos += self.fusion_frontback_embeds(scaled_front_mask) # (t h w) b c
            output = self.decoder(
                tgt=text_feat, query_pos=None,
                memory=visual_feat, memory_key_padding_mask=None, pos=visual_pos)
            output = rearrange(output, 's b c -> b s c')
        elif query == 'visual_ask_text':
            pass
        else:
            raise ValueError()
        
        return output

@register_fusion
def SingleCrossAttention(configs, d_model):
    return SingleCrossAttention_cls(configs, d_model)


class StackNMNs_Find(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, x, textual_query):
        return self.conv2(self.conv1(x) * self.linear(textual_query)[..., None, None]).squeeze(1) # b 1 h w
        
class StackNMNs_Transform(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        
        self.linear1 = nn.Linear(dim, dim) 
        self.linear2 = nn.Linear(dim, dim)
        
    def forward(self, a_1, x, textual_query):
        """
        Input: 
            a_1: b h w
            x: b c h w
            textual_query: b c
        """
        x1 = x * a_1.unsqueeze(1).sum((-2, -1)) # b c
        
        x = self.conv1(x) * self.linear1(x1)[..., None, None] * self.linear2(textual_query)[..., None, None] # b c * b c h w
        x = self.conv2(x).squeeze(1) # b 1 h w    
        return x
    
class StackNMNs_And(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    
    def forward(self, a_1, a_2):
        """
        Input:
            a_1: b h w
            a_2: b h w
        """
        return torch.minimum(a_1, a_2)
    
class StackNMNs_Or(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    def forward(self, a_1, a_2, **kwargs):
        return torch.maximum(a_1, a_2)

class StackNMNs_Filter(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.find_module = StackNMNs_Find(dim,)
        self.and_module = StackNMNs_And()
    
    def forward(self, a_1, x, textual_query):
        return self.and_module(a_1, self.find_module(x, textual_query))

class StackNMNs_Scene(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x).squeeze(1) # b h w

class StackNMNs_Answer(nn.Module):
    def __init__(self, dim, num_labels) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, num_labels)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
    
    def forward(self, a_1, x, textual_query):
        x1 = x * a_1.unsqueeze(1).sum((-2, -1)) # b c
        return self.linear1(self.linear2(x1) * self.linear3(textual_query)) # b #num_labes

class StackNMNs_Compare(nn.Module):
    def __init__(self, dim, num_labels) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, num_labels)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.linear4 = nn.Linear(dim, dim)
        
    def forward(self, a_1, a_2, x, textual_query):
        x1 = x * a_1.unsqueeze(1).sum((-2, -1)) # b c
        x2 = x * a_2.unsqueeze(1).sum((-2, -1)) # b c

        return self.linear1(self.linear2(x1) * self.linear3(x2) * self.linear4(textual_query))

class StackNMNs_NoOP(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
class Stack2D(nn.Module):
    def __init__(self, L, resolution, batch_size):
        super().__init__()
        # dim is defined as $H\times W$ in StackNMNs
        self.memory_length = L
        h, w = resolution
        self.memory = nn.Parameter(torch.zeros([batch_size, self.memory_length, h, w]), requires_grad=False) # b L h w
        self.pointer =  nn.Parameter(torch.tensor([[0, 1, 0]]), requires_grad=False).repeat(batch_size, 1) # b 3
        
    def push(self, z):
        # b c
        pointer = F.conv1d(self.pointer, torch.tensor([0,0,1])) # L
        memory = self.memory * (1 - pointer) + z * (pointer)
        return memory

    def pop(self):
        z = (self.memory * self.pointer).sum(0) # d
        pointer = F.conv1d(pointer, torch.tensor([1,0,0])) # L
        return z, self.memory

    def push_(self, z):
        self.pointer = self.conv1d(self.pointer, torch.tensor([0,0,1])) # L
        self.memory = self.memory * (1- self.pointer) + z * (self.pointer)
    
    def pop_(self):
        z = (self.memory * self.pointer).sum(0) # d
        self.pointer = F.conv1d(self.pointer, torch.tensor([1,0,0])) # L
        return z   
    
    def top(sef):
        return None  
    
from abc import abstractclassmethod
from .encoder_text import TemplateDistribution_WordsSoftDistribution
from .criterion_video import matching_entrypoints

class StackNMNs(nn.Module):
    def __init__(self, stack_length, dim, 
                 maximum_time_steps,
                 criterion_configs,
                 num_labels,
                 
                 ) -> None:
        super().__init__()
        self.num_templates = 8
        self.stack_length = stack_length
        self.dim = dim
        self.maximum_time_steps = maximum_time_steps        
        self.template_vocabulary = nn.ModuleList(
            [
            StackNMNs_Find(dim=dim),
            StackNMNs_Filter(dim),
            StackNMNs_And(),
            StackNMNs_Answer(dim, num_labels=num_labels),
            StackNMNs_Compare(dim, num_labels=num_labels),
            StackNMNs_NoOP(),
            StackNMNs_Scene(dim),
            StackNMNs_Or(),
            StackNMNs_Transform(dim)
            ]
        )
        self.continuous_layout_generator = TemplateDistribution_WordsSoftDistribution(dim = dim,
                                                                                      maximum_time_steps=maximum_time_steps,
                                                                                      num_templates=self.num_templates)
        create_criterion = matching_entrypoints(criterion_configs.name)
        self.criterion = create_criterion(criterion_configs)
        
        self.template_names = []
        self.template_arty = []

    
    def forward(self, visual_feats, text_feats, text_sentence_feats,
                return_loss=False, targets=None):
        """
        """
        batch_size, _, h, w = visual_feats.shape
        
        stack = Stack2D(self.stack_length, resolution=(h, w), batch_size=batch_size)
        
        textual_query = None
        for time_step in range(self.maximum_time_steps):
            # b L, b s, b c
            template_distribution_t, words_soft_attention_t, textual_query = self.continuous_layout_generator(
                text_feats, text_sentence_feats, textual_query, time_step,
            )
                        
            hidden_stacks = []
            for template, arty, template_name in zip(self.template_vocabulary, self.template_arty, self.template_names):
                if template_name == 'NoOp':
                    hidden_stacks.append(stack.memory)
                    continue
                
                if arty == 0:
                    output = template(textual_query, visual_feats, text_feats)
                    memory_weight = stack.push(output)
                    hidden_stacks.append(memory_weight)
                    
                elif arty == 1:
                    arty1, _ = stack.pop()
                    output = template(arty1, textual_query, visual_feats, text_feats)
                    memory_weight = stack.push(output)
                    hidden_stacks.append(memory_weight)
                    
                elif arty == 2:
                    arty1, _ = stack.pop()
                    arty2, _ = stack.pop()
                    output = template(arty1, arty2, textual_query, visual_feats, text_feats)
                    memory_weight = stack.push(output)
                    hidden_stacks.append(memory_weight)
                    
                else:
                    raise NotImplementedError()             
            
            hidden_stacks = torch.stack(hidden_stacks, dim=0) # b L d
            
            stack.memory = hidden_stacks * template_distribution_t.unsqueeze(-1) 
            stack.pointer = None #TODO
        
        final_out = stack.top()
        output = {
            'pred_masks': final_out, # b h w
            'pred_boxes': final_out # b 4
        }
        
        if return_loss:
            assert targets is not None
            loss_dict = self.criterion(output, targets)
            return output, loss_dict
        return output

@register_fusion
def stack_NMNs(configs, d_model):
    return StackNMNs(
        stack_length=configs['stack_length'],
        dim=d_model,
        maximum_time_steps=configs['maximum_T'],
        criterion_configs=configs.criterion_configs,
        num_labels=configs['num_labels']
    )


class XNMs_AttenNode(nn.Module):
    def __init__(self, dim, ) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, query, scene_graph):
        return node_attention

class XNMs_AttenEdge(nn.Module):
    def __init__(self, dim, ) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, query, scene_graph):
        return edge_attention

class XNMs_Transfer(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        
    def forward(self, scene_graph, node_attention, edge_attention):
        return new_node_attention
    
class XNMs_And(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    
    def forward(self, a_1, a_2):
        return torch.minimum(a_1, a_2)
    
class XNMs_Or(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    def forward(self, a_1, a_2, **kwargs):
        return torch.maximum(a_1, a_2)

class XNMs_Not(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    def forward(self, a):
        return 1 - a
    
class XNMs_Intersect(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.and_module = XNMs_And()
    
    def forward(self, a_1, a_2):
        return self.and_module(a_1, a_2)

class XNMs_Union(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.or_module = XNMs_Or()
    
    def forward(self, a_1, a_2):
        return self.or_module(a_1, a_2)    

class XNMs_Filter(nn.Module):
    def __init__(self, dim, ) -> None:
        super().__init__()
        self.atten_node = XNMs_AttenNode(dim,)
        self.and_module = XNMs_And()
    def forward(self, a_1, query):
        return self.filter_module(a_1, query)

        

class XNMs(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

    def forward(self, visual_feats, text_feats, text_sentence_feats,
                return_loss=False, targets=None):
        pass

        
class StanfordDependencyParsing(nn.Module):
    def __init__(self, 
                 num_vocabs,
                 num_pos_tag_vocab, 
                 num_dependenciy_vocab,
                 
                 num_chosen_words,
                 num_chosen_pos,
                 num_chosen_arc,
                 
                 dim) -> None:
        super().__init__()
        self.vocabs = nn.Embedding(num_vocabs, dim)
        self.tokenizer = None
        
        self.pos_tag_vocabs = nn.Embedding(num_pos_tag_vocab, dim) # keyword
        self.pos_tags = ['PRP', 'VBZ', 'JJ', 'NN']
        self.pos_tag_classifer = nn.Linear(dim, num_pos_tag_vocab)
        
        self.dependency_vocabs = nn.Embedding(num_dependenciy_vocab, dim)
        self.dependencies = ['amod', 'tmod', 'nsubj', 'csubj', 'dojb',]
        self.dependency_classifier = nn.Linear(dim, num_dependenciy_vocab)

        self.actions = ['shift', 
                        *[f'left_arc({s})' for s in self.dependencies],
                        *[f'right_arc({s})' for s in self.dependencies],]
        self.num_actions = 1 + 2 * num_dependenciy_vocab
        self.action_classifier = nn.Linear(dim, self.num_actions)      

        self.word_linear = nn.Linear(dim*num_chosen_words, dim)
        self.pos_linear = nn.Linear(dim*num_chosen_pos, dim)
        self.arc_linear = nn.Linear(dim*num_chosen_arc, dim) 
    
    def forward(self, text, text_gt_parse_tree):
        
        token_ids, words = self.tokenizer(text) # s
        num_tokens = len(token_ids)
        text_feats = self.vocabulary(token_ids) # s c
        
        stack = []
        buffer = token_ids # list[id]
        config = None
        
        number_steps = text_gt_parse_tree.get_number_steps
        
        for seq_idx in range(number_steps):
            
            chosen_words = choose_words(config).flatten() # nw c
            chosen_pos_tages = chho0se_pos(config).flatten(0) # np c
            chosen_arc = choose_arc(config).flatten(0)  # na c
            
            input_feats = self.word_linear(chosen_words) + \
                          self.pos_linear(chosen_pos_tages) + \
                          self.arc_linear(chosen_arc) # c
                          
            action_distribution = self.action_classifier(input_feats) # num_actions
            
            config = update_config(config, action_distribution)

        
class RNN_image_sentence_parser(nn.Module):
    def __init__(self,
                 num_vocabs,
                 dim,
                 
                 num_labels) -> None:
        super().__init__()
        self.hand_craft = None
        self.vocabulary = nn.Embedding(num_vocabs, dim)
        self.tokenizer = None
        
        self.img_class_embed = nn.Linear()
        self.text_pos_embed = nn.Linear()
        
        self.rnn_linear = nn.Linear(2*dim, dim)
        self.rnn_linear2 = nn.Linear(dim, 1)
        
        
    def forward(self, text, image,  img_gt_segment_classes, text_gt_pos_tags, text_gt_parse_tree):
        """
        img_gt_segment_classes: hw class
        
        text_gt_parse_tree: binary tree
        """
        token_ids = self.tokenizer(text) # s
        text_feats = self.vocabulary(token_ids) # s c
        pred_pos_tag = self.text_pos_embed(text_feats) # s pos_tag
        pred_text_sym_adj = same_cls_as_neighbor(pred_pos_tag) # s s
        
        pred_text_parsing_tree = self.rnn(text_feats, pred_text_sym_adj)
        
        
        img_feats = self.hand_craft(image) # hw c
        pred_img_segment_classes = self.img_class_embed(img_feats) # hw c
        pred_img_sym_adj = same_cls_as_neighbor(pred_img_segment_classes) # hw hw
        # 相同class的就是1; which pair is a neighbor in the parsing tree
        # A visual tree is correct if all adjacent segments 
        # that belong to the same class are merged into one super 
        # segment before merges occur with super segments of different classes       
        pred_img_parsing_tree = self.rnn(img_feats, pred_img_sym_adj)
        
        
        loss = tree_loss(img_gt_segment_classes, pred_img_parsing_tree)
        loss += tree_loss(pred_text_parsing_tree, text_gt_parse_tree)
        
    
    def rnn(self, feats, neighborhood_matrix):
        """
        feats: hw c / s c
        neighborhood_matrix: hw hw / s s
        """
        # number of steps for this rnn is fixed to sequence_length,
        # because each step choose a pair, delete two row/columns and add another parent row/column
        
        seq_len = len(neighborhood_matrix)
        
        total_score = 0.
        parse_tree = []
        for seq_idx in range(seq_len):
            all_neighbors = one_as_neighbors(neighborhood_matrix) #list[(i, j)], number of pairs
            # hw c, hw c -> hw hw 2c
            pair_embeds =  torch.cat([feats.unsqueeze(0), feats.unsqueeze(1)], dim=-1) # hw hw 2c
            
            parent_feats = self.rnn_linear(pair_embeds) # hw hw c, used as the parent nonterminal feat 
            scores = self.rnn_linear2(parent_feats) # hw hw 1
            
            
            # 从当前的neighborhood中找到最高score, 防止最高score是非neighbor导致
            neighbors_scores = scores.index(all_neighbors) # number of pairs
            neighbor_with_max_score_idx, score = neighbors_scores.max()
            
            total_score += score 
            i, j = all_neighbors[neighbor_with_max_score_idx]
            parse_tree.append((i, j))
            neighborhood_matrix.update((i, j))
            
            arange_idx = torch.arange(len(feats))
            arange_idx[i] == 0
            arange_idx[j] == 0
            feats = feats[arange_idx]
            feats = torch.cat([feats, parent_feats[i, j]]) # 去掉i, j新增他们的parent
            
            if seq_idx == len(seq_len) - 1:
                assert (len(feats) == 1) and (len(neighborhood_matrix) == 1)

        return parse_tree

        
        
        
        
        

