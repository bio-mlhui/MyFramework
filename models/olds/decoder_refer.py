import torch
from einops import rearrange, repeat, reduce

from torch.nn import functional as F

from .layers_unimodal_attention import *

import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .criterion_video import matching_entrypoints
import torch.nn as nn
from util.misc import find_scale_from_multiscales, find_scales_from_multiscales
from torch_geometric.data import Data,Batch
from .transformer import get_norm

def pad_1d_feats(feat_list):
    # list[ni c] -> b nmax c
    feat_len = [len(feat) for feat in feat_list]
    n_max = max(feat_len) 
    batch_size = len(feat_list)
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
    for i in range(batch_size):
        feat_list[i] = F.pad(feat_list[i].clone(), pad=[0, 0, 0, n_max-feat_len[i]])
        pad_mask[i, :feat_len[i]] = False
    feat_list = torch.stack(feat_list, dim=0) # b nmax c
    return feat_list, pad_mask

def pad_token_ids(token_ids, pad_id, device):
    # list[list[int], ni], batch -> T(batch, n_max)
    batch_size = len(token_ids)
    n_max = max(len(t_ids) for t_ids in token_ids)
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        num_tokens = len(token_ids[i])
        token_ids[i] = token_ids[i] + [pad_id] * (n_max - num_tokens)
        pad_mask[i][:num_tokens] = False

    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    return token_ids, pad_mask  
  
_refer_decoder_entrypoints = {}
def register_refer_decoder(fn):
    refer_decoder_name = fn.__name__
    _refer_decoder_entrypoints[refer_decoder_name] = fn
    return fn
def refer_decoder_entrypoints(refer_decoder_name):
    try:
        return _refer_decoder_entrypoints[refer_decoder_name]
    except KeyError as e:
        print(f'Refer Decoder {refer_decoder_name} not found')


class Mask2_Video_Refer(nn.Module):
    def __init__(
        self, # decoder 
        num_queries: int, 
        query_feat: str,  # 是文本 还是学到的  
              
        in_channels,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        concate_text,
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        add_position=False,
   
    ):
        super().__init__()
        # 定义object query pos & feat
        if query_feat == 'learn':
            assert num_queries > 10
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.num_queries = num_queries
        elif query_feat == 'word':
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.num_queries = num_queries
            pass
        elif query_feat == 'sentence':
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.num_queries = num_queries
            pass
        elif query_feat == 'word_noquery':
            assert num_queries == 0
            pass
        else:
            raise ValueError()
        self.query_feat_des = query_feat
        
         
        
        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.concate_text = concate_text # for those where the video and text are not fused, just concate the text and video
        self.add_position = add_position
        
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']

        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]

        token_feats, token_pad_mask, token_sentence \
            = text_args['token_feats'], text_args['token_pad_masks'], text_args['token_sentence_feats']
        token_feats = rearrange(token_feats, 'b s c -> s b c')
        
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        if self.query_feat_des == 'learn':
            query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        elif self.query_feat_des == 'sentence':
            query_feats = repeat(token_sentence, 'b c -> s b c', s=self.num_queries)
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        elif self.query_feat_des == 'word':
            query_feats = token_feats
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)[:len(query_feats)]
        elif self.query_feat_des == 'word_noquery':
            query_feats = token_feats
            query_pos = None
        output = query_feats
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
        predictions_class = [] # list[b nq 2], init -> 32x -> 16x -> 8x
        predictions_mask = [] # list[b t nq H/4 W/4], 
        
        attn_mask_size = size_list[0]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size)
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
                tgt_key_padding_mask=None if self.query_feat_des != 'word' else token_pad_mask, # b n 
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

@register_refer_decoder
def mask2former_video_refer(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_Refer(
                            num_queries=configs['nqueries'],
                            query_feat=configs['query_feat'],
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            dim_feedforward=configs['dff'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            concate_text=configs['concate_text'],
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],)


# 文本那边是linearized graph, 没有matching, 没有position embedding
class Mask2_Video_LinearizedGraph(nn.Module):
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
    
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        graph_which_to_cross,
        compute_layer0_loss=True,
   
    ):
        super().__init__()
        # 定义object query pos & feat

        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()


        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        self.compute_layer0_loss = compute_layer0_loss
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.graph_which_to_cross = graph_which_to_cross
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.text_pos = nn.Embedding(1, hidden_dim)

        
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
       
        predictions_mask = [] # list[b t nq H/4 W/4],        
        linearized_graph_feats, linearized_graph_pad_masks, each_token_lengths, node_token_indexs, edge_token_indexs\
            = text_args['token_feats'], text_args['token_pad_masks'], text_args['each_token_lengths'], text_args['node_token_indexs'], text_args['edge_token_indexs']

        
        attn_mask_size = size_list[0]
        
        
        # 拆开 -> output
        if self.graph_which_to_cross == '整个linearized graph':
            output = linearized_graph_feats.permute(1, 0, 2) # b s c -> s b c
            output_key_padding_mask = linearized_graph_pad_masks # b s
            chosen_index = 2
        else:
            whole_graph = linearized_graph_feats.clone() # b s c
            output = [] # list[nj c] -> b nmax c
            extracted_indexs_by_batch = []
            for batch_idx, (graph_feat, pad_mask, token_lengths, node_indexs, edge_indexs) in \
                    enumerate(zip(whole_graph.clone(), linearized_graph_pad_masks, \
                        each_token_lengths, node_token_indexs, edge_token_indexs)):
                # graph_feat: max c[max] -> si_after c
                # token_lengths: list[int] 加起来等于 si_after
                # node_indexs/edge_indexs: list[int]
                graph_feat = graph_feat[~pad_mask]
                graph_feat = torch.split(graph_feat, token_lengths) # list[T(ni c)], 原先的长度
                if self.graph_which_to_cross == '只有node和edge':
                    extracted_indexs = node_indexs.extend(edge_indexs)
                elif self.graph_which_to_cross == '只有node':
                    extracted_indexs = node_indexs
                else:
                    raise ValueError()
                extracted_indexs_by_batch.append(extracted_indexs)
                # list[T(ni c)] -> ns c
                ect_feats = torch.cat([graph_feat[ect_idx] for ect_idx in extracted_indexs], dim=0)
                output.append(ect_feats)
            # list[nsi, c] -> b nmax c
            output, output_key_padding_mask = Mask2_Video_LinearizedGraph.pad_1d_feats(output)
            output = output.permute(1, 0, 2)
            chosen_index = 0
            
        token_max = output.shape[0]            
        output_pos = repeat(self.text_pos.weight, '1 c -> s b c', b=batch_size, s=token_max)            
        _, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size,
                                                                    return_cls=False)
        
        # t nq h w -> list[t h w] -> b t h w
        outputs_mask = outputs_mask[:, :, chosen_index, ...] # 因为第0个是top的concept
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # b*h n thw
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            thw = srcs[level_index].shape[0]
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # n b c
                memory=srcs[level_index], # thw b c
                memory_mask=attn_mask, # bh n thw
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses[level_index] + \
                    repeat(self.video_pos.weight, '1 c -> thw b c',b=batch_size, thw=thw),  # thw b c
                query_pos=output_pos, # n b c
            )

            # 对号入座
            if self.graph_which_to_cross == '整个linearized graph':
                pass
            else:
                for batch_idx, (cross_feat, pad_mask, extracted_indexs, each_token_length) in \
                    enumerate(zip(output.permute(1,0,2), output_key_padding_mask,extracted_indexs_by_batch, each_token_lengths)):
                    cross_feat = cross_feat[~pad_mask]
                    left_len = 0
                    for idx, ext_idx in enumerate(extracted_indexs):
                        split_len = each_token_length[ext_idx]
                        
                        start_idx = sum(each_token_length[:ext_idx])
                        end_idx = start_idx + split_len
                        
                        whole_graph[batch_idx, start_idx:end_idx] = cross_feat[left_len: (left_len+split_len)]
                        left_len += split_len
                    assert left_len == len(cross_feat)
                
            whole_graph = self.transformer_self_attention_layers[i](
                whole_graph.permute(1,0,2), # n b c
                tgt_mask=None,
                tgt_key_padding_mask=linearized_graph_pad_masks, # b n 
                query_pos=None, # n b c
            )
            
            # # 1是padding
            # self_attention_mask = repeat(linearized_graph_pad_masks, 'b src -> b 1 tgt src',tgt=whole_graph.shape[1])
            # self_attention_mask = self_attention_mask.float() * torch.finfo(whole_graph.dtype).min
            # whole_graph = self.transformer_self_attention_layers[i](
            #     hidden_states = whole_graph.permute(1,0,2), # n b c
            #     attention_mask = self_attention_mask,
            #     layer_head_mask=None
            # )[0]
            whole_graph = whole_graph.permute(1,0,2) # b s c
            # 再拆开
            if self.graph_which_to_cross == '整个linearized graph':
                pass
            else:
                output = [] # list[nj c] -> b nmax c
                for batch_idx, (graph_feat, pad_mask, token_lengths, extracted_indexs) in \
                        enumerate(zip(whole_graph.clone(), linearized_graph_pad_masks, each_token_lengths, extracted_indexs_by_batch)):
                    graph_feat = graph_feat[~pad_mask]
                    graph_feat = torch.split(graph_feat, token_lengths) # list[T(ni c)], 原先的长度
                    # list[T(ni c)] -> ns c
                    ect_feats = torch.cat([graph_feat[ect_idx] for ect_idx in extracted_indexs], dim=0)
                    output.append(ect_feats)
                # list[nsi, c] -> b nmax c
                output, output_key_padding_mask = Mask2_Video_LinearizedGraph.pad_1d_feats(output)
                output = output.permute(1, 0, 2)

            output = self.transformer_ffn_layers[i](
                output # n b c
            )
            
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
            # (b nq 2, real), (b t nq H W, real), bh n thw
            _, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, 
                                                                                   attn_mask_target_size=attn_mask_size,
                                                                                   return_cls=False)
            # t nq h w -> list[t h w] -> b t h w
            outputs_mask = outputs_mask[:, :, chosen_index, ...] # 因为第3个肯定是top的concept
            predictions_mask.append(outputs_mask)

        assert len(predictions_mask) == self.num_layers + 1
        
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
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
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                if (i == 0) and (not self.compute_layer0_loss):
                    continue
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
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
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        if return_cls:
            raise ValueError()
            
        return None, outputs_mask, attn_mask

@register_refer_decoder
def mask2former_video_linearized_graph(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_LinearizedGraph(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            dim_feedforward=configs['dff'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],
                            graph_which_to_cross=configs['graph_which_to_cross'],
                            compute_layer0_loss=configs['compute_layer0_loss'] if 'compute_layer0_loss' in configs else True)    




# 每一层的graph module不共享
class Mask2_Video_Graph_Cross_Multiscale(nn.Module):
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        
        nheads: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        graph_which_to_cross_strategy,
        graph_layer_configs,
        share_graph_layers
   
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        
        from .layers_graph import graph_layer_entrypoints
        from .transformer import _get_clones
        create_graph_layer  = graph_layer_entrypoints(graph_layer_configs.name)
        graph_layer = create_graph_layer(graph_layer_configs, hidden_dim)
        self.share_graph_layers = share_graph_layers
        if share_graph_layers:
            self.transformer_self_attention_layers = graph_layer
        else:
            self.transformer_self_attention_layers = _get_clones(graph_layer, N=self.num_layers) 
        self.transformer_cross_attention_layers = _get_clones(CrossAttentionLayer(
                                                    d_model=hidden_dim,
                                                    nhead=nheads,
                                                    dropout=0.0,
                                                    normalize_before=pre_norm,
                                                ), N=self.num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.graph_which_to_cross_strategy = graph_which_to_cross_strategy
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.text_pos = nn.Embedding(1, hidden_dim)

    def get_cross_attention_index(self, node_identifer: Tensor, edge_identifier, layer_index):
        """
        # 0:variable, 1:concept_val, 2: constant
        # 0: /    1: regular edge 2: constant edge
        Input:
            node_identifier: tensor, num_node
            edge_identifider: tensor, num_edge
        Output:
            list[bool], list[bool]
        """
        device = node_identifer.device
        if self.graph_which_to_cross_strategy == '0层只有concept nodes, 之后所有nodes都用, edge不变':
            if layer_index == 0:
                return torch.arange(len(node_identifer), device=device)[node_identifer == 1], torch.tensor([], device=device).long()
            else:
                return torch.arange(len(node_identifer), device=device), torch.tensor([], device=device).long()
            
        if self.graph_which_to_cross_strategy == '0':
            # 每一层都只有concept nodes
            return torch.arange(len(node_identifer), device=device)[node_identifer == 1], torch.tensor([], device=device).long()
        
        if self.graph_which_to_cross_strategy == '0层只有concept nodes, 之后所有node, edge都用':
            if layer_index == 0:
                return node_identifer == 1, torch.zeros_like(edge_identifier).bool()
            else:
                return torch.ones_like(node_identifer).bool(), torch.ones_like(edge_identifier).bool()
            
        raise ValueError()

        
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
       
        
        # 我们定义referent的mask是最终的top variable的embedding
        predictions_mask = [] # list[b t nq H/4 W/4],
        for batch_idx, (graph, node_identifier, edge_identifier)  in \
            enumerate(zip(text_args['graphs'], text_args['node_identifiers'], text_args['edge_identifiers'])):

            nodes_feats = graph.x.clone() # num_node c
            edges_feats = graph.edge_attr.clone() # num_edge c
            attn_mask_size = size_list[0]        
            # 1 t n h w, 1*head n thw
            all_mask, all_attn_mask = self.forward_prediction_heads(torch.cat([nodes_feats, edges_feats], dim=0).unsqueeze(1),
                                                                    mask_features[[batch_idx]],
                                                                    attn_mask_target_size=attn_mask_size)
            
            # 第1层cross attention之前只有concept nodes能计算到attention mask
            # 其他nodes的attention mask都没有
            cross_node_idx, cross_edge_idx = self.get_cross_attention_index(node_identifier, edge_identifier, layer_index=0)
            # n_cross 1 c
            output = torch.cat([nodes_feats.clone()[cross_node_idx], edges_feats.clone()[cross_edge_idx]], dim=0).unsqueeze(1) 
            output_pos = repeat(self.video_pos.weight, '1 c -> n 1 c',n=output.shape[0])
            # 1*head num_cross size
            attn_mask = all_attn_mask[:, torch.cat([cross_node_idx, cross_edge_idx + len(node_identifier)])]

            layer_output_mask = []
            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels 
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
                thw = srcs[level_index].shape[0]
                output = self.transformer_cross_attention_layers[i](
                    tgt=output,  # num_cross 1 c
                    memory=srcs[level_index][:, [batch_idx]], # thw 1 c
                    memory_mask=attn_mask, # 
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=poses[level_index][:, [batch_idx]] + \
                        repeat(self.video_pos.weight, '1 c -> thw 1 c', thw=thw),  # thw 1 c
                    query_pos=output_pos, # num_cross 1 c
                )
                
                nodes_feats[cross_node_idx] = output[:len(cross_node_idx)].squeeze(1)
                edges_feats[cross_edge_idx] = output[len(cross_node_idx):].squeeze(1)

                if self.share_graph_layers:
                    nodes_feats = self.transformer_self_attention_layers(x=nodes_feats,
                                                                    edge_index=graph.edge_index,
                                                                    edge_attr=edges_feats.clone())
                else:
                    nodes_feats = self.transformer_self_attention_layers[i](x=nodes_feats,
                                                                        edge_index=graph.edge_index,
                                                                        edge_attr=edges_feats.clone())   
                
                attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
                # 1 t n h w, 1*head n thw
                all_mask, all_attn_mask = self.forward_prediction_heads(torch.cat([nodes_feats, edges_feats], dim=0).unsqueeze(1),
                                                                        mask_features[[batch_idx]],
                                                                        attn_mask_target_size=attn_mask_size)
                cross_node_idx, cross_edge_idx = self.get_cross_attention_index(node_identifier, edge_identifier, layer_index=i)
                # n_cross 1 c
                if i != (self.num_layers - 1):
                    output = torch.cat([nodes_feats.clone()[cross_node_idx], edges_feats.clone()[cross_edge_idx]], dim=0).unsqueeze(1) 
                    output_pos = repeat(self.video_pos.weight, '1 c -> n 1 c',n=output.shape[0])
                    # 1*head num_cross size
                    attn_mask = all_attn_mask[:, torch.cat([cross_node_idx, cross_edge_idx + len(node_identifier)])] 
                layer_output_mask.append(all_mask[:, :, 0])  
            
            predictions_mask.append(layer_output_mask) # list[ list[1 t h w], 每一层] batch_size
    
        predictions_mask = list(zip(*predictions_mask)) # list[list[1 t h w], batch] 层数
        # list[b t n h w]
        predictions_mask = [torch.cat(predictions_mask[layer_idx], dim=0) for layer_idx in range(len(predictions_mask))]
        assert len(predictions_mask) == self.num_layers
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
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
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_attn_mask=True,):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        return outputs_mask, attn_mask

class Mask2_Video_Graph_Cross_Multiscale_batched(nn.Module):
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        
        nheads: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        graph_which_to_cross_strategy,
        graph_layer_configs,
        share_graph_layers
   
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        
        from .layers_graph import graph_layer_entrypoints
        from .transformer import _get_clones
        create_graph_layer  = graph_layer_entrypoints(graph_layer_configs.name)
        graph_layer = create_graph_layer(graph_layer_configs, hidden_dim)
        self.share_graph_layers = share_graph_layers
        if share_graph_layers:
            self.transformer_self_attention_layers = graph_layer
        else:
            self.transformer_self_attention_layers = _get_clones(graph_layer, N=self.num_layers) 
        self.transformer_cross_attention_layers = _get_clones(CrossAttentionLayer(
                                                    d_model=hidden_dim,
                                                    nhead=nheads,
                                                    dropout=0.0,
                                                    normalize_before=pre_norm,
                                                ), N=self.num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.graph_which_to_cross_strategy = graph_which_to_cross_strategy
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.text_pos = nn.Embedding(1, hidden_dim)

    def get_cross_attention_index(self, all_seg_ids, layer_index, device):
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
        max_len = all_seg_ids.shape[1]
        if self.graph_which_to_cross_strategy == '0层只有concept nodes, 之后所有nodes都用, edge不变':
            if layer_index == 0:
                return  torch.logical_or(all_seg_ids==1, all_seg_ids==2)
            else:
                return all_seg_ids > 0
            # if layer_index == 0:
            #     return [torch.arange(max_len, device=device)[bool_mask] for bool_mask in torch.logical_or(all_seg_ids==1, all_seg_ids==2)]
            # else:
            #     return [torch.arange(max_len, device=device)[bool_mask] for bool_mask in (all_seg_ids > 0)]
            
        if self.graph_which_to_cross_strategy == '0':
            # 每一层都只有concept nodes
            # 能够防止bias发生, 因为root variable永远都没有看到过video
            return torch.logical_or(all_seg_ids==1, all_seg_ids==2)
        
        if self.graph_which_to_cross_strategy == '0层只有concept nodes, 之后所有node, edge都用':
            if layer_index == 0:
                return node_identifer == 1, torch.zeros_like(edge_identifier).bool()
            else:
                return torch.ones_like(node_identifer).bool(), torch.ones_like(edge_identifier).bool()
            
        raise ValueError()
    
    
    def get_pred(self, prediction_masks, seg_ids):
        """
        b t n h w
        b max [1 2 3 -1 -2 -3 0]
        """
        return torch.stack([p_mask[:, 0] for p_mask in prediction_masks], dim=0)

    def forward(self, video_args, text_args, return_loss=False, targets=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
       
        
        predictions_mask = [] # list[b t nq H/4 W/4],
        
        all_feats, all_seg_ids = text_args['all_feats'].permute(1,0,2), text_args['all_seg_ids']
        
        all_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=all_feats.shape[0], b=batch_size)
        attn_mask_size = size_list[0] 
        
        all_pred_masks, all_attn_mask = self.forward_prediction_heads(all_feats, mask_features, attn_mask_target_size=attn_mask_size)
        # predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
        # b max [True/False]
        who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=0, device=device)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels 
            all_attn_mask[torch.where(all_attn_mask.sum(-1) == all_attn_mask.shape[-1])] = False 
            thw = srcs[level_index].shape[0]
            
            output = self.transformer_cross_attention_layers[i](
                tgt=all_feats.clone(),  # max b c
                memory=srcs[level_index], # thw b c
                memory_mask=all_attn_mask, # 
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses[level_index] + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size),  # thw b c
                query_pos=all_pos, # max b c
            )
            
            all_feats = all_feats * (1 - who_does_cross_attention_mask.permute(1,0).float().unsqueeze(-1)) + output * (who_does_cross_attention_mask.permute(1,0).float().unsqueeze(-1))
            # for batch_idx, cross_mask in enumerate(who_does_cross_attention_mask):
            #     all_feats[cross_mask, batch_idx] = output[cross_mask, batch_idx]
            
            graphs : Batch = copy.deepcopy(text_args['graph'])
            edge_index = graphs.edge_index.to(device)
            nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
            edge_feats = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
            
            if self.share_graph_layers:
                nodes_feats, edge_feats = self.transformer_self_attention_layers(nodes_feats, edge_index, edge_feats.clone())
            else:
                nodes_feats, edge_feats = self.transformer_self_attention_layers[i](nodes_feats, edge_index, edge_feats.clone())
            graphs = graphs.to_data_list()
            num_nodes = [g.num_nodes for g in graphs]
            num_edges = [g.num_edges for g in graphs]
            assert sum(num_nodes) == len(nodes_feats)
            assert sum(num_edges) == len(edge_feats)
            batch_node_feats = torch.split(nodes_feats, num_nodes)
            batch_edge_feats = torch.split(edge_feats, num_edges)
            for batch_idx, seg_ids in enumerate(all_seg_ids):
                all_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
                all_feats[seg_ids < 0, batch_idx] = batch_edge_feats[batch_idx]
                
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

            all_pred_masks, all_attn_mask = self.forward_prediction_heads(all_feats, mask_features, attn_mask_target_size=attn_mask_size)
            
            predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
            
            if i < self.num_layers - 1:
                who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=i+1, device=device)
            
        assert len(predictions_mask) == self.num_layers
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
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
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_attn_mask=True,):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        return outputs_mask, attn_mask

@register_refer_decoder
def mask2_video_graph_cross_multiscale(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_Graph_Cross_Multiscale_batched(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],
                            graph_which_to_cross_strategy=configs['graph_which_to_cross_strategy'],
                            graph_layer_configs=decoder_configs.graph_layer,
                            share_graph_layers=configs['share_graph_layers'] if 'share_graph_layers' in configs else False)



class Mask2_Video_Graph_Cross_Multiscale_batched_notIterative(Mask2_Video_Graph_Cross_Multiscale_batched):
    def __init__(self, in_channels, hidden_dim: int, nheads: int, pre_norm: bool, mask_dim: int, enforce_input_project: bool, dec_layers: int, used_scales, conved_scale, matching_configs, aux_loss, graph_which_to_cross_strategy, graph_layer_configs, share_graph_layers):
        super().__init__(in_channels, hidden_dim, nheads, pre_norm, mask_dim, enforce_input_project, dec_layers, used_scales, conved_scale, matching_configs, aux_loss, graph_which_to_cross_strategy, graph_layer_configs, share_graph_layers)

    def do_inference(self, text_args, all_feats, all_seg_ids, 
                     mask_features,
                     layer_index):
        device = all_feats.device
        graphs : Batch = copy.deepcopy(text_args['graph'])
        edge_index = graphs.edge_index.to(device)
        nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
        edge_feats = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
        
        inference_layer = self.transformer_self_attention_layers if self.share_graph_layers else self.tranformer_self_attn_layers[layer_index]
        
        nodes_feats = inference_layer(nodes_feats, edge_index, edge_feats.clone())

        graphs = graphs.to_data_list()
        num_nodes = [g.num_nodes for g in graphs]
        num_edges = [g.num_edges for g in graphs]
        assert sum(num_nodes) == len(nodes_feats)
        assert sum(num_edges) == len(edge_feats)
        batch_node_feats = torch.split(nodes_feats, num_nodes)
        batch_edge_feats = torch.split(edge_feats, num_edges)
        for batch_idx, seg_ids in enumerate(all_seg_ids):
            all_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            all_feats[seg_ids < 0, batch_idx] = batch_edge_feats[batch_idx] 
        
        # b t n h w
        all_pred_masks = self.get_predication_masks(all_feats, mask_features)
        return all_pred_masks
        
    def get_predication_masks(self, all_feats, mask_features):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(all_feats)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        return outputs_mask
    
    def get_attn_mask(self, pred_masks, attn_mask_target_size):
        bs, nf, *_ = pred_masks.shape
        attn_mask = pred_masks.detach().flatten(0,1) # bt n h w
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
        attn_mask = (attn_mask.sigmoid() < 0.5).bool() 
        return attn_mask
                 
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
       
        predictions_mask = [] # list[b t nq H/4 W/4]
        all_feats, all_seg_ids = text_args['all_feats'].permute(1,0,2), text_args['all_seg_ids']
        all_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=all_feats.shape[0], b=batch_size)
        
        all_pred_masks = self.do_inference(text_args, all_feats.clone(), all_seg_ids, mask_features, layer_index=-1)
        predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
        
        attn_mask_size = size_list[0]
        all_attn_mask = self.get_attn_mask(all_pred_masks, attn_mask_size)
        who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=0, device=device)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels 
            all_attn_mask[torch.where(all_attn_mask.sum(-1) == all_attn_mask.shape[-1])] = False 
            thw = srcs[level_index].shape[0]
            
            output = self.transformer_cross_attention_layers[i](
                tgt=all_feats.clone(),  # max b c
                memory=srcs[level_index], # thw b c
                memory_mask=all_attn_mask, # 
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses[level_index] + repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size),  # thw b c
                query_pos=all_pos, # max b c
            )
            
            for batch_idx, cross_mask in enumerate(who_does_cross_attention_mask):
                all_feats[cross_mask, batch_idx] = output.clone()[cross_mask, batch_idx]
            
            all_pred_masks = self.do_inference(text_args, all_feats.clone(), all_seg_ids, mask_features, layer_index=i)
            predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
            
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
            all_attn_mask = self.get_attn_mask(all_pred_masks, attn_mask_size)
            
            who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=i+1, device=device)
            
        assert len(predictions_mask) == self.num_layers + 1
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }

        if return_loss:
            assert targets is not None
            losses = self.forward_refer_loss(outputs, targets)
            return outputs, losses
        else:
            assert targets is None
            return outputs, None
      


@register_refer_decoder
def mask2_video_graph_cross_multiscale_notIterative(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_Graph_Cross_Multiscale_batched_notIterative(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],
                            graph_which_to_cross_strategy=configs['graph_which_to_cross_strategy'],
                            graph_layer_configs=decoder_configs.graph_layer,
                            share_graph_layers=configs['share_graph_layers'] if 'share_graph_layers' in configs else False)

class ObjectDetector(nn.Module):
    def __init__(
        self, # decoder 
        num_classes,
        in_channels,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        video_feat_proj,
        scale_before_fuse_configs,
        num_queries: int,
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
   
    ):
        super().__init__()
        self.build_feat_proj(video_feat_proj, d_model=hidden_dim)
        from .encoder_multiscale import multiscale_encoder_entrypoints        
        create_scale_before_fusion = multiscale_encoder_entrypoints(scale_before_fuse_configs.name)
        self.scale_before_encoder = create_scale_before_fusion(scale_before_fuse_configs, d_model=hidden_dim)
        assert num_queries > 10
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.num_queries = num_queries

        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs, num_classes=num_classes)

        
        from transformers import BartForConditionalGeneration
        amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.linear_to_textbb = nn.Linear(hidden_dim, amr2text_model.config.d_model, bias=False)
        self.token_classifier = amr2text_model.lm_head
        for p in self.token_classifier.parameters():
            p.requires_grad_(False)
        

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


    def forward(self, video_args,return_loss=False, targets=None):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        multiscales = [scale_feat.clone() for scale_feat in video_args['multiscales']]
        multiscale_masks = [pad_mask.clone() for pad_mask in video_args['multiscale_pad_masks']]
        multiscale_poses = [pos.clone() for pos in video_args['multiscale_poses']]
        multiscale_dec = copy.deepcopy(video_args['multiscale_des'])
        
        if self.video_feat_proj_name != 'no_proj':
            multiscales = self.proj_bakcbone_out(multiscales,)
        
        if self.scale_before_encoder is not None:
            multiscales = self.scale_before_encoder((multiscales, multiscale_masks, 
                                                    multiscale_poses, copy.deepcopy(multiscale_dec)))
        
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]

        batch_size, nf, *_, device = *mask_features.shape, mask_features.device

        query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        output = query_feats
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
        predictions_class = [] # list[b nq k+1], init -> 32x -> 16x -> 8x
        predictions_mask = [] # list[b nq t H/4 W/4], 
        predictions_token = [] # list[b nq num_tokens]
        attn_mask_size = size_list[0]
        outputs_class, outputs_mask, attn_mask, token_logits = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size)
        predictions_token.append(token_logits)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # b*h n thw
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # n b c
                memory=srcs[level_index], # thw b c
                memory_mask=attn_mask, # bh n thw
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses[level_index],  # thw b c
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
            # (b nq 2, real), (b nq t H W, real), bh n thw
            outputs_class, outputs_mask, attn_mask, token_logits = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_token.append(token_logits)

        assert len(predictions_class) == self.num_layers + 1
        outputs = {
            'object_embeds': output.permute(1, 0, 2), # b n c
            'pred_logits': predictions_class[-1], # b nq k+1
            'pred_masks': predictions_mask[-1], # b nq t H W
            'pred_token': predictions_token[-1],
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask, predictions_token)
        } # {'object_embeds': b n c, 'object_box_diff':..}

        if return_loss:
            assert targets is not None
            losses, indices = self.forward_object_loss(outputs, targets)
            losses.update({'matching_results': indices})
            return outputs, losses
        else:
            assert targets is None
            return outputs, None
    
    def forward_object_loss(self, out, targets):
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
        return losses, indices
    
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, predictions_token):
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
            {"pred_logits": a, "pred_masks": b, "pred_token": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_token[:-1])
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_cls=True, return_attn_mask=True, return_box=False):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features).permute(0, 2, 1, 3, 4)  # b n t h w
        token_logits = self.token_classifier(self.linear_to_textbb(decoder_output)) # b n num_tokens
        
        attn_mask = None
        outputs_class = None
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bn t h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bn t h w, real
            attn_mask = repeat(attn_mask, '(b n) t h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads,
                               n=self.num_queries) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        if return_cls:
            outputs_class = self.class_embed(decoder_output)  # b n k+1
            
        return outputs_class, outputs_mask, attn_mask, token_logits

@register_refer_decoder
def object_detector(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return ObjectDetector(
        num_classes=configs['num_classes'],
        in_channels=d_model,
        hidden_dim=d_model,
        nheads=configs['nheads'],
        dim_feedforward=configs['dff'],
        pre_norm=configs['pre_norm'],
        mask_dim=configs['mask_dim'],
        enforce_input_project=configs['enforce_proj_input'],
        # important
        video_feat_proj=decoder_configs.video_feat_proj,
        scale_before_fuse_configs=decoder_configs.scale_encoder,
        num_queries=configs['num_queries'],
        dec_layers=configs['nlayers'],
        used_scales=configs['used_scales'],
        conved_scale=configs['conved_scale'],
        matching_configs=decoder_configs.matching,
        aux_loss=configs['aux_loss'],)

class Referent_Decoder(nn.Module):
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        
        nheads: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        graph_which_to_cross_strategy,
        graph_layer_configs,
        share_graph_layers
   
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        
        from .layers_graph import graph_layer_entrypoints
        from .transformer import _get_clones
        create_graph_layer  = graph_layer_entrypoints(graph_layer_configs.name)
        graph_layer = create_graph_layer(graph_layer_configs, hidden_dim)
        self.share_graph_layers = share_graph_layers
        if share_graph_layers:
            self.transformer_self_attention_layers = graph_layer
        else:
            self.transformer_self_attention_layers = _get_clones(graph_layer, N=self.num_layers) 
        self.transformer_cross_attention_layers = _get_clones(CrossAttentionLayer(
                                                    d_model=hidden_dim,
                                                    nhead=nheads,
                                                    dropout=0.0,
                                                    normalize_before=pre_norm,
                                                ), N=self.num_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.graph_which_to_cross_strategy = graph_which_to_cross_strategy
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.object_pos = nn.Embedding(1, hidden_dim)
        self.text_pos = nn.Embedding(1, hidden_dim)

    def get_cross_attention_index(self, all_seg_ids, layer_index, device):
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
        if self.graph_which_to_cross_strategy == '0层只有concept nodes, 之后所有nodes都用, edge不变':
            if layer_index == 0:
                return  torch.logical_or(all_seg_ids==1, all_seg_ids==2)
            else:
                return all_seg_ids > 0
            # if layer_index == 0:
            #     return [torch.arange(max_len, device=device)[bool_mask] for bool_mask in torch.logical_or(all_seg_ids==1, all_seg_ids==2)]
            # else:
            #     return [torch.arange(max_len, device=device)[bool_mask] for bool_mask in (all_seg_ids > 0)]
        
        if self.graph_which_to_cross_strategy == '0':
            # 每一层都只有concept nodes
            # 能够防止bias发生, 因为root variable永远都没有看到过video
            return torch.logical_or(all_seg_ids==1, all_seg_ids==2)
        
        if self.graph_which_to_cross_strategy == 'edges, nodes都去做cross attention':
            # 适合 arg1的word embedding是 predicate-specific embedding, 也就是说为每个roleset的argx学一个embedding
            return all_seg_ids != 0
        
        raise ValueError()
    
    
    def get_pred(self, prediction_masks, seg_ids):
        """
        b t n h w
        b max [1 2 3 -1 -2 -3 0]
        """
        return torch.stack([p_mask[:, 0] for p_mask in prediction_masks], dim=0)
    
    def build_multimodal_features_along_edge(self, memory, all_seg_ids):
        """
        memory: thw b c
        all_seg_ids: b max, >0, <0, =0
        """
        memory_by_edge = []
        for bt_memory, bt_seg_ids in zip(memory.permute(1,0,2), all_seg_ids):
            num_edges = (bt_seg_ids<0).int().sum()
            memory_by_edge.append(repeat(bt_memory, 'thw c -> E thw c', E=num_edges))
        return torch.cat(memory_by_edge, dim=0) # E_batch thw c
    
    def forward(self, 
                video_features_args,
                object_args,
                text_args,
                return_loss=False,
                targets=None,
                matching_results=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_features_args['multiscales'], video_features_args['multiscale_pad_masks'], \
                video_features_args['multiscale_poses'], video_features_args['multiscale_des']
                
        # n b c
        objects_queries = object_args['object_embeds'].permute(1,0,2)
        num_objects, batch_size, _ = objects_queries.shape
        
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        cross_memories_by_scale = []
        cross_memory_poses_by_scale = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            thw = scale_feats.shape[0]
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            memory = torch.cat([scale_feats, objects_queries], dim=0) # (thw + n) b c
            pos = torch.cat([rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c') +\
                                repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size),
                               repeat(self.object_pos.weight, '1 c -> n b c', n=num_objects, b=batch_size)], dim=0)
            cross_memories_by_scale.append(memory) # thw+n b c
            cross_memory_poses_by_scale.append(pos) # thw+n b c
       
        predictions_mask = [] # list[b t H/4 W/4],
        # Vi+Ei_max b c
        all_feats, all_seg_ids = text_args['all_feats'].permute(1,0,2), text_args['all_seg_ids']
        
        all_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=all_feats.shape[0], b=batch_size)
        attn_mask_size = size_list[0] 
        # b t n h w, b*h n thw+num_objects
        all_pred_masks, all_attn_mask = self.forward_prediction_heads(all_feats, mask_features, 
                                                                      attn_mask_target_size=attn_mask_size,
                                                                      num_objects=num_objects)
        # predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
        # b max [True/False]
        who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=0, device=device)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels 
            all_attn_mask[torch.where(all_attn_mask.sum(-1) == all_attn_mask.shape[-1])] = False 
            
            output = self.transformer_cross_attention_layers[i](
                tgt=all_feats.clone(),  # max b c
                memory=cross_memories_by_scale[level_index], # thw b c
                memory_mask=all_attn_mask, # 
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=cross_memory_poses_by_scale[level_index],  # thw b c
                query_pos=all_pos, # max b c
            )
            
            all_feats = all_feats * (1 - who_does_cross_attention_mask.permute(1,0).float().unsqueeze(-1)) + output * (who_does_cross_attention_mask.permute(1,0).float().unsqueeze(-1))
            # for batch_idx, cross_mask in enumerate(who_does_cross_attention_mask):
            #     all_feats[cross_mask, batch_idx] = output[cross_mask, batch_idx]
            
            graphs : Batch = copy.deepcopy(text_args['graph'])
            edge_index = graphs.edge_index.to(device)
            nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
            edge_feats = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(all_feats.permute(1,0,2), all_seg_ids)], dim=0)
            multimodal_features = self.build_multimodal_features_along_edge(cross_memories_by_scale[level_index],all_seg_ids)
            assert len(multimodal_features) == len(edge_feats)
            if self.share_graph_layers:
                nodes_feats, edge_feats = self.transformer_self_attention_layers(nodes_feats, edge_index, edge_feats.clone(),
                                                                                 multimodal_features=multimodal_features)
            else:
                nodes_feats, edge_feats = self.transformer_self_attention_layers[i](nodes_feats, edge_index, edge_feats.clone(),
                                                                                    multimodal_features=multimodal_features)
            graphs = graphs.to_data_list()
            num_nodes = [g.num_nodes for g in graphs]
            num_edges = [g.num_edges for g in graphs]
            assert sum(num_nodes) == len(nodes_feats)
            assert sum(num_edges) == len(edge_feats)
            batch_node_feats = torch.split(nodes_feats, num_nodes)
            batch_edge_feats = torch.split(edge_feats, num_edges)
            for batch_idx, seg_ids in enumerate(all_seg_ids):
                all_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
                all_feats[seg_ids < 0, batch_idx] = batch_edge_feats[batch_idx]
                
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

            all_pred_masks, all_attn_mask = self.forward_prediction_heads(all_feats, mask_features,
                                                                          attn_mask_target_size=attn_mask_size,
                                                                          num_objects=num_objects)
            
            predictions_mask.append(self.get_pred(all_pred_masks, all_seg_ids))
            
            who_does_cross_attention_mask = self.get_cross_attention_index(all_seg_ids, layer_index=i+1, device=device)
            
        assert len(predictions_mask) == self.num_layers
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }

        if return_loss:
            assert targets is not None and matching_results is not None
            losses = self.forward_refer_loss(outputs, targets, matching_results)
            return outputs, losses
        else:
            assert targets is None
            return outputs, None

    def forward_refer_loss(self, out, targets, matching_results):
        """
        Params:
            targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
            matching_results: list[[0,20,75],[1,2,0]]
        """
        losses = {}
        
        outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_attn_mask=True,
                                 num_objects=None):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        if return_attn_mask:
            assert attn_mask_target_size is not None and num_objects is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
            
            pad_objects_cross = F.pad(attn_mask.float(), pad=[0, num_objects, 0, 0], value=0.).bool() # b*h n thw+num_objects
            
        return outputs_mask, pad_objects_cross

@register_refer_decoder
def referent_decoder(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Referent_Decoder(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],
                            graph_which_to_cross_strategy=configs['graph_which_to_cross_strategy'],
                            graph_layer_configs=decoder_configs.graph_layer,
                            share_graph_layers=configs['share_graph_layers'] if 'share_graph_layers' in configs else False)


class Referent_Decoder_forSequence_LinearizedAMR(nn.Module):
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        
        nheads: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        dim_feedforward,

        # important
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
   
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )        

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.object_pos = nn.Embedding(1, hidden_dim)
        self.text_pos = nn.Embedding(1, hidden_dim)

    def get_pred(self, prediction_masks):
        """
        b t n h w -> 第一个pointer对应的
        """
        return prediction_masks[:, :, 1]

    def forward(self, 
                video_features_args,
                object_args,
                text_args,
                return_loss=False,
                targets=None,
                matching_results=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_features_args['multiscales'], video_features_args['multiscale_pad_masks'], \
                video_features_args['multiscale_poses'], video_features_args['multiscale_des']
                
        # n b c
        objects_queries = object_args['object_embeds'].permute(1,0,2)
        num_objects, batch_size, _ = objects_queries.shape
        
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        cross_memories_by_scale = []
        cross_memory_poses_by_scale = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            thw = scale_feats.shape[0]
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            memory = torch.cat([scale_feats, objects_queries], dim=0) # (thw + n) b c
            pos = torch.cat([rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c') +\
                                repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size),
                               repeat(self.object_pos.weight, '1 c -> n b c', n=num_objects, b=batch_size)], dim=0)
            cross_memories_by_scale.append(memory) # thw+n b c
            cross_memory_poses_by_scale.append(pos) # thw+n b c
       
        predictions_mask = [] # list[b t H/4 W/4],

        
        output, output_pad_mask = text_args['token_feats'].permute(1,0,2), text_args['token_pad_masks']
        output_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=output.shape[0], b=batch_size)
        attn_mask_size = size_list[0] 
        # b t n h w, b*h n thw+num_objects
        pred_masks, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size,num_objects=num_objects)
        predictions_mask.append(self.get_pred(pred_masks))
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels 
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # max b c
                memory=cross_memories_by_scale[level_index], # thw b c
                memory_mask=attn_mask, # 
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=cross_memory_poses_by_scale[level_index],  # thw b c
                query_pos=output_pos, # max b c
            )

            output = self.transformer_self_attention_layers[i](
                output, # n b c
                tgt_mask=None,
                tgt_key_padding_mask=output_pad_mask, # b n 
                query_pos=output_pos, # n b c
            )
            
            output = self.transformer_ffn_layers[i](
                output # n b c
            )
                
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

            pred_masks, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                          attn_mask_target_size=attn_mask_size,
                                                                          num_objects=num_objects)
            
            predictions_mask.append(self.get_pred(pred_masks))
            
        assert len(predictions_mask) == self.num_layers + 1
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }

        if return_loss:
            assert targets is not None and matching_results is not None
            losses = self.forward_refer_loss(outputs, targets, matching_results)
            return outputs, losses
        else:
            assert targets is None
            return outputs, None

    def forward_refer_loss(self, out, targets, matching_results):
        """
        Params:
            targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
            matching_results: list[[0,20,75],[1,2,0]]
        """
        losses = {}
        
        outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
        ]
        
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_attn_mask=True,
                                 num_objects=None):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        if return_attn_mask:
            assert attn_mask_target_size is not None and num_objects is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
            
            pad_objects_cross = F.pad(attn_mask.float(), pad=[0, num_objects, 0, 0], value=0.).bool() # b*h n thw+num_objects
            
        return outputs_mask, pad_objects_cross

@register_refer_decoder
def referent_decoder_forSequence(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Referent_Decoder_forSequence_LinearizedAMR(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            dim_feedforward=configs['dff'],
                            # important
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],)


class Referent_Decoder_forSequenceText(nn.Module):
    def __init__(
        self, # decoder               
        in_channels,
        hidden_dim: int,
        
        nheads: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        dim_feedforward,

        # important
        nqueries,
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
   
    ):
        super().__init__()
        self.query_pos = nn.Embedding(nqueries, hidden_dim)
        self.nqueries = nqueries
        self.hidden_dim = hidden_dim
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )        

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss
        
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)     
        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        
        # cross
        self.video_pos = nn.Embedding(1, hidden_dim)
        self.object_pos = nn.Embedding(1, hidden_dim)
        # self.text_pos = nn.Embedding(1, hidden_dim)

    def forward(self, 
                video_features_args,
                object_args,
                text_args,
                return_loss=False,
                targets=None,
                matching_results=None):
        # b t c h w
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_features_args['multiscales'], video_features_args['multiscale_pad_masks'], \
                video_features_args['multiscale_poses'], video_features_args['multiscale_des']
                
        # n b c
        objects_queries = object_args['object_embeds'].permute(1,0,2)
        num_objects, batch_size, _ = objects_queries.shape
        
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        cross_memories_by_scale = []
        cross_memory_poses_by_scale = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            thw = scale_feats.shape[0]
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            memory = torch.cat([scale_feats, objects_queries], dim=0) # (thw + n) b c
            pos = torch.cat([rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c') +\
                                repeat(self.video_pos.weight, '1 c -> thw b c', thw=thw, b=batch_size),
                               repeat(self.object_pos.weight, '1 c -> n b c', n=num_objects, b=batch_size)], dim=0)
            cross_memories_by_scale.append(memory) # thw+n b c
            cross_memory_poses_by_scale.append(pos) # thw+n b c
       

        token_sentence_feats = text_args['token_sentence_feats'] # b c
        output = repeat(token_sentence_feats, 'b c -> n b c', n=self.nqueries)
        # output_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=output.shape[0], b=batch_size)
        output_pos = repeat(self.query_pos.weight, 'n c -> n b c', b=batch_size)
        
        
        predictions_mask = [] # list[b t H/4 W/4],
        predictions_class = []
        attn_mask_size = size_list[0] 
        # b t n h w, b*h n thw+num_objects
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size,
                                                 num_objects=len(objects_queries))
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels 
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # max b c
                memory=cross_memories_by_scale[level_index], # thw b c
                memory_mask=attn_mask, # 
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=cross_memory_poses_by_scale[level_index],  # thw b c
                query_pos=output_pos, # max b c
            )

            output = self.transformer_self_attention_layers[i](
                output, # n b c
                tgt_mask=None,
                tgt_key_padding_mask=None, # b n 
                query_pos=output_pos, # n b c
            )
            
            output = self.transformer_ffn_layers[i](
                output # n b c
            )
                
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size,
                                                 num_objects=len(objects_queries))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            
        assert len(predictions_mask) == self.num_layers + 1
        outputs = {
            'pred_logits': predictions_class[-1], # b nq 2
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
        }

        if return_loss:
            assert targets is not None and matching_results is not None
            losses = self.forward_refer_loss(outputs, targets, matching_results)
            return outputs, losses
        else:
            assert targets is None
            return outputs, None

    def forward_refer_loss(self, out, targets, matching_results):
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
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]  
    
    def forward_prediction_heads(self, output, mask_features, 
                                 attn_mask_target_size=None, 
                                 return_attn_mask=True,
                                 num_objects=None):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_mask = None
        if return_attn_mask:
            assert attn_mask_target_size is not None and num_objects is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
            
            pad_objects_cross = F.pad(attn_mask.float(), pad=[0, num_objects, 0, 0], value=0.).bool() # b*h n thw+num_objects
            
        outputs_class = self.class_embed(decoder_output) 
           
        return outputs_class, outputs_mask, pad_objects_cross

@register_refer_decoder
def referent_decoder_forSequenceText(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Referent_Decoder_forSequenceText(
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            dim_feedforward=configs['dff'],
                            # important
                            nqueries=configs['nqueries'],
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],)





# 之前的module里没有fusion, 
# self attention 初始化成bart encoder
# video encoder之后没有fusion
# 不同scale 的特征 concate到一块成一个sequence
class Mask2_Video_Self1024(nn.Module):
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


    def build_proj(self, proj_configs, visual_dim, text_dim):
        configs = vars(proj_configs)
        if configs == {}:
            pass
        elif configs['name'] == 'resizer':
            self.text_proj_to_vision = FeatureResizer(
                input_feat_size=text_dim,
                output_feat_size=visual_dim,
                dropout=configs['dropout'],
                do_ln=configs['do_ln']
            )
            self.vision_proj_to_text = FeatureResizer(
                input_feat_size=visual_dim,
                output_feat_size=text_dim,
                dropout=configs['dropout'],
                do_ln=configs['do_ln']
            )
        elif configs['name'] == 'resizer_multilayer':
            self.text_proj_to_vision = FeatureResizer_MultiLayer(
                input_feat_size=text_dim,
                hidden_size=text_dim,
                output_feat_size=visual_dim,
                num_layers=configs['nlayers'],
                dropout=configs['dropout'],
                do_ln=configs['do_ln'],
                activation='relu'
            )
            self.vision_proj_to_text = FeatureResizer_MultiLayer(
                input_feat_size=visual_dim,
                hidden_size=visual_dim,
                output_feat_size=text_dim,
                num_layers=configs['nlayers'],
                dropout=configs['dropout'],
                do_ln=configs['do_ln'],
                activation='relu'
            )
        elif configs['name'] == 'linear':
            self.text_proj_to_vision = nn.Linear(text_dim, visual_dim, bias=False)
            self.vision_proj_to_text = nn.Linear(visual_dim, text_dim, bias=False)
        else:
            raise NotImplementedError()
        
    def __init__(
        self, # decoder               
        in_channels, # 256
        visual_dim: int, # 256
        mask_dim: int, # 256
        nheads: int,
        pre_norm: bool,
        enforce_input_project: bool,

        # important
        used_scales, # 3
        conved_scale, # [1, 4]
        matching_configs, # no matching
        aux_loss,
        graph_which_to_cross, # '只有node'
        proj_configs, # 'linear'
        freeze_self_attention_layers
   
    ):
        super().__init__()
        # 定义object query pos & feat

        self.visual_dim = visual_dim # 256
        self.used_scales = used_scales # 3
        self.conved_scale = conved_scale # [1, 4]
        self.num_feature_levels = len(used_scales) # 3
        self.level_embed = nn.Embedding(self.num_feature_levels, visual_dim) # 3
        
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != self.visual_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        
        from transformers import BartForConditionalGeneration
        amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
        amr2text_model = amr2text_model.get_encoder()
        if freeze_self_attention_layers:
            for p in amr2text_model.parameters():
                p.requires_grad_(False) 
        self.text_dim = amr2text_model.config.d_model

        self.num_layers = len(amr2text_model.layers)

        self.transformer_self_attention_layers = amr2text_model.layers
        
        # cross attention的时候区分哪边是visual/text
        self.transformer_cross_attention_layers = nn.ModuleList() # 256
        self.video_pos = nn.Embedding(1, self.visual_dim)
        self.text_pos = nn.Embedding(1, self.visual_dim)
        for i in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=self.visual_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.build_proj(proj_configs,
                        visual_dim=self.visual_dim,
                        text_dim=self.text_dim)
        self.decoder_norm = nn.LayerNorm(self.visual_dim)
        self.aux_loss = aux_loss
        self.mask_embed = MLP(self.visual_dim, self.visual_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.graph_which_to_cross = graph_which_to_cross
        
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
        srcs = torch.cat(srcs, dim=0) # \sigma thw b c
        poses = torch.cat(poses, dim=0) # \sigma thw b c
        
        predictions_mask = [] # list[b t nq H/4 W/4],        
        linearized_graph_feats, linearized_graph_pad_masks, each_token_lengths, node_token_indexs, edge_token_indexs\
            = text_args['token_feats'], text_args['token_pad_masks'], text_args['each_token_lengths'], text_args['node_token_indexs'], text_args['edge_token_indexs']
        
        # 拆开 -> output
        if self.graph_which_to_cross == '整个linearized graph':
            output = linearized_graph_feats.permute(1, 0, 2) # b s c -> s b c
            output_key_padding_mask = linearized_graph_pad_masks # b s
            chosen_index = 2
        else:
            whole_graph = linearized_graph_feats.clone() # b s c
            output = [] # list[nj c] -> b nmax c
            extracted_indexs_by_batch = []
            for batch_idx, (graph_feat, pad_mask, token_lengths, node_indexs, edge_indexs) in \
                    enumerate(zip(whole_graph.clone(), linearized_graph_pad_masks, \
                        each_token_lengths, node_token_indexs, edge_token_indexs)):
                # graph_feat: max c[max] -> si_after c
                # token_lengths: list[int] 加起来等于 si_after
                # node_indexs/edge_indexs: list[int]
                graph_feat = graph_feat[~pad_mask]
                graph_feat = torch.split(graph_feat, token_lengths) # list[T(ni c)], 原先的长度
                if self.graph_which_to_cross == '只有node和edge':
                    extracted_indexs = node_indexs.extend(edge_indexs)
                elif self.graph_which_to_cross == '只有node':
                    extracted_indexs = node_indexs
                else:
                    raise ValueError()
                extracted_indexs_by_batch.append(extracted_indexs)
                # list[T(ni c)] -> ns c
                ect_feats = torch.cat([graph_feat[ect_idx] for ect_idx in extracted_indexs], dim=0)
                output.append(ect_feats)
            # list[nsi, c] -> b nmax c
            output, output_key_padding_mask = Mask2_Video_LinearizedGraph.pad_1d_feats(output)
            output = self.text_proj_to_vision(output)
            
            output = output.permute(1, 0, 2)
            chosen_index = 0
            
        token_max = output.shape[0]            
        output_pos = repeat(self.text_pos.weight, '1 c -> s b c', b=batch_size, s=token_max)  
        # list[b*head tgt src_i]          
        outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_sizes=size_list)
        attn_mask = torch.cat(attn_mask, dim=-1)
        # t nq h w -> list[t h w] -> b t h w
        outputs_mask = outputs_mask[:, :, chosen_index, ...] # 因为第0个是top的concept
        predictions_mask.append(outputs_mask)
        
        for i in range(self.num_layers):
            # b*h n thw
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            thw = srcs.shape[0]
            output = self.transformer_cross_attention_layers[i](
                tgt=output,  # n b c
                memory=srcs, # thw b c
                memory_mask=attn_mask, # bh n thw
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=poses + repeat(self.video_pos.weight, '1 c -> thw b c',b=batch_size, thw=thw),  # thw b c
                query_pos=output_pos, # n b c
            )

            # 对号入座
            if self.graph_which_to_cross == '整个linearized graph':
                pass
            else:
                output = self.vision_proj_to_text(output)
                for batch_idx, (cross_feat, pad_mask, extracted_indexs, each_token_length) in \
                    enumerate(zip(output.permute(1,0,2), output_key_padding_mask,extracted_indexs_by_batch, each_token_lengths)):
                    cross_feat = cross_feat[~pad_mask]
                    left_len = 0
                    for idx, ext_idx in enumerate(extracted_indexs):
                        split_len = each_token_length[ext_idx]
                        
                        start_idx = sum(each_token_length[:ext_idx])
                        end_idx = start_idx + split_len
                        
                        whole_graph[batch_idx, start_idx:end_idx] = cross_feat[left_len: (left_len+split_len)]
                        left_len += split_len
                    assert left_len == len(cross_feat)

            # 1是padding
            self_attention_mask = repeat(linearized_graph_pad_masks, 'b src -> b 1 tgt src',tgt=whole_graph.shape[1])
            self_attention_mask = self_attention_mask.float() * torch.finfo(whole_graph.dtype).min
            whole_graph = self.transformer_self_attention_layers[i](
                hidden_states = whole_graph, # b n c
                attention_mask = self_attention_mask,
                layer_head_mask=None
            )[0]
            
            # 再拆开
            if self.graph_which_to_cross == '整个linearized graph':
                pass
            else:
                output = [] # list[nj c] -> b nmax c
                for batch_idx, (graph_feat, pad_mask, token_lengths, extracted_indexs) in \
                        enumerate(zip(whole_graph.clone(), linearized_graph_pad_masks, each_token_lengths, extracted_indexs_by_batch)):
                    graph_feat = graph_feat[~pad_mask]
                    graph_feat = torch.split(graph_feat, token_lengths) # list[T(ni c)], 原先的长度
                    # list[T(ni c)] -> ns c
                    ect_feats = torch.cat([graph_feat[ect_idx] for ect_idx in extracted_indexs], dim=0)
                    output.append(ect_feats)
                # list[nsi, c] -> b nmax c
                output, output_key_padding_mask = Mask2_Video_LinearizedGraph.pad_1d_feats(output)
                output = self.text_proj_to_vision(output)
                output = output.permute(1, 0, 2)
                
            # (b nq 2, real), (b t nq H W, real), bh n thw
            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, 
                                                                    attn_mask_target_sizes=size_list)
            attn_mask = torch.cat(attn_mask, dim=-1)
            # t nq h w -> list[t h w] -> b t h w
            outputs_mask = outputs_mask[:, :, chosen_index, ...] # 因为第3个肯定是top的concept
            predictions_mask.append(outputs_mask)

        assert len(predictions_mask) == self.num_layers + 1
        
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
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
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
        ]
        
    def forward_prediction_heads(self, output, 
                                 mask_features, 
                                 attn_mask_target_sizes,
                                 return_attn_mask=True):
        bs, nf, *_= mask_features.shape # b t c h w
        decoder_output = self.decoder_norm(output)  # n b c
        decoder_output = decoder_output.transpose(0, 1)  # b n c
        
        mask_embed = self.mask_embed(decoder_output)  # b n c
        mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
        attn_masks = None
        if return_attn_mask:
            assert attn_mask_target_sizes is not None
            attn_masks = []
            for tgt_size in attn_mask_target_sizes:
                attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
                attn_mask = F.interpolate(attn_mask, size=tgt_size, mode="bilinear", align_corners=False) # bt n h w, real
                attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
                attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
                attn_masks.append(attn_mask)
            
        return outputs_mask, attn_masks

@register_refer_decoder
def mask2former_video_lingraph_Self1024(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_Self1024(
                            in_channels=d_model,
                            visual_dim=d_model,

                            mask_dim=configs['mask_dim'],
                            nheads=configs['nheads'],
                            pre_norm=configs['pre_norm'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],
                            graph_which_to_cross=configs['graph_which_to_cross'],
                            proj_configs=decoder_configs.proj,
                            freeze_self_attention_layers=decoder_configs.freeze_self_attention_layers)    


# 没有 class embedding, forward要知道哪个object query是要输出的
class Mask2_Video_Refer_no_matching(nn.Module):
    def __init__(
        self, # decoder 
        num_queries: int, 
        query_feat: str,  # 是文本 还是学到的  
              
        in_channels,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        concate_text,
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        add_position=False,
   
    ):
        super().__init__()
        # 定义object query pos & feat
        assert 'word' in query_feat
        
        if query_feat == 'word':
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
            self.num_queries = num_queries 
        elif query_feat == 'word_noquery':
            assert num_queries == 0
            pass
        else:
            raise ValueError()
        self.query_feat_des = query_feat
       
        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.concate_text = concate_text # for those where the video and text are not fused, just concate the text and video
        self.add_position = add_position
        
    def forward(self, video_args, text_args, return_loss=False, targets=None):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
            = video_args['multiscales'], video_args['multiscale_pad_masks'], video_args['multiscale_poses'], video_args['multiscale_des']

        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]

        # b
        token_feats, token_pad_mask, token_sentence \
            = text_args['token_feats'], text_args['token_pad_masks'], text_args['token_sentence_feats']
        token_feats = rearrange(token_feats, 'b s c -> s b c')
        
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        if self.query_feat_des == 'word':
            query_feats = token_feats
            query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)[:len(query_feats)]
        elif self.query_feat_des == 'word_noquery':
            query_feats = token_feats
            query_pos = None
        else:
            raise ValueError()
        output = query_feats
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
        predictions_mask = [] # list[b t nq H/4 W/4], 
        
        attn_mask_size = size_list[0]
        _, outputs_mask, attn_mask = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size,
                                                 return_cls=False)
        # t nq h w -> list[t h w] -> b t h w
        outputs_mask = outputs_mask[:, :, 2, ...] # 因为第3个肯定是top的concept
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
                tgt_key_padding_mask=None if self.query_feat_des != 'word' else token_pad_mask, # b n 
                query_pos=query_pos, # n b c
            )
            output = self.transformer_ffn_layers[i](
                output # n b c
            )
            
            attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
            # (b nq 2, real), (b t nq H W, real), bh n thw
            _, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, 
                                                                                   attn_mask_target_size=attn_mask_size,
                                                                                   return_cls=False)
            # t nq h w -> list[t h w] -> b t h w
            outputs_mask = outputs_mask[:, :, 2, ...] # 因为第3个肯定是top的concept
            predictions_mask.append(outputs_mask)

        assert len(predictions_mask) == self.num_layers + 1
        
        outputs = {
            'pred_masks': predictions_mask[-1], # b t nq H W
            'aux_outputs': self._set_aux_loss(predictions_mask)
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
               
        losses = self.criterion(outputs_without_aux, targets)
        if self.aux_loss:
            for i, aux_outputs in enumerate(out['aux_outputs']):
                l_dict_i = self.criterion(aux_outputs, targets)
                
                for k in l_dict_i.keys():
                    assert k in losses
                    losses[k] += l_dict_i[k]  
        return losses
    
    def _set_aux_loss(self, outputs_seg_masks):
        """
        Input:
            - outputs_seg_masks:
                list[T(tb n H W)]
        """
        return [
            {"pred_masks": b}
            for b in outputs_seg_masks[:-1]
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
        if return_attn_mask:
            assert attn_mask_target_size is not None
            attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
            attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
            attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
        if return_cls:
            raise ValueError()
            
        return None, outputs_mask, attn_mask

@register_refer_decoder
def mask2former_video_no_matching(decoder_configs, d_model):
    configs = vars(decoder_configs)
    return Mask2_Video_Refer_no_matching(
                            num_queries=configs['nqueries'],
                            query_feat=configs['query_feat'],
                            in_channels=d_model,
                            hidden_dim=d_model,
                            nheads=configs['nheads'],
                            dim_feedforward=configs['dff'],
                            pre_norm=configs['pre_norm'],
                            mask_dim=configs['mask_dim'],
                            enforce_input_project=configs['enforce_proj_input'],
                            # important
                            concate_text=configs['concate_text'],
                            dec_layers=configs['nlayers'],
                            used_scales=configs['used_scales'],
                            conved_scale=configs['conved_scale'],
                            matching_configs=decoder_configs.matching,
                            aux_loss=configs['aux_loss'],)    



class ObjectGraph_TextGraph_CrossAttention(nn.Module):
    def __init__(self, in_dim, nheads, dropout=0.) -> None:
        super().__init__()
        assert in_dim % nheads == 0
        head_dim = in_dim // nheads
        self.scale = head_dim ** -0.5
        self.nheads = nheads
        self.head_dim = head_dim
        
        self.q_emb = nn.Linear(in_dim, nheads * head_dim, bias=False)
        self.k_emb = nn.Linear(in_dim, nheads * head_dim, bias=False)
        self.v_emb = nn.Linear(in_dim, nheads * head_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(dropout)
        )
        
    def get_weights(self, name, cpu=True):
        if name == 'q':
            out = self.q_emb.weight.detach().data
        elif name == 'k':
            out = self.k_emb.weight.detach().data
        elif name == 'v':
            out = self.v_emb.weight.detach().data
        elif name == 'out':
            out = self.to_out[0].weight.detach().data
            
        if cpu:
            return out.cpu()
        else:
            return out   
        
    def forward(self, query, key, value, key_padding_mask, attn_mask=None):
        """
        query: b n c
        key: b m c (value with pos)
        value: 
        attn_mask: float, b*h n m
        key_padding_mask: b m, 0代表是padding区域
        """
        h = self.nheads
        
        q = self.q_emb(query)
        k = self.k_emb(key)
        v = self.v_emb(value)
        
        q, k, v = map(lambda t: rearrange(t, 'b s (h d) -> (b h) s d', h=h), (q, k, v))
        weights = torch.einsum('bnc,bmc->bnm', q, k) * self.scale
        max_neg_value = - torch.finfo(weights.dtype).max
        
        if key_padding_mask is not None:
            key_padding_mask = repeat(key_padding_mask, 'b m ->  (b h) 1 m', h=h)
            weights.masked_fill_(key_padding_mask, max_neg_value)
        
        if attn_mask is not None:
            weights.masked_fill_(attn_mask, max_neg_value)
        
        attn_weights = weights.softmax(dim=-2)
        
        out = torch.einsum('bnm,bmc->bnc', attn_weights, v)
        out = rearrange(out, '(b h) n c -> b n (h c)',h=h)
        out = self.to_out(out)  # b n d
        return out, weights


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.hidden_dim = dim

        self.W = nn.Parameter(torch.empty(size=(dim, dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # N c
        # N N
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # N N
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1) # 
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # b N c
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) # b N c
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# mask2former负责parse objects
# 针对text parse出的scene graph()
class Mask2_ParseObjects_ExecuteSceneGraph(nn.Module):
    def __init__(
        self, # decoder         
        in_channels,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,

        # important
        concate_text,
        dec_layers: int,
        used_scales,
        conved_scale,
        matching_configs,
        aux_loss,
        add_position=False,
   
    ):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.num_feature_levels = len(used_scales)
        self.used_scales = used_scales
        assert dec_layers % self.num_feature_levels == 0
        self.conved_scale = conved_scale
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                # should be 
                raise NotImplementedError()
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())  
                     
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, 2)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        create_criterion = matching_entrypoints(matching_configs.name)
        self.criterion = create_criterion(matching_configs)
        self.concate_text = concate_text # for those where the video and text are not fused, just concate the text and video
        self.add_position = add_position
        
    def forward(self,
                query_feats,
                query_pos, 
                video_args,
                text_args,
                targets=None, return_loss=False):
        """
        query_feats: n b c
        video: b t c h w
        text: b s c
        """
        # make sure that the video features are fused with the text features before
        
        multiscales, multiscale_masks, multiscale_poses, multiscale_dec = video_args
        used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
        used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
        used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
        conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
        mask_features = multiscales[conved_feat_idx]

        text_graph = text_args
        
        batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
        output = query_feats
        
        srcs = []
        poses = []
        size_list = []
        for i in range(self.num_feature_levels):
            # 32x -> 16x -> 8x
            size_list.append(used_video_feats[i].shape[-2:])
            scale_feats = used_video_feats[i]
            scale_feats = self.input_proj[i](scale_feats) # b t c h w
            scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
            scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
            srcs.append(scale_feats) # thw b c
            poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
        predictions_class = [] # list[b nq 2], init -> 32x -> 16x -> 8x
        predictions_mask = [] # list[b t nq H/4 W/4], 
        predictions_referent_mask = []
        
        attn_mask_size = size_list[0]
        referent_embed = self.bottomup_tree(output, feats=None, text_graph=text_graph)
        # all objects
        outputs_class, outputs_mask, attn_mask, referent_mask = self.forward_prediction_heads(
                                                 output, mask_features, attn_mask_target_size=attn_mask_size,
                                                 referent_embed = referent_embed)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_referent_mask.append(referent_mask)
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # b*h n thw
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
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
            referent_embed = self.bottomup_tree(output, feats=cross_memory, text_graph=text_graph)
            outputs_class, outputs_mask, attn_mask, referent_mask = self.forward_prediction_heads(output, mask_features, 
                                                                                   attn_mask_target_size=attn_mask_size,
                                                                                   referent_embed=referent_embed)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_referent_mask.append(referent_mask)

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
    
    # def bottomup_tree(self, output, feats, text_graph=text_graph):
    #     """
    #     output: b nq c
    #     feats: b t c h w
    #     text_graph
    #     """
    #     pass