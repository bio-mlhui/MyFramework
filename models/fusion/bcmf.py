import torch
import torch.nn as nn
from models.layers.position_encoding import build_position_encoding
from detectron2.modeling import META_ARCH_REGISTRY
from einops import repeat, rearrange, reduce
import torch.nn.functional as F
from torch_geometric.data import Data
import dgl
import torch_geometric.utils as tg_util
import networkx as nx
from models.layers.utils import zero_module
# multiscale + text
import numpy as np

class CommonCrossAttentionWeights(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                  dropout=0.0,
                  text_trans='dot',
                  visual_trans='dot'):
        super().__init__()

        assert d_model % nheads == 0
        head_dim = d_model // nheads
        self.scale = head_dim ** -0.5
        self.nheads = nheads
        self.head_dim = head_dim
        self.text_trans = text_trans # dot/add_dot/add/none
        self.visual_trans = visual_trans
        assert text_trans in ['dot', 'add', 'add_dot', 'none']
        assert visual_trans in ['dot', 'add', 'add_dot', 'none'] 

        self.amr_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
        self.amr_v_emb = nn.Linear(d_model, nheads * head_dim, bias=False)

        self.vis_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
        self.vis_v_emb = nn.Linear(d_model, nheads * head_dim, bias=False)
  
        self.visual_out = nn.Sequential( nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.amr_out = nn.Sequential( nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_vanilla(self, amr_feats, amr_pad_mask, visual_feats, visual_pads=None):
        amr_pad_mask = repeat(amr_pad_mask, 'b s -> (b h) s',h=self.nheads)
        # b s c, b s, b s c
        amr_feats_qk = self.amr_emb(amr_feats)
        visual_feats_qk = self.vis_emb(visual_feats)
        amr_feats_qk, visual_feats_qk = map(lambda t: rearrange(t, 'b s (h c) -> (b h) s c', h=self.nheads), (amr_feats_qk, visual_feats_qk))

        amr_values = self.amr_v_emb(amr_feats)
        visual_values = self.vis_v_emb(visual_feats)
        amr_values, visual_values = map(lambda t: rearrange(t, 'b s (h c) -> (b h) s c', h=self.nheads), (amr_values, visual_values))

        common_weights = torch.einsum('bnc,bmc->bnm', amr_feats_qk, visual_feats_qk) * self.scale

        amr_vis_weights = common_weights.clone()
        # amr cross vis:
        if visual_pads is not None:
            visual_pads = repeat(visual_pads, 'b m -> (b h) m',h=self.nheads)
            amr_vis_weights.masked_fill_(visual_pads.unsqueeze(1), torch.finfo(common_weights.dtype).min)
        amr_vis_weights = amr_vis_weights.softmax(-1) # b s_amr s_vis
        amr_feats_2 = amr_vis_weights @ visual_values
        amr_feats_2 = rearrange(amr_feats_2, '(b h) s c -> b s (h c)', h=self.nheads)
        amr_feats_2 = self.amr_out(amr_feats_2)

        # visual cross text:
        vis_text_weights = common_weights.permute(0, 2, 1) # bh s_vis s_amr
        vis_text_weights.masked_fill_(amr_pad_mask.unsqueeze(1), torch.finfo(common_weights.dtype).min)
        vis_text_weights = vis_text_weights.softmax(dim=-1)
        visual_feats_2 = vis_text_weights @ amr_values
        visual_feats_2 = rearrange(visual_feats_2, '(b h) s c -> b s (h c)', h=self.nheads)
        visual_feats_2 = self.visual_out(visual_feats_2)

        ret = []
        if self.text_trans == 'dot':
            ret.append(amr_feats_2 * amr_feats)
        elif self.text_trans == 'add_dot':
            ret.append(amr_feats + (amr_feats * amr_feats_2))
        elif self.text_trans == 'add':
            ret.append(amr_feats_2 + amr_feats)
        elif self.text_trans == 'none':
            ret.append(amr_feats)
        else:
            raise ValueError()
        
        if self.visual_trans == 'dot':
            ret.append(visual_feats * visual_feats_2)
        elif self.visual_trans == 'add_dot':
            ret.append(visual_feats + (visual_feats * visual_feats_2))
        elif self.visual_trans =='add':
            ret.append(visual_feats + visual_feats_2)
        elif self.visual_trans == 'none':
            ret.append(visual_feats)
        else:
            raise ValueError()

        return ret
     
    def forward_mem_eff(self, amr_feats, amr_pad_mask, visual_feats, visual_pads=None):
        # b s c, b s, b s c
        amr_feats_qk = self.amr_emb(amr_feats)
        visual_feats_qk = self.vis_emb(visual_feats)
        amr_feats_qk, visual_feats_qk = map(lambda t: rearrange(t, 'b s (h c) -> b h s c', h=self.nheads), (amr_feats_qk, visual_feats_qk))

        amr_values = self.amr_v_emb(amr_feats)
        visual_values = self.vis_v_emb(visual_feats)
        amr_values, visual_values = map(lambda t: rearrange(t, 'b s (h c) -> b h s c', h=self.nheads), (amr_values, visual_values))

        amr_length = amr_feats.shape[1]
        visual_length = visual_feats.shape[1]
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=True):
            amr_feats_2 = F.scaled_dot_product_attention(query=amr_feats_qk,
                                                         key=visual_feats_qk,
                                                         value=visual_values,
                                                         attn_mask=None)
            visual_feats_2 = F.scaled_dot_product_attention(query=visual_feats_qk,
                                                         key=amr_feats_qk,
                                                         value=amr_values,
                                                         attn_mask=None) 
        amr_feats_2 = rearrange(amr_feats_2, 'b h s c -> b s (h c)')
        visual_feats_2 = rearrange(visual_feats_2, 'b h s c -> b s (h c)')
        amr_feats_2 = self.amr_out(amr_feats_2)
        visual_feats_2 = self.visual_out(visual_feats_2)
        # torch.backends.cuda.enable_mem_efficient_sdp() 
        assert self.text_trans == 'none'
        ret = []
        if self.text_trans == 'dot':
            ret.append(amr_feats_2 * amr_feats)
        elif self.text_trans == 'add_dot':
            ret.append(amr_feats + (amr_feats * amr_feats_2))
        elif self.text_trans == 'add':
            ret.append(amr_feats_2 + amr_feats)
        elif self.text_trans == 'none':
            ret.append(amr_feats)
        else:
            raise ValueError()
        
        if self.visual_trans == 'dot':
            ret.append(visual_feats * visual_feats_2)
        elif self.visual_trans == 'add_dot':
            ret.append(visual_feats + (visual_feats * visual_feats_2))
        elif self.visual_trans =='add':
            ret.append(visual_feats + visual_feats_2)
        elif self.visual_trans == 'none':
            ret.append(visual_feats)
        else:
            raise ValueError()

        return ret

    def forward(self, amr_feats, amr_pad_mask, visual_feats, visual_pads=None):
        if amr_feats.shape[1] > 10000:
            return self.forward_mem_eff(amr_feats, amr_pad_mask, visual_feats, visual_pads)
        else:
            return self.forward_vanilla(amr_feats, amr_pad_mask, visual_feats, visual_pads=None)

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, 
                 dropout=0.0, dot_or_add='dot'):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dot_or_add = dot_or_add
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None):
        tgt2, attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask) # b tgt src, float, 0,1
        if self.dot_or_add == 'add':
            return tgt + tgt2, attn_weights
        else:
            return tgt * tgt2, attn_weights

@META_ARCH_REGISTRY.register()
class BCMF(nn.Module):
    def __init__(self, 
                 configs,) -> None:
        d_model=configs['d_model']
        nheads=configs['nheads']
        dropout=configs['dropout']
        amr_pos_type=configs['amr_pos_type']
        text_pos_type=configs['text_pos_type']
        use_text=configs['use_text']
        text_trans=configs['text_trans']
        visual_trans=configs['visual_trans']
        super().__init__()
        self.d_model = d_model
        self.amr_pos_type = amr_pos_type
        self.text_pos_type = text_pos_type   # 但是transoform amr text控制是否把内部的text用起来
        self.use_text = use_text
        self.transform_amr_text= True
        self.multiscale_text_module = CommonCrossAttentionWeights(d_model=d_model,
                                                                    nheads=nheads,
                                                                        dropout=dropout,
                                                                        text_trans=text_trans,
                                                                        visual_trans=visual_trans)
        if 'refer' in amr_pos_type:
            # 最大深度
            self.depth_amr_pos = nn.Embedding(100, embedding_dim=d_model)
            zero_module(self.depth_amr_pos)
        if 'learned' in text_pos_type:
            self.lrn_text_pos = nn.Embedding(512, embedding_dim=512)
            zero_module(self.lrn_text_pos)
        elif 'sin' in self.text_pos_type:
            self.sin_text_pos = build_position_encoding(position_embedding_name='1d')
        elif 'refer_sin' in self.text_pos_type:
            self.amr_refer_sin_pos = build_position_encoding(position_embedding_name='1d')

    def amr_positional_encoding(self, g: dgl.DGLGraph, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        import scipy as sp
        import dgl
        # Laplacian
        A = g.adj().to_dense().numpy().astype(float)
        N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float).toarray()
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L)
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        return torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

        # Eigenvectors with scipy
        #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        # EigVal, EigVec = sp.sparse.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
        # EigVec = EigVec[:, EigVal.argsort()] # increasing order
        # return torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    def get_amr_poses(self, amrs : Data =None, amr_token_feats=None, amr_token_seg_ids=None):
        # b (v+e)_max c
        enc_dim = amr_token_feats.shape[-1]
        amr_poses = torch.zeros_like(amr_token_feats)
        if 'lap' in self.amr_pos_type:
            for btch_idx, amr in enumerate(amrs):
                num_nodes = amr.num_nodes
                lap_pos = self.amr_positional_encoding(tg_util.to_dgl(amr), pos_enc_dim=1).squeeze(-1)
                amr_poses[btch_idx][:num_nodes] += lap_pos
        if 'refer' in self.amr_pos_type:
            # compute shortest path
            for btch_idx, amr in enumerate(amrs):
                num_nodes = amr.num_nodes
                nx_graph = tg_util.to_networkx(amr)
                all_lengths = dict(nx.single_target_shortest_path_length(nx_graph, 0)) # 0, 1, 2, 3
                pos_lengths = torch.tensor([all_lengths[idx] for idx in range(num_nodes)], dtype=torch.int64).to(amr_token_feats.device)
                pos = self.depth_amr_pos(pos_lengths)
                add_amr_poses = torch.zeros_like(amr_poses)
                add_amr_poses[btch_idx][:num_nodes] = pos
                amr_poses = amr_poses + add_amr_poses
        if 'refer_sin' in self.amr_pos_type:
            for btch_idx, amr in enumerate(amrs):
                num_nodes = amr.num_nodes
                nx_graph = tg_util.to_networkx(amr)
                all_lengths = dict(nx.single_target_shortest_path_length(nx_graph, 0)) # 0, 1, 2, 3
                max_length = max(list(all_lengths.values()))
                foo_pad_mask = amr_token_feats.new_zeros([1, max_length])
                pos = self.amr_refer_sin_pos(foo_pad_mask, hidden_dim=amr_token_feats.shape[-1]).permute(2, 1, 0) # 1 s -> 1 c s
                pos = self.depth_amr_pos(pos_lengths)
                amr_poses[btch_idx][:num_nodes] += pos            
        return amr_poses

    def get_text_poses(self, text_feats=None, text_pad_masks=None):
        # b s c
        text_poses = torch.zeros_like(text_feats)
        if 'sin' in self.text_pos_type:
            sin_pos = self.sin_text_pos(text_pad_masks, hidden_dim=text_feats.shape[-1]).permute(0, 2, 1) # b s c
            text_poses += sin_pos
        return text_poses

    def forward_video_multiscale(self,
                multiscale_feats, multiscale_poses, multiscale_is_flattened,
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None,
                amr_text_add_pos=True):
        B = len(amrs)
        if multiscale_is_flattened:
            # bt \sigmaHE c
            assert len(multiscale_feats.shape) == 3
            BT = multiscale_is_flattened.shape[0]
            multiscale_feats = multiscale_feats + multiscale_poses # bt s c
        else:
            # list[bt c h w]
            BT = multiscale_feats[0].shape[0]
            scale_sizes = [mf.shape[-2:] for mf in multiscale_feats]
            scale_length = [mf.shape[-2:][0] * mf.shape[-2:][1]  for mf in multiscale_feats]
            flattened_multiscale = torch.cat([mf.flatten(2) for mf in multiscale_feats], dim=-1) # bt c \sigmaHW
            flattened_poses = torch.cat([mf.flatten(2) for mf in multiscale_poses], dim=-1)
            flattened_multiscale = flattened_multiscale + flattened_poses
            multiscale_feats = flattened_multiscale.permute(0, 2, 1) # bt s c

        T = BT // B
        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = repeat(amr_token_feats, 'b s c -> (b t) s c', t=T)    
        memory_pad_masks = repeat(amr_token_seg_ids==0, 'b s -> (b t) s',t=T)
        memory_poses = repeat(amr_poses, 'b s c -> (b t) s c', t=T)

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, 
                                             text_pad_masks=text_pad_masks)
            memory = torch.cat([memory, 
                                repeat(text_feats, 'b s c -> (b t) s c', t=T)], dim=1)
            
            memory_pad_masks = torch.cat([memory_pad_masks, 
                                          repeat(text_pad_masks, 'b s -> (b t) s',t=T)], dim=1)
            
            memory_poses = torch.cat([memory_poses, repeat(text_poses, 'b s c -> (b t) s c', t=T)], dim=1)
        
        if amr_text_add_pos:
            memory = memory + memory_poses # bt s c

        if self.transform_amr_text:
            memory, multiscale_feats = self.multiscale_text_module(amr_feats=memory,
                                                                   amr_pad_mask=memory_pad_masks, 
                                                                   visual_feats=multiscale_feats)
        else:
            multiscale_feats = self.multiscale_text_module(tgt=multiscale_feats.permute(1,0,2),
                                                            memory=memory.permute(1,0,2), 
                                                            memory_key_padding_mask=memory_pad_masks,
                                                            pos=None, 
                                                            query_pos=multiscale_feats.permute(1,0,2))[0]
            multiscale_feats = multiscale_feats.permute(1, 0, 2)
        
        if not multiscale_is_flattened:
            multiscale_feats = multiscale_feats.split(scale_length, dim=1) # list[bt hw c]
            multiscale_feats = [rearrange(mf, 'bt  (h w) c -> bt c h w', h=sz[0], w=sz[1]) for mf, sz in zip(multiscale_feats, scale_sizes)]

        if not self.transform_amr_text:
            return multiscale_feats, amr_token_feats, text_feats
        else:
            amr_length = amr_token_feats.shape[1]
            if self.use_text:
                amr_token_feats = rearrange(memory[:, :amr_length], '(b t) s c -> b t s c', b=B, t=T)
                text_feats = rearrange(memory[:, amr_length:], '(b t) s c -> b t s c', b=B, t=T)
                amr_token_feats = amr_token_feats.mean(1)
                text_feats = text_feats.mean(1)
            else:
                amr_token_feats = memory
                amr_token_feats = amr_token_feats.mean(1)
            # bt -> b
            return multiscale_feats, amr_token_feats, text_feats

    def forward_video_frameQuery(self,
                frame_queries, # b t nq c
                time_pad, # b t
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None,
                amr_text_add_pos=True):

        B, T, nq, _ = frame_queries.shape

        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)
        memory = amr_token_feats.clone()    
        memory_pad_masks = amr_token_seg_ids==0
        memory_poses = amr_poses

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, 
                                             text_pad_masks=text_pad_masks)
            memory = torch.cat([memory,text_feats], dim=1)
            memory_pad_masks = torch.cat([memory_pad_masks, text_pad_masks], dim=1)
            memory_poses = torch.cat([memory_poses, text_poses], dim=1)
        
        if amr_text_add_pos:
            memory = memory + memory_poses # b s c
        
        frame_queries = frame_queries.flatten(1, 2) # b tnq c
        frame_queries_time_pad = repeat(time_pad, 'b t -> b (t nq)',nq=nq)
 
        memory, frame_queries = self.multiscale_text_module(amr_feats=memory,
                                                            amr_pad_mask=memory_pad_masks, 
                                                            visual_feats=frame_queries,
                                                            visual_pads=frame_queries_time_pad) # b s c
        frame_queries = rearrange(frame_queries, 'b (t nq) c -> b t nq c',t=T, nq=nq)
        amr_length = amr_token_feats.shape[1]
        if self.use_text:
            amr_token_feats = memory[:, :amr_length]
            text_feats = memory[:, amr_length:]
        else:
            amr_token_feats = memory
        return frame_queries, amr_token_feats, text_feats

    def forward_image_multiscale(self,
                multiscale_feats, multiscale_poses, multiscale_is_flattened,
                amrs, amr_token_feats, amr_token_seg_ids,
                text_feats=None, text_pad_masks=None,
                amr_text_add_pos=True): # 假设multiscale肯定转换
        B = len(amr_token_feats)
        if multiscale_is_flattened:
            # b \sigmaHE c
            assert len(multiscale_feats.shape) == 3
            multiscale_feats = multiscale_feats + multiscale_poses # b s c
            BT = multiscale_feats.shape[0]
        else:
            # list[b c h w]
            scale_sizes = [mf.shape[-2:] for mf in multiscale_feats]
            scale_length = [(mf.shape[-2:][0] * mf.shape[-2:][1]) for mf in multiscale_feats]
            flattened_multiscale = torch.cat([mf.flatten(2) for mf in multiscale_feats], dim=-1) # b c \sigmaHW
            flattened_poses = torch.cat([mf.flatten(2) for mf in multiscale_poses], dim=-1)
            flattened_multiscale = flattened_multiscale + flattened_poses
            multiscale_feats = flattened_multiscale.permute(0, 2, 1) # b s c
            BT = multiscale_feats.shape[0]
        T = BT // B
        amr_poses = self.get_amr_poses(amrs=amrs, amr_token_feats=amr_token_feats, amr_token_seg_ids=amr_token_seg_ids)

        memory = repeat(amr_token_feats, 'b s c -> (b t) s c', t=T)    
        memory_pad_masks = repeat(amr_token_seg_ids==0, 'b s -> (b t) s',t=T)
        memory_poses = repeat(amr_poses, 'b s c -> (b t) s c', t=T)

        if self.use_text:
            assert text_feats is not None
            text_poses = self.get_text_poses(text_feats=text_feats, text_pad_masks=text_pad_masks)
            memory = torch.cat([memory, 
                                repeat(text_feats, 'b s c -> (b t) s c', t=T)], dim=1)
            
            memory_pad_masks = torch.cat([memory_pad_masks, 
                                          repeat(text_pad_masks, 'b s -> (b t) s',t=T)], dim=1)
            
            memory_poses = torch.cat([memory_poses, 
                                      repeat(text_poses, 'b s c -> (b t) s c', t=T)], dim=1)        
        if amr_text_add_pos:
            memory = memory + memory_poses # b s c

        memory, multiscale_feats = self.multiscale_text_module(amr_feats=memory,
                                                                amr_pad_mask=memory_pad_masks, 
                                                                visual_feats=multiscale_feats) # b s c

        if not multiscale_is_flattened:
            multiscale_feats = multiscale_feats.permute(0, 2, 1) # b c s
            multiscale_feats = multiscale_feats.split(scale_length, dim=-1) # list[bt c hw]
            multiscale_feats = [rearrange(mf, 'bt c (h w) -> bt c h w', h=sz[0], w=sz[1]) for mf, sz in zip(multiscale_feats, scale_sizes)]

        amr_length = amr_token_feats.shape[1]
        if self.use_text:
            amr_token_feats = rearrange(memory[:, :amr_length], '(b t) s c -> b t s c', b=B, t=T)
            text_feats = rearrange(memory[:, amr_length:], '(b t) s c -> b t s c', b=B, t=T)
            amr_token_feats = amr_token_feats.mean(1)
            text_feats = text_feats.mean(1)
        else:
            amr_token_feats = rearrange(memory, '(b t) s c -> b t s c', b=B, t=T)
            amr_token_feats = amr_token_feats.mean(1)
        # bt -> b
        return multiscale_feats, amr_token_feats, text_feats

    def forward(self,
                amrs=None, amr_token_feats=None, amr_token_seg_ids=None,
                is_image_multiscale=None,
                is_video_multiscale=None,
                is_video_frame_query=None,
                text_feats=None, text_pad_masks=None,
                amr_text_add_pos=None,
                frame_queries=None, # b t nq c
                multiscale_feats=None, multiscale_poses=None, multiscale_is_flattened=None,
                time_pad=None,
                ):
        if is_video_multiscale:
            return self.forward_video_multiscale(multiscale_feats=multiscale_feats,
                                           multiscale_poses=multiscale_poses,
                                           multiscale_is_flattened=multiscale_is_flattened,
                                           amrs=amrs, 
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            amr_text_add_pos=amr_text_add_pos,
                                            text_pad_masks=text_pad_masks)
        elif is_image_multiscale:
            return self.forward_image_multiscale(multiscale_feats=multiscale_feats,
                                           multiscale_poses=multiscale_poses,
                                           multiscale_is_flattened=multiscale_is_flattened,
                                           amrs=amrs, 
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            amr_text_add_pos=amr_text_add_pos,
                                            text_pad_masks=text_pad_masks)
        elif is_video_frame_query:
            assert time_pad is not None
            return self.forward_video_frameQuery(frame_queries=frame_queries,
                                            amrs=amrs, 
                                            time_pad=time_pad,
                                            amr_token_feats=amr_token_feats, 
                                            amr_token_seg_ids=amr_token_seg_ids, 
                                            text_feats=text_feats, 
                                            amr_text_add_pos=amr_text_add_pos,
                                            text_pad_masks=text_pad_masks)
        pass
