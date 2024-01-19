from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
import copy
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat, rearrange

from models.layers.utils import _get_clones
from detectron2.config import configurable
from models.layers.decoder_layers import FFNLayer, SelfAttentionLayer
from detectron2.modeling import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class FrameQuerySwin(nn.Module):
    def __init__(
        self,
        configs,
        fusion_module,
    ):
        super().__init__()
        swin_attn = configs['swin_attn']
        self.nlayers = configs['nlayers']
        self.window_size = configs['window_size']
        d_model = configs['d_model']
        inputs_projs_config = configs['inputs_projs']

        self.attn_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=swin_attn['nheads'],
                                                          dropout=0.0,
                                                          activation=swin_attn['activation'],
                                                          normalize_before=swin_attn['normalize_before'],),
                                      self.nlayers) 
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=swin_attn['dim_feedforward'],
                                               dropout=0.0,
                                               activation=swin_attn['activation'],
                                               normalize_before=swin_attn['normalize_before']),
                                      self.nlayers)  
        self.inputs_proj = META_ARCH_REGISTRY.get(inputs_projs_config['name'])(inputs_projs_config,
                                                                             out_dim=d_model, text_dim=None, )
        self.fusion_module = fusion_module

    def forward(self,
                frame_query_by_layer=None, # list[b t nq c]
                text_inputs=None):
        B, T, nqf, _ = frame_query_by_layer[0].shape
        # list[b t nq c]
        frame_query = frame_query_by_layer[-self.used_layers:] # 训练的时候用后三层, 测试的时候用最后一层
        L = len(frame_query)
        frame_query = torch.stack(frame_query, dim=0).flatten(0, 1) # lb t nq c
        frame_query = self.input_proj(frame_query)

        if self.early_fusion[0] is not None:
            frame_query, text_inputs = self.early_fusion[0](frame_queries=frame_query,
                                                            is_video_frame_query=True,
                                                            text_inputs=text_inputs)
            
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous() # t nq lb c

        if (self.window_size != 0) and self.window_size < T:
            pad = int(ceil(T / self.window_size)) * self.window_size - T # 让window能
            _T = pad + T
            frame_query = F.pad(frame_query, (0,0,0,0,0,0,0,pad))  # t_pad
            enc_mask = frame_query.new_ones(L*B, _T).bool()        # lb t_pad
            enc_mask[:, :T] = False
        else:
            enc_mask = frame_query.new_zeros([L*B, T]).bool()

        # t nq LB c
        frame_query, text_inputs = self.encode_frame_query(frame_query, enc_mask, text_inputs)
        frame_query = frame_query[:T].flatten(0,1)              # tnq LB c

        return frame_query, text_inputs

    def encode_frame_query(self, frame_query, attn_mask, text_inputs):
        """
        _t nq LB c, LB _t
        b s c, b s
        """
        t_pad = frame_query.shape[0]
        if self.window_size == 0 or (self.window_size >= t_pad):
            return_shape = frame_query.shape 
            frame_query = frame_query.flatten(0, 1)
            for i in range(self.nlayers):
                frame_query = self.attn_layers[i](frame_query) # (t nqf) b c
                if self.layer_fusion_modules[i] is not None:
                    frame_query = rearrange(frame_query, '(t nq) b c -> b t nq c',t=t_pad)
                    frame_query, text_inputs = self.layer_fusion_modules[i](frame_queries=frame_query,
                                                                    is_video_frame_query=True,
                                                                    time_pad = attn_mask, # b t
                                                                    text_inputs=text_inputs)
                    frame_query = rearrange(frame_query, 'b t nq c -> (t nq) b c')
                frame_query = self.ffn_layers[i](frame_query)
            frame_query = frame_query.view(return_shape)
            return frame_query, text_inputs
        else:
            T, fQ, LB, C = frame_query.shape
            W = self.window_size
            Nw = T // W
            half_W = int(ceil(W / 2))  # 滑动一半

            # b t -> bN w nq -> bN w_nq
            window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)
            # b t -> b N t' t'
            _attn_mask  = torch.roll(attn_mask, half_W, 1)
            _attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
            _attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
            _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1) # Padding size肯定小于window size
            _attn_mask[:, 0, :half_W, half_W:] = True
            _attn_mask[:, 0, half_W:, :half_W] = True # 开头不关注结尾，结尾不关注开头
            _attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.nlayers):
                if layer_idx % 2 == 0:
                    frame_query = self._window_attn(frame_query, window_mask, layer_idx,)
                else:
                    frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx,)
                if self.layer_fusion_modules[layer_idx] is not None:
                    frame_query = rearrange(frame_query, 't nq b c -> b t nq c')
                    frame_query, text_inputs = self.layer_fusion_modules[layer_idx](frame_queries=frame_query, # b t nq c
                                                                    is_video_frame_query=True,
                                                                    time_pad = attn_mask, # b t
                                                                    text_inputs=text_inputs)
                    frame_query = rearrange(frame_query, 'b t nq c -> t nq b c')
                frame_query = self.ffn_layers[layer_idx](frame_query)
            return frame_query, text_inputs

    def _window_attn(self, frame_query, attn_mask, layer_idx):
        # t nq b c
        T, fQ, LB, C = frame_query.shape
        # bN t'
        W = self.window_size
        Nw = T // W

        # t nq b c -> N t' nq b c -> t' nq b N c -> t'nq bN c
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.attn_layers[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_query

    def _shift_window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.attn_layers[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_query = torch.roll(frame_query, -half_W, 0)

        return frame_query


@META_ARCH_REGISTRY.register()
class FrameQuerySwin_VIS111(nn.Module):
    def __init__(
        self,
        configs,
    ):
        super().__init__()
        swin_attn = configs['swin_attn']
        self.nlayers = configs['nlayers']
        self.window_size = configs['window_size']
        d_model = configs['d_model']
        inputs_projs_config = configs['inputs_projs']
        self.num_heads = swin_attn['nheads']
        self.attn_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=swin_attn['nheads'],
                                                          dropout=0.0,
                                                          activation=swin_attn['activation'],
                                                          normalize_before=swin_attn['normalize_before'],),
                                      self.nlayers) 
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=swin_attn['dim_feedforward'],
                                               dropout=0.0,
                                               activation=swin_attn['activation'],
                                               normalize_before=swin_attn['normalize_before']),
                                      self.nlayers)  
        self.inputs_proj = META_ARCH_REGISTRY.get(inputs_projs_config['name'])(inputs_projs_config,
                                                                             out_dim=d_model, query_dim=None)

    def forward(self,
                frame_queries): # Lb t nqf c
        B, T, nqf, _ = frame_queries.shape
        frame_queries = self.inputs_proj(frame_queries)
        frame_queries = frame_queries.permute(1, 2, 0, 3) # t nq lb c

        if self.window_size != 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T # 让window能
            _T = pad + T
            frame_queries = F.pad(frame_queries, (0,0,0,0,0,0,0,pad))  # t_pad
            enc_mask = frame_queries.new_ones(B, _T).bool()        # lb t_pad
            enc_mask[:, :T] = False
        else:
            enc_mask = frame_queries.new_zeros([B, T]).bool()

        # t nq LB c
        frame_queries = self.encode_frame_queries(frame_queries, enc_mask)
        frame_queries = frame_queries[:T] # t nq lb c
        frame_queries = rearrange(frame_queries, 't nq lb c -> lb t nq c')
        return frame_queries

    def encode_frame_queries(self, frame_queries, attn_mask):
        """
        _t nq LB c, LB _t
        b s c, b s
        """
        t_pad = frame_queries.shape[0]
        if self.window_size == 0 or (self.window_size >= t_pad):
            return_shape = frame_queries.shape 
            frame_queries = frame_queries.flatten(0, 1) # tnq b c
            for i in range(self.nlayers):
                frame_queries = self.attn_layers[i](frame_queries) # (t nqf) b c
                frame_queries = self.ffn_layers[i](frame_queries)
            frame_queries = frame_queries.view(return_shape)
            return frame_queries
        else:
            T, fQ, LB, C = frame_queries.shape
            W = self.window_size
            Nw = T // W
            half_W = int(ceil(W / 2))  # 滑动一半

            # b t -> bN w nq -> bN w_nq
            window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)
            # b t -> b N t' t'
            _attn_mask  = torch.roll(attn_mask, half_W, 1)
            _attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
            _attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
            _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1) # Padding size肯定小于window size
            _attn_mask[:, 0, :half_W, half_W:] = True
            _attn_mask[:, 0, half_W:, :half_W] = True # 开头不关注结尾，结尾不关注开头
            _attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.nlayers):
                if layer_idx % 2 == 0:
                    frame_queries = self._window_attn(frame_queries, window_mask, layer_idx,)
                else:
                    frame_queries = self._shift_window_attn(frame_queries, shift_window_mask, layer_idx,)
                frame_queries = self.ffn_layers[layer_idx](frame_queries)
            return frame_queries

    def _window_attn(self, frame_queries, attn_mask, layer_idx):
        # t nq b c
        T, fQ, LB, C = frame_queries.shape
        # bN t'
        W = self.window_size
        Nw = T // W

        # t nq b c -> N t' nq b c -> t' nq b N c -> t'nq bN c
        frame_queries = frame_queries.view(Nw, W, fQ, LB, C)
        frame_queries = frame_queries.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_queries = self.attn_layers[layer_idx](frame_queries, tgt_key_padding_mask=attn_mask)
        frame_queries = frame_queries.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_queries

    def _shift_window_attn(self, frame_queries, attn_mask, layer_idx):
        T, fQ, LB, C = frame_queries.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_queries = torch.roll(frame_queries, half_W, 0)
        frame_queries = frame_queries.view(Nw, W, fQ, LB, C)
        frame_queries = frame_queries.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_queries = self.attn_layers[layer_idx](frame_queries, tgt_mask=attn_mask)
        frame_queries = frame_queries.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_queries = torch.roll(frame_queries, -half_W, 0)

        return frame_queries




