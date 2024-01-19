# obj queries, temp queries, graph -> module -> b nq, nodes
_reason_module_entrypoints = {}
def register_reason_module(fn):
    reason_module_name = fn.__name__
    _reason_module_entrypoints[reason_module_name] = fn

    return fn
def reason_module_entrypoint(reason_module_name):
    try:
        return _reason_module_entrypoints[reason_module_name]
    except KeyError as e:
        print(f'RVOS moel {reason_module_name} not found')

import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Union
import torch.nn.functional as F
# norm
# 使用memory
from torch_geometric.nn import Aggregation
import dgl
from torch_geometric.nn.aggr import Aggregation
from einops import rearrange, repeat
import math
import networkx as nx
from torch_geometric.data import Batch
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot

def batching_graph(amrs,
                    amr_token_feats,
                    amr_seg_ids,
                    memories, # 
                    memories_pos,
                    text_feats, node_alignments
                    ):
    """
    Args:
        amrs: list[Graph]
        amr_token_feats: b (v+e)max c
        amr_seg_ids: b (v+e)max
        memories: b nq c
        memories_pos: b nq c
        text_feats: b smax c
        node_alignments: list[list[int], si] batch
    Returns:
        _type_: _description_
    """
    device = amr_token_feats.device
    nodes_batch_ids = []
    edges_batch_ids = []
    num_nodes_by_batch = [g.num_nodes for g in amrs]
    for bch_idx, nnode in enumerate(num_nodes_by_batch):
        nodes_batch_ids.extend([bch_idx] * nnode)
    num_edges_by_batch = [g.num_edges for g in amrs]
    for bch_idx, nedge in enumerate(num_edges_by_batch):
        edges_batch_ids.extend([bch_idx] * nedge)
    nodes_batch_ids = torch.tensor(nodes_batch_ids, device=device)
    edges_batch_ids = torch.tensor(edges_batch_ids, device=device)
    # edge_depth = get_edge_depth(amrs) # list[Ei], batch
    batched_amrs = Batch.from_data_list(amrs) # concate
    edge_index = batched_amrs.edge_index.to(device)

    node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
    edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
    node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0)
    edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0)

    if memories_pos != None:
        # V nq c
        node_memories_feats = torch.stack([memories[bid] for bid in nodes_batch_ids], dim=0)
        node_memories_poses = torch.stack([memories_pos[bid] for bid in nodes_batch_ids], dim=0)

        edge_memories_feats = torch.stack([memories[bid] for bid in edges_batch_ids], dim=0)
        edge_memories_poses = torch.stack([memories_pos[bid] for bid in edges_batch_ids], dim=0)

        node_memories = {'feat': node_memories_feats, 'pos': node_memories_poses}
        edge_memories = {'feat': edge_memories_feats, 'pos': edge_memories_poses}
    else:
        # V nq c
        node_memories_feats = torch.stack([memories[bid] for bid in nodes_batch_ids], dim=0)
        _, nq, d_model = node_memories_feats.shape
        if len(edges_batch_ids) == 0:
            edge_memories_feats = torch.zeros([0, nq, d_model]).to(node_memories_feats)
        else:
            edge_memories_feats = torch.stack([memories[bid] for bid in edges_batch_ids], dim=0)
        node_memories = {'feat': node_memories_feats, 'pos': None}
        edge_memories = {'feat': edge_memories_feats, 'pos': None}        

    node_subseqs = [] # list[s c], V
    for btc_text_feat, btc_node_alis in zip(text_feats, node_alignments):
        # s c, list[int]
        for node_ali in btc_node_alis:
            node_subseqs.append(btc_text_feat[:(node_ali+1)])

    node_dsends = [] # list[si c], V
    icgd = list(zip(edge_index[0, :].tolist(), edge_index[1, :].tolist()))
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(list(range(len(nodes_batch_ids))))
    nx_graph.add_edges_from(icgd)
    for node_id in range(len(nodes_batch_ids)):
        # s c, list[int]
        dsends = list(nx.descendants(nx_graph, node_id))
        dsends = [node_id] + dsends
        node_dsends.append(node_feats[dsends])  

    return nodes_batch_ids, edges_batch_ids, \
        node_seg_ids, edges_seg_ids, \
            node_feats, edge_feats,\
            node_memories, edge_memories, edge_index, node_subseqs, node_dsends

def build_batch_along_edge(sequence, num_edges_by_batch):
    """
    sequence: b ..
    num_edges_batch: E
    E ..
    """
    num_dims = sequence.dim() - 1
    batched_sequence = []
    for bt_seq, num_edges in zip(sequence, num_edges_by_batch):
        rep = [num_edges] + [1] * num_dims
        batched_sequence.append(bt_seq.unsqueeze(0).repeat(rep))
    return torch.cat(batched_sequence, dim=0) 

def build_batch_along_node(sequence, num_nodes_by_batch):
    """
    sequence: b ..
    num_edges_batch: V
    V ..
    """
    num_dims = sequence.dim() - 1
    batched_sequence = []
    for bt_seq, num_nodes in zip(sequence, num_nodes_by_batch):
        rep = [num_nodes] + [1] * num_dims
        batched_sequence.append(bt_seq.unsqueeze(0).repeat(rep))

    return torch.cat(batched_sequence, dim=0) 



class Spatial_Temporal_Grounding_v1(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 score_aggr='sum',
                 obj_query_proj=None,
                 temp_query_proj=None,
                 frame_query_proj=None
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.head_dim = d_model // nheads
        self.nheads = nheads
        self.score_aggr = score_aggr

        self.node_linear = nn.Linear(d_model, self.head_dim * self.nheads, bias=False)
        self.edge_linear = nn.Linear(d_model, self.head_dim * self.nheads, bias=False)
        obj_query_proj_name = obj_query_proj.pop('name')
        if  obj_query_proj_name == 'FeatureResizer':
            self.obj_query_proj = FeatureResizer(**obj_query_proj)
        elif obj_query_proj_name == 'linear':
            self.obj_query_proj = nn.Linear(**obj_query_proj)
        elif obj_query_proj_name == 'mlp':
            self.obj_query_proj = MLP(d_model, d_model, d_model, 3)
        else:
            raise ValueError()         
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 2*self.head_dim, self.head_dim])) # 1 h 2c c
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1

        self.ref_2 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, self.head_dim])) # 1 h c c
        self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1

        if temp_query_proj is not None:
            temp_query_proj_name = temp_query_proj.pop('name')
            if  temp_query_proj_name == 'FeatureResizer':
                self.temp_query_proj = FeatureResizer(**temp_query_proj)
            elif temp_query_proj_name == 'linear':
                self.temp_query_proj = nn.Linear(**temp_query_proj)
            elif temp_query_proj_name == 'mlp':
                self.temp_query_proj = MLP(d_model, d_model, d_model, 3)
            else:
                raise ValueError()
        if frame_query_proj is not None:
            frame_query_proj_name = frame_query_proj.pop('name')
            if  frame_query_proj_name == 'FeatureResizer':
                self.frame_query_proj = FeatureResizer(**frame_query_proj)
            elif frame_query_proj_name == 'linear':
                self.frame_query_proj = nn.Linear(**frame_query_proj)
            elif frame_query_proj_name == 'mlp':
                self.frame_query_proj = MLP(d_model, d_model, d_model, 3)
            else:
                raise ValueError()   
        self._reset_parameters()

        # temporal的参数
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        glorot(self.context_2)
        glorot(self.context_1)
        glorot(self.ref_1)
        glorot(self.ref_2)

    def batching_graph(self, 
                       amrs=None,
                        amr_token_feats=None,
                        amr_seg_ids=None
                        ):
        """
        Args:
            amrs: list[Graph]
            amr_token_feats: b (v+e)max c
            amr_seg_ids: b (v+e)max
            memories: b nq c
            memories_pos: b nq c
            text_feats: b smax c
            node_alignments: list[list[int], si] batch
        Returns:
            _type_: _description_
        """
        device = amr_token_feats.device
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        nodes_batch_ids = torch.tensor(nodes_batch_ids, device=device)
        edges_batch_ids = torch.tensor(edges_batch_ids, device=device)
        # edge_depth = get_edge_depth(amrs) # list[Ei], batch
        batched_amrs = Batch.from_data_list(amrs) # concate
        edge_index = batched_amrs.edge_index.to(device)

        # V c
        node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0) # V
        # E c
        if sum(num_edges_by_batch) == 0:
            edge_feats = node_feats.new_zeros([0, node_feats.shape[-1]])
            edges_seg_ids = node_seg_ids.new_ones([0])
        else:
            edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
            edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0) 
 

        return nodes_batch_ids, edges_batch_ids, \
                  node_seg_ids, edges_seg_ids, edge_index, \
                        node_feats, edge_feats 

    def batching_memory(self, tensor, nodes_batch_ids, edges_batch_ids):
        # b ... -> V ... + E ...
        # V nq c
        node_mem = torch.stack([tensor[bid] for bid in nodes_batch_ids], dim=0)
        if len(edges_batch_ids) == 0: 
            edge_mem = torch.zeros([0, *node_mem.shape[1:]]).to(node_mem)
        else: 
            edge_mem = torch.stack([tensor[bid] for bid in edges_batch_ids], dim=0) 

        return node_mem, edge_mem

    def forward_2d(self,obj_queries=None,
                    amrs=None, 
                    amr_token_feats=None,
                    amr_token_seg_ids=None, 
                    node_alignments=None,
                    text_feats=None, text_pad_masks=None):
        batch_size = obj_queries.shape[0]

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        obj_queries = self.obj_query_proj(obj_queries)
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo
        node_obj_queries, edge_obj_queries = self.batching_memory(obj_queries, nodes_batch_ids, edges_batch_ids)

        grounding_score = self.reason_2d(node_feats=node_feats, 
                                      edge_feats=edge_feats,
                                      node_obj_queries=node_obj_queries, 
                                      edge_obj_queries=edge_obj_queries,
                                      edge_index=edge_index,) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq

        return g_score_by_batch

    def reason_2d(self, 
                node_feats=None, 
                edge_feats=None, 
                edge_index=None,
                node_obj_queries=None,
                edge_obj_queries=None):
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype

        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_obj_queries = rearrange(node_obj_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_obj_queries @ self.ref_2) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.mean(1).squeeze(-1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        is_2d=True, is_3d=False,
                                        edge_attr=edge_feats[order_eid, :].clone(), # E hc
                                        node_obj_query=node_obj_queries.flatten(1), # V h_nq_c
                                        )
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
        return scores # V nq

    def message_2d(self,edge_attr=None,  # E hc
                    x_j=None,   # E nq
                    node_obj_query_j=None, # E h_nq_c
                    node_obj_query_i=None):
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = repeat(x_j, 'E nq -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_obj_query_j = rearrange(node_obj_query_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_obj_query_i = rearrange(node_obj_query_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)

        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_obj_query_j
        context_feat_j = context_feat_j.repeat(1,1, nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_obj_query_i], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h 1 c -> E h nq c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        return context_score.mean(1) # E nq   

    def forward_3d(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights=None, # b nq T nqf
                frame_queries_grounding_score=None, # list[Vi T nqf]
                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        batch_size, nq, _, = temporal_queries.shape
        frame_queries = self.frame_query_proj(frame_queries)
        temporal_queries = self.temp_query_proj(temporal_queries)
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo
        if frame_queries_grounding_score is not None:
            node_frame_query_gscore = torch.cat(frame_queries_grounding_score, dim=0) # V T nqf
        else:
            node_frame_query_gscore = None
        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        node_cross_attns, edge_cross_attns = self.batching_memory(cross_attn_weights, nodes_batch_ids, edges_batch_ids)
        node_frame_queries, edge_frame_queries = self.batching_memory(frame_queries, nodes_batch_ids, edges_batch_ids)
        grounding_score = self.reason_3d(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                    node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                    node_cross_attns=node_cross_attns, edge_cross_attns=edge_cross_attns,
                                    node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries,
                                    node_frame_query_gscore=node_frame_query_gscore                                   
                                    ) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
   
    def reason_3d(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_cross_attns=None, edge_cross_attns=None, # V nq T nqf
                node_frame_queries=None, edge_frame_queries=None, # V T nqf c
                node_frame_query_gscore=None # V T nqf
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype
        
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.mean(1).squeeze(-1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        is_2d=False, is_3d=True
                                        )
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
        return scores # V nq
    

    def message_3d(self, 
                edge_attr,  # E hc
                x_j,   # E nq
                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_cross_attn_j=None, edge_cross_attn_i=None,):
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = repeat(x_j, 'E nq -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_temp_query_j = rearrange(node_temp_query_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_temp_query_i = rearrange(node_temp_query_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)

        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_temp_query_j
        context_feat_j = context_feat_j.repeat(1,1, nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h 1 c -> E h nq c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        return context_score.mean(1) # E nq         

    def message(self, 
                is_2d=True,
                is_3d=False,
                edge_attr=None,  # E hc
                x_j=None,   # E nq
                node_obj_query_j=None, # E h_nq_c
                node_obj_query_i=None,

                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_cross_attn_j=None, edge_cross_attn_i=None,

                ) -> Tensor: # E h_nq_c
        if is_2d:
            return self.message_2d(edge_attr=edge_attr, x_j=x_j, node_obj_query_j=node_obj_query_j,
                                   node_obj_query_i=node_obj_query_i)
        if is_3d:
            return self.message_3d(edge_attr=edge_attr,x_j=x_j, 
                                    node_temp_query_j=node_temp_query_j, node_temp_query_i=node_temp_query_i,
                                    node_frame_query_j=node_frame_query_j, node_frame_query_i=node_frame_query_i,
                                    node_cross_attn_j=node_cross_attn_j, edge_cross_attn_i=edge_cross_attn_i)
        
    def aggregate(self, 
                  inputs, # E nq
                  x, # V nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        out = [] # list[nq] 
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(x[tgt_node_idx])
            else:
                self_score = x[tgt_node_idx]
                # Msg+1 nq
                msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
                node_aggr_scores = torch.cat([msgs, self_score.unsqueeze(0)], dim=0)
                out.append(self.aggr_msgs(node_aggr_scores))

        return torch.stack(out, dim=0) # V nq
    
    def aggr_msgs(self, msgs):
        # msg nq
        if self.score_aggr == 'sum':
            return msgs.sum(dim=0)
        elif self.score_aggr == 'min':
            return msgs.min(dim=0)[0]
        else:
            raise ValueError()


    def forward(self,
                is_2d=True,
                is_3d=False,

                obj_queries=None,

                temporal_queries=None,  # b nq c
                frame_queries=None,  # b T nqf c
                cross_attn_weights=None, # b nq T nqf

                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, text_pad_masks=None,
                frame_queries_grounding_score=None):
        
        assert is_2d or is_3d
        assert not(is_2d and is_3d)
        if is_2d:
            return self.forward_2d(obj_queries=obj_queries,
                                   amrs=amrs, 
                                    amr_token_feats=amr_token_feats,
                                    amr_token_seg_ids=amr_token_seg_ids, 
                                    node_alignments=node_alignments,
                                    text_feats=text_feats, 
                                    text_pad_masks=text_pad_masks)
        if is_3d:
            return self.forward_3d(temporal_queries=temporal_queries,  # b nq c
                                    frame_queries=frame_queries,  # b T nqf c
                                    cross_attn_weights=cross_attn_weights, # b nq T nqf
                                    frame_queries_grounding_score=frame_queries_grounding_score, # list[Vi T nqf]
                                    amrs=amrs, 
                                    amr_token_feats=amr_token_feats,
                                    amr_token_seg_ids=amr_token_seg_ids, 
                                    node_alignments=node_alignments,
                                    text_feats=text_feats, 
                                    text_pad_masks=text_pad_masks)



@register_reason_module
def spatial_temporal_grounding_v1(configs):
    return Spatial_Temporal_Grounding_v1(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        obj_query_proj=configs['obj_query_obj'],
                        temp_query_proj=configs['temp_query_proj'] if 'temp_query_proj' in configs else None,
                        frame_query_proj=configs['frame_query_proj'] if 'frame_query_proj' in configs else None)



class Spatial_Temporal_Grounding_v2(Spatial_Temporal_Grounding_v1):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 score_aggr='sum',
                 obj_query_proj=None,
                 temp_query_proj=None,
                 frame_query_proj=None,
                 detach_weight=False,
                 ):
        super().__init__(flow=flow,
                         d_model=d_model,
                         nheads=nheads,
                         score_aggr=score_aggr,
                         obj_query_proj=obj_query_proj,
                         temp_query_proj=temp_query_proj,
                         frame_query_proj=frame_query_proj)
        self.detach_weight = detach_weight

    def forward_3d(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights: torch.Tensor =None, # b nq T nqf
                frame_queries_grounding_score=None, # list[Vi T nqf]
                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        batch_size, nq, _, = temporal_queries.shape
        _, T, nqf,_ = frame_queries.shape
        frame_queries = self.frame_query_proj(frame_queries) # linear
        temporal_queries = self.temp_query_proj(temporal_queries) # linear
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo

        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        # 对每个query选择每帧最大cross weight的query作为它t时刻的query
        # b nq T nqf -> b nq T
        if self.detach_weight:
            cross_attn_weights = cross_attn_weights.detach()
        max_frame_weights, max_frame_idxs = cross_attn_weights.max(dim=-1) 
        frame_queries = frame_queries.unsqueeze(1).repeat(1, nq, 1, 1, 1) # b nq T nqf c
        frame_queries = frame_queries.flatten(0, 2) # b_nq_T nqf c
        max_frame_idxs = max_frame_idxs.flatten() # b_nq_T
        chosen_frame_queries = torch.stack([fq.clone()[cidx] for cidx, fq in zip(max_frame_idxs, frame_queries)], dim=0)
        chosen_frame_queries = rearrange(chosen_frame_queries, '(b nq T) c -> b nq T c', b=batch_size, nq=nq, T=T)
        # b nq T c, b nq c, b nq T 1

        # V nq T
        node_frame_weights, edge_frame_weights = self.batching_memory(max_frame_weights, nodes_batch_ids, edges_batch_ids)
        # V nq T c
        node_frame_queries, edge_frame_queries = self.batching_memory(chosen_frame_queries, nodes_batch_ids, edges_batch_ids)
        grounding_score = self.reason_3d(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                        node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                        node_frame_weights=node_frame_weights, edge_frame_weights=edge_frame_weights,

                                        node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries                                  
                                        ) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
   
    def reason_3d(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_frame_weights=None, edge_frame_weights=None, # V nq T
                node_frame_queries=None, edge_frame_queries=None, # V nq T c
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_frame_weights = repeat(node_frame_weights, 'V nq T -> V h nq T',h=self.nheads)
        node_frame_queries = rearrange(node_frame_queries, 'V nq T (h c) -> V h nq T c',h=self.nheads)

        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2)) # V h nq c * V h 1 c ->  V h nq c
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.squeeze(-1) # V h nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores.flatten(1), # V h_nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        node_frame_query=node_frame_queries.flatten(1), # V h_nq_T_c
                                        is_2d=False, is_3d=True,
                                        node_frame_weight=node_frame_weights.flatten(1) # V h_nq_T
                                        )
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
        return scores.mean(1) # V nq
    

    def message_3d(self, 
                edge_attr,  # E hc
                x_j,   # E nq
                node_temp_query_j=None, node_temp_query_i=None, # V h_nq_c
                node_frame_query_j=None, node_frame_query_i=None, # V h_nq_T_c
                node_frame_weight_j=None, node_frame_weight_i=None,): # V h_nq_T
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_frame_weight_i = rearrange(node_frame_weight_i, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        T = node_frame_weight_i.shape[-1]
        node_frame_query_i = rearrange(node_frame_query_i, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)

        node_frame_weight_j = rearrange(node_frame_weight_j, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq,T=T)


        node_frame_query_j = rearrange(node_frame_query_j, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)
        # E h 1 nq
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        # E h 1 1 nq @ E h T nq c -> E h T 1 c
        context_feat_j = soft_attn_j.unsqueeze(2) @ node_frame_query_j
        # E h 1 nq @ E h nq T -> E h 1 T
        context_weight_j = soft_attn_j @ node_frame_weight_j

        context_feat_j = context_feat_j.repeat(1,1,1, nq, 1) # E h T nq c
        cat_feat = torch.cat([context_feat_j, node_frame_query_i], dim=-1) # E h T nq 2c
        cat_feat = cat_feat.flatten(2,3) # E h T_nq 2c

        # E h T_nq 2c @ 1 h 2c c -> E h T_nq c
        # E h T_nq c * E h 1 c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h T_nq c @ 1 h c 1 -> E h T_nq 1 -> E h T_nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        context_score = rearrange(context_score, 'E h (T nq) -> E h nq T',T=T,nq=nq)

        # E h nq T * E h 1 T -> E h nq T 
        # E h nq T * E h nq T
        context_score = ((node_frame_weight_i * context_weight_j) * context_score).sum(-1) # E h nq

        return context_score.flatten(1) # E h_nq         


    def message(self, 
                is_2d=True,
                is_3d=False,
                edge_attr=None,  # E hc
                x_j=None,   # E nq
                node_obj_query_j=None, # E h_nq_c
                node_obj_query_i=None,

                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_frame_weight_j=None, node_frame_weight_i=None,

                ) -> Tensor: # E h_nq_c
        if is_2d:
            return self.message_2d(edge_attr=edge_attr, x_j=x_j, node_obj_query_j=node_obj_query_j,
                                   node_obj_query_i=node_obj_query_i)
        if is_3d:
            return self.message_3d(edge_attr=edge_attr,x_j=x_j, 
                                    node_temp_query_j=node_temp_query_j, node_temp_query_i=node_temp_query_i,
                                    node_frame_query_j=node_frame_query_j, node_frame_query_i=node_frame_query_i,
                                    node_frame_weight_j=node_frame_weight_j, node_frame_weight_i=node_frame_weight_i)

@register_reason_module
def spatial_temporal_grounding_v2(configs):
    return Spatial_Temporal_Grounding_v2(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        obj_query_proj=configs['obj_query_obj'],
                        temp_query_proj=configs['temp_query_proj'] if 'temp_query_proj' in configs else None,
                        frame_query_proj=configs['frame_query_proj'] if 'frame_query_proj' in configs else None,
                        detach_weight=configs['detach_weight'])


class Spatial_Temporal_Grounding_v3(Spatial_Temporal_Grounding_v1):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 score_aggr='sum',
                 obj_query_proj=None,
                 temp_query_proj=None,
                 frame_query_proj=None,
                 detach_weight=None,
                 only_component_1=False,
                 only_component_2=False,
                 all_zeros = False,
                 ):
        super().__init__(flow=flow,
                         d_model=d_model,
                         nheads=nheads,
                         score_aggr=score_aggr,
                         obj_query_proj=obj_query_proj,
                         temp_query_proj=temp_query_proj,
                         frame_query_proj=frame_query_proj)
        self.detach_weight = detach_weight
        self.only_component_1 = only_component_1
        self.only_component_2 = only_component_2
        self.all_zeros = all_zeros

    def message_3d(self, 
                edge_attr,  # E hc
                x_j,   # E nq
                node_temp_query_j=None, node_temp_query_i=None, # V h_nq_c
                node_frame_query_j=None, node_frame_query_i=None, # V h_nq_T_c
                node_frame_weight_j=None, node_frame_weight_i=None,): # V h_nq_T
        # version = 'v4'

        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_frame_weight_i = rearrange(node_frame_weight_i, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        T = node_frame_weight_i.shape[-1]
        node_frame_query_i = rearrange(node_frame_query_i, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)
        node_temp_query_j = rearrange(node_temp_query_j, 'E (h nq c) -> E h nq c',h=self.nheads, nq=nq)
        # E h 1 nq
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        # E h 1 nq @ E h nq c -> E h 1 c
        context_feat_j = soft_attn_j @ node_temp_query_j

        # E h 1 1 c -> E h T nq c
        context_feat_j = context_feat_j.unsqueeze(2).repeat(1,1,T, nq, 1) # E h T nq c
        cat_feat = torch.cat([context_feat_j, node_frame_query_i], dim=-1) # E h T nq 2c
        cat_feat = cat_feat.flatten(2,3) # E h T_nq 2c

        # E h T_nq 2c @ 1 h 2c c -> E h T_nq c
        # E h T_nq c * E h 1 c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h T_nq c @ 1 h c 1 -> E h T_nq 1 -> E h T_nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        context_score = rearrange(context_score, 'E h (T nq) -> E h nq T',T=T,nq=nq)

        # E h nq T * E h nq T
        context_score = (node_frame_weight_i * context_score).sum(-1) # E h nq

        return context_score.flatten(1) # E h_nq 
            
        # # elif version == 'v4':
        # edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        # x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        # nq = x_j.shape[-1]
        # node_frame_weight_j = rearrange(node_frame_weight_j, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        # T = node_frame_weight_j.shape[-1]
        # node_frame_query_j = rearrange(node_frame_query_j, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)

        # node_temp_query_i = repeat(node_temp_query_i, 'E (h nq c) -> E h T nq c',h=self.nheads, nq=nq, T=T)
        # # E h 1 nq
        # soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        # # E h 1 1 nq @ E h T nq c -> E h T 1 c
        # context_feat_j = soft_attn_j.unsqueeze(2) @ node_frame_query_j

        # # E h 1 1 c -> E h T nq c
        # context_feat_j = context_feat_j.repeat(1,1,1, nq, 1) # E h T nq c
        # cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E h T nq 2c
        # cat_feat = cat_feat.flatten(2,3) # E h T_nq 2c

        # # E h T_nq 2c @ 1 h 2c c -> E h T_nq c
        # # E h T_nq c * E h 1 c
        # context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        # context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # # E h T_nq c @ 1 h c 1 -> E h T_nq 1 -> E h T_nq
        # context_score = (context_score @ self.context_1).squeeze(-1)

        # context_score = rearrange(context_score, 'E h (T nq) -> E h nq T',T=T,nq=nq)

        # # E h nq T * E h nq T
        # context_score = (node_frame_weight_j * context_score).sum(-1) # E h nq

        # return context_score.flatten(1) # E h_nq 


    def forward_3d(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights: torch.Tensor =None, # b nq T nqf
                frame_queries_grounding_score=None, # list[Vi T nqf]
                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        batch_size, nq, _, = temporal_queries.shape
        _, T, nqf,_ = frame_queries.shape
        frame_queries = self.frame_query_proj(frame_queries) # linear
        temporal_queries = self.temp_query_proj(temporal_queries) # linear
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo

        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        # 对每个query选择每帧最大cross weight的query作为它t时刻的query
        # b nq T nqf -> b nq T
        if self.detach_weight:
            cross_attn_weights = cross_attn_weights.detach()
        max_frame_weights, max_frame_idxs = cross_attn_weights.max(dim=-1) 
        frame_queries = frame_queries.unsqueeze(1).repeat(1, nq, 1, 1, 1) # b nq T nqf c
        frame_queries = frame_queries.flatten(0, 2) # b_nq_T nqf c
        max_frame_idxs = max_frame_idxs.flatten() # b_nq_T
        chosen_frame_queries = torch.stack([fq.clone()[cidx] for cidx, fq in zip(max_frame_idxs, frame_queries)], dim=0)
        chosen_frame_queries = rearrange(chosen_frame_queries, '(b nq T) c -> b nq T c', b=batch_size, nq=nq, T=T)
        # b nq T c, b nq c, b nq T 1

        # V nq T
        node_frame_weights, edge_frame_weights = self.batching_memory(max_frame_weights, nodes_batch_ids, edges_batch_ids)
        # V nq T c
        node_frame_queries, edge_frame_queries = self.batching_memory(chosen_frame_queries, nodes_batch_ids, edges_batch_ids)
        grounding_score = self.reason_3d(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                        node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                        node_frame_weights=node_frame_weights, edge_frame_weights=edge_frame_weights,

                                        node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries                                  
                                        ) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
   
    def reason_3d(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_frame_weights=None, edge_frame_weights=None, # V nq T
                node_frame_queries=None, edge_frame_queries=None, # V nq T c
                ):
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_frame_weights = repeat(node_frame_weights, 'V nq T -> V h nq T',h=self.nheads)
        node_frame_queries = rearrange(node_frame_queries, 'V nq T (h c) -> V h nq T c',h=self.nheads)

        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2)) # V h nq c * V h 1 c ->  V h nq c
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.squeeze(-1) # V h nq

        if self.all_zeros:
            return  torch.zeros_like(scores).mean(1)
        if self.only_component_1:
            return scores.mean(1)
        if self.only_component_2:
            scores = torch.zeros_like(scores)
        
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
            return scores.mean(1)
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores.flatten(1), # V h_nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        node_frame_query=node_frame_queries.flatten(1), # V h_nq_T_c
                                        is_2d=False, is_3d=True,
                                        node_frame_weight=node_frame_weights.flatten(1) # V h_nq_T
                                        )
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(1) # V nq

    def message(self, 
                is_2d=True,
                is_3d=False,
                edge_attr=None,  # E hc
                x_j=None,   # E nq
                node_obj_query_j=None, # E h_nq_c
                node_obj_query_i=None,

                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_frame_weight_j=None, node_frame_weight_i=None,

                ) -> Tensor: # E h_nq_c
        if is_2d:
            return self.message_2d(edge_attr=edge_attr, x_j=x_j, node_obj_query_j=node_obj_query_j,
                                   node_obj_query_i=node_obj_query_i)
        if is_3d:
            return self.message_3d(edge_attr=edge_attr,x_j=x_j, 
                                    node_temp_query_j=node_temp_query_j, node_temp_query_i=node_temp_query_i,
                                    node_frame_query_j=node_frame_query_j, node_frame_query_i=node_frame_query_i,
                                    node_frame_weight_j=node_frame_weight_j, node_frame_weight_i=node_frame_weight_i)


@register_reason_module
def spatial_temporal_grounding_v3(configs):
    return Spatial_Temporal_Grounding_v3(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        obj_query_proj=configs['obj_query_obj'],
                        temp_query_proj=configs['temp_query_proj'] if 'temp_query_proj' in configs else None,
                        frame_query_proj=configs['frame_query_proj'] if 'frame_query_proj' in configs else None,
                        only_component_2=configs['only_component_2'] if 'only_component_2' in configs else False,
                        only_component_1=configs['only_component_1'] if 'only_component_1' in configs else False,
                        all_zeros=configs['all_zeros'] if 'all_zeros' in configs else False,
                        detach_weight=configs['detach_weight'])


class Spatial_Temporal_Grounding_v3_sigmoid(Spatial_Temporal_Grounding_v1):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 score_aggr='sum',
                 obj_query_proj=None,
                 temp_query_proj=None,
                 frame_query_proj=None,
                 detach_weight=None,
                 ):
        super().__init__(flow=flow,
                         d_model=d_model,
                         nheads=nheads,
                         score_aggr=score_aggr,
                         obj_query_proj=obj_query_proj,
                         temp_query_proj=temp_query_proj,
                         frame_query_proj=frame_query_proj)
        self.detach_weight = detach_weight

    def message_3d(self, 
                edge_attr,  # E hc
                x_j,   # E nq
                node_temp_query_j=None, node_temp_query_i=None, # V h_nq_c
                node_frame_query_j=None, node_frame_query_i=None, # V h_nq_T_c
                node_frame_weight_j=None, node_frame_weight_i=None,): # V h_nq_T
        # version = 'v4'

        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_frame_weight_i = rearrange(node_frame_weight_i, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        T = node_frame_weight_i.shape[-1]
        node_frame_query_i = rearrange(node_frame_query_i, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)
        node_temp_query_j = rearrange(node_temp_query_j, 'E (h nq c) -> E h nq c',h=self.nheads, nq=nq)
        # E h 1 nq
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        # E h 1 nq @ E h nq c -> E h 1 c
        context_feat_j = soft_attn_j @ node_temp_query_j

        # E h 1 1 c -> E h T nq c
        context_feat_j = context_feat_j.unsqueeze(2).repeat(1,1,T, nq, 1) # E h T nq c
        cat_feat = torch.cat([context_feat_j, node_frame_query_i], dim=-1) # E h T nq 2c
        cat_feat = cat_feat.flatten(2,3) # E h T_nq 2c

        # E h T_nq 2c @ 1 h 2c c -> E h T_nq c
        # E h T_nq c * E h 1 c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h T_nq c @ 1 h c 1 -> E h T_nq 1 -> E h T_nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        context_score = rearrange(context_score, 'E h (T nq) -> E h nq T',T=T,nq=nq)

        # E h nq T * E h nq T
        context_score = (node_frame_weight_i * context_score).sum(-1) # E h nq

        return context_score.flatten(1) # E h_nq 
            
        # # elif version == 'v4':
        # edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        # x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        # nq = x_j.shape[-1]
        # node_frame_weight_j = rearrange(node_frame_weight_j, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        # T = node_frame_weight_j.shape[-1]
        # node_frame_query_j = rearrange(node_frame_query_j, 'E (h nq T c) -> E h T nq c',nq=nq,h=self.nheads,T=T)

        # node_temp_query_i = repeat(node_temp_query_i, 'E (h nq c) -> E h T nq c',h=self.nheads, nq=nq, T=T)
        # # E h 1 nq
        # soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        # # E h 1 1 nq @ E h T nq c -> E h T 1 c
        # context_feat_j = soft_attn_j.unsqueeze(2) @ node_frame_query_j

        # # E h 1 1 c -> E h T nq c
        # context_feat_j = context_feat_j.repeat(1,1,1, nq, 1) # E h T nq c
        # cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E h T nq 2c
        # cat_feat = cat_feat.flatten(2,3) # E h T_nq 2c

        # # E h T_nq 2c @ 1 h 2c c -> E h T_nq c
        # # E h T_nq c * E h 1 c
        # context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        # context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # # E h T_nq c @ 1 h c 1 -> E h T_nq 1 -> E h T_nq
        # context_score = (context_score @ self.context_1).squeeze(-1)

        # context_score = rearrange(context_score, 'E h (T nq) -> E h nq T',T=T,nq=nq)

        # # E h nq T * E h nq T
        # context_score = (node_frame_weight_j * context_score).sum(-1) # E h nq

        # return context_score.flatten(1) # E h_nq 


    def forward_3d(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights: torch.Tensor =None, # b nq T nqf
                frame_queries_grounding_score=None, # list[Vi T nqf]
                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        batch_size, nq, _, = temporal_queries.shape
        _, T, nqf,_ = frame_queries.shape
        frame_queries = self.frame_query_proj(frame_queries) # linear
        temporal_queries = self.temp_query_proj(temporal_queries) # linear
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo

        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        # 对每个query选择每帧最大cross weight的query作为它t时刻的query
        # b nq T nqf -> b nq T
        if self.detach_weight:
            cross_attn_weights = cross_attn_weights.detach()
        max_frame_weights, max_frame_idxs = cross_attn_weights.max(dim=-1) 
        frame_queries = frame_queries.unsqueeze(1).repeat(1, nq, 1, 1, 1) # b nq T nqf c
        frame_queries = frame_queries.flatten(0, 2) # b_nq_T nqf c
        max_frame_idxs = max_frame_idxs.flatten() # b_nq_T
        chosen_frame_queries = torch.stack([fq.clone()[cidx] for cidx, fq in zip(max_frame_idxs, frame_queries)], dim=0)
        chosen_frame_queries = rearrange(chosen_frame_queries, '(b nq T) c -> b nq T c', b=batch_size, nq=nq, T=T)
        # b nq T c, b nq c, b nq T 1

        # V nq T
        node_frame_weights, edge_frame_weights = self.batching_memory(max_frame_weights, nodes_batch_ids, edges_batch_ids)
        # V nq T c
        node_frame_queries, edge_frame_queries = self.batching_memory(chosen_frame_queries, nodes_batch_ids, edges_batch_ids)
        grounding_score = self.reason_3d(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                        node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                        node_frame_weights=node_frame_weights, edge_frame_weights=edge_frame_weights,

                                        node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries                                  
                                        ) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
   
    def reason_3d(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_frame_weights=None, edge_frame_weights=None, # V nq T
                node_frame_queries=None, edge_frame_queries=None, # V nq T c
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_frame_weights = repeat(node_frame_weights, 'V nq T -> V h nq T',h=self.nheads)
        node_frame_queries = rearrange(node_frame_queries, 'V nq T (h c) -> V h nq T c',h=self.nheads)

        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2)) # V h nq c * V h 1 c ->  V h nq c
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.squeeze(-1) # V h nq

        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
            return scores.mean(1)
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores.flatten(1), # V h_nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        node_frame_query=node_frame_queries.flatten(1), # V h_nq_T_c
                                        is_2d=False, is_3d=True,
                                        node_frame_weight=node_frame_weights.flatten(1) # V h_nq_T
                                        )
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(1) # V nq

    def message(self, 
                is_2d=True,
                is_3d=False,
                edge_attr=None,  # E hc
                x_j=None,   # E nq
                node_obj_query_j=None, # E h_nq_c
                node_obj_query_i=None,

                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_frame_weight_j=None, node_frame_weight_i=None,

                ) -> Tensor: # E h_nq_c
        if is_2d:
            return self.message_2d(edge_attr=edge_attr, x_j=x_j, node_obj_query_j=node_obj_query_j,
                                   node_obj_query_i=node_obj_query_i)
        if is_3d:
            return self.message_3d(edge_attr=edge_attr,x_j=x_j, 
                                    node_temp_query_j=node_temp_query_j, node_temp_query_i=node_temp_query_i,
                                    node_frame_query_j=node_frame_query_j, node_frame_query_i=node_frame_query_i,
                                    node_frame_weight_j=node_frame_weight_j, node_frame_weight_i=node_frame_weight_i)


@register_reason_module
def spatial_temporal_grounding_v3_sigmoid(configs):
    return Spatial_Temporal_Grounding_v3_sigmoid(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        obj_query_proj=configs['obj_query_obj'],
                        temp_query_proj=configs['temp_query_proj'] if 'temp_query_proj' in configs else None,
                        frame_query_proj=configs['frame_query_proj'] if 'frame_query_proj' in configs else None,
                        detach_weight=configs['detach_weight'])




class Spatial_Temporal_Grounding_v5(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 flow='source_to_target',
                 score_aggr='sum',
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.score_aggr = score_aggr

        self.node_linear = nn.Linear(d_model, d_model, bias=False)
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.obj_query_proj = MLP(d_model, d_model, d_model, 3)
        self.frame_query_proj = MLP(d_model, d_model, d_model, 3)
        self.temp_query_proj = MLP(d_model, d_model, d_model, 3)

        # 2c -> c
        self.context_2 = MLP(2*d_model, d_model, d_model, 3) 
        self.context_1 = nn.Linear(d_model, 1, bias=False)
        # c -> c
        self.ref_2 = MLP(d_model, d_model, d_model, 3)
        self.ref_1 = nn.Linear(d_model, 1, bias=False)


    def batching_graph(self, 
                       amrs=None,
                        amr_token_feats=None,
                        amr_seg_ids=None
                        ):
        """
        Args:
            amrs: list[Graph]
            amr_token_feats: b (v+e)max c
            amr_seg_ids: b (v+e)max
            memories: b nq c
            memories_pos: b nq c
            text_feats: b smax c
            node_alignments: list[list[int], si] batch
        Returns:
            _type_: _description_
        """
        device = amr_token_feats.device
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        nodes_batch_ids = torch.tensor(nodes_batch_ids, device=device)
        edges_batch_ids = torch.tensor(edges_batch_ids, device=device)
        # edge_depth = get_edge_depth(amrs) # list[Ei], batch
        batched_amrs = Batch.from_data_list(amrs) # concate
        edge_index = batched_amrs.edge_index.to(device)

        # V c
        node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0) # V
        # E c
        if sum(num_edges_by_batch) == 0:
            edge_feats = node_feats.new_zeros([0, node_feats.shape[-1]])
            edges_seg_ids = node_seg_ids.new_ones([0])
        else:
            edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
            edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0) 
 

        return nodes_batch_ids, edges_batch_ids, \
                  node_seg_ids, edges_seg_ids, edge_index, \
                        node_feats, edge_feats 

    def batching_memory(self, tensor, nodes_batch_ids, edges_batch_ids):
        # b ... -> V ... + E ...
        # V nq c
        node_mem = torch.stack([tensor[bid] for bid in nodes_batch_ids], dim=0)
        if len(edges_batch_ids) == 0: 
            edge_mem = torch.zeros([0, *node_mem.shape[1:]]).to(node_mem)
        else: 
            edge_mem = torch.stack([tensor[bid] for bid in edges_batch_ids], dim=0) 

        return node_mem, edge_mem

    def forward_2d(self,obj_queries=None,
                    amrs=None, 
                    amr_token_feats=None,
                    amr_token_seg_ids=None, 
                    node_alignments=None,
                    text_feats=None, text_pad_masks=None):
        batch_size = obj_queries.shape[0]

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        obj_queries = self.obj_query_proj(obj_queries)
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo
        node_obj_queries, edge_obj_queries = self.batching_memory(obj_queries, nodes_batch_ids, edges_batch_ids)

        grounding_score = self.reason_2d(node_feats=node_feats, 
                                      edge_feats=edge_feats,
                                      node_obj_queries=node_obj_queries, 
                                      edge_obj_queries=edge_obj_queries,
                                      edge_index=edge_index,) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq

        return g_score_by_batch

    def reason_2d(self, 
                node_feats=None,  # V c
                edge_feats=None, 
                edge_index=None,
                node_obj_queries=None, # V nq c
                edge_obj_queries=None):
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype

        # V nq c * V 1 c
        ref_score = self.ref_2(node_obj_queries) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = self.ref_1(ref_score) # V nq 1
        scores = scores.squeeze(-1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        is_2d=True, is_3d=False,
                                        edge_attr=edge_feats[order_eid, :].clone(), # E c
                                        node_obj_query=node_obj_queries.flatten(1), # V nq_c
                                        )
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
        return scores # V nq

    def message_2d(self,edge_attr=None,  # E c
                    x_j=None,   # E nq
                    node_obj_query_j=None, # E nq_c
                    node_obj_query_i=None):
        nq = x_j.shape[-1]
        node_obj_query_j = rearrange(node_obj_query_j, 'E (nq c) -> E nq c',nq=nq)
        node_obj_query_i = rearrange(node_obj_query_i, 'E (nq c) -> E nq c',nq=nq)

        # E 1 nq
        soft_attn_j = x_j.softmax(-1).unsqueeze(1)
        # E 1 nq @ E nq c
        context_feat_j = soft_attn_j @ node_obj_query_j
        context_feat_j = context_feat_j.repeat(1, nq, 1) # E nq c
        cat_feat = torch.cat([context_feat_j, node_obj_query_i], dim=-1) # E nq 2c


        # E nq c * E 1 c
        context_score = self.context_2(cat_feat) * (edge_attr.unsqueeze(1))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E nq 1
        context_score = self.context_1(context_score).squeeze(-1)

        return context_score # E nq   

    def forward_3d(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights=None, # b nq T nqf
                frame_queries_grounding_score=None, # list[Vi T nqf]
                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        V, E = node_feats.shape[0], edge_feats.shape[0]
        batch_size, nq, _, = temporal_queries.shape
        frame_queries = self.frame_query_proj(frame_queries)
        temporal_queries = self.temp_query_proj(temporal_queries)
        node_feats = self.node_linear(node_feats)
        if E > 0:
            edge_feats = self.edge_linear(edge_feats) 
        else:
            zero_foo = (self.edge_linear.weight * torch.zeros_like(self.edge_linear.weight)).sum()
            node_feats = node_feats + zero_foo
        if frame_queries_grounding_score is not None:
            node_frame_query_gscore = torch.cat(frame_queries_grounding_score, dim=0) # V T nqf
        else:
            node_frame_query_gscore = None
        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        node_cross_attns, edge_cross_attns = self.batching_memory(cross_attn_weights, nodes_batch_ids, edges_batch_ids)
        node_frame_queries, edge_frame_queries = self.batching_memory(frame_queries, nodes_batch_ids, edges_batch_ids)
        grounding_score = self.reason_3d(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                    node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                    node_cross_attns=node_cross_attns, edge_cross_attns=edge_cross_attns,
                                    node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries,
                                    node_frame_query_gscore=node_frame_query_gscore                                   
                                    ) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
   
    def reason_3d(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_cross_attns=None, edge_cross_attns=None, # V nq T nqf
                node_frame_queries=None, edge_frame_queries=None, # V T nqf c
                node_frame_query_gscore=None # V T nqf
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype
        

        ref_score = self.ref_2(node_temp_queries) * (node_feats.unsqueeze(1))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = self.ref_1(ref_score) # V nq 1
        scores = scores.squeeze(-1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        edge_attr=edge_feats[order_eid, :], # E c
                                        node_temp_query=node_temp_queries.flatten(1), # V nq_c
                                        is_2d=False, is_3d=True
                                        )
        if E == 0:
            scores = scores + self.context_2.sum() * 0. + self.context_1.sum() * 0.
        return scores # V nq
    

    def message_3d(self, 
                edge_attr,  # E c
                x_j,   # E nq
                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_cross_attn_j=None, edge_cross_attn_i=None,):
        nq = x_j.shape[-1]
        node_temp_query_j = rearrange(node_temp_query_j, 'E (nq c) -> E nq c',nq=nq)
        node_temp_query_i = rearrange(node_temp_query_i, 'E (nq c) -> E nq c',nq=nq)

        # E 1 nq
        soft_attn_j = x_j.softmax(-1).unsqueeze(1)
        context_feat_j = soft_attn_j @ node_temp_query_j
        context_feat_j = context_feat_j.repeat(1,nq, 1) # E nq c
        cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E nq 2c

        context_score = self.context_2(cat_feat) * (edge_attr.unsqueeze(1))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)

        context_score = self.context_1(context_score).squeeze(-1)

        return context_score        

    def message(self, 
                is_2d=True,
                is_3d=False,
                edge_attr=None,  # E hc
                x_j=None,   # E nq
                node_obj_query_j=None, # E h_nq_c
                node_obj_query_i=None,

                node_temp_query_j=None, node_temp_query_i=None,
                node_frame_query_j=None, node_frame_query_i=None,
                node_cross_attn_j=None, edge_cross_attn_i=None,

                ) -> Tensor: # E h_nq_c
        if is_2d:
            return self.message_2d(edge_attr=edge_attr, x_j=x_j, node_obj_query_j=node_obj_query_j,
                                   node_obj_query_i=node_obj_query_i)
        if is_3d:
            return self.message_3d(edge_attr=edge_attr,x_j=x_j, 
                                    node_temp_query_j=node_temp_query_j, node_temp_query_i=node_temp_query_i,
                                    node_frame_query_j=node_frame_query_j, node_frame_query_i=node_frame_query_i,
                                    node_cross_attn_j=node_cross_attn_j, edge_cross_attn_i=edge_cross_attn_i)
        
    def aggregate(self, 
                  inputs, # E nq
                  x, # V nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        out = [] # list[nq] 
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(x[tgt_node_idx])
            else:
                self_score = x[tgt_node_idx]
                # Msg+1 nq
                msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
                node_aggr_scores = torch.cat([msgs, self_score.unsqueeze(0)], dim=0)
                out.append(self.aggr_msgs(node_aggr_scores))

        return torch.stack(out, dim=0) # V nq
    
    def aggr_msgs(self, msgs):
        # msg nq
        if self.score_aggr == 'sum':
            return msgs.sum(dim=0)
        elif self.score_aggr == 'min':
            return msgs.min(dim=0)[0]
        else:
            raise ValueError()


    def forward(self,
                is_2d=True,
                is_3d=False,

                obj_queries=None,

                temporal_queries=None,  # b nq c
                frame_queries=None,  # b T nqf c
                cross_attn_weights=None, # b nq T nqf

                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, text_pad_masks=None,
                frame_queries_grounding_score=None):
        
        assert is_2d or is_3d
        assert not(is_2d and is_3d)
        if is_2d:
            return self.forward_2d(obj_queries=obj_queries,
                                   amrs=amrs, 
                                    amr_token_feats=amr_token_feats,
                                    amr_token_seg_ids=amr_token_seg_ids, 
                                    node_alignments=node_alignments,
                                    text_feats=text_feats, 
                                    text_pad_masks=text_pad_masks)
        if is_3d:
            return self.forward_3d(temporal_queries=temporal_queries,  # b nq c
                                    frame_queries=frame_queries,  # b T nqf c
                                    cross_attn_weights=cross_attn_weights, # b nq T nqf
                                    frame_queries_grounding_score=frame_queries_grounding_score, # list[Vi T nqf]
                                    amrs=amrs, 
                                    amr_token_feats=amr_token_feats,
                                    amr_token_seg_ids=amr_token_seg_ids, 
                                    node_alignments=node_alignments,
                                    text_feats=text_feats, 
                                    text_pad_masks=text_pad_masks)



@register_reason_module
def spatial_temporal_grounding_v5(configs):
    return Spatial_Temporal_Grounding_v5(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],)





# 不用任何temporal
# head在feature 上
from .layers_unimodal_attention import FeatureResizer, MLP
class Temporal_Grounding_v1(geo_nn.MessagePassing):

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )


    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 score_aggr='sum',
                 temp_query_proj=None,
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.head_dim = d_model // nheads
        self.nheads = nheads
        self.score_aggr = score_aggr

        self.node_linear = nn.Linear(d_model, self.head_dim * self.nheads, bias=False)
        self.edge_linear = nn.Linear(d_model, self.head_dim * self.nheads, bias=False)
        if temp_query_proj.pop('name') == 'FeatureResizer':
            self.temp_query_proj = FeatureResizer(**temp_query_proj)
        elif temp_query_proj.pop('name') == 'linear':
            self.temp_query_proj = nn.Linear(**temp_query_proj)
        else:
            raise ValueError()
        
        # 1 h 2c c
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 2*self.head_dim, self.head_dim])) 
        # 1 h c 1
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1]))
        # 1 h c c
        self.ref_2 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, self.head_dim])) 
        # 1 h c 1
        self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) 

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        glorot(self.context_2)
        glorot(self.context_1)
        glorot(self.ref_1)
        glorot(self.ref_2)
        
    def batching_graph(self, 
                       amrs=None,
                        amr_token_feats=None,
                        amr_seg_ids=None
                        ):
        """
        Args:
            amrs: list[Graph]
            amr_token_feats: b (v+e)max c
            amr_seg_ids: b (v+e)max
            memories: b nq c
            memories_pos: b nq c
            text_feats: b smax c
            node_alignments: list[list[int], si] batch
        Returns:
            _type_: _description_
        """
        device = amr_token_feats.device
        nodes_batch_ids = []
        edges_batch_ids = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            nodes_batch_ids.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            edges_batch_ids.extend([bch_idx] * nedge)
        nodes_batch_ids = torch.tensor(nodes_batch_ids, device=device)
        edges_batch_ids = torch.tensor(edges_batch_ids, device=device)
        # edge_depth = get_edge_depth(amrs) # list[Ei], batch
        batched_amrs = Batch.from_data_list(amrs) # concate
        edge_index = batched_amrs.edge_index.to(device)

        node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
        node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0)
        edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0)
 

        return nodes_batch_ids, edges_batch_ids, \
            node_seg_ids, edges_seg_ids, \
                node_feats, edge_feats, edge_index

    def batching_memory(self, tensor, nodes_batch_ids, edges_batch_ids):
        # b ... -> V ... + E ...
        # V nq c
        node_mem = torch.stack([tensor[bid] for bid in nodes_batch_ids], dim=0)
        if len(edges_batch_ids) == 0: 
            edge_mem = torch.zeros([0, *node_mem.shape[1:]]).to(node_mem)
        else: 
            edge_mem = torch.stack([tensor[bid] for bid in edges_batch_ids], dim=0) 

        return node_mem, edge_mem

    def forward(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights=None, # b nq T nqf

                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        
        batch_size, nq, _, = temporal_queries.shape

        # 线性normalize, b nq T
        chosen_frame_weights = chosen_frame_weights / (chosen_frame_weights.sum(-1).unsqueeze(-1))

        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        
        grounding_score = self.reason(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                    node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
    
    def reason(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype

        node_temp_queries = self.temp_query_proj(node_temp_queries)
        node_feats = self.node_linear(node_feats)
        edge_feats = self.edge_linear(edge_feats)
        
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.mean(1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        )
        return scores # V nq
    
    def message(self, 
                edge_attr,  # E hc
                x_j,   # E nq
                node_temp_query_j, # E h_nq_c
                node_temp_query_i,) -> Tensor: # E h_nq_c
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = repeat(x_j, 'E nq -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_temp_query_j = rearrange(node_temp_query_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_temp_query_i = rearrange(node_temp_query_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)

        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_temp_query_j
        context_feat_j = context_feat_j.repeat(1,1, nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h 1 c -> E h nq c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)

        return context_score.mean(1) # E nq
    
    def aggregate(self, 
                  inputs, # E nq
                  x, # V nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        out = [] # list[nq] 
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(x[tgt_node_idx])
            else:
                self_score = x[tgt_node_idx]
                # Msg+1 nq
                msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
                node_aggr_scores = torch.cat([msgs, self_score.unsqueeze(0)], dim=0)
                out.append(self.aggr_msgs(node_aggr_scores))

        return torch.stack(out, dim=0) # V nq
    
    def aggr_msgs(self, msgs):
        # msg nq
        if self.score_aggr == 'sum':
            return msgs.sum(dim=0)
        elif self.score_aggr == 'min':
            return msgs.min(dim=0)[0]
        else:
            raise ValueError()
        
@register_reason_module
def temporal_grounding_v1(configs):
    return Temporal_Grounding_v1(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        temp_query_proj=configs['temp_query_proj'])


class Temporal_Grounding_v1_MLP(geo_nn.MessagePassing):
    def __init__(self, 
                  d_model,
                  flow,
                  score_aggr,
                  temp_query_proj) -> None:
        super().__init__(aggr=None,
                        flow=flow,)
        self.score_aggr = score_aggr

        self.node_linear = nn.Linear(d_model, d_model, bias=False)
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)

        if temp_query_proj.pop('name') == 'FeatureResizer':
            self.temp_query_proj = FeatureResizer(**temp_query_proj)
        elif temp_query_proj.pop('name') == 'linear':
            self.temp_query_proj = nn.Linear(**temp_query_proj)
        else:
            raise ValueError()

        self.context_2 = MLP(2 * d_model, 2 * d_model, d_model, num_layers=3)
        self.context_1 = MLP(d_model, d_model, 1, num_layers=3)
        # 1 h c c
        self.ref_2 = MLP(d_model, d_model, d_model, num_layers=3)
        # 1 h c 1
        self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) 

# 有temporal
class Temporal_Grounding_v2(Temporal_Grounding_v1):
    def message(self, 
                edge_attr, x_j,  
                node_temp_query_j,
                node_temp_query_i,) -> Tensor:
        """
        Args:
            edge_attr: E hc
            x_j: E h_nq
            node_temp_query_i, # E h_nq_c
            node_frame_query_i, # E h_nq_T_c
            node_cross_weight_i, # E h_nq_T
            yp_i: E c
        """
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_temp_query_j = rearrange(node_temp_query_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_temp_query_i = rearrange(node_temp_query_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_cross_weight_i = rearrange(node_cross_weight_i, 'E (h nq T) -> E h nq T',h=self.nheads, nq=nq)
        T = node_cross_weight_i.shape[-1]
        node_frame_query_i = rearrange(node_frame_query_i, 'E (h nq T c) -> E h nq T c',h=self.nheads, nq=nq, T=T)

        # temporal query
        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_temp_query_j
        context_feat_j = context_feat_j.repeat(1,1, nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_temp_query_i], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h 1 c -> E h nq c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)
        return context_score.flatten(1)
   
    def forward(self,
                 temporal_queries=None,  # b nq c
                 frame_queries=None,  # b T nqf c
                 cross_attn_weights=None, # b nq T nqf

                amrs=None, 
                amr_token_feats=None,
                amr_token_seg_ids=None, 
                node_alignments=None,
                text_feats=None, 
                text_pad_masks=None):

        nodes_batch_ids, edges_batch_ids,\
            node_seg_ids, edges_seg_ids, \
            edge_index, \
            node_feats, edge_feats,= self.batching_graph(amrs=amrs,
                                                amr_token_feats=amr_token_feats,
                                                amr_seg_ids=amr_token_seg_ids,)
        
        batch_size, nq, T, nqf = cross_attn_weights.shape

        chosen_frame_weights, chosen_frame_query_idxs = cross_attn_weights.max(-1) # b nq T
        # b nq T, b T nqf c -> b nq T c
        frame_queries = frame_queries.unsqueeze(1) # b nq T nqf c
        frame_queries = frame_queries.flatten(0, 2) # b_nq_T nqf c
        chosen_frame_query_idxs = chosen_frame_query_idxs.flatten() # b_nq_T
        chosen_frame_queries = torch.stack([fq[cidx] for cidx, fq in zip(chosen_frame_query_idxs, frame_queries)], dim=0)
        chosen_frame_queries = rearrange(chosen_frame_queries, '(b nq T) c -> b nq T c', b=batch_size, nq=nq, T=T)

        # 线性normalize, b nq T
        chosen_frame_weights = chosen_frame_weights / (chosen_frame_weights.sum(-1).unsqueeze(-1))

        node_temp_queries, edge_temp_queries = self.batching_memory(temporal_queries, nodes_batch_ids, edges_batch_ids)
        node_frame_queries, edge_frame_queries = self.batching_memory(chosen_frame_queries, nodes_batch_ids, edges_batch_ids)
        node_cross_weights, edge_cross_weights = self.batching_memory(chosen_frame_weights, nodes_batch_ids, edges_batch_ids)
        
        grounding_score = self.reason(node_feats=node_feats, edge_feats=edge_feats,edge_index=edge_index,
                                    node_temp_queries=node_temp_queries, edge_temp_queries=edge_temp_queries,
                                    node_frame_queries=node_frame_queries, edge_frame_queries=edge_frame_queries,
                                    node_cross_weights=node_cross_weights, edge_cross_weights=edge_cross_weights,) # V nq
        g_score_by_batch = [] # list[vi nq]
        for bch_idx in range(batch_size):
            bch_node_score = torch.stack([grounding_score[idx] for idx, batch_id in enumerate(nodes_batch_ids) if batch_id == bch_idx], dim=0)
            g_score_by_batch.append(bch_node_score) # vi nq
        
        return g_score_by_batch
    
    def reason(self, 
                node_feats=None, edge_feats=None,edge_index=None, # V c
                node_temp_queries=None, edge_temp_queries=None, # V nq c
                node_frame_queries=None,edge_frame_queries=None, # V nq T c
                node_cross_weights=None,edge_cross_weights=None, # V nq T
                ):
        
        V, E, device, dtype = node_feats.shape[0], edge_feats.shape[0], node_feats.device, node_feats.dtype

        node_temp_queries = self.temp_query_proj(node_temp_queries)
        node_feats = self.node_linear(node_feats)
        edge_feats = self.edge_linear(edge_feats)
        
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
         # V h nq c @ 1 h c 1 -> V h nq 1
        node_temp_queries = rearrange(node_temp_queries, 'V nq (h c) -> V h nq c',h=self.nheads)
        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        ref_score = (node_temp_queries @ self.ref_2) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        scores = ref_score @ self.ref_1
        scores = scores.mean(1) # V nq

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=V)
        traversal_order = dgl.topological_nodes_generator(dgl_graph)
        for idx, frontier_nodes in enumerate(traversal_order):
            frontier_nodes = frontier_nodes.to(device)
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes, form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_temp_query=node_temp_queries.flatten(1), # V h_nq_c
                                        )
        return scores # V nq
    