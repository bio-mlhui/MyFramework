
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
from einops import rearrange
_graphLayer_entrypoints = {}
def register_graphLayer(fn):
    graphLayer_name = fn.__name__
    _graphLayer_entrypoints[graphLayer_name] = fn

    return fn
def graphLayer_entrypoint(graphLayer_name):
    try:
        return _graphLayer_entrypoints[graphLayer_name]
    except KeyError as e:
        print(f'RVOS moel {graphLayer_name} not found')

class Graph_Layer_v0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        return x, edge_attr
     
class Graph_Layer_v1(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        memory: |E| hw c
        out: |V| dim_out
        """
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return x2 + x, edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):
        """ Compute message for each neighbord
        Input:
            x_j: E dim
            x_i: E dim
            edge_embeds: E dim
        Output:
            message: E dim
        """     
        # 1. 你是一个man
        # x_j: man
        # x_i: m
        # edge: /    
        
        # 2. 你是(:arg1为car)的look, 
        # x_j: c  (/ car)
        # x_i: l  (/ look)
        # edge: :ARG1
        
        # message如果是describable的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car)
        # message如果是entity的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car的look)
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_v1_updateEdge(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        memory: |E| hw c
        out: |V| dim_out
        """
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return x2 + x, edge_attr_2 + edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):

        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_gatHeadv2(nn.Module):
    def __init__(self, d_model, nhead, flow, aggr,):
        super().__init__()
        self.self_attn = geo_nn.GATv2Conv(in_channels=d_model,
                                        out_channels=d_model,
                                        add_self_loops=False,
                                        hedas=nhead,
                                        dropout=0.0,
                                        edge_dim=d_model,
                                        flow=flow,
                                        aggr=aggr
                                        )

        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        x2 = self.self_attn(x, edge_index, edge_attr,) 
        x2 = self.norm(self.dropout(x + x2), batch=batch_id[:len(x2)])
        return x2, edge_attr

class Graph_Layer_v1_norm(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        memory: |E| hw c
        out: |V| dim_out
        """
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return self.norm(x2 + x), edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):
        """ Compute message for each neighbord
        Input:
            x_j: E dim
            x_i: E dim
            edge_embeds: E dim
        Output:
            message: E dim
        """     
        # 1. 你是一个man
        # x_j: man
        # x_i: m
        # edge: /    
        
        # 2. 你是(:arg1为car)的look, 
        # x_j: c  (/ car)
        # x_i: l  (/ look)
        # edge: :ARG1
        
        # message如果是describable的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car)
        # message如果是entity的话,
            # l收到的信息是: (/look, 是look), (:ARG1 c, ARG1是car的look)
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_v1_normv2(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.norm = geo_nn.LayerNorm(d_model)
        
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        # batch_id: |V|+|E|
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return self.norm(x2 + x, batch=batch_id[:len(x)]), edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_v1_dropout(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        # batch_id: |V|+|E|
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return self.dropout(x2 + x), edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_v1_norm_edgeUpdate(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        """
        batch_id: |V|+|E|, int
        """
        x2 = self.node_linear(x.clone())
        edge_attr_2 = self.edge_linear(edge_attr)
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
        
        return self.norm(x2 + x), self.norm(edge_attr+ edge_attr_2)
        
    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message

class Graph_Layer_v1bi(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        x2 = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2) # arguments
    
        # residual
        return x2 + x, edge_attr
    
  
    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message
class Graph_Layer_bidirection(nn.Module):
    def __init__(self,
                 d_model,
                 aggr
                 ) -> None:
        super().__init__()
        self.source_to_target_conv = Graph_Layer_v1bi(d_model=d_model, flow='source_to_target', aggr=aggr)
        self.target_to_source_conv = Graph_Layer_v1bi(d_model=d_model, flow='target_to_source', aggr=aggr)
        
    def forward(self, x, edge_index, edge_attr,memory, batch_id):
        """
        x: num_nodes c
        edge_index: 2 num_edges
        edge_attr: num_edges 2*c
        """
        tgt2src_nodes_feats, _ = self.target_to_source_conv(x=x.clone(),
                                                        edge_attr=edge_attr.clone(),
                                                        edge_index=edge_index, memory=memory, batch_id=batch_id)
        src2tgt_nodes_feats, _ = self.source_to_target_conv(x=x.clone(),
                                                        edge_attr=edge_attr.clone(),
                                                        edge_index=edge_index, memory=memory, batch_id=batch_id)
          
        return (src2tgt_nodes_feats + tgt2src_nodes_feats ) / 2, edge_attr

class Graph_Layer_gatHeadFullStep(nn.Module):
    def __init__(self, d_model, nhead, flow, aggr,):
        super().__init__()
        self.self_attn = geo_nn.GATConv(in_channels=d_model,
                                        out_channels=d_model,
                                        add_self_loops=False,
                                        hedas=nhead,
                                        dropout=0.0,
                                        flow=flow,
                                        aggr=aggr
                                        )
        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        orders = []
        for idx, frontier in enumerate(traversal_order):
            src, tgt, eid =  dgl_graph.in_edges(frontier.to(x.device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(eid) == 0
                continue
            orders.append(eid)
        
        x2 = x.clone()
        for edge_order in orders: 
            x2 = self.self_attn(x2, edge_index[:, edge_order], edge_attr[edge_order, :]) # arguments

        return self.norm(self.dropout(x + x2), batch=batch_id[:len(x2)]), edge_attr



class Graph_Layer_gatHead(nn.Module):
    def __init__(self, d_model, nhead, flow, aggr,):
        super().__init__()
        self.self_attn = geo_nn.GATConv(in_channels=d_model,
                                        out_channels=d_model,
                                        add_self_loops=False,
                                        hedas=nhead,
                                        dropout=0.0,
                                        flow=flow,
                                        aggr=aggr
                                        )

        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        x2 = self.self_attn(x, edge_index, edge_attr,) 
        x2 = self.norm(self.dropout(x + x2), batch=batch_id[:len(x2)])
        return x2, edge_attr

class Graph_Layer_v3(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,):
        super().__init__(aggr=V3_Aggregation(),
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.memory_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        x2 = self.node_linear(x)  # |V| c
        edge_attr_2 = self.edge_linear(edge_attr)  # |E| c
        
        video_mem = self.memory_linear(memory['video_mem']) # |E| c
        video_mem += memory['video_mem_pos']
        video_pad_mask = memory['video_mem_pad_mask']
        x2 = self.propagate(edge_index, size=None,  # keywords
                             x=x2, edge_attr=edge_attr_2, video_mem={'feat': video_mem, 'pad': video_pad_mask}) # arguments
    
        # residual
        return self.norm(self.dropout(x2 + x), 
                         batch=batch_id[:len(x)]), edge_attr
    
    def aggregate(self, inputs: Tensor, x:Tensor, video_mem:Tensor, index: Tensor, 
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim, kwargs={'video_mem':video_mem, 'x':x})
   

    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message


class V3_Aggregation(Aggregation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, msgs: Tensor, index: Tensor=None, ptr=None, dim_size: int=None, dim: int = -2,
                kwargs=None) -> Tensor:
        # dim_size:|V|, x: |E| c, video_mem: |E| hw c
        """
        x: |E| c
        x_i: |V| c
        video_mem:|E| hw c
        index: |E|, int, 从0到|V|, 表示每个message属于哪个, [0, 1, 3, 4, 5, 6]
        """
        # E hw c, E c 1 -> E hw
        video_mem = kwargs['video_mem']['feat']
        video_mem_pad_mask = kwargs['video_mem']['pad']
        x = kwargs['x']
        attn_mask = torch.matmul(video_mem, msgs.unsqueeze(-1)).squeeze(-1)
        # list[c], |V|
        aggrated_message = []
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                aggrated_message.append(x[tgt_node_idx])
                continue
            # mi hw c
            tgt_node_memory = torch.stack([video_mem[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            tgt_node_mem_pad = torch.stack([video_mem_pad_mask[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            for i in range(1, len(tgt_node_memory)):
                assert (tgt_node_memory[i] - tgt_node_memory[0]).sum() == 0
            tgt_node_memory = tgt_node_memory[0].permute(1, 0) # hw c -> c hw
            tgt_node_mem_pad = tgt_node_mem_pad[0]
            # mi+1 hw
            masks = [attn_mask[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx]
            masks.append(torch.einsum('sc,c->s',tgt_node_memory.permute(1, 0), x[tgt_node_idx]))
            masks = torch.stack(masks, dim=0)
            # hw
            min_mask: Tensor = masks.min(dim=0)[0]
            min_mask.masked_fill_(tgt_node_mem_pad, torch.finfo(min_mask.dtype).min)
            min_mask = F.softmax(min_mask, dim=0)
            aggrated_message.append(torch.einsum('cs,s->c',tgt_node_memory, min_mask))
        return torch.stack(aggrated_message, dim=0)
    
class Graph_Layer_v3_fullstep(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,):
        super().__init__(aggr=V3_Aggregation(),
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        device = x.device          
        x2 = self.node_linear(x)  # |V| c
        
        video_mem = memory['video_mem'] # |E| hw c
        video_mem += memory['video_mem_pos']
        video_pad_mask = memory['video_mem_pad_mask']
        
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        orders = []
        for idx, frontier in enumerate(traversal_order):
            src, tgt, eid =  dgl_graph.in_edges(frontier.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(eid) == 0
                continue
            orders.append(eid)
            
        for edge_order in orders: 
            x2 = self.propagate(edge_index[:, edge_order], 
                                size=None,  # keywords
                                x=x2, 
                                edge_attr=edge_attr[edge_order, :], 
                                video_mem={'feat': video_mem[edge_order, :], 
                                           'pad': video_pad_mask[edge_order, :]}) # arguments
    
        # residual
        return self.norm(self.dropout(x2 + x), 
                         batch=batch_id[:len(x)]), edge_attr
    
    def aggregate(self, inputs: Tensor, x:Tensor, video_mem:Tensor, index: Tensor, 
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim, kwargs={'video_mem':video_mem, 'x':x})
   

    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message


class V3_Aggregation_obj(Aggregation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, msgs: Tensor, index: Tensor=None, ptr=None, dim_size: int=None, dim: int = -2,
                kwargs=None) -> Tensor:
        # dim_size:|V|, x: |E| c, video_mem: |E| hw c
        """
        x: |E| c
        x_i: |V| c
        index: |E|, int, 从0到|V|, 表示每个message属于哪个, [0, 1, 3, 4, 5, 6]
        """
        # E nq c, E c 1 -> E nq
        obj_queries = kwargs['obj_queries']['feat']
        x = kwargs['x']
        attn_mask = torch.matmul(obj_queries, msgs.unsqueeze(-1)).squeeze(-1)
        # list[c], |V|
        aggrated_message = []
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                aggrated_message.append(x[tgt_node_idx])
                continue
            # mi nq c
            tgt_node_memory = torch.stack([obj_queries[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            for i in range(1, len(tgt_node_memory)):
                assert (tgt_node_memory[i] - tgt_node_memory[0]).sum() == 0
            tgt_node_memory = tgt_node_memory[0].permute(1, 0) # nq c -> c nq
            # mi+1 nq
            masks = [attn_mask[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx]
            masks.append(torch.einsum('sc,c->s',tgt_node_memory.permute(1, 0), x[tgt_node_idx]))
            masks = torch.stack(masks, dim=0)
            # nq
            min_mask: Tensor = masks.min(dim=0)[0]
            min_mask = F.softmax(min_mask, dim=0)
            aggrated_message.append(torch.einsum('cs,s->c',tgt_node_memory, min_mask))
        return torch.stack(aggrated_message, dim=0)
    
class Graph_Layer_v3_fullstep_obj(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,):
        super().__init__(aggr=V3_Aggregation_obj(),
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.norm = geo_nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, memory, batch_id):
        device = x.device          
        x2 = self.node_linear(x)  # |V| c
        
        obj_queries = memory['feat'] # |E| nq c
        
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        orders = []
        for idx, frontier in enumerate(traversal_order):
            src, tgt, eid =  dgl_graph.in_edges(frontier.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(eid) == 0
                continue
            orders.append(eid)
            
        for edge_order in orders: 
            x2 = self.propagate(edge_index[:, edge_order], 
                                size=None,  # keywords
                                x=x2, 
                                edge_attr=edge_attr[edge_order, :], 
                                obj_queries={'feat': obj_queries[edge_order, :],}) # arguments
    
        # residual
        return self.norm(self.dropout(x2 + x), 
                         batch=batch_id[:len(x)]), edge_attr
    
    def aggregate(self, inputs: Tensor, x:Tensor, obj_queries:Tensor, index: Tensor, 
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim, kwargs={'obj_queries':obj_queries, 'x':x})
   

    def message(self, x_j, x_i, edge_attr):
        message = torch.cat([x_j, x_i, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message



@register_graphLayer
def graph_layer_inferfullstep(configs):
    return Graph_Layer_v3_fullstep(
        d_model=configs['d_model'],
        flow=configs['flow']
    )

@register_graphLayer
def graph_layer_inferfullstep_obj(configs):
    return Graph_Layer_v3_fullstep_obj(
        d_model=configs['d_model'],
        flow=configs['flow']
    )
  

@register_graphLayer
def graph_layer_gatHeadFullStep(configs):
    return Graph_Layer_gatHeadFullStep(d_model=configs['d_model'],
                                        nhead=configs['nhead'],
                                        flow=configs['flow'],
                                        aggr=configs['aggr'])

 
@register_graphLayer
def graph_layer_bidirection(configs):
    return Graph_Layer_bidirection(
        d_model=configs['d_model'],
        aggr=configs['aggr'],
    )


@register_graphLayer
def graph_layer_infer(configs):
    return Graph_Layer_v3(d_model=configs['d_model'],
                          flow=configs['flow'])
@register_graphLayer
def graph_layer_v0(configs):
    return Graph_Layer_v0()

@register_graphLayer
def graph_layer_v1(configs):
    return Graph_Layer_v1(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_v1_updateEdge(configs):
    return Graph_Layer_v1_updateEdge(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_v1_norm(configs):
    return Graph_Layer_v1_norm(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_v1_normv2(configs):
    return Graph_Layer_v1_normv2(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_v1_norm_updateEdge(configs):
    return Graph_Layer_v1_norm_edgeUpdate(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_v3(configs):
    return Graph_Layer_v3(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])

@register_graphLayer
def graph_layer_gatHead(configs):
    return Graph_Layer_gatHead(d_model=configs['d_model'],
                               nhead=configs['nhead'],
                               flow=configs['flow'],
                               aggr=configs['aggr'])

@register_graphLayer
def graph_layer_gatHeadv2(configs):
    return Graph_Layer_gatHeadv2(d_model=configs['d_model'],
                               nhead=configs['nhead'],
                               flow=configs['flow'],
                               aggr=configs['aggr'])



@register_graphLayer
def graph_layer_dropout(configs):
    return Graph_Layer_v1_dropout(d_model=configs['d_model'],
                          flow=configs['flow'],
                          aggr=configs['aggr'])


import networkx as nx
from torch_geometric.data import Batch
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
    batched_amrs = Batch.from_data_list(amrs) # concate
    edge_index = batched_amrs.edge_index.to(device)

    node_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
    edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats, amr_seg_ids)], dim=0)
    node_seg_ids = torch.cat([seg_ids[seg_ids>0] for seg_ids in amr_seg_ids], dim=0)
    edges_seg_ids = torch.cat([seg_ids[seg_ids<0] for seg_ids in amr_seg_ids], dim=0)

    # V nq c
    node_memories_feats = torch.stack([memories[bid] for bid in nodes_batch_ids], dim=0)
    node_memories_poses = torch.stack([memories_pos[bid] for bid in nodes_batch_ids], dim=0)

    edge_memories_feats = torch.stack([memories[bid] for bid in edges_batch_ids], dim=0)
    edge_memories_poses = torch.stack([memories_pos[bid] for bid in edges_batch_ids], dim=0)

    node_memories = {'feat': node_memories_feats, 'pos': node_memories_poses}
    edge_memories = {'feat': edge_memories_feats, 'pos': edge_memories_poses}

    node_subseqs = [] # list[s c], V
    for btc_text_feat, btc_node_alis in zip(text_feats, node_alignments):
        # s c, list[int]
        for node_ali in btc_node_alis:
            node_subseqs.append(btc_text_feat[:(node_ali+1)])

    node_dsends = [] # list[si c], V
    icgd = list(zip(edge_index[0, :].tolist(), edge_index[1, :].tolist()))
    nx_graph = nx.DiGraph(icgd)
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

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
# self layer, head/tail representation + ffn layer
class GAT_Attention(geo_nn.MessagePassing):
    def __init__(self, flow,
                 d_model, nheads, dropout,
                 add_self_loop=True,
                 fill_loop_value=0.):
        super().__init__(aggr=None, flow=flow)
        self.d_model = d_model
        self.dropout = dropout

        self.head_dim = d_model // nheads
        self.scale = (3 * self.head_dim) ** -0.5
        self.nheads = nheads

        self.node_linear = Linear(d_model, nheads * self.head_dim, bias=False, weight_initializer='glorot')
        self.edge_linear = Linear(d_model, nheads * self.head_dim, bias=False, weight_initializer='glorot')

        self.attn_weight = nn.Parameter(torch.zeros([1, nheads, 3 * self.head_dim]))
        glorot(self.attn_weight)
        self.to_out = nn.Sequential(Linear(nheads * self.head_dim, d_model, weight_initializer='glorot'), 
                                    nn.Dropout(dropout))
        self.add_self_loop = add_self_loop
        self.fill_loop_value = fill_loop_value
    
    def forward(self, x, edge_index,
                edge_attr, 
                size= None,):
        """
        Args:
            x: V c
            edge_index: 2 E
            edge_attr: E c
        """
        H, C = self.nheads, self.d_model
        # V c -> V hc
        x = self.node_linear(x)
        # E c -> E hc
        edge_attr = self.edge_linear(edge_attr)

        # V hc
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        return self.to_out(out)

    def message(self, x_j, x_i, edge_attr) -> Tensor:
        # x_j: E hc
        # x_i: E hc
        # edge_attr: E hc
        x_j = rearrange(x_j, 'E (h c) -> E h c',h=self.nheads)
        x_i = rearrange(x_i, 'E (h c) -> E h c',h=self.nheads)
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c',h=self.nheads)

        # E h 3c * 1 h 3c -> E h
        alpha = (torch.cat([x_j, x_i, edge_attr], dim=-1) * self.attn_weight).sum(-1) * self.scale
        return alpha
    
    def aggregate(self, inputs, index, x, x_j, dim_size= None) -> Tensor:
        """
        Args:
            inputs: E h
            index: E
            x_j: E hc
            x: V hc
            dim_size: V
        """
        x = rearrange(x, 'V (h c) -> V h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h c) -> E h c', h=self.nheads)

        out = [] # V hc
        for tgt_node_idx in range(dim_size):
            self_feat = x[tgt_node_idx] # h c
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(self_feat)
                continue
            # Msg h
            msg_alpha = [inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx]
            if self.add_self_loop:
                self_loop_feat = torch.ones_like(self_feat).float() * self.fill_loop_value
                # h 3c * h 3c -> h
                self_alpha = (torch.cat([self_feat, self_feat, self_loop_feat], dim=-1) * (self.attn_weight.squeeze(0))).sum(-1) * self.scale
                msg_alpha.append(self_alpha)
            msg_alpha = torch.stack(msg_alpha, dim=0)
            msg_alpha = msg_alpha.softmax(dim=0)

            # Msg h c
            income_feat = [x_j[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx]
            if self.add_self_loop:
                income_feat.append(self_feat)
            income_feat = torch.stack(income_feat, dim=0)
            
            # h c Msg @ h MSg 1 -> h c
            aggr_feat = (income_feat.permute(1,2, 0)) @ (msg_alpha.permute(1, 0).unsqueeze(-1))
            aggr_feat = aggr_feat.squeeze(-1)
            out.append(aggr_feat)
        return torch.stack(out, dim=0).flatten(1)


class Subseq_YsYp(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.ys_softattn = nn.Linear(d_model, 1, bias=False)
        self.yp_softattn = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, node_feats=None, edge_feats=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """
        node_subseqs: list[si c], V
        """
        ys_by_node = []
        yp_by_node = []
        for subseq in node_subseqs:
            # si c -> si
            soft_attn = self.ys_softattn(subseq).squeeze(-1).softmax(dim=0)
            # 1 si @ si c
            ys = soft_attn.unsqueeze(0) @ subseq # 1 c
            ys_by_node.append(ys)

            soft_attn = self.yp_softattn(subseq).squeeze(-1).softmax(dim=0)
            yp = soft_attn.unsqueeze(0) @ subseq # 1 c
            yp_by_node.append(yp)

        ys_by_node = torch.cat(ys_by_node, dim=0)
        yp_by_node = torch.cat(yp_by_node, dim=0)
        return ys_by_node, yp_by_node

class Desends_YsYp(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.ys_softattn = nn.Linear(d_model, 1, bias=False)
        self.yp_softattn = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, node_feats=None, edge_feats=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """
        node_subseqs: list[si c], V
        """
        ys_by_node = []
        yp_by_node = []
        for descends in node_dsends:
            # si c -> si
            soft_attn = self.ys_softattn(descends).squeeze(-1).softmax(dim=0)
            # 1 si @ si c
            ys = soft_attn.unsqueeze(0) @ descends # 1 c
            ys_by_node.append(ys)

            soft_attn = self.yp_softattn(descends).squeeze(-1).softmax(dim=0)
            yp = soft_attn.unsqueeze(0) @ descends # 1 c
            yp_by_node.append(yp)

        ys_by_node = torch.cat(ys_by_node, dim=0)
        yp_by_node = torch.cat(yp_by_node, dim=0)
        return ys_by_node, yp_by_node


class TopDown_Bottomup_YsYp(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                 dropout,
                 add_self_loop=True,
                 fill_loop_value=0.):
        super().__init__()
        self.bottomup_self_attn = GAT_Attention(flow='source_to_target',
                                              d_model=d_model,
                                              nheads=nheads,
                                              dropout=dropout,
                                              add_self_loop=add_self_loop,
                                              fill_loop_value=fill_loop_value)
        self.topdown_self_attn = GAT_Attention(flow='source_to_target',
                                              d_model=d_model,
                                              nheads=nheads,
                                              dropout=dropout,
                                              add_self_loop=add_self_loop,
                                              fill_loop_value=fill_loop_value)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                node_feats=None, edge_feats=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """不对node/edge的feature进行转换
        Args:
            node_batch_ids: V
            edge_batch_ids: E

            node_seg_ids: V
            edge_seg_ids: E
            node_feats: V c
            edge_feats: E c
            edge_memories: V nq c
            node_memories: V nq c
            edge_index: 2 E

            node_subseqs: list[s c], V
        """
        bottomup_node_feats, topdown_node_feats = node_feats.clone(), node_feats.clone()
        bottomup_edge_index, topdown_edge_index = edge_index, edge_index[[1,0],:]
        
        bottomup_node_feats = self.forward_graph(bottomup_edge_index, bottomup_node_feats, edge_feats, direction='bottomup')
        topdown_node_feats = self.forward_graph(topdown_edge_index, topdown_node_feats, edge_feats, direction='topdown')
        # ys, yp
        return topdown_node_feats, bottomup_node_feats
    
    def forward_graph(self, edge_index, node_feats, edge_attr, direction):
        device = node_feats.device
        try:
            dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
            topo_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        for idx, frontier_nodes in enumerate(topo_order):
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V c
                if direction == 'bottomup':
                    node_feats_2 = self.bottomup_self_attn(x=node_feats,
                                                        edge_index=edge_index[:, order_eid],
                                                        edge_attr=edge_attr[order_eid, :])
                elif direction == 'topdown':
                    node_feats_2 = self.topdown_self_attn(x=node_feats,
                                                          edge_index=edge_index[:, order_eid],
                                                          edge_attr=edge_attr[order_eid, :])
                trans_mask = torch.zeros_like(node_feats).bool() # V c
                trans_mask[frontier_nodes, :] = True
                node_feats = torch.where(trans_mask, node_feats_2, node_feats)
        return node_feats

class TopDown_TopDown_YsYp(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                 dropout,
                 add_self_loop=True,
                 fill_loop_value=0.):
        super().__init__()
        self.ys_self_attn = GAT_Attention(flow='source_to_target',
                                              d_model=d_model,
                                              nheads=nheads,
                                              dropout=dropout,
                                              add_self_loop=add_self_loop,
                                              fill_loop_value=fill_loop_value)
        self.yp_self_attn = GAT_Attention(flow='source_to_target',
                                              d_model=d_model,
                                              nheads=nheads,
                                              dropout=dropout,
                                              add_self_loop=add_self_loop,
                                              fill_loop_value=fill_loop_value)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                node_feats=None, edge_feats=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """不对node/edge的feature进行转换
        Args:
            node_batch_ids: V
            edge_batch_ids: E

            node_seg_ids: V
            edge_seg_ids: E
            node_feats: V c
            edge_feats: E c
            edge_memories: V nq c
            node_memories: V nq c
            edge_index: 2 E

            node_subseqs: list[s c], V
        """
        ys_node_feats, yp_node_feats = node_feats.clone(), node_feats.clone()
        ys_edge_index, yp_edge_index = edge_index[[1, 0], :], edge_index[[1, 0], :]
        
        ys_node_feats = self.forward_graph(ys_edge_index, ys_node_feats, edge_feats, direction='ys')
        yp_node_feats = self.forward_graph(yp_edge_index, yp_node_feats, edge_feats, direction='yp')
        # ys, yp
        return ys_node_feats, yp_node_feats
    
    def forward_graph(self, edge_index, node_feats, edge_attr, direction):
        device = node_feats.device
        try:
            dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
            topo_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        for idx, frontier_nodes in enumerate(topo_order):
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V c
                if direction == 'ys':
                    node_feats_2 = self.ys_self_attn(x=node_feats,
                                                        edge_index=edge_index[:, order_eid],
                                                        edge_attr=edge_attr[order_eid, :])
                elif direction == 'yp':
                    node_feats_2 = self.yp_self_attn(x=node_feats,
                                                          edge_index=edge_index[:, order_eid],
                                                          edge_attr=edge_attr[order_eid, :])
                trans_mask = torch.zeros_like(node_feats).bool() # V c
                trans_mask[frontier_nodes, :] = True
                node_feats = torch.where(trans_mask, node_feats_2, node_feats)
        return node_feats


class Grounding_v1(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 get_ysyp={'name': 'subseq'},# subseq/topdown_bottomup
                 flow='source_to_target',
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.ref_2 = nn.Linear(d_model, d_model, bias=False)
        self.ref_1 = nn.Linear(d_model, 1, bias=False)
        
        self.context_2 = nn.Linear(3*d_model, d_model, bias=False)
        self.context_1 = nn.Linear(d_model, 1, bias=False)
        get_ysyp_name = get_ysyp.pop('name')
        if get_ysyp_name == 'subseq':
            self.get_ysyp = Subseq_YsYp(d_model)
        elif get_ysyp_name == 'topdown_bottomup':
            self.get_ysyp = TopDown_Bottomup_YsYp(**get_ysyp)
        elif get_ysyp_name == 'topdown_topdown':
            self.get_ysyp = TopDown_TopDown_YsYp(**get_ysyp)
        elif get_ysyp_name == 'descends':
            self.get_ysyp = Desends_YsYp(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                node_batch_ids=None, edge_batch_ids=None, 
                node_seg_ids=None, edge_seg_ids=None,
                node_feats=None, edge_feats=None,
                edge_memories=None, node_memories=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """不对node/edge的feature进行转换
        Args:
            node_batch_ids: V
            edge_batch_ids: E

            node_seg_ids: V
            edge_seg_ids: E
            node_feats: V c
            edge_feats: E c
            edge_memories: V nq c
            node_memories: V nq c
            edge_index: 2 E

            node_subseqs: list[s c], V
        """
        device = edge_feats.device
        node_query_feats, node_query_pos  = node_memories['feat'], node_memories['pos']
        node_query_feats = self.with_pos_embed(node_query_feats, node_query_pos) # V nq c

        # V c, V c
        ys_by_node, yp_by_node = self.get_ysyp(node_subseqs=node_subseqs,
                                                       node_feats=node_feats,
                                                       edge_feats=edge_feats,
                                                       edge_index=edge_index,
                                                       node_dsends=node_dsends)

        # intialize score V nq
        # S(xi, v) = S_s(xi, y_s^v)
        # V nq c \odot V 1 c -> V nq c -> V nq
        ref_match = self.ref_2(node_query_feats) * (ys_by_node.unsqueeze(1))
        ref_match = ref_match / ref_match.norm(dim=-1, keepdim=True)
        scores = self.ref_1(ref_match).squeeze(-1)

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        for idx, frontier_nodes in enumerate(traversal_order):
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        yp=yp_by_node, # V c
                                        edge_attr=edge_feats[order_eid, :], # E c
                                        node_nq=node_query_feats.flatten(1), # V nq*c
                                        ) # arguments
        return scores 
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i, yp_i) -> Tensor:
        """
        Args:
            edge_attr: E c
            x_j: E nq
            node_nq_j/i: E nqc
            yp_i: E c
        """
        nq = x_j.shape[-1]
        node_nq_j = rearrange(node_nq_j, 'E (nq c) -> E nq c',nq=nq)
        node_nq_i = rearrange(node_nq_i, 'E (nq c) -> E nq c',nq=nq)
        score = x_j.clone()
        # E 1 nq @ E nq c -> E 1 c
        soft_attn_j = x_j.softmax(-1)
        context_feat_j = (soft_attn_j.unsqueeze(1)) @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,nq,1) # E nq c
        edge_attr = edge_attr.unsqueeze(1).repeat(1, nq, 1) # E nq c
        cat_feat = torch.cat([context_feat_j, node_nq_i, edge_attr], dim=-1) # E nq 3c
        cat_feat = self.context_2(cat_feat) * (yp_i.unsqueeze(1)) # E nq c \odot E 1 c -> E nq c
        cat_feat = cat_feat / cat_feat.norm(dim=-1, keepdim=True) # E nq c
        context_score = self.context_1(cat_feat).squeeze(-1) # E nq
        return score + context_score 
    
    def aggregate(self, 
                  inputs, # E nq
                  x, # V nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        out = [] # V nq
        for tgt_node_idx in range(dim_size):
            self_score = x[tgt_node_idx] # nq
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(self_score)
                continue
            # Msg nq
            msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            msgs = msgs.sum(dim=0) 
            out.append(msgs + self_score)
        return torch.stack(out, dim=0)

@register_graphLayer
def grounding_v1(configs):
    return Grounding_v1(d_model=configs['d_model'],
                        flow=configs['flow'],
                        get_ysyp=configs['get_ysyp'] if 'get_ysyp' in configs else {'name': 'subseq'})


class Desends_YsYp_multihead(nn.Module):
    def __init__(self, 
                 d_model,
                 nheads,
                 dropout,) -> None:
        super().__init__()
        self.head_dim = d_model // nheads
        self.nheads = nheads
        # 1 h c
        self.ys_softattn = nn.Parameter(torch.zeros([nheads, self.head_dim, 1])) # h c 1
        self.yp_softattn = nn.Parameter(torch.zeros([nheads, self.head_dim, 1])) # h c 1
        glorot(self.ys_softattn)
        glorot(self.yp_softattn)
        self.ys_to_out = nn.Sequential(nn.Linear(self.head_dim * self.nheads, d_model), # hc c
                                    nn.Dropout(dropout))
        self.yp_to_out = nn.Sequential(nn.Linear(self.head_dim * self.nheads, d_model), # hc c
                                    nn.Dropout(dropout))
    
    def forward(self, node_feats=None, edge_feats=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """
        node_subseqs: list[si c], V
        """
        ys_by_node = [] # V hc
        yp_by_node = [] # V hc
        for descends in node_dsends:
            # h si c * h 1 c -> h si
            descends = rearrange(descends, 'si (h c) -> h si c', h=self.nheads)

            # h si c @ h c 1 -> h si
            soft_attn = (descends @ self.ys_softattn).squeeze(-1).softmax(dim=-1)
            # h 1 si @ h si c -> h c -> hc
            ys = (soft_attn.unsqueeze(1) @ descends).squeeze(1).flatten()
            ys_by_node.append(ys)

            # h si c @ h c 1 -> h si
            soft_attn = (descends @ self.yp_softattn).squeeze(-1).softmax(dim=-1)
            # h 1 si @ h si c -> h c -> hc
            yp = (soft_attn.unsqueeze(1) @ descends).squeeze(1).flatten()
            yp_by_node.append(yp)

        ys_by_node = torch.stack(ys_by_node, dim=0)
        yp_by_node = torch.stack(yp_by_node, dim=0)

        return self.ys_to_out(ys_by_node), self.yp_to_out(yp_by_node)

class Grounding_v1_multihead(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 nheads,
                 get_ysyp={'name': 'subseq'},# subseq/topdown_bottomup
                 flow='source_to_target',
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.head_dim = d_model // nheads
        self.nheads = nheads
        self.ref_2 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, self.head_dim])) # 1 h c c
        self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.ref_1)
        glorot(self.ref_2)
        
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 3*self.head_dim, self.head_dim])) # 1 h 3c c
        glorot(self.context_2)
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.context_1)
        get_ysyp_name = get_ysyp.pop('name')
        if get_ysyp_name == 'descends_multihead':
            self.get_ysyp = Desends_YsYp_multihead(**get_ysyp)
        else:
            raise ValueError()

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                node_batch_ids=None, edge_batch_ids=None, 
                node_seg_ids=None, edge_seg_ids=None,
                node_feats=None, edge_feats=None,
                edge_memories=None, node_memories=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None):
        """不对node/edge的feature进行转换
        Args:
            node_batch_ids: V
            edge_batch_ids: E

            node_seg_ids: V
            edge_seg_ids: E
            node_feats: V c
            edge_feats: E c
            edge_memories: V nq c
            node_memories: V nq c
            edge_index: 2 E

            node_subseqs: list[s c], V
        """
        device = edge_feats.device
        node_query_feats, node_query_pos = node_memories['feat'], node_memories['pos']
        node_query_feats = self.with_pos_embed(node_query_feats, node_query_pos) # V nq c
        node_query_feats = rearrange(node_query_feats, 'V nq (h c) -> V h nq c',h=self.nheads)
        # V c, V c
        ys_by_node, yp_by_node = self.get_ysyp(node_subseqs=node_subseqs,
                                                node_feats=node_feats,
                                                edge_feats=edge_feats,
                                                edge_index=edge_index,
                                                node_dsends=node_dsends)
        ys_by_node = rearrange(ys_by_node, 'V (h c) -> V h c', h=self.nheads)
        yp_by_node = rearrange(yp_by_node, 'V (h c) -> V h c', h=self.nheads)
        # intialize score V h_nq
        # S(xi, v) = S_s(xi, y_s^v)
        # V h nq c @ 1 h c c -> V h nq c
        node_query_feats = node_query_feats @ self.ref_2
        # V h nq c * V h 1 c -> V h nq c
        ref_match = node_query_feats * (ys_by_node.unsqueeze(2))
        ref_match = ref_match / ref_match.norm(dim=-1, keepdim=True)
        # V h nq c @ 1 h c 1 -> V h nq 1
        scores = ref_match @ self.ref_1
        # V h_nq
        scores = scores.flatten(1)

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        for idx, frontier_nodes in enumerate(traversal_order):
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V h_nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V h_nq
                                        yp=yp_by_node.flatten(1), # V hc
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_nq=node_query_feats.flatten(1), # V h_nq_c
                                        ) # arguments
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(dim=1) # V nq
    
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i, yp_i) -> Tensor:
        """
        Args:
            edge_attr: E hc
            x_j: E h_nq
            node_nq_j/i: E nqc
            yp_i: E c
        """
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_nq_i = rearrange(node_nq_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_nq_j = rearrange(node_nq_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        yp_i = rearrange(yp_i, 'E (h c) -> E h c', h=self.nheads)
        
        score = x_j.clone()
        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,1, nq,1) # E h nq c
        edge_attr = edge_attr.unsqueeze(2).repeat(1,1,nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_nq_i, edge_attr], dim=-1) # E h nq 3c

        # E h nq 3c @ 1 h 3c c -> E h nq c
        cat_feat = cat_feat @ self.context_2 
        # E h nq c * E h 1 c -> E h nq c
        cat_feat = cat_feat * (yp_i.unsqueeze(2))
        cat_feat = cat_feat / cat_feat.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (cat_feat @ self.context_1).squeeze(-1)
        return (score + context_score).flatten(1)
    
    def aggregate(self, 
                  inputs, # E h_nq
                  x, # V h_nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        x = rearrange(x, 'V (h nq) -> V h nq',h=self.nheads)
        inputs = rearrange(inputs, 'E (h nq) -> E h nq',h=self.nheads)
        out = [] # V  h nq
        for tgt_node_idx in range(dim_size):
            self_score = x[tgt_node_idx] # h nq
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(self_score)
                continue
            # Msg h nq
            msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            msgs = msgs.sum(dim=0) 
            out.append(msgs + self_score)
        return torch.stack(out, dim=0).flatten(1)

@register_graphLayer
def grounding_v1_multihead(configs):
    return Grounding_v1_multihead(d_model=configs['d_model'],
                        flow=configs['flow'],
                        nheads=configs['nheads'],
                        get_ysyp=configs['get_ysyp'] if 'get_ysyp' in configs else {'name': 'subseq'})


class Grounding_v1_multihead_noysyp(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.head_dim = d_model // nheads
        self.nheads = nheads
        self.ref_2 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, self.head_dim])) # 1 h c c
        self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.ref_1)
        glorot(self.ref_2)
        
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 3*self.head_dim, self.head_dim])) # 1 h 3c c
        glorot(self.context_2)
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.context_1)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                node_score,
                node_batch_ids=None, edge_batch_ids=None, 
                node_seg_ids=None, edge_seg_ids=None,
                node_feats=None, edge_feats=None,
                edge_memories=None, node_memories=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None,
                ys_node_feats=None, yp_node_feats=None):
        """不对node/edge的feature进行转换
        Args:
            node_score: V nq
            node_batch_ids: V
            edge_batch_ids: E

            node_seg_ids: V
            edge_seg_ids: E
            node_feats: V c
            edge_feats: E c
            edge_memories: V nq c
            node_memories: V nq c
            edge_index: 2 E

            node_subseqs: list[s c], V
        """
        device = edge_feats.device
        node_query_feats, node_query_pos = node_memories['feat'], node_memories['pos']
        node_query_feats = self.with_pos_embed(node_query_feats, node_query_pos) # V nq c
        node_query_feats = rearrange(node_query_feats, 'V nq (h c) -> V h nq c',h=self.nheads)
        ys_by_node = rearrange(ys_node_feats, 'V (h c) -> V h c', h=self.nheads)
        yp_by_node = rearrange(yp_node_feats, 'V (h c) -> V h c', h=self.nheads)
        # intialize score V h_nq
        # S(xi, v) = S_s(xi, y_s^v)
        # V h nq c @ 1 h c c -> V h nq c
        node_query_feats = node_query_feats @ self.ref_2
        # V h nq c * V h 1 c -> V h nq c
        ref_match = node_query_feats * (ys_by_node.unsqueeze(2))
        ref_match = ref_match / ref_match.norm(dim=-1, keepdim=True)
        # V h nq c @ 1 h c 1 -> V h nq 1
        scores = ref_match @ self.ref_1
        # V h_nq
        scores = scores.flatten(1)

        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        for idx, frontier_nodes in enumerate(traversal_order):
            src, tgt, order_eid =  dgl_graph.in_edges(frontier_nodes.to(device), form='all')
            if idx == 0:
                assert len(src) == 0 and len(tgt) == 0 and len(order_eid) == 0
            else:
                # V h_nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V h_nq
                                        yp=yp_by_node.flatten(1), # V hc
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_nq=node_query_feats.flatten(1), # V h_nq_c
                                        ) # arguments
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        scores = scores.mean(dim=1)

        return scores + node_score # V nq
    
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i, yp_i) -> Tensor:
        """
        Args:
            edge_attr: E hc
            x_j: E h_nq
            node_nq_j/i: E nqc
            yp_i: E c
        """
        edge_attr = rearrange(edge_attr, 'E (h c) -> E h c', h=self.nheads)
        x_j = rearrange(x_j, 'E (h nq) -> E h nq', h=self.nheads)
        nq = x_j.shape[-1]
        node_nq_i = rearrange(node_nq_i, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        node_nq_j = rearrange(node_nq_j, 'E (h nq c) -> E h nq c',nq=nq,h=self.nheads)
        yp_i = rearrange(yp_i, 'E (h c) -> E h c', h=self.nheads)
        
        score = x_j.clone()
        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,1, nq,1) # E h nq c
        edge_attr = edge_attr.unsqueeze(2).repeat(1,1,nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_nq_i, edge_attr], dim=-1) # E h nq 3c

        # E h nq 3c @ 1 h 3c c -> E h nq c
        cat_feat = cat_feat @ self.context_2 
        # E h nq c * E h 1 c -> E h nq c
        cat_feat = cat_feat * (yp_i.unsqueeze(2))
        cat_feat = cat_feat / cat_feat.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (cat_feat @ self.context_1).squeeze(-1)
        return (score + context_score).flatten(1)
    
    def aggregate(self, 
                  inputs, # E h_nq
                  x, # V h_nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        x = rearrange(x, 'V (h nq) -> V h nq',h=self.nheads)
        inputs = rearrange(inputs, 'E (h nq) -> E h nq',h=self.nheads)
        out = [] # V  h nq
        for tgt_node_idx in range(dim_size):
            self_score = x[tgt_node_idx] # h nq
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(self_score)
                continue
            # Msg h nq
            msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
            msgs = msgs.sum(dim=0) 
            out.append(msgs + self_score)
        return torch.stack(out, dim=0).flatten(1)

@register_graphLayer
def grounding_v1_multihead_noysyp(configs):
    return Grounding_v1_multihead_noysyp(d_model=configs['d_model'],
            flow=configs['flow'],
            nheads=configs['nheads'],)

@register_graphLayer
def desends_multihead(configs):
    return Desends_YsYp_multihead(d_model=configs['d_model'],
                                  nheads=configs['nheads'],
                                  dropout=configs['dropout'])
