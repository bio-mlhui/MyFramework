
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

def bfs_traversal(edge_index):
    # E 2
    # top is 0
    edge_index = edge_index.permute(1,0)
    edge_to_eid = {f'{(ei[0].item(),ei[1].item())}':id for id, ei in enumerate(edge_index)}
    pids = edge_index[:, 0].unique()
    nodeid_to_childs = {}
    for pid in pids:
        childs = [ei[1] for ei in edge_index if ei[0] == pid]
        nodeid_to_childs[pid] = childs

    edge_depth = [None] * len(edge_index)
    current_depth = 1
    frontiers = [0]
    while len(nodeid_to_childs) != 0:
        for ft in frontiers:
            if ft not in nodeid_to_childs:
                continue
            childs = nodeid_to_childs[ft]
            for chi in childs:
                edge_depth[edge_to_eid[f'{(ft, chi)}']] = current_depth

def get_edge_depth(amrs):
    edge_index_by_batch = [g.edge_index for g in amrs]
    nx_graphs = [nx.DiGraph(list(zip(ei[1, :], ei[0, :])) for ei in edge_index_by_batch)]

    # top都是0
    
    depth_by_batch = [max(ed) for ed in edge_depth_by_batch]
    max_depth = max(depth_by_batch)
    added_depth_by_batch = [max_depth - dbb for dbb in depth_by_batch]
    edge_depth_by_batch = [edbb + adbb for edbb, adbb in zip(edge_depth_by_batch, added_depth_by_batch)]
    edge_depth = torch.cat(edge_depth_by_batch)
    return depth

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

class Grounding_v1_multihead_v2(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 nheads,
                 flow='source_to_target',
                 self_score='dot', # dot/zero
                 score_aggr='sum',
                 random_drop=False,
                 drop_p=None,
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        self.head_dim = d_model // nheads
        self.nheads = nheads
        self.random_drop = random_drop
        self.drop_p=drop_p
        
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 2*self.head_dim, self.head_dim])) # 1 h 2c c
        glorot(self.context_2)
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.context_1)

        self.self_score = self_score
        if self_score == 'dot':
            self.ref_2 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, self.head_dim])) # 1 h c c
            self.ref_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
            glorot(self.ref_1)
            glorot(self.ref_2)
        elif self_score == 'zero':
            pass
        else:
            raise ValueError()
        
        self.score_aggr = score_aggr
        
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
        num_nodes, nq, _ = node_query_feats.shape
        node_query_feats = rearrange(node_query_feats, 'V nq (h c) -> V h nq c',h=self.nheads)
        if self.self_score == 'dot':
            node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
            # intialize score V h_nq
            # S(xi, v) = S_s(xi, y_s^v)
            # V h nq c @ 1 h c c -> V h nq c
            # V h nq c * V h 1 c -> V h nq c
            ref_score = (node_query_feats @ self.ref_2) * (node_feats.unsqueeze(-2))
            ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
            # V h nq c @ 1 h c 1 -> V h nq 1
            scores = ref_score @ self.ref_1
            # V h_nq
            scores = scores.flatten(1)
        elif self.self_score == 'zero':
            # V h_nq
            scores = torch.zeros([num_nodes, self.nheads * nq]).float().to(device)

        # 如果整个batch都没有edge                                                                             
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
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_nq=node_query_feats.flatten(1), # V h_nq_c
                                        ) # arguments
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(dim=1) # V nq
    
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i) -> Tensor:
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
        
        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,1, nq,1) # E h nq c
        cat_feat = torch.cat([context_feat_j, node_nq_i], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h 1 c -> E h nq c
        context_score = (cat_feat @ self.context_2) * (edge_attr.unsqueeze(-2))
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)
        return context_score.flatten(1)
    
    def aggregate(self, 
                  inputs, # E h_nq
                  x, # V h_nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        x = rearrange(x, 'V (h nq) -> V h nq',h=self.nheads)
        inputs = rearrange(inputs, 'E (h nq) -> E h nq',h=self.nheads)
        out = [] # V  h nq
        for tgt_node_idx in range(dim_size):
            # 如果没有节点连向x_i, 那么message就是x_i本身
            if tgt_node_idx not in index:
                out.append(x[tgt_node_idx])
            else:
                self_score = x[tgt_node_idx]
                # Msg+1 h nq
                msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
                node_aggr_scores = torch.cat([msgs, self_score.unsqueeze(0)], dim=0)
                if (self.training) and (self.random_drop):
                    num_msgs = len(node_aggr_scores)
                    save_prob = (1 - self.drop_p) * torch.ones(num_msgs)
                    save_mask = torch.bernoulli(save_prob).bool().to(node_aggr_scores.device)
                    if save_mask.any():
                        dropped_scores = node_aggr_scores[save_mask]
                        out.append(self.aggr_multiple(dropped_scores))
                    else:
                        out.append(torch.zeros_like(node_aggr_scores[0])) # h nq
                else:
                    out.append(self.aggr_multiple(node_aggr_scores))

        return torch.stack(out, dim=0).flatten(1)
    
    def aggr_multiple(self, msgs):
        # msg h nq
        if self.score_aggr == 'sum':
            return msgs.sum(dim=0)
        elif self.score_aggr == 'min':
            num_msgs, head, nq = msgs.shape
            msgs = msgs.flatten(1) # msg nq
            intersect_msgs, _ = msgs.min(dim=0)
            intersect_msgs = rearrange(intersect_msgs, '(h nq) -> h nq',h=head, nq=nq)
            return intersect_msgs
        else:
            raise ValueError()
@register_graphLayer
def grounding_v1_multihead_v2(configs):
    return Grounding_v1_multihead_v2(d_model=configs['d_model'],
                        flow=configs['flow'],
                        self_score=configs['self_score'] if 'self_score' in configs else 'dot',
                        score_aggr=configs['score_aggr'] if 'score_aggr' in configs else 'sum',
                        nheads=configs['nheads'],
                        random_drop=configs['random_drop'] if 'random_drop' in configs else False,
                        drop_p=configs['drop_p'] if 'drop_p' in configs else None)

class Grounding_v1_multihead_v2_MLP(geo_nn.MessagePassing):
    def __init__(self, 
                 d_model,
                 flow='source_to_target',
                 score_aggr='sum',
                 ):
        super().__init__(aggr=None,
                         flow=flow,)
        

        self.query_proj = FeatureResizer(d_model, d_model, dropout=0.0, do_ln=True)

        self.context_2 = MLP(2 * d_model, d_model, d_model, num_layers=3)
        self.context_1 = MLP(d_model, d_model, 1, num_layers=3)

        self.ref_2 = MLP(d_model, d_model, d_model, num_layers=3)
        self.ref_1 = MLP(d_model, d_model, 1, num_layers=3)
        
        self.score_aggr = score_aggr

       
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
        device = edge_feats.device
        node_query_feats, node_query_pos = node_memories['feat'], node_memories['pos']
        node_query_feats = self.with_pos_embed(node_query_feats, node_query_pos) # V nq c

        node_query_feats = self.query_proj(node_query_feats)
        # c -> c
        # c * c -> 1
        scores = self.ref_1(self.ref_2(node_query_feats) * (node_feats.unsqueeze(-2))).squeeze(-1) # V nq

        # 如果整个batch都没有edge                                                                             
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
                # V nq
                scores = self.propagate(edge_index[:, order_eid], 
                                        size=None,
                                        x=scores, # V nq
                                        edge_attr=edge_feats[order_eid, :], # E c
                                        node_nq=node_query_feats.flatten(1), # V nq_c
                                        ) # arguments
        return scores
    
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i) -> Tensor:
        """
        Args:
            edge_attr: E c
            x_j: E nq
            node_nq_j/i: E nqc
        """
        nq = x_j.shape[-1]
        node_nq_i = rearrange(node_nq_i, 'E (nq c) -> E nq c',nq=nq)
        node_nq_j = rearrange(node_nq_j, 'E (nq c) -> E nq c',nq=nq)

        # E 1 nq  @ E nq c -> E 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(1)
        context_feat_j = soft_attn_j @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,nq,1) # E nq c
        cat_feat = torch.cat([context_feat_j, node_nq_i], dim=-1) # E nq 2c

        context_score = self.context_1(self.context_2(cat_feat) * (edge_attr.unsqueeze(1)))
        return context_score.squeeze(-1)
    
    def aggregate(self, 
                  inputs, # E nq
                  x, # V nq
                  index, # E, int 每个信息指向哪个节点
                  dim_size=None):
        out = [] # V  nq
        for tgt_node_idx in range(dim_size):
            if tgt_node_idx not in index:
                out.append(x[tgt_node_idx])
            else:
                self_score = x[tgt_node_idx]
                # Msg+1 nq
                msgs = torch.stack([inputs[idx] for idx in range(len(index)) if index[idx] == tgt_node_idx], dim=0)
                node_aggr_scores = torch.cat([msgs, self_score.unsqueeze(0)], dim=0)
                out.append(self.aggr_multiple(node_aggr_scores))

        return torch.stack(out, dim=0)
    
    def aggr_multiple(self, msgs):
        # msg nq
        if self.score_aggr == 'sum':
            return msgs.sum(dim=0)
        elif self.score_aggr == 'min':
            return msgs.min(dim=0)[0]
        else:
            raise ValueError()
        
@register_graphLayer
def grounding_v1_multihead_v2_mlp(configs):
    return Grounding_v1_multihead_v2_MLP(d_model=configs['d_model'],
                                        flow=configs['flow'],
                                        score_aggr=configs['score_aggr'] if 'score_aggr' in configs else 'sum')
import math
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



@register_graphLayer
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

@register_graphLayer
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


@register_graphLayer
def spatial_temporal_grounding_v3(configs):
    return Spatial_Temporal_Grounding_v3(d_model=configs['d_model'],
                        flow=configs['flow'],
                        score_aggr=configs['score_aggr'],
                        nheads=configs['nheads'],
                        obj_query_proj=configs['obj_query_obj'],
                        temp_query_proj=configs['temp_query_proj'] if 'temp_query_proj' in configs else None,
                        frame_query_proj=configs['frame_query_proj'] if 'frame_query_proj' in configs else None,
                        only_component_2=configs['only_component_2' if 'only_component_2' in configs else False],
                        only_component_1=configs['only_component_1'] if 'only_component_1' in configs else False,
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


@register_graphLayer
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



@register_graphLayer
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
        
@register_graphLayer
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
    

class Grounding_v1_multihead_v3(geo_nn.MessagePassing):
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
        
        self.context_2 = nn.Parameter(torch.zeros([1, self.nheads, 2*self.head_dim, self.head_dim])) # 1 h 2c c
        glorot(self.context_2)
        self.context_1 = nn.Parameter(torch.zeros([1, self.nheads, self.head_dim, 1])) # 1 h c 1
        glorot(self.context_1)

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

        node_feats = rearrange(node_feats, 'V (h c) -> V h c',h=self.nheads)
        # intialize score V h_nq
        # S(xi, v) = S_s(xi, y_s^v)
        # V h nq c @ 1 h c c -> V h nq c
        # V h nq c * V h 1 c -> V h nq c
        ref_score = (node_query_feats @ self.ref_2) * (node_feats.unsqueeze(-2))
        ref_score = ref_score / ref_score.norm(dim=-1, keepdim=True)
        # V h nq c @ 1 h c 1 -> V h nq 1
        scores = ref_score @ self.ref_1
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
                                        edge_attr=edge_feats[order_eid, :], # E hc
                                        node_nq=node_query_feats.flatten(1), # V h_nq_c
                                        ) # arguments
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(dim=1) # V nq
    
    def message(self, edge_attr, x_j, node_nq_j, node_nq_i) -> Tensor:
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
        
        # E h 1 nq @ E h nq c -> E h 1 c
        soft_attn_j = x_j.softmax(-1).unsqueeze(2)
        context_feat_j = soft_attn_j @ node_nq_j
        context_feat_j = context_feat_j.repeat(1,1, nq,1) # E h nq c
        edge_attr = edge_attr.unsqueeze(-2).repeat(1,1,nq, 1) # E h nq c
        cat_feat = torch.cat([context_feat_j, edge_attr], dim=-1) # E h nq 2c

        # E h nq 2c @ 1 h 2c c -> E h nq c
        # E h nq c * E h nq c -> E h nq c
        context_score = (cat_feat @ self.context_2) * node_nq_i
        context_score = context_score / context_score.norm(dim=-1, keepdim=True)
        # E h nq c @ 1 h c 1 -> E h nq 1 -> E h nq
        context_score = (context_score @ self.context_1).squeeze(-1)
        return context_score.flatten(1)
    
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
def grounding_v1_multihead_v3(configs):
    return Grounding_v1_multihead_v3(d_model=configs['d_model'],
                        flow=configs['flow'],
                        nheads=configs['nheads'],)


# 多层
class Score_Module(geo_nn.MessagePassing):
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
                node_scores,
                node_batch_ids=None, edge_batch_ids=None, 
                node_seg_ids=None, edge_seg_ids=None,
                node_feats=None, edge_feats=None,
                edge_memories=None, node_memories=None,
                edge_index=None,
                node_subseqs=None,
                node_dsends=None,
                ys_by_node=None,
                yp_by_node=None):
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

        scores = self.propagate(edge_index, 
                                size=None,
                                x=scores, # V h_nq
                                yp=yp_by_node.flatten(1), # V hc
                                edge_attr=edge_feats, # E hc
                                node_nq=node_query_feats.flatten(1), # V h_nq_c
                                ) # arguments
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(dim=1) + node_scores # V nq
    
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

from .transformer import _get_clones
class Grounding_v1_multihead_multiLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 get_ysyp={'name': 'subseq'},# subseq/topdown_bottomup
                 score_module={'name': 'grounding_v1',},
                 strategies=None,
                 num_layers=None
                 ):
        super().__init__()
        get_ysyp_name = get_ysyp.pop('name')
        if get_ysyp_name == 'descends_multihead':
            self.get_ysyp = Desends_YsYp_multihead(**get_ysyp)
        else:
            raise ValueError()
        score_module = Score_Module(d_model=d_model,
                                         nheads=score_module['nheads'],
                                         flow=score_module['flow'])
        self.score_layers = _get_clones(score_module, num_layers)
        self.num_layers = num_layers
 
    def forward(self,
                node_batch_ids=None, edge_batch_ids=None, 
                node_seg_ids=None, edge_seg_ids=None,
                node_feats=None, edge_feats=None,
                edge_memories=None, node_memories=None,
                edge_index=None,
                edge_depth=None,
                node_subseqs=None,
                node_dsends=None):
        ys_by_node, yp_by_node = self.get_ysyp(node_feats=node_feats, edge_feats=edge_feats,
                                                edge_index=edge_index,
                                                node_subseqs=node_subseqs,
                                                node_dsends=node_dsends)
        for layer_idx in range(self.num_layers):
            scores = self.score_layers[layer_idx](
                                            scores,
                                            node_batch_ids=node_batch_ids, edge_batch_ids=edge_batch_ids, 
                                            node_seg_ids=node_seg_ids, edge_seg_ids=edge_seg_ids,
                                            node_feats=node_feats, edge_feats=edge_feats,
                                            edge_memories=edge_memories, node_memories=node_memories,
                                            edge_index=edge_index,
                                            node_subseqs=node_subseqs,
                                            node_dsends=node_dsends,
                                            ys_by_node=ys_by_node,
                                            yp_by_node=yp_by_node)
        scores = rearrange(scores, 'V (h nq) -> V h nq',h=self.nheads)
        return scores.mean(dim=1) # V nq



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

        return scores # V nq
    
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
