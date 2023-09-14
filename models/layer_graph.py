
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
# norm
# 使用memory
from torch_geometric.nn import Aggregation
import dgl
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