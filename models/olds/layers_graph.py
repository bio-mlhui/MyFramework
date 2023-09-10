_graph_layer_entrypoints = {}

def register_graph_layer(fn):
    graph_layer_name = fn.__name__
    _graph_layer_entrypoints[graph_layer_name] = fn

    return fn

def graph_layer_entrypoints(graph_layer_name):
    try:
        return _graph_layer_entrypoints[graph_layer_name]
    except KeyError as e:
        print(f'graph_layer {graph_layer_name} not found')

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as geo_nn


# 假设: 假设在此之前用了cross attention, 即每个特征是 一个概率加权
# graph: 是一棵树, root->leaf, 即source是parent, target是child
# 维度: edge和node具有相同的维度, 输出的维度也是一样
# 改变: 不改变edge的特征
# aggregation: 想找到red shirt; shirt是一个attention分布p1; red是一个attention分布p2; red shirt的attention分布应该是min(p1, p2)
# flow: target2source, 因为是一棵树
# message: s(shirt) --(:mod)-> r(red); 每个message是entity, 而不是descriable.
# residual链接
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
    
    def forward(self, x, edge_index, edge_attr, multimodal_features):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        out = self.propagate(edge_index, size=None,  # keywords
                             x=x, edge_attr=edge_attr_2) # arguments
    
        # residual
        return out + x, edge_attr
    
  
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
 
@register_graph_layer
def graph_layer_v1(configs, d_model):
    return Graph_Layer_v1(
        d_model=d_model,
        flow=configs.flow,
        aggr=configs.aggr,
        
    )

class TreeLSTM_layer_pyg(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_lin = nn.Linear(d_model, d_model, bias=False) 
        self.edge_lin = nn.Linear(d_model, d_model, bias=False) 
        self.message_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
     
    def forward(self, x, edge_index, edge_attr):
        device = x.device
        dgl_graph = dgl.graph((edge_index[0, :], edge_index[1, :]))
        try:
            traversal_order = dgl.topological_nodes_generator(dgl_graph)
        except:
            exit()
        orders = []
        for idx, frontier in enumerate(traversal_order):
            if idx == 0:
                src, tgt, eid =  dgl_graph.in_edges(frontier.to(device), form='all')
                assert len(src) == 0 and len(tgt) == 0 and len(eid) == 0
                continue
            src, tgt, eid =  dgl_graph.in_edges(frontier.to(device), form='all')
            orders.append(eid)
            
        x = self.node_lin(x)
        edge_attr_2 = self.edge_lin(edge_attr) # |E| c
        
        for edge_order in orders:
            x = self.propagate(edge_index[:, edge_order], 
                               size=None,  # keywords
                                x=x, edge_attr=edge_attr_2[edge_order, :]) # arguments

        return x, edge_attr
    
  
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
 

@register_graph_layer
def graph_layer_v2(configs, d_model):
    return TreeLSTM_layer_pyg(
        d_model=d_model,
        aggr=configs.aggr,
        flow='source_to_target'
    )
            


class Graph_Layer_v3(nn.Module):
    def __init__(self,
                 d_model,
                 aggr
                 ) -> None:
        super().__init__()
        self.source_to_target_conv = Graph_Layer_v1(d_model=d_model, flow='source_to_target', aggr=aggr)
        self.target_to_source_conv = Graph_Layer_v1(d_model=d_model, flow='target_to_source', aggr=aggr)
        
    def forward(self, x, edge_index, edge_attr,multimodal_features):
        """
        x: num_nodes c
        edge_index: 2 num_edges
        edge_attr: num_edges 2*c
        """
        tgt2src_nodes_feats = self.target_to_source_conv(x=x.clone(),
                                                        edge_attr=edge_attr.clone(),
                                                        edge_index=edge_index)
        src2tgt_nodes_feats = self.source_to_target_conv(x=x.clone(),
                                                        edge_attr=edge_attr.clone(),
                                                        edge_index=edge_index)
          
        return (src2tgt_nodes_feats + tgt2src_nodes_feats ) / 2, edge_attr

 
@register_graph_layer
def graph_layer_v3(configs, d_model):
    return Graph_Layer_v3(
        d_model=d_model,
        aggr=configs.aggr,
    )



class Graph_Layer_v4(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(2*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr,multimodal_features):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        
        
        out = self.propagate(edge_index, size=None,  # keywords
                             x=x, edge_attr=edge_attr_2) # arguments

        # residual
        return out + x, edge_attr
    
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
        message = torch.cat([x_j, edge_attr], dim=-1) # E 3*dim
        message = self.message_linear(message)  # E c
        return message
 
@register_graph_layer
def graph_layer_v4(configs, d_model):
    return Graph_Layer_v4(
        d_model=d_model,
        flow=configs.flow,
        aggr=configs.aggr,
        
    )


class Graph_Layer_v5(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.edge_update_linear = nn.Linear(3*d_model, d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr,multimodal_features):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.node_linear(x)
        edge_attr_2 = self.edge_updater(edge_index=edge_index,
                                      x=x,
                                      edge_attr=edge_attr.clone())
        edge_attr = edge_attr + edge_attr_2

        x_2 = self.propagate(edge_index, size=None,  # keywords
                             x=x, edge_attr=edge_attr) # arguments
        x = x + x_2
        # residual
        return x, edge_attr
    
    def edge_update(self, x_j, x_i, edge_attr) -> Tensor:
        return self.edge_update_linear(torch.cat([x_j, x_i, edge_attr], dim=-1))
    
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
 
@register_graph_layer
def graph_layer_v5(configs, d_model):
    return Graph_Layer_v5(
        d_model=d_model,
        flow=configs.flow,
        aggr=configs.aggr,
        
    )


class Graph_Layer_v6(nn.Module):
    def __init__(self, configs, d_model) -> None:
        super().__init__()
        self.layer = geo_nn.GATConv(
        in_channels=d_model,
        out_channels=d_model,
        heads=configs.nheads,
        concat=configs.concat,
        negative_slope=configs.negative_slope,
        add_self_loops=configs.add_self_loops,
        dropout=configs.dropout,
        edge_dim=d_model,
        fill_value=configs.fill_value,
        bias=configs.bias,
        flow=configs.flow
    )
        
    def forward(self, x, edge_index, edge_attr,multimodal_features):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.layer(x, edge_index, edge_attr.clone())
        
        return x, edge_attr

@register_graph_layer
def graph_layer_v6(configs, d_model):
    return Graph_Layer_v6(configs=configs, d_model=d_model
    )
    

class Graph_Layer_v7(nn.Module):
    def __init__(self, configs, d_model) -> None:
        super().__init__()
        self.layer = geo_nn.GATv2Conv(
        in_channels=d_model,
        out_channels=d_model,
        heads=configs.nheads,
        concat=configs.concat,
        negative_slope=configs.negative_slope,
        add_self_loops=configs.add_self_loops,
        dropout=configs.dropout,
        edge_dim=d_model,
        fill_value=configs.fill_value,
        bias=configs.bias,
        share_weights=False,
        flow=configs.flow
    )
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        
        out: |V| dim_out
        """
        x = self.layer(x, edge_index, edge_attr.clone())
        
        return x, edge_attr

@register_graph_layer
def graph_layer_v7(configs, d_model):
    return Graph_Layer_v7(configs=configs, d_model=d_model
    )


class Graph_Layer_v8(geo_nn.MessagePassing):
    def __init__(self,
                 d_model,
                 flow,
                 aggr):
        super().__init__(aggr=aggr, 
                         flow=flow)   
        self.node_linear = nn.Linear(d_model, d_model, bias=False) 
        self.edge_linear = nn.Linear(d_model, d_model, bias=False)
        self.message_linear = nn.Linear(3*d_model, d_model)
        self.multimodal_linear = nn.Linear(d_model, d_model)
        # self.norm = nn.LayerNorm(d_model)
    
    def reset_parameters(self):
        self.message_linear.bias.data.zero_()
    
    def forward(self, x, edge_index, edge_attr, multimodal_features):
        """
        x: |V| dim_in
        edge_index: 2 |E|
        edge_embeds: |E| dim_in
        multimodal_features: thw b c
        out: |V| dim_out
        """
        x = self.node_linear(x)
        edge_attr_2 = self.edge_linear(edge_attr)
        multimodal_features_2 = self.multimodal_linear(multimodal_features)
        
        out = self.propagate(edge_index, size=None,  # keywords
                             x=x, edge_attr=edge_attr_2, multimodal_features=multimodal_features_2) # arguments
    
        # residual
        return out + x, edge_attr
    
  
    def message(self, x_j, x_i, edge_attr, multimodal_features):
        """ Compute message for each neighbord
        Input:
            x_j: E dim
            x_i: E dim
            edge_embeds: E dim
            multimodal_features: E thw c
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
        message = self.message_linear(message)  # E 1 c
        soft_attention = torch.einsum('eac,ebc->eab', message.unsqueeze(1), multimodal_features)
        soft_attention = F.softmax(soft_attention, dim=-1)
        message_2 = torch.einsum('eab,ebc->eac',soft_attention, multimodal_features).squeeze(1) # E c
        # message = self.norm(message + message_2)
        message = message + message_2
        return message
 
@register_graph_layer
def graph_layer_v8(configs, d_model):
    return Graph_Layer_v8(
        d_model=d_model,
        flow=configs.flow,
        aggr=configs.aggr,
        
    )


















from dgl.utils import expand_as_pair
from dgl import DGLGraph
import torch.nn.functional as F
import dgl
import dgl.function as fn
# user-defined message passing
def udf_u_mul_e(edges):
    return {'m': edges.src['h'] * edges.data['w']}
# user-defined reduce
def udf_max(nodes):
   return {'h_max' : torch.max(nodes.mailbox['m'], 1)[0]}


# role的特征就是word embedding, 可能的改进: :ARG0是predicate-specific role
# 只有一个方向, 即从leaf到root, 可能的改进: bi-TreeLSTM
# messag是linear([src, tgt, role])
# 推断的时候, leaf没有更新
from dgl.udf import EdgeBatch, NodeBatch
class TreeLSTM_layer_util(nn.Module):
    def __init__(
        self,
        d_model,
    ):
        super().__init__()
        self.message_linear = nn.Linear(3*d_model, d_model, bias=False)
        
    def message_func(self, edges: EdgeBatch):
        print('tes')
        print(edges.edges())
        print(edges.data.keys())
        role_feat = edges.data['e']  # 要传送message的edge的数量 c
        src_feat = edges.src['x'] 
        tgt_feat = edges.dst['x']
        message = self.message_linear(torch.cat([src_feat, tgt_feat, role_feat], dim=-1))
        return {"m": message}  
    
    def apply_node_func(self, nodes):
        return {'x': nodes.data['x']}  

class TreeLSTM_layer_dgl(nn.Module):
    def __init__(self, d_model, reduce) -> None:
        super().__init__()
        self.reduce = reduce
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.cell = TreeLSTM_layer_util(d_model=d_model)
        
    # def reduce_func(self, nodes : NodeBatch):  # data: N 1; mailbox: N D 1
    #     if self.reduce == 'min': # N D 1
    #         return {'x': torch.min(nodes.mailbox['m'], dim=1)[0]}
    #     else:
    #         raise ValueError()
    

        
    def batch_graph(self, connectivitys, all_seg_ids, all_feats):
        batch_size, device = len(connectivitys), all_feats.device
        
        # build graphs
        graphs = []
        for connectivity, seg_ids, feats in zip(connectivitys, all_seg_ids, all_feats.permute(1, 0, 2)):
            is_node, is_edge = seg_ids > 0, seg_ids < 0
            node_feats = feats[is_node]
            edge_feats = feats[is_edge]

            graph : DGLGraph = dgl.graph((connectivity[0, :], connectivity[1, :]))
            graph.ndata['x'] = node_feats
            graph.edata['e'] = edge_feats
            graphs.append(graph)
        # The numbers of nodes and edges of the input graphs are accessible via the
        # DGLGraph.batch_num_nodes and DGLGraph.batch_num_edges attributes of the resulting grap
        graphs = dgl.batch(graphs)
        return graphs
    
    def forward(self, connectivitys, all_seg_ids, all_feats):
        """
        connectivitys: list[T(2 |E|)]
        """
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        all_feats = self.lin(all_feats)
        graphs : DGLGraph = self.batch_graph(connectivitys, all_seg_ids, all_feats.clone())

        dgl.prop_nodes_topo(
            graph=graphs,
            message_func=self.cell.message_func,
            reduce_func=dgl.function.min('m', 'x'),
            apply_node_func=self.cell.apply_node_func
            # reverse=True
        )
        
        unbatched_graphs = dgl.unbatch(graphs)
        
        for batch_idx, seg_ids in enumerate(all_seg_ids):
            max_len = seg_ids.shape[0]
            node_index, edge_index = torch.arange(max_len, device=all_feats.device)[seg_ids > 0], torch.arange(max_len, device=all_feats.device)[seg_ids < 0]
            
            all_feats[node_index, batch_idx] = unbatched_graphs[batch_idx].ndata.pop('x')
            all_feats[edge_index, batch_idx] = unbatched_graphs[batch_idx].edata.pop('e')
            
        return all_feats



# @register_graph_layer
# def graph_layer_v5(configs, d_model):
#     return geo_nn.GATConv(
#         in_channels=d_model,
#         out_channels=d_model,
#         heads=configs.nheads,
#         concat=configs.concat,
#         negative_slope=configs.negative_slope,
#         add_self_loops=configs.add_self_loops,
#         dropout=configs.dropout,
#         edge_dim=d_model,
#         fill_value=configs.fill_value,
#         bias=configs.bias,
#         flow=configs.flow
#     )



# @register_graph_layer
# def graph_layer_v6(configs, d_model):
#     return geo_nn.GATConv(
#         in_channels=d_model,
#         out_channels=d_model,
#         heads=configs.nheads,
#         concat=configs.concat,
#         negative_slope=configs.negative_slope,
#         add_self_loops=configs.add_self_loops,
#         dropout=configs.dropout,
#         edge_dim=d_model,
#         fill_value=configs.fill_value,
#         bias=configs.bias,
#         flow=configs.flow
#     )

