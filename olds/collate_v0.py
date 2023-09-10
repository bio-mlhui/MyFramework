
from util.misc import nested_tensor_from_videos_list
from util.misc import nested_tensor_from_videos_list_with_stride
from torch_geometric.data import Data, Batch
import penman
import torch

class Image_Collator:
    def __call__(self, batch):
        # batch: imgs, targets
        batch = list(zip(*batch)) 
        batch[0] = nested_tensor_from_videos_list(batch[0], size_divisibility=32) 
        # batch[0]: samples: NestedTensor(tensor, mask)
        #           tensor: [B, T, C, H, W], mask: [B, T, H, W]
        # batch[1]: targets: list[dict]
        return tuple(batch) 

class Video_Collator:
    def __call__(self, batch):
        # list[T(t 3 hi wi)], list[ [None,..,dict,..None] ], list[str], list[dict]
        samples, targets, text_query, auxiliary  = list(zip(*batch))
        samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=32)
        *_, H, W = samples.tensors.shape
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        
        batch_dict = {
            'samples': samples, # NT(t b 3 h w)
            'targets': targets, # list[[None...None], [dict...dict]], t
            'text_query': text_query, # list[str], b,
            'auxiliary':{
                'exist_queries': [s_dic['exist_queries'] for s_dic in auxiliary], # list[list[str], N], b
                'amr_tree_string': [s_dic['amr_tree_string'] for s_dic in auxiliary], # list[Graph], b
                'amr_tree_string_linearization_dict': [s_dic['amr_tree_string_linearization_dict'] for s_dic in auxiliary], # list[dict], b
                'first_noun': [s_dic['first_noun'] for s_dic in auxiliary]
            }
        }
        return batch_dict


def penman_to_graph(amr_string):
    amr = penman.decode(amr_string)
    nodes_id = {var:id for id, var in enumerate(amr.variables())} 
    nodes_concept = [] # node_id:concept
    edges_role = [] # edge_id:role
    
    
    for node_id, node in enumerate(amr.instances()):
        nodes_id[node.source]= node_id
        nodes_concept.append(node.target.split('-')[0])
    connectivity = []
    for edge_id, edge in enumerate(amr.edges()):
        connectivity.append([nodes_id[edge.source], nodes_id[edge.target]])
        edges_role.append(edge.role)
    connectivity = torch.tensor(connectivity, dtype=torch.long)
    
    return Data(edge_index=connectivity,
                nodes_concept = nodes_concept,
                edges_role = edges_role)
