# coding:utf-8
import re
import sys
import copy
import json
import yaml
import penman
from tqdm import tqdm
from pathlib import Path
from .IO import read_raw_amr_data

from penman.codec import PENMANCodec
from penman import layout
from .get_linearized_alignment import format

def dfs_linearize(graph, use_pointer_tokens):
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}

    codec = PENMANCodec(model=None)
    tree = layout.configure(graph_, top=None, model=codec.model)
    linearized, instance_index, edge_index, attribute_index = format(tree, variables=graph_.variables())
    
    for piece in linearized.split():
        if '(' in piece or ')' in piece or '/' in piece:
            assert len(piece) == 1
        
    linearized_nodes = linearized.split()
    cum_sum = {}
    cum = 0
    for id, split_li in enumerate(linearized_nodes):
        cum_sum[cum] = id
        cum += len(split_li)
        
    for key, index_list in instance_index.items():
        instance_index[key] = [cum_sum[idx] for idx in index_list]
    for key, index_list in edge_index.items():
        edge_index[key] = [cum_sum[idx] for idx in index_list]
    for key, index_list in attribute_index.items():
        attribute_index[key] = [cum_sum[idx] for idx in index_list]

    new_instance_index = copy.deepcopy(instance_index)
    new_edge_index = copy.deepcopy(edge_index)
    new_attribute_index = copy.deepcopy(attribute_index)
    if use_pointer_tokens:
        remap = {}
        for i in range(1, len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes[i - 1]
            if nxt == "/":
                remap[lst] = f"<pointer:{len(remap)}>"
        i = 1
        linearized_nodes_ = [linearized_nodes[0]]
        while i < (len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes_[-1]
            if nxt in remap:
                if lst == "(" and linearized_nodes[i + 1] == "/":
                    nxt = remap[nxt]
                    
                    # 因为除去了 /, 所以所有大于它的index都要减去1
                    for var, n_idx in instance_index.items():
                        for iii in range(len(n_idx)):
                            if n_idx[iii] > (i + 1):
                                new_instance_index[var][iii] -= 1
                    
                    for edge, e_idx in edge_index.items():
                        for iii in range(len(e_idx)):
                            if e_idx[iii] > (i + 1):
                                new_edge_index[edge][iii] -= 1
                                
                    for source, a_idx in attribute_index.items():
                        for iii in range(len(a_idx)):
                            if a_idx[iii] > (i + 1):
                                new_attribute_index[source][iii] -= 1
                    i += 1
                elif lst.startswith(":"): #:ARG0 m, coreference
                    nxt = remap[nxt]
            linearized_nodes_.append(nxt)
            i += 1
        linearized_nodes = linearized_nodes_
    return linearized_nodes, remap, new_instance_index, new_edge_index, new_attribute_index


# if __name__ == "__main__":
#     from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#     parser = ArgumentParser(
#         description="AMR processing script",
#         formatter_class=ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument('--config', type=Path, default='default.yaml',
#         help='Use the following config for hparams.')
#     parser.add_argument('--input_file', type=str,
#         help='The input AMR file.')
#     parser.add_argument('--output_prefix', type=str,
#         help='The output_prefix.')
    
#     args, unknown = parser.parse_known_args()

#     with args.config.open() as y:
#         config = yaml.load(y, Loader=yaml.FullLoader)

#     remove_pars = False
#     use_pointer_tokens = True
#     graphs = read_raw_amr_data(
#         [args.input_file],
#         use_recategorization=config["use_recategorization"],
#         remove_wiki=config["remove_wiki"],
#         dereify=config["dereify"],
#     )

#     line_amr = []

#     for g in tqdm(graphs):
#         lin_tokens, pointer_variable_mapping, node_index, edge_index, attribute_index = dfs_linearize(g)
        
#         line_amr.append({
#             'linearized_amr': " ".join(lin_tokens),
#             'var_pointer_map': pointer_variable_mapping,
#             'instance_linearized_ali': node_index,
#             'edge_linearized_ali': edge_index,
#             'attribute_linearized_ali': attribute_index,
#             'sentence': g.metadata["snt"]
#         })

