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

from typing import Optional, Union, List, Iterable
from collections import defaultdict
from penman.tree import (Tree, is_atomic)

instance_links = defaultdict(dict) # (var, :instances, var_value): [val_index, var_value_index]
role_links = defaultdict(dict) # (src, relation, tgt_node): [relation_index, tgt_index]
attribute_links = defaultdict(dict) # (var, :relation, constant) : [realtion_index, constant_index]


variable_list = []

def format(tree: Tree, variables) -> str:
    """
    Format *tree* into a PENMAN string.

    Args:
        tree: a Tree object
        indent: how to indent formatted strings
        compact: if ``True``, put initial attributes on the first line
    Returns:
        the PENMAN-serialized string of the Tree *t*
    Example:
        >>> import penman
        >>> print(penman.format(
        ...     ('b', [('/', 'bark-01'),
        ...            (':ARG0', ('d', [('/', 'dog')]))])))
        (b / bark-01
           :ARG0 (d / dog))
    """
    global instance_links, role_links, attribute_links, variable_list
    variable_list = variables
    instance_links = defaultdict(dict)
    role_links = defaultdict(dict)
    attribute_links = defaultdict(dict)
    assert tree.metadata == {}
    
    part, column = _format_node(tree.node, 0)
    assert column == sum([len(split) for split in part.split()])
    
    return part, instance_links, role_links, attribute_links

def _format_node(node,
                 column) -> str:
    """
    Format tree *node* into a PENMAN string.
    """
    var, edges = node # (var, list[Branch]), Branch: Tuple[Role, Any]
    
    assert len(edges) > 0 

    parts = []
    source_column = column + 1
    
    column += len(str(var)) + 1
    for edge in edges:
        part, column = _format_edge(edge, column, source=var, source_column=source_column)
        parts.append(part)
    column += 1
    return f'( {var!s} {" ".join(parts)} )', column

def _format_edge(edge, column, source, source_column):
    global role_links, instance_links, attribute_links
    role, target = edge # Branch(role, any), Any: target: Node(variable, List[Branch]) / Union[str, int...]

    if role == '/':
        assert type(target) == str # instance
        key = f'{source}, :instance, {target}'
        target_column = column + 1
        instance_links[key]['ali'] = [source_column, target_column]
        instance_links[key]['is_coreference'] = 0

        column += len(target) + 1
        branch_string = f' {target!s} ' 
    
    elif role.startswith(':'):
        
        if type(target) == str: # constant/coreference
            if target not in variable_list:  # constant
                key = f'{source}, {role}, {target}'
                attribute_links[key]['ali'] = [source_column, column, column + len(role)]
                attribute_links[key]['is_coreference'] = 0
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
                
            else: # coference
                key = f'{source}, {role}, {target}'
                role_links[key]['ali'] = [source_column, column, column+len(role)]
                role_links[key]['is_coreference'] =  1
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
            
        elif type(target) == tuple and (len(target) > 0): # edge
            key = f'{source}, {role}, {target[0]}'
            role_links[key]['ali'] = [source_column, column, column + len(role) + 1]
            role_links[key]['is_coreference'] = 0
            column += len(role)
            branch_string, column = _format_node(target, column)
            assert branch_string[0] == '('
            
        else:
            raise ValueError()
    else:
        raise ValueError()
        
    return f' {role} {branch_string} ', column


def dfs_linearize(graph, use_pointer_tokens):
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}

    codec = PENMANCodec(model=None)
    tree = layout.configure(graph_, top=None, model=codec.model)
    linearized, instance_links, role_links, attrubite_links = format(tree, variables=graph_.variables())
    # key: dict['ali':list[int], 'is_coreference':0/1]
    
    for piece in linearized.split():
        if '(' in piece or ')' in piece or '/' in piece:
            assert len(piece) == 1
        
    linearized_nodes = linearized.split()
    cum_sum = {}
    cum = 0
    for id, split_li in enumerate(linearized_nodes):
        cum_sum[cum] = id
        cum += len(split_li)
        
    # 把column alignment换成 单词alignment
    for key, link_dict in instance_links.items():
        instance_links[key]['ali'] = [cum_sum[idx] for idx in link_dict['ali']]
    for key, link_dict in role_links.items():
        role_links[key]['ali'] = [cum_sum[idx] for idx in link_dict['ali']]
    for key, link_dict in attrubite_links.items():
        attrubite_links[key]['ali'] = [cum_sum[idx] for idx in link_dict['ali']]

    new_instance_links = copy.deepcopy(instance_links)
    new_role_links = copy.deepcopy(role_links)
    new_attrubite_links = copy.deepcopy(attrubite_links)
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
                    for key, link_dict in instance_links.items():
                        n_idx = link_dict['ali']
                        for iii in range(len(n_idx)):
                            if n_idx[iii] > (i + 1):
                                new_instance_links[key]['ali'][iii] -= 1
                    
                    for key, link_dict in role_links.items():
                        e_idx = link_dict['ali']
                        for iii in range(len(e_idx)):
                            if e_idx[iii] > (i + 1):
                                new_role_links[key]['ali'][iii] -= 1
                                
                    for key, link_dict in attrubite_links.items():
                        a_idx = link_dict['ali']
                        for iii in range(len(a_idx)):
                            if a_idx[iii] > (i + 1):
                                new_attrubite_links[key]['ali'][iii] -= 1
                    i += 1
                elif lst.startswith(":"): #:ARG0 m, coreference
                    nxt = remap[nxt]
            linearized_nodes_.append(nxt)
            i += 1
        linearized_nodes = linearized_nodes_
    return linearized_nodes, remap, new_instance_links, new_role_links, new_attrubite_links


from datasets.preprocess.AMR_Process.penman_interface import _remove_wiki
def dfs_linearized_penmen(amr_penman_string, 
                        use_pointer_tokens,
                        remove_pars=None,
                        use_recategorization = False,
                        remove_wiki = False,
                        dereify= False,):
    amr_graph = penman.decode(amr_penman_string)
    if remove_wiki:
        amr_graph = _remove_wiki(amr_graph)
    if use_recategorization:
        metadata = amr_graph.metadata
        metadata["snt_orig"] = metadata["snt"]
        tokens = eval(metadata["tokens"])
        metadata["snt"] = " ".join(
            [
                t
                for t in tokens
                if not ((t.startswith("-L") or t.startswith("-R")) and t.endswith("-"))
            ]
        )    
    lin_tokens, pointer_variable_mapping, instance_links, role_links, attribute_links = dfs_linearize(amr_graph,
                                                                                                    use_pointer_tokens=use_pointer_tokens)
    
    # 去掉alignment
    alignment_pattern = re.compile(r'~(?:[a-z]\.?)?(?P<ali>[0-9]+)(?:,[0-9]+)*')
            
    new_instance_links = {}
    for ins_key, link_dict in instance_links.items():
        ins_key = re.sub(alignment_pattern, repl='', string=ins_key)
        new_instance_links[ins_key] = link_dict
    
    new_attribute_links = {}  
    for attr_key, link_dict in attribute_links.items():
        attr_key = re.sub(alignment_pattern, repl='', string=attr_key)
        new_attribute_links[attr_key] = link_dict
    
    new_role_links = {}  
    for edge_key, link_dict in role_links.items():
        edge_key = re.sub(alignment_pattern, repl='', string=edge_key)
        new_role_links[edge_key] = link_dict  
    
    amr_tree_string_linearized = ' '.join(lin_tokens)
    
    return amr_tree_string_linearized, pointer_variable_mapping, new_instance_links, new_attribute_links, new_role_links  
