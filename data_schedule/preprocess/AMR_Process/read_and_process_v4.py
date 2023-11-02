# coding:utf-8
import re
import copy
from penman.codec import PENMANCodec
from penman import layout

from collections import defaultdict
from penman.tree import (Tree, is_atomic)
import networkx as nx
import penman
instances_index = defaultdict(list) # (var, :instances, var_value): [val_index, var_value_index]
edge_index = defaultdict(list) # (src, relation, tgt_node): [relation_index, tgt_index]
attributes_index = defaultdict(list) # (var, :relation, constant) : [realtion_index, constant_index]

nx_graph = nx.DiGraph()

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
    global instances_index, edge_index, attributes_index, variable_list, nx_graph
    nx_graph = nx.DiGraph(top=None)
    variable_list = variables
    instances_index = defaultdict(list)
    edge_index = defaultdict(list)
    attributes_index = defaultdict(list)
    assert tree.metadata == {}
    
    part, column = _format_node(tree.node, 0)
    # assert column == sum([len(split) for split in part.split()])
    
    return nx_graph

def _format_node(node, column) -> str:
    """
    Format tree *node* into a PENMAN string.
    """
    var, edges = node # (var, list[Branch]), Branch: Tuple[Role, Any]
    global nx_graph
    assert var in variable_list and len(edges) > 0
    if var not in nx_graph.nodes:
        nx_graph.add_node(var, seg_id=1, alignment=-100)
        if nx_graph.graph['top'] is None:
            nx_graph.graph['top'] = var
        
    parts = []
    source_column = column + 1
    column += len(str(var)) + 1
    for edge in edges:
        part, column = _format_edge(edge, column, source=var, source_column=source_column)
        parts.append(part)
    column += 1
    return f'( {var!s} {" ".join(parts)} )', column

def _format_edge(edge, column, source, source_column):
    alignment_pattern = re.compile(r'~(?:[a-z]\.?)?(?P<ali>[0-9]+)(?:,[0-9]+)*')
    global edge_index, instances_index, attributes_index, nx_graph
    
    role, target = edge # Branch(role, any), Any: target: Node(variable, List[Branch]) / Union[str, int...]
    role = re.sub(alignment_pattern, "", role)
    role_of = role[:-3] if '-of' in role else f'{role}-of'
    
    if role == '/':
        assert type(target) == str # instance
        match_res = re.search(alignment_pattern, target)
        ali = int(match_res.group(1))
        target = re.sub(alignment_pattern, "", target)
        key = f'{source}, :instance, {target}'
        target_column = column + 1
        instances_index[key] = [source_column, target_column]

        if target not in nx_graph.nodes:
            nx_graph.add_node(target, seg_id=2, alignment=ali)
        nx_graph.add_edge(target, source, role=':instance', seg_id=-2)
        
        column += len(target) + 1
        branch_string = f' {target!s} ' 
    
    elif role.startswith(':'):
        
        if type(target) == str: # constant/coreference
            ori_target = copy.deepcopy(target)
            target = re.sub(alignment_pattern, "", target)
            if target not in variable_list:  # constant
                match_res = re.search(alignment_pattern, ori_target)
                ali = int(match_res.group(1))
                
                key = f'{source}, {role}, {target}'
                attributes_index[key] = [source_column, column, column + len(role)]
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
                if target not in nx_graph.nodes:
                    nx_graph.add_node(target, seg_id=3, alignment=ali)
                nx_graph.add_edge(target, source, role=role_of, seg_id=-3)
                
            else: # coference
                key = f'{source}, {role}, {target}'
                edge_index[key] = [source_column, column, column+len(role)]
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
                
                if target not in nx_graph.nodes:
                    nx_graph.add_node(target, seg_id=1, alignment=-100)
                    nx_graph.add_edge(target, source, role=role_of, seg_id=-1)
                else:
                    # 需要确定source的所有parents没有target
                    if target in nx.descendants(nx_graph, source):
                        nx_graph.add_edge(source, target, role=role, seg_id=-1)
                    else:
                        nx_graph.add_edge(target, source, role=role_of, seg_id=-1)
                
        elif type(target) == tuple and (len(target) > 0): # edge
            target_variable = target[0]
            target_variable = re.sub(alignment_pattern, "", target_variable)
            
            key = f'{source}, {role}, {target_variable}'
            edge_index[key] = [source_column, column, column + len(role) + 1]
            
            if target_variable not in nx_graph.nodes:
                nx_graph.add_node(target_variable, seg_id=1, alignment=-100)
                nx_graph.add_edge(target_variable, source, role=role_of, seg_id=-1)
            else:
                # 需要确定source的所有parents没有target
                if target_variable in nx.descendants(nx_graph, source):
                    nx_graph.add_edge(source, target_variable, role=role, seg_id=-1)
                else:
                    nx_graph.add_edge(target_variable, source, role=role_of, seg_id=-1)
            
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
    return format(tree, variables=graph_.variables())


from data_schedule.preprocess.AMR_Process.penman_interface import _remove_wiki
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
    return  dfs_linearize(amr_graph, use_pointer_tokens=use_pointer_tokens)
