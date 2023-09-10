from typing import Optional, Union, List, Iterable

from penman.types import BasicTriple
from penman.tree import (Tree, is_atomic)
from collections import defaultdict

# triples: var/var_value; var :rel var2; var :rel constant;
# instances + attributes = nodes


instances_index = defaultdict(list) # (var, :instances, var_value): [val_index, var_value_index]
edge_index = defaultdict(list) # (src, relation, tgt_node): [relation_index, tgt_index]
attributes_index = defaultdict(list) # (var, :relation, constant) : [realtion_index, constant_index]

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
    global instances_index, edge_index,attributes_index, variable_list
    variable_list = variables
    instances_index = defaultdict(list)
    edge_index = defaultdict(list)
    attributes_index = defaultdict(list)
    assert tree.metadata == {}
    
    part, column = _format_node(tree.node, 0)
    assert column == sum([len(split) for split in part.split()])
    
    return part, instances_index, edge_index, attributes_index

def _format_node(node,
                 column) -> str:
    """
    Format tree *node* into a PENMAN string.
    """
    global instances_index
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
    global edge_index
    role, target = edge # Branch(role, any), Any: target: Node(variable, List[Branch]) / Union[str, int...]

    if role == '/':
        assert type(target) == str # instance
        key = f'{source}, :instance, {target}'
        target_column = column + 1
        instances_index[key] = [source_column, target_column]
        column += len(target) + 1
        branch_string = f' {target!s} ' 
    
    elif role.startswith(':'):
        
        if type(target) == str: # constant/coreference
            if target not in variable_list: 
                key = f'{source}, {role}, {target}'
                attributes_index[key] = [source_column, column, column + len(role)]
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
                
            else: # constant
                key = f'{source}, {role}, {target}'
                edge_index[key] = [source_column, column, column+len(role)]
                column += (len(role) + len(target))
                branch_string = f' {target!s} '
            
        elif type(target) == tuple and (len(target) > 0): # edge
            key = f'{source}, {role}, {target[0]}'
            edge_index[key] = [source_column, column, column + len(role) + 1]
            
            column += len(role)
            branch_string, column = _format_node(target, column)
            assert branch_string[0] == '('
            
        else:
            raise ValueError()
    else:
        raise ValueError()
        
    return f' {role} {branch_string} ', column
