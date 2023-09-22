
import os
import pandas
import h5py
from glob import glob
import json
import numpy as np
import torch
from transition_amr_parser.parse import AMRParser
import re
from tqdm import tqdm
import networkx as nx
from stanza.server import CoreNLPClient
from penman import surface
import penman
from penman.models.amr import model as penman_amr_model
from data_schedule.preprocess.AMR_Process.read_and_process_v3 import dfs_linearized_penmen
# region
pos_targets = ['NN', 'NNS', 'NNP', 'NNPS']
def get_first_noun(ann, sent:str):
    words = sent.split()
    tokens = ann['tokens'] # list[dict]
    for id, token in enumerate(tokens):
        if token['pos'] in pos_targets:
            assert token['word'] == words[id]
            return id, words[id]
import copy        
def get_root(ann, sent:str):
    words = sent.split()
    tokens = ann['tokens'] # list[dict]
    depparse = ann['basicDependencies']
    depparse = {f"{dep['dependent']-1} {dep['governor']-1}": dep['dep'] for dep in depparse}
    is_compound = False
    for id, token in enumerate(tokens):
        if token['pos'] in pos_targets:
            assert token['word'] == words[id]
            if token['word'] == 'being' and ((id-1) >= 0) and (words[id-1] == 'human'):
                is_compound = True
                return id, 'being', is_compound, depparse
            
            # 第一个名词和之后的某个单词组成compound关系
            out_idx = id
            for dep_key, dep_val in depparse.items():
                dependent, governer = dep_key.split(' ')
                if dep_val == 'compound' and (int(dependent) == out_idx):
                    out_idx = int(governer)
                    is_compound = True
                    break
                
            # while ((cnt+1) < len(words)) and (f"{cnt} {cnt+1}" in depparse) and depparse[f"{cnt} {cnt+1}"] == 'compound':
            #     is_compound = True
            #     cnt += 1
                
            return out_idx, words[out_idx], is_compound, depparse
from collections import defaultdict
#endregion

pt_dir = '/hpc2hdd/home/testuser21/pt/DATA/AMR3.0/models/amr3.0-structured-bart-large-neur-al/seed42'   


def a2ds_normalize_text(text_query):
    # 非法输入
    if text_query == 'The left with yellow t shirt on the left running':
        text_query = 'the man with yellow tshirt on the left running'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1
# from data_schedule.rvos.a2ds_schedule import a2ds_normalize_text
def a2ds_perWindow_perExp(root): 
        parser = AMRParser.from_pretrained('AMR3-structbart-L')
                            
        instances = pandas.read_csv(os.path.join(root, 'a2d_annotation.txt'))
        text_to_aux = defaultdict(dict)
        torch.hub.load
        # for each query
        for text_query in tqdm(instances['query'].to_list()):
            # 非法输入
            text_1 = a2ds_normalize_text(text_query)
            # first one
            if text_1 not in text_to_aux:
                tokens, _ = parser.tokenize(text_1)
                amr_string = parser.parse_sentence(tokens)[0]
                text_to_aux[text_1]['initial_parsed_amr'] = amr_string

            # second one
            text_2 = text_1.replace('left', '@').replace('right', 'left').replace('@', 'right')
            if text_2 not in text_to_aux:
                tokens, _ = parser.tokenize(text_2)
                amr_string = parser.parse_sentence(tokens)[0]
                text_to_aux[text_2]['initial_parsed_amr'] = amr_string
        # for each text
        pos_tagger = CoreNLPClient(annotators=['tokenize','pos','parse', 'depparse'], 
                                        timeout=30000, output_format='json', 
                                        memory='6G',
                                        be_quiet=True,) 
        
        new_text_to_amr_aux = copy.deepcopy(text_to_aux)
        for text_query, aux_dict  in tqdm(text_to_aux.items()):
            initial_parsed_amr = aux_dict['initial_parsed_amr']
            tokenized_string = penman.decode(initial_parsed_amr).metadata['tok']
            # 找到第一个名词
            pos_ann = pos_tagger.annotate(tokenized_string)['sentences'][0]
            if tokenized_string == 'man climbing on right':
                pos_ann['tokens'][0]['pos'] = 'NN'
            try:
                first_noun_idx, first_noun = get_first_noun(pos_ann, tokenized_string)
            except:
                pass
            # 改变amr的top
            amr_graph = penman.decode(initial_parsed_amr, model=penman_amr_model)
            amr_graph.metadata = {}
            alignments = surface.alignments(amr_graph)
            concept_keys = list(alignments.keys())
            concept_start_alignments = [idx for idx, ali in enumerate(list(alignments.values())) if ali.indices[0] == first_noun_idx]
            if len(concept_start_alignments) == 0:
                for idx, key in enumerate(concept_keys):  # 一个句子中出现了两个light, 但是amr parser只识别了后一个light, 并且pos tagger认为第一个是top
                    if key[-1].startswith(first_noun.lower()):
                        concept_start_alignments.append(idx)
            concept_start_alignments = concept_start_alignments[0]
            top_variable, _, top_concept = concept_keys[concept_start_alignments]
            amr_tree_string_change_top = penman.encode(amr_graph, top=top_variable,model=penman_amr_model)
            
            # 把改变了top之后的amr linearize
            amr_tree_string_linearized, pointer_variable_mapping, instances_index, edge_index, attributes_index, nx_graph  \
                    = dfs_linearized_penmen(amr_tree_string_change_top, use_pointer_tokens=True,)

            new_text_to_amr_aux[text_query]['amr_tree_string'] = amr_tree_string_change_top
            new_text_to_amr_aux[text_query]['inference_graph'] = nx.node_link_data(nx_graph)
            new_text_to_amr_aux[text_query]['first_noun'] = top_concept
            new_text_to_amr_aux[text_query]['amr_tree_string_linearization_dict'] = {
                'amr_tree_string_linearized': amr_tree_string_linearized,
                'var_pointer_map': pointer_variable_mapping,
                'instance_linearized_ali':instances_index,
                'edge_linearized_ali':edge_index,
                'attribute_linearized_ali':attributes_index
            }
        with open(os.path.join(root, 'text_to_aux2.json'), 'w') as f:
            json.dump(new_text_to_amr_aux, f)            
        # if os.path.exists(os.path.join(root, 'text_to_aux.json')):
        #     with open(os.path.join(root, 'text_to_aux.json'), 'e') as f:
        #         text_to_aux = json.load(f)
        #     for text_query, amr_aux_dict in new_text_to_amr_aux.items():
        #         assert text_query in text_to_aux # 保证只有text query; hfliped text; 没有数据集中的非法语言
        #         for amr_key, amr_value in amr_aux_dict.items():
        #             assert amr_key not in text_to_aux[text_query]
        #             text_to_aux[text_query][amr_key] = amr_value
        # else:
        #     text_to_aux = new_text_to_amr_aux
        


    
def yrvos_perWindow_perExp(root):         
    # pasing
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    for use in ['test', 'train', 'valid']:
        sentence_annotation_file = os.path.join(root, 'meta_expressions', use, 'meta_expressions.json')
        with open(sentence_annotation_file, 'r') as f:
            json_file = json.load(f)
            
        new_json_file = json_file['videos']
        for video_id, vid_ann in json_file['videos'].items():
            for text_query_id, exp_dict in vid_ann['expressions'].items():
                new_text_query = copy.deepcopy(exp_dict['exp'])
                if new_text_query == 'cannot describe too little':
                    new_text_query = 'an airplane not moving'
                elif new_text_query == 'a red clothe':
                    new_text_query = 'a red cloth'
                normalized_text_query = new_text_query.replace('right most', 'rightmost')
                normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
                normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
                tokens, _ = parser.tokenize(normalized_text_query)
                amr_string = parser.parse_sentence(tokens)[0]
                amr_string = f'# ::snt {text_query}\n' + amr_string
                new_json_file[video_id]['expressions'][text_query_id]['initial_parsed_amr'] = amr_string 
                
        with open(os.path.join(root, 'meta_expressions', use, 'meta_expressinos_parsedAmr.json'), 'w') as f:
            json.dumps({'videos': new_json_file}, f)

    pos_tagger = CoreNLPClient(annotators=['tokenize','pos','parse'], 
                                    timeout=30000, output_format='json', 
                                    memory='6G',
                                    be_quiet=True,endpoint='http://localhost:65222') 
    
    for use in ['test', 'train', 'valid']:
        assert os.path.exists(os.path.join(sentence_annotation_directory, use, 'meta_expressinos_only_parsed.json'))
        with open(os.path.join(sentence_annotation_directory, use, 'meta_expressinos_only_parsed.json'), 'r') as f:
            json_file = json.load(f)
            
        new_json_file = json_file['videos']
        for video_id, vid_ann in json_file['videos'].items():
            for text_query_id, exp_dict in vid_ann['expressions'].items():
                text_query = exp_dict['exp']
                initial_parsed_amr = exp_dict['amr_penman']
                
                if text_query == 'cannot describe too little':
                    parser = AMRParser.from_pretrained('AMR3-structbart-L')
                    text_query = 'an airplane not moving'
                    tokens, _ = parser.tokenize(text_query)
                    initial_parsed_amr = parser.parse_sentence(tokens)[0]
                    initial_parsed_amr = f'# ::snt {text_query}\n' + initial_parsed_amr
                if text_query == 'a red clothe':
                    # parser = AMRParser.from_pretrained('AMR3-structbart-L')
                    text_query = 'a red cloth'
                    tokens, _ = parser.tokenize(text_query)
                    initial_parsed_amr = parser.parse_sentence(tokens)[0]
                    initial_parsed_amr = f'# ::snt {text_query}\n' + initial_parsed_amr
                    
                amr_graph = penman.decode(initial_parsed_amr, model=penman_amr_model)
                # tagging
                amr_toks = amr_graph.metadata['tok']
                pos_ann = pos_tagger.annotate(amr_toks)['sentences'][0]
                
                # change top according to the top
                root_idx, root, is_compound, depparse_result = get_root(pos_ann, amr_toks)
                amr_graph.metadata = {}
                alignments = surface.alignments(amr_graph)
                concept_keys = list(alignments.keys())
                concept_start_alignments = []
                
                for idx, ali in enumerate(list(alignments.values())):
                    if ali.indices[0] == root_idx:
                        concept_start_alignments.append(idx)
                        
                if len(concept_start_alignments) == 0:                        
                    for idx, key in enumerate(concept_keys):  # 一个句子中出现了两个light, 但是amr parser只识别了后一个light, 并且pos tagger认为第一个是top
                        if key[-1].startswith(root):
                            concept_start_alignments.append(idx)
                    if len(concept_start_alignments) == 0:
                        # a motor bike driven -> motorcycle 本来是bike, 被abstract成motorcycle
                        if is_compound:
                            root_idx -= 1
                            root = amr_toks.split(' ')[root_idx]
                            for idx, ali in enumerate(list(alignments.values())):
                                if ali.indices[0] == root_idx:
                                    concept_start_alignments.append(idx)
                            assert len(concept_start_alignments) > 0
                        elif root == 'being': # a black being 
                            root = 'thing'
                            for idx, key in enumerate(concept_keys):  # 一个句子中出现了两个light, 但是amr parser只识别了后一个light, 并且pos tagger认为第一个是top
                                if key[-1].startswith(root):
                                    concept_start_alignments.append(idx)
                        else:
                            # a pair of hands, nmod
                            for dep_key, dep_val in depparse_result.items():
                                dependent, governer = dep_key.split(' ')
                                if (int(governer) == root_idx) and (dep_val == 'nmod'):
                                    root_idx = int(dependent)
                                    root = amr_toks.split(' ')[int(dependent)]
                                    for idx, ali in enumerate(list(alignments.values())):
                                        if ali.indices[0] == root_idx:
                                            concept_start_alignments.append(idx)
                                    break
                    else:
                        assert len(concept_start_alignments) > 0
                        if len(concept_start_alignments) != 1:
                            pass
                # 一个word可能对应很多个concept assert len(concept_start_alignments) == 1
                concept_start_alignments = concept_start_alignments[0]
                top_variable, _, top_concept = concept_keys[concept_start_alignments]
                # 改变top
                amr_tree_string_change_top = penman.encode(amr_graph,
                                                top=top_variable,
                                                model=penman_amr_model)
                
                # linearize
                amr_tree_string_linearized, pointer_variable_mapping, instances_index, edge_index, attributes_index, nx_graph   \
                        = dfs_linearized_penmen(amr_tree_string_change_top,use_pointer_tokens=True,)
                
                new_json_file[video_id]['expressions'][text_query_id].pop('amr_penman')
                new_json_file[video_id]['expressions'][text_query_id]['initial_parsed_amr'] = initial_parsed_amr
                new_json_file[video_id]['expressions'][text_query_id]['amr_tree_string'] = amr_tree_string_change_top
                new_json_file[video_id]['expressions'][text_query_id]['first_noun'] = top_concept
                new_json_file[video_id]['expressions'][text_query_id]['amr_tree_string_linearization_dict'] = {
                    'amr_tree_string_linearized': amr_tree_string_linearized,
                    'var_pointer_map': pointer_variable_mapping,
                    'instance_linearized_ali':instances_index,
                    'edge_linearized_ali':edge_index,
                    'attribute_linearized_ali':attributes_index
                }     
        with open(os.path.join(sentence_annotation_directory, use, f'meta_expressions_changeTop.json'), 'w') as f:
            json.dump({'videos': new_json_file}, f)

    return
     
                
if __name__ == '__main__':
    
    a2ds_perWindow_perExp(root='/hpc2hdd/home/testuser21/datasets/a2d_sentences')
    # yrvos_perWindow_perExp(root='/hpc2hdd/home/testuser21/datasets/youtube_rvos',
    #                        do_parse=True)