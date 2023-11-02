
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

pt_dir = '/home/xuhuihui/pt'   


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

def a2ds_perWindow_perExp(root): 
    with open(os.path.join(root, 'text_to_parseAMRaux.json'), 'r') as f:
        text_to_aux = json.load(f)
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
        new_text_to_amr_aux[text_query]['toknized_string'] = tokenized_string
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
    with open(os.path.join(root, 'text_to_aux.json'), 'w') as f:
        json.dump(new_text_to_amr_aux, f)            

def yrvos_normalize_text(text_query):
    if new_text_query == 'cannot describe too little':
        new_text_query = 'an airplane not moving'
    elif new_text_query == 'a red clothe':
        new_text_query = 'a red cloth'
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

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
                normalized_text_query = yrvos_normalize_text()
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
     
def refcocog_normalize_text(text_query):
    # 非法输入
    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

def refcoco_parseAMR(root, name):
    # save to refer/text_to_aux.json
    # refcoco: 142210 个句子
    # refcoco+: 141564
    # refcocog: 95010
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    with open(os.path.join(root, 'refer', name, 'global_mappings.json'), 'r') as f:
        all_sents = json.load(f)['sent']

    text_to_parsedAMR_file = os.path.join(root, 'refer', name, 'text_to_parseAMRaux.json')
    assert not os.path.exists(text_to_parsedAMR_file)
    text_to_parsedAMR = defaultdict(dict)
    failed_sent_ids = []
    for sent_id, sent_data in tqdm(all_sents.items()):
        text_query = sent_data['sent']
        try:
            all_auged_texts = [text_query]
            if ('left' in text_query) or ('right' in text_query):
                text_2 = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
                all_auged_texts.append(text_2)
            for auged_text in all_auged_texts:
                auged_text = refcocog_normalize_text(auged_text)
                if auged_text not in text_to_parsedAMR:
                    tokens, _ = parser.tokenize(auged_text)
                    initial_parsed_amr = parser.parse_sentence(tokens)[0]
                    text_to_parsedAMR[auged_text]['initial_parsed_amr'] = initial_parsed_amr
        except:
            failed_sent_ids.append(sent_id)

    with open(text_to_parsedAMR_file, 'w') as f:
        json.dump({'text_to_parsedAMR': text_to_parsedAMR, 
                   'failed_sent_ids': failed_sent_ids}, f)

def refcocog_normalize_text_v2(text_query,name, sent_id):
    if name == 'refcoco':
        if sent_id == '':
            pass
    if name == 'refcoco+':
        if sent_id == '126910':
            assert text_query == 'sis'
            text_query = 'sister'
            pass
    if name == 'refcocog':
        if sent_id == '29232':
            assert text_query == '{}'
            text_query = 'man wearing black pants'
        if sent_id == '268':
            assert text_query == '{}'
            text_query = 'woman surfing above the water'

    normalized_text_query = text_query.replace('right most', 'rightmost')
    normalized_text_query = normalized_text_query.replace('left most', 'leftmost')
    normalized_text_query = normalized_text_query.replace('front most', 'frontmost')
    # first one
    text_1 = " ".join(normalized_text_query.lower().split())
    return text_1

def refcoco_handle_failed_ids(root, name):
    assert name != 'refcoco'
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    with open(os.path.join(root, 'refer', name, 'global_mappings.json'), 'r') as f:
        all_sents = json.load(f)['sent']
    text_to_parseAMRaux_file = os.path.join(root, 'refer', name, 'text_to_parseAMRaux.json')
    with open(text_to_parseAMRaux_file, 'r') as f:
        text_to_parseAMRaux = json.load(f)
        failed_sent_ids:list = text_to_parseAMRaux['failed_sent_ids']
        text_to_parseAMRaux = text_to_parseAMRaux['text_to_parsedAMR']

    for sent_id, sent_data in tqdm(all_sents.items()):
        if sent_id not in failed_sent_ids:
            continue
        else:
            try:
                text_query = sent_data['sent']
                text_query = refcocog_normalize_text_v2(text_query, name=name, sent_id=sent_id)
                all_auged_texts = [text_query]
                if ('left' in text_query) or ('right' in text_query):
                    text_2 = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
                    all_auged_texts.append(text_2) 

                for auged_text in all_auged_texts:
                    tokens, _ = parser.tokenize(auged_text)
                    initial_parsed_amr = parser.parse_sentence(tokens)[0]
                    assert auged_text not in text_to_parseAMRaux
                    text_to_parseAMRaux[auged_text] = {'initial_parsed_amr': initial_parsed_amr}
            except:
                with open(text_to_parseAMRaux_file, 'w') as f:
                    json.dump({'failed_sent_ids': failed_sent_ids, 'text_to_parsedAMR':text_to_parseAMRaux}, f)
            else:
                failed_sent_ids.remove(sent_id) 
    with open(text_to_parseAMRaux_file, 'w') as f:
        json.dump(text_to_parseAMRaux, f)              

from data_schedule.preprocess.AMR_Process.read_and_process_v4 import dfs_linearized_penmen as dfs_linearized_penmen_v4
def refcoco_changeTop(root, name, nlp_endpoint):
    pos_tagger = CoreNLPClient(annotators=['tokenize','pos','parse', 'depparse'],
                               endpoint=nlp_endpoint, 
                                timeout=30000, output_format='json', 
                                memory='6G',
                                be_quiet=True,)
    with open(os.path.join(root, 'refer', name, 'global_mappings.json'), 'r') as f:
        global_mapping = json.load(f)
        all_sents = global_mapping['sent'] # # sence_id: tokens, raw, sent_id, sent
        refs = global_mapping['refs'] # # ref_id :[sent_ids: [0, 1, 2], 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences'(list[]), 'category_id']
    
    with open(os.path.join(root, 'refer', name, 'text_to_parseAMRaux.json'), 'r') as f:
        text_to_parseAMR = json.load(f)     # refcoco: 106420, refcoco+: 89842

    text_to_aux_file = os.path.join(root, 'refer', name, 'text_to_aux.json')    
    if os.path.exists(text_to_aux_file):
        with open(text_to_aux_file, 'r') as f:
            text_to_aux = json.load(f)
    else:
        text_to_aux = {}
    conditions_statics = {1:0, 2:0, 3:0, 4:0, 5: 0, 6: 0, 7: 0}

    for sent_id, sent_data in tqdm(all_sents.items()): # refcoco: 142210, refcoco+: 141564
        text_query = sent_data['sent']
        text_query = refcocog_normalize_text_v2(text_query, name, sent_id)
        all_auged_texts = [text_query]
        if ('left' in text_query) or ('right' in text_query):
            text_2 = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
            all_auged_texts.append(text_2)
        auged_rets = {}
        for auged_text in all_auged_texts:
            if auged_text in text_to_aux:
                continue
            assert auged_text in text_to_parseAMR
            initial_parsed_amr = text_to_parseAMR[auged_text]['initial_parsed_amr']
            ret = {}
            ret['initial_parsed_amr'] = initial_parsed_amr
            tokenized_string = penman.decode(initial_parsed_amr).metadata['tok']

            amr_tokens = tokenized_string.split()
            amr_graph = penman.decode(initial_parsed_amr, model=penman_amr_model)
            amr_graph.metadata = {}
            alignments = surface.alignments(amr_graph) # (val, _, concept): Alignment(1,)
            concept_to_tok = {key:amr_tokens[value.indices[0]] for key, value in alignments.items()}
            concept_aligns = {key:value.indices[0] for key, value in alignments.items()}
            def get_min_ali_concept(concept_aligns):
                concepts = list(concept_aligns.keys())
                aligns = list(concept_aligns.values())
                min_ali = min(aligns)
                return concepts[aligns.index(min_ali)]
            if len(amr_tokens) == 1:
                top_concepts = [key for key,value in alignments.items() if value.indices[0] == 0]
                if len(top_concepts) == 0:
                    conditions_statics[5] += 1
                    raise ValueError()
                elif len(top_concepts) == 1:
                    conditions_statics[7] += 1
                elif len(top_concepts) > 1:
                    conditions_statics[6] += 1
                top_concept = top_concepts[0]
                top_variable, _, top_concept = top_concept
            else:
                first_noun_idx = None
                pos_ann = pos_tagger.annotate(tokenized_string)['sentences'][0]
                pos_tokenized_tokens = pos_ann['tokens'] # list[dict]
                for idx, pos_tok in enumerate(pos_tokenized_tokens):
                    if pos_tok['pos'] in pos_targets:
                        first_noun_idx, first_noun = idx, pos_tok['word']
                        break
                # pos中的first_noun_idx到amr tok的first_noun_idx
                if first_noun_idx is not None:
                    # 找到这个名词对应的concepts集合
                    def transform_pos_to_amr(first_idx_pos, amr_toks, foo_pos_toks):
                        pos_toks = [foo_pos_toks[foo_i]['word'] for foo_i in range(len(foo_pos_toks))]
                        if (first_idx_pos < len(amr_tokens)) and (amr_toks[first_idx_pos] == pos_toks[first_idx_pos]):
                            return first_idx_pos
                        else:
                            before_sum = sum([len(pos_toks[foo_i]) for foo_i in range(first_idx_pos)])
                            cnt = 0
                            for idx, atok in enumerate(amr_tokens):
                                if cnt >= before_sum:
                                    return idx
                                else:
                                    cnt += len(atok)
                            if cnt > before_sum:
                                return len(amr_tokens) - 1
                            raise ValueError()
                    first_noun_idx = transform_pos_to_amr(first_noun_idx, amr_tokens, pos_tokenized_tokens)
                    first_noun_concepts = [key for key,val in concept_aligns.items() if val == first_noun_idx]
                    if len(first_noun_concepts) == 1:
                        conditions_statics[4] += 1
                        top_variable, _, top_concept = first_noun_concepts[0]
                    elif len(first_noun_concepts) > 1:
                        conditions_statics[3] += 1
                        top_variable, _, top_concept = first_noun_concepts[0]
                    else:
                        # 这个名词没有被parse 出句子
                        conditions_statics[2] += 1
                        # alignment最小的concept
                        top_variable, _, top_concept = get_min_ali_concept(concept_aligns)
                else:
                    conditions_statics[1] += 1
                    top_variable, _, top_concept = get_min_ali_concept(concept_aligns)

            amr_tree_string_change_top = penman.encode(amr_graph, top=top_variable,model=penman_amr_model)
            # if len(concept_start_alignments) == 0:
            #     for idx, key in enumerate(concept_keys):  # 一个句子中出现了两个light, 但是amr parser只识别了后一个light, 并且pos tagger认为第一个是top
            #         if key[-1].startswith(first_noun.lower()):
            #             concept_start_alignments.append(idx)
            nx_graph = dfs_linearized_penmen_v4(amr_tree_string_change_top, use_pointer_tokens=True,)
            ret['toknized_string'] = tokenized_string
            ret['amr_tree_string'] = amr_tree_string_change_top
            ret['inference_graph'] = nx.node_link_data(nx_graph)
            ret['first_noun'] = top_concept
            auged_rets[auged_text] = ret
        
        for key, value in auged_rets.items():
            text_to_aux[key] = value   # refcocog: {1: 52, 2: 412, 3: 4230, 4: 83756, 5: 0, 6: 41, 7: 91}
    # refcoco+: {1: 610, 2: 551, 3: 1996, 4: 30814, 5: 0, 6: 347, 7: 242}
    with open(text_to_aux_file, 'w') as f: # 记得看看condition statics
        json.dump(text_to_aux, f) # refcoco: {1: 605, 2: 430, 3: 2464, 4: 36260, 5: 0, 6: 237, 7: 194}

import argparse
if __name__ == '__main__':
    # a2ds_perWindow_perExp(root='/home/xuhuihui/datasets/a2d_sentences')
    # yrvos_perWindow_perExp(root='/hpc2hdd/home/testuser21/datasets/youtube_rvos', do_parse=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,)
    parser.add_argument('--endpoint', type=str,)
    parser.add_argument('--task_name', type=str,default='parseAMR')
    args = parser.parse_args()
    if args.task_name == 'parseAMR':
        refcoco_parseAMR(root='/home/xuhuihui/datasets', name=args.name)
    elif args.task_name == 'changeTop':
        refcoco_changeTop(root='/home/xuhuihui/datasets', name=args.name, nlp_endpoint=args.endpoint)
    elif args.task_name == 'handle_fail':
        refcoco_handle_failed_ids(root='/home/xuhuihui/datasets', name=args.name)

    # name = 'refcocog' # 
    # refcoco_parseAMR(root='/home/xuhuihui/datasets', name=name)

    # name = 'refcoco+' # 
    # refcoco_parseAMR(root='/home/xuhuihui/datasets', name=name)