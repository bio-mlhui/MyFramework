
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

pos_targets = ['NN', 'NNS', 'NNP', 'NNPS']
def get_first_noun(ann, sent:str):
    words = sent.split()
    tokens = ann['tokens'] # list[dict]
    for id, token in enumerate(tokens):
        if token['pos'] in pos_targets:
            assert token['word'] == words[id]
            return id, words[id]
        
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


from stanza.server import CoreNLPClient
from penman import surface
import penman
from penman.models.amr import model as penman_amr_model
from datasets.preprocess.AMR_Process.read_and_process_v2 import dfs_linearized_penmen

# 返回amr的tokeniz
def a2ds(root, 
        do_parse=False,
        do_change_top=False,
        do_generate_dataset=False,
        chage_role_attribute=True): 

    if do_parse:
        batch_size = 10
        assert not os.path.exists(os.path.join(root, f'text_annotations/a2d_annotation.json'))
        parser = AMRParser.from_pretrained('AMR3-structbart-L')
        instances = pandas.read_csv(os.path.join(root, 'text_annotations/a2d_annotation.txt'))
        number_of_instances = len(instances['query'].to_list())
        initial_parsed_amrs = []
        cnt = 0
        for _ in tqdm(range(int(number_of_instances // batch_size + 1))):
            batch_texts = instances['query'][cnt:(cnt+batch_size)]
            cnt += len(batch_texts)

            batch_tokens = [parser.tokenize(text)[0] for text in batch_texts]
            batch_amrs = parser.parse_sentences(batch_tokens, batch_size=len(batch_tokens))[0]
            batch_amrs = [f'# ::snt {text}\n' + amr for amr, text in zip(batch_texts, batch_amrs)]
            initial_parsed_amrs += batch_amrs
        instances['initial_parsed_amr'] = initial_parsed_amrs
        with open(os.path.join(root, 'text_annotations/a2d_annotation.json'), 'w') as f:
            instances.to_json(f, orient='records')
        return 

        # assert not os.path.exists(os.path.join(root, f'text_annotations/a2d_annotation.json'))
        # parser = AMRParser.from_pretrained('AMR3-structbart-L')
        # instances = pandas.read_csv(os.path.join(root, 'text_annotations/a2d_annotation.txt'))
        # number_of_instances = len(instances['query'].to_list())
        # initial_parsed_amrs = []
        # for i in tqdm(range(number_of_instances)):
        #     text_annotation = instances['query'][i]
        #     # amr parsing
        #     tokens, _ = parser.tokenize(text_annotation)
        #     amr_tree_string = parser.parse_sentence(tokens)[0]
        #     amr_tree_string = f'# ::snt {text_annotation}\n' + amr_tree_string
        #     initial_parsed_amrs.append(amr_tree_string)
        # instances['initial_parsed_amr'] = initial_parsed_amrs
        # with open(os.path.join(root, 'text_annotations/a2d_annotation.json'), 'w') as f:
        #     instances.to_json(f, orient='records')
        # return 
    
    if do_change_top:
        if os.path.exists(os.path.join(root, f'text_annotations/a2d_annotation_changeTop.json')):
            with open(os.path.join(root, f'text_annotations/a2d_annotation_changeTop.json'), 'r') as f:
                instances = json.load(f)
        else:
            with open(os.path.join(root, f'text_annotations/a2d_annotation.json'), 'r') as f:
                instances = json.load(f)
            
        pos_tagger = CoreNLPClient(annotators=['tokenize','pos','parse', 'depparse'], 
                                        timeout=30000, output_format='json', 
                                        memory='6G',
                                        be_quiet=True,) 
        
        for i in tqdm(range(len(instances))):
            if 'amr_tree_string_linearization_dict' in instances[i]:
                continue
            initial_parsed_amr = instances[i]['initial_parsed_amr']
            tokenized_string = penman.decode(initial_parsed_amr).metadata['tok']
            # 找到第一个名词
            pos_ann = pos_tagger.annotate(tokenized_string)['sentences'][0]
            first_noun_idx, first_noun = get_first_noun(pos_ann, tokenized_string)
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
            amr_tree_string_linearized, pointer_variable_mapping, instance_links, attribute_links, role_links  \
                    = dfs_linearized_penmen(amr_tree_string_change_top, use_pointer_tokens=True,)

            instances[i]['amr_tree_string'] = amr_tree_string_change_top
            instances[i]['first_noun'] = top_concept
            instances[i]['amr_tree_string_linearization_dict'] = {
                'amr_tree_string_linearized': amr_tree_string_linearized,
                'var_pointer_map': pointer_variable_mapping,
                'instance_linearized_ali':instance_links,
                'edge_linearized_ali':role_links,
                'attribute_linearized_ali':attribute_links
                }
            
        with open(os.path.join(root, 'text_annotations/a2d_annotation_changeTop.json'), 'w') as f:
            json.dump(instances, f)
        return
    
    if do_generate_dataset:
        with open(os.path.join(root, f'text_annotations/a2d_annotation_changeTop.json'), 'r') as f:
            instances = pandas.read_json(f)
            
        videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
        videos.columns = ['video_id', '', '', '', '', '','', '', 'usage']
        with open(os.path.join(root, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
            unused_ids = f.read().splitlines()
            
        for usage, train_or_test, in {0: 'train', 1:'test'}.items():
            used_videos_ids = list(videos[(~videos.video_id.isin(unused_ids)) & (videos.usage==usage)]['video_id'])
            
            whether_used = instances.video_id.isin(used_videos_ids)
            used_instances = instances[whether_used]
            
            assert used_instances.keys()[0] == 'video_id'
            assert used_instances.keys()[1] == 'instance_id'
            assert used_instances.keys()[2] == 'query'
            assert used_instances.keys()[3] == 'initial_parsed_amr'
            assert used_instances.keys()[4] == 'amr_tree_string'
            assert used_instances.keys()[5] == 'first_noun'
            assert used_instances.keys()[6] == 'amr_tree_string_linearization_dict'
            used_instances = used_instances.astype({'instance_id': int}, errors='raise')
            groupby_obj = used_instances.groupby(by='video_id')
            
            used_idxs = np.arange(len(whether_used))[whether_used]
            used_instances = list(used_instances.to_records(index=False))
            
            # 比如一个视频有三帧被标注，且该视频里有三个instance被标注，则一个视频就输出了9个sample
            instances_by_frame = []
            for video_id, instance_id, text_annotation, _, amr_tree_string, first_noun, amr_tree_string_linearization_dict in used_instances:            
                frames = sorted(glob((os.path.join(root, f'text_annotations/a2d_annotation_with_instances/{video_id}', '*.h5'))))
                exist_texts = groupby_obj.get_group(video_id) # 这个video中所有出现的instances
                exist_texts = exist_texts.set_index('instance_id')
            
                for frame in frames:
                    frame_idx = int(frame.split('/')[-1].split('.')[0])
                    f = h5py.File(frame)
                    frame_instances = list(f['instance'])
                    if int(instance_id) in frame_instances:
                        masks = torch.from_numpy(np.array(f['reMask'])).transpose(-1, -2)
                        masks = masks.unsqueeze(dim=0) if masks.dim() == 2 else masks
                        assert len(masks) >= len(frame_instances)
                        exist_text_queries = exist_texts.loc[frame_instances]['query'].tolist()
                        
                        instances_by_frame.append({'video_id':video_id,
                                                'frame_idx':frame_idx, 
                                                'instance_id': int(instance_id), 
                                                'text_query': text_annotation, 
                                                'exist_queries': exist_text_queries,
                                                'amr_tree_string': amr_tree_string,
                                                'first_noun':first_noun,
                                                'amr_tree_string_linearization_dict':amr_tree_string_linearization_dict
                                                })
                    f.close()

            with open(os.path.join(root, f'a2ds_{train_or_test}.json'), 'w') as f:
                json.dump(instances_by_frame, f)
        return
        
    
def youtube_rvos(do_parse=False,
                 do_change_top=False,
                 sentence_annotation_directory='/home/xhh/datasets/youtube_rvos/meta_expressions',):         
    if do_parse:
        # pasing
        parser = AMRParser.from_pretrained('AMR3-structbart-L')
        for use in ['test', 'train', 'valid']:
            assert not os.path.exists(os.path.join(sentence_annotation_directory, use, 'meta_expressinos_only_parsed.json'))
            sentence_annotation_file = os.path.join(sentence_annotation_directory, use, 'meta_expressions.json')
            with open(sentence_annotation_file, 'r') as f:
                json_file = json.load(f)
                
            new_json_file = json_file['videos']
            for video_id, vid_ann in json_file['videos'].items():
                for text_query_id, exp_dict in vid_ann['expressions'].items():
                    text_query = exp_dict['exp']
                    text_query = ' '.join(text_query.strip().split()).lower()
                    
                    tokens, _ = parser.tokenize(text_query)
                    amr_string = parser.parse_sentence(tokens)[0]
                    amr_string = f'# ::snt {text_query}\n' + amr_string
                    
                    new_json_file[video_id]['expressions'][text_query_id]['exp'] = text_query
                    new_json_file[video_id]['expressions'][text_query_id]['amr_penman'] = amr_string 
                    
            with open(os.path.join(sentence_annotation_directory, use, 'meta_expressinos_only_parsed.json'), 'w') as f:
                json.dumps({'videos': new_json_file}, f)
        return
    
    if do_change_top:
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
                    amr_tree_string_linearized, pointer_variable_mapping, instance_links, attribute_links, role_links,   \
                            = dfs_linearized_penmen(amr_tree_string_change_top,use_pointer_tokens=True,)
                    
                    new_json_file[video_id]['expressions'][text_query_id].pop('amr_penman')
                    new_json_file[video_id]['expressions'][text_query_id]['initial_parsed_amr'] = initial_parsed_amr
                    new_json_file[video_id]['expressions'][text_query_id]['amr_tree_string'] = amr_tree_string_change_top
                    new_json_file[video_id]['expressions'][text_query_id]['first_noun'] = top_concept
                    new_json_file[video_id]['expressions'][text_query_id]['amr_tree_string_linearization_dict'] = {
                        'amr_tree_string_linearized': amr_tree_string_linearized,
                        'var_pointer_map': pointer_variable_mapping,
                        'instance_linearized_ali':instance_links,
                        'edge_linearized_ali':role_links,
                        'attribute_linearized_ali':attribute_links
                    }     
            with open(os.path.join(sentence_annotation_directory, use, f'meta_expressions_changeTop.json'), 'w') as f:
                json.dump({'videos': new_json_file}, f)
    
        return
                
if __name__ == '__main__':

    a2ds(root='/home/xhh/datasets/a2d_sentences', chage_role_attribute=True)
    
    # youtube_rvos(do_parse=False, do_change_top=True)