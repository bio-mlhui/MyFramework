
import os
import pandas
import h5py
from glob import glob
import json
import numpy as np
import torch
from transition_amr_parser.parse import AMRParser

def save_a2ds_json_withamr_perAnnClip(root, postfix, 
                                      amr_file='tam_a2ds_penmans.txt', 
                                      linearized_amr_file='tam_a2ds_penmans_linearized.amr', 
                                      sentence_annotation_file='text_annotations/a2d_annotation.txt'):
    """
    每个(video, instance_id, text) pair, 再根据annotation的情况, 生成一个sample    
    """    
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'usage']
    with open(os.path.join(root, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
        
    sentences = []
    amr_penmans = []    
    with open(os.path.join(root, amr_file), 'r') as ifp:

        amrblock = []
        for line in ifp:
            line = line.rstrip()
            if not line:
                amr_penmans.append("\n".join(amrblock))
                amrblock = []
            elif line.startswith("# ::id "):
                continue
            elif line.startswith("# ::snt "):
                sentences.append(line[8:])
            elif line.startswith("# ::save-data "):
                continue
            elif line.startswith("#"):
                continue
            else:
                amrblock.append(line)     
    
    with open(os.path.join(root, linearized_amr_file), 'r') as ifp:
        linearized_penmans = ifp.read().splitlines()
    
    # 'video_id', 'instance_id', 'query'
    instances = pandas.read_csv(os.path.join(root, sentence_annotation_file))  
    
    # check validity
    assert len(sentences) == len(amr_penmans)
    assert len(sentences) == len(linearized_penmans)
    assert len(sentences) == len(instances)
    ck_instances = list(instances.to_records(index=False))
    for (_, _, text, *_), sent in zip(ck_instances, sentences):
        assert text.rstrip() == sent
    
    for usage, train_or_test, in {0:'train', 1:'test'}.items():
        used_videos_ids = list(videos[(~videos.video_id.isin(unused_ids)) & (videos.usage==usage)]['video_id'])
        
        whether_used = instances.video_id.isin(used_videos_ids)
        used_instances = instances[whether_used]
        
        used_instances = used_instances.astype({'instance_id': int}, errors='raise')
        groupby_obj = used_instances.groupby(by='video_id')
        
        used_idxs = np.arange(len(whether_used))[whether_used]
        used_instances = list(used_instances.to_records(index=False))
        
        # 比如一个视频有三帧被标注，且该视频里有三个instance被标注，则一个视频就输出了9个sample
        instances_by_frame = []
        for (video_id, instance_id, text_annotation, *_), origin_idx in zip(used_instances, used_idxs):
            # all annotated frames of this video
            assert text_annotation.rstrip() == sentences[origin_idx]
            
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
                    exist_text_queries = [q.lower() for q in exist_text_queries]
                    
                    instances_by_frame.append({'video_id':video_id,
                                               'frame_idx':frame_idx, 
                                               'instance_id': int(instance_id), 
                                               'text_query': text_annotation.lower(), 
                                               'exist_queries': exist_text_queries,
                                               'amr_penman': amr_penmans[origin_idx],
                                               'linearized_penman': linearized_penmans[origin_idx]
                                               })
                f.close()
        with open(os.path.join(root, f'a2ds_{train_or_test}{postfix}.json'), 'w') as f:
            json.dump(instances_by_frame, f)


def save_a2ds_json_withamr_perAnnClip_linarized_file_is_json(root, postfix,
                                      linearized_amr_file,  
                                      amr_file='tam_a2ds_penmans.txt', 
                                      sentence_annotation_file='text_annotations/a2d_annotation.txt'):
    """
    每个(video, instance_id, text) pair, 再根据annotation的情况, 生成一个sample    
    """    
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'usage']
    with open(os.path.join(root, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()
        
    sentences = []
    amr_penmans = []    
    with open(os.path.join(root, amr_file), 'r') as ifp:

        amrblock = []
        for line in ifp:
            line = line.rstrip()
            if not line:
                amr_penmans.append("\n".join(amrblock))
                amrblock = []
            elif line.startswith("# ::id "):
                continue
            elif line.startswith("# ::snt "):
                sentences.append(line[8:])
            elif line.startswith("# ::save-data "):
                continue
            elif line.startswith("#"):
                continue
            else:
                amrblock.append(line)     
    
    with open(os.path.join(root, linearized_amr_file), 'r') as ifp:
        linearized_penmans = json.load(ifp) # list[dict]
    
    # 'video_id', 'instance_id', 'query'
    instances = pandas.read_csv(os.path.join(root, sentence_annotation_file))  
    
    # check validity
    assert len(sentences) == len(amr_penmans)
    assert len(sentences) == len(linearized_penmans)
    assert len(sentences) == len(instances)
    ck_instances = list(instances.to_records(index=False))
    for (_, _, text, *_), sent, lin_dict in zip(ck_instances, sentences, linearized_penmans):
        assert text.rstrip() == sent
        assert lin_dict['sentence'] == sent 
    
    for usage, train_or_test, in {0:'train', 1:'test'}.items():
        used_videos_ids = list(videos[(~videos.video_id.isin(unused_ids)) & (videos.usage==usage)]['video_id'])
        
        whether_used = instances.video_id.isin(used_videos_ids)
        used_instances = instances[whether_used]
        
        used_instances = used_instances.astype({'instance_id': int}, errors='raise')
        groupby_obj = used_instances.groupby(by='video_id')
        
        used_idxs = np.arange(len(whether_used))[whether_used]
        used_instances = list(used_instances.to_records(index=False))
        
        # 比如一个视频有三帧被标注，且该视频里有三个instance被标注，则一个视频就输出了9个sample
        instances_by_frame = []
        for (video_id, instance_id, text_annotation, *_), origin_idx in zip(used_instances, used_idxs):
            # all annotated frames of this video
            assert text_annotation.rstrip() == sentences[origin_idx]
            
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
                    exist_text_queries = [q.lower() for q in exist_text_queries]
                    
                    instances_by_frame.append({'video_id':video_id,
                                               'frame_idx':frame_idx, 
                                               'instance_id': int(instance_id), 
                                               'text_query': text_annotation.lower(), 
                                               'exist_queries': exist_text_queries,
                                               'amr_penman': amr_penmans[origin_idx],
                                               'linearized_penman': linearized_penmans[origin_idx]
                                               })
                f.close()
        with open(os.path.join(root, f'a2ds_{train_or_test}{postfix}.json'), 'w') as f:
            json.dump(instances_by_frame, f)

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

        
def dfs_linearized_penmen(amr_penman_string, 
                        use_pointer_tokens,
                        remove_pars=None,
                        use_recategorization = False,
                        remove_wiki = False,
                        dereify= False,):
    from datasets.preprocess.AMR_Process.read_and_process import dfs_linearize
    from datasets.preprocess.AMR_Process.penman_interface import _remove_wiki
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
    lin_tokens, pointer_variable_mapping, node_index, edge_index, attribute_index = dfs_linearize(amr_graph,
                                                                                                    use_pointer_tokens=use_pointer_tokens)
    amr_tree_string_linearized = ' '.join(lin_tokens)
    return amr_tree_string_linearized, pointer_variable_mapping, node_index, edge_index, attribute_index      
              


from stanza.server import CoreNLPClient
from penman import surface
import penman
from penman.models.amr import model as penman_amr_model
# 返回amr的tokeniz
def a2ds_augument_with_amr_string(root, 
                                postfix,
                                sentence_annotation_file='text_annotations/a2d_annotation.txt',
                                code_v = '1'): 
    
       
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
    videos.columns = ['video_id', '', '', '', '', '','', '', 'usage']
    with open(os.path.join(root, 'text_annotations/a2d_missed_videos.txt'), 'r') as f:
        unused_ids = f.read().splitlines()

    if code_v == '1':
        instances = pandas.read_csv(os.path.join(root, sentence_annotation_file))
        amr_tree_strings = []
        first_noun_idxs = []
        first_nouns = []
        amr_tree_string_linearization_dict = []
        for i in range(len(instances['query'].to_list())):
            text_annotation = instances['query'][i]
            text_annotation = ' '.join(text_annotation.strip().split()).lower()
            
            pos_ann = pos_tagger.annotate(text_annotation)['sentences'][0]
            first_noun_idx, first_noun = get_first_noun(pos_ann, text_annotation)
            
            # amr parsing
            tokens, _ = parser.tokenize(text_annotation)
            amr_tree_string = parser.parse_sentence(tokens)[0]
            amr_tree_string = f'# ::snt {text_annotation}\n' + amr_tree_string
            
            # change top according to the top
            amr_graph = penman.decode(amr_tree_string, model=penman_amr_model)
            amr_graph.metadata = {}
            alignments = surface.alignments(amr_graph)
            concept_keys = list(alignments.keys())
            concept_start_alignments = [ idx for idx, ali in enumerate(list(alignments.values())) if ali.indices[0] == first_noun_idx]
            # 一个word可能对应很多个concept assert len(concept_start_alignments) == 1
            concept_start_alignments = concept_start_alignments[0]
            top_variable, _, top_concept = concept_keys[concept_start_alignments]
            # 改变top
            amr_tree_string = penman.encode(amr_graph,
                                            top=top_variable,
                                            model=penman_amr_model)
            
            # linearize
            amr_tree_string_linearized, pointer_variable_mapping, node_index, edge_index, attribute_index  \
                    = dfs_linearized_penmen(amr_tree_string,use_pointer_tokens=True,)
            
            amr_tree_strings.append(amr_tree_string)
            first_noun_idxs.append(first_noun_idx)
            first_nouns.append(first_noun)
            amr_tree_string_linearization_dict.append({
                'amr_tree_string_linearized': amr_tree_string_linearized,
                'var_pointer_map': pointer_variable_mapping,
                'instance_linearized_ali':node_index,
                'edge_linearized_ali':edge_index,
                'attribute_linearized_ali':attribute_index
                })
            
            instances['query'][i] = text_annotation
        instances['amr_tree_string'] = amr_tree_strings
        instances['first_noun_idx'] = first_noun_idxs
        instances['first_noun'] = first_noun
        instances['amr_tree_string_linearization_dict'] = amr_tree_string_linearization_dict
        with open(os.path.join(root, f'text_augumented_{postfix}.json'), 'w') as f:
            instances.to_json(f, orient='records')
    elif code_v == '2':
        with open(os.path.join(root, f'text_augumented_change_root.json'), 'r') as f:
            instances = pandas.read_json(f, orient='records')
    
    for usage, train_or_test, in {0: 'train', 1:'test'}.items():
        used_videos_ids = list(videos[(~videos.video_id.isin(unused_ids)) & (videos.usage==usage)]['video_id'])
        
        whether_used = instances.video_id.isin(used_videos_ids)
        used_instances = instances[whether_used]
        
        used_instances = used_instances.astype({'instance_id': int}, errors='raise')
        groupby_obj = used_instances.groupby(by='video_id')
        
        used_idxs = np.arange(len(whether_used))[whether_used]
        used_instances = list(used_instances.to_records(index=False))
        
        # 比如一个视频有三帧被标注，且该视频里有三个instance被标注，则一个视频就输出了9个sample
        instances_by_frame = []
        for video_id, instance_id, text_annotation, amr_tree_string, first_noun_idx, first_noun, amr_tree_string_linearization_dict in used_instances:            
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
                                               'first_noun_idx':first_noun_idx,
                                               'first_noun':first_noun,
                                               'amr_tree_string_linearization_dict':amr_tree_string_linearization_dict
                                               })
                f.close()
        for idx, ib in enumerate(instances_by_frame):
            fni = instances_by_frame[idx]['first_noun_idx']
            instances_by_frame[idx]['first_noun_idx'] = int(fni)
            
        with open(os.path.join(root, f'a2ds_{train_or_test}_{postfix}.json'), 'w') as f:
            json.dump(instances_by_frame, f)


# 每个句子加上penman amr
def youtube_rvos_augument_with_amr_string( 
                                postfix,
                                sentence_annotation_directory='/home/xhh/datasets/youtube_rvos/meta_expressions',
                                code_v = '1'): 
    if code_v == '1':
        parser = AMRParser.from_pretrained('AMR3-structbart-L')
        for use in ['test', 'train', 'valid']:
            sentence_annotation_file = os.path.join(sentence_annotation_directory, use, 'meta_expressions.json')
            with open(sentence_annotation_file, 'r') as f:
                json_file = json.load(f)
                # "videos": {"video id": {}}
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
            with open(os.path.join(sentence_annotation_directory, use, f'meta_expressinos_{postfix}.json'), 'w') as f:
                json.dump({'videos': new_json_file}, f)
    elif code_v == '2':
        assert postfix != 'only_amr'
        # 只进行了amr parsing
        # 重新load, 改变top, linearize, 然后存到另一个文件中
        pos_tagger = CoreNLPClient(annotators=['tokenize','pos','parse'], 
                                        timeout=30000, output_format='json', 
                                        memory='6G',
                                        be_quiet=True,endpoint='http://localhost:65222') 
        
        for use in ['train', 'valid']:
            if os.path.exists(os.path.join(sentence_annotation_directory, use, 'meta_expressinos_changeRoot_Linearized.json')):
                sentence_annotation_file = os.path.join(sentence_annotation_directory, use, 'meta_expressinos_changeRoot_Linearized.json')
            else:
                sentence_annotation_file = os.path.join(sentence_annotation_directory, use, 'meta_expressinos_only_parsed.json')
                
            with open(sentence_annotation_file, 'r') as f:
                json_file = json.load(f)
                
            new_json_file = json_file['videos']
            for video_id, vid_ann in json_file['videos'].items():
                for text_query_id, exp_dict in vid_ann['expressions'].items():
                    if 'amr_tree_string_linearization_dict' in exp_dict:
                        continue
                    text_query = exp_dict['exp']
                    amr_tree_string = exp_dict['amr_penman']
                    
                    if text_query == 'cannot describe too little':
                        parser = AMRParser.from_pretrained('AMR3-structbart-L')
                        text_query = 'an airplane not moving'
                        tokens, _ = parser.tokenize(text_query)
                        amr_tree_string = parser.parse_sentence(tokens)[0]
                        amr_tree_string = f'# ::snt {text_query}\n' + amr_tree_string
                        new_json_file[video_id]['expressions'][text_query_id]['exp'] = text_query
                        new_json_file[video_id]['expressions'][text_query_id]['amr_penman'] = amr_tree_string
                    if text_query == 'a red clothe':
                        # parser = AMRParser.from_pretrained('AMR3-structbart-L')
                        text_query = 'a red cloth'
                        tokens, _ = parser.tokenize(text_query)
                        amr_tree_string = parser.parse_sentence(tokens)[0]
                        amr_tree_string = f'# ::snt {text_query}\n' + amr_tree_string
                        new_json_file[video_id]['expressions'][text_query_id]['exp'] = text_query
                        new_json_file[video_id]['expressions'][text_query_id]['amr_penman'] = amr_tree_string
                        
                    amr_graph = penman.decode(amr_tree_string, model=penman_amr_model)
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
                    amr_tree_string = penman.encode(amr_graph,
                                                    top=top_variable,
                                                    model=penman_amr_model)
                    
                    # linearize
                    amr_tree_string_linearized, pointer_variable_mapping, node_index, edge_index, attribute_index  \
                            = dfs_linearized_penmen(amr_tree_string,use_pointer_tokens=True,)
                    
                    new_json_file[video_id]['expressions'][text_query_id]['first_noun_idx'] = root_idx
                    new_json_file[video_id]['expressions'][text_query_id]['first_noun'] = root
                    new_json_file[video_id]['expressions'][text_query_id]['amr_tree_string_linearization_dict'] = {
                        'amr_tree_string_linearized': amr_tree_string_linearized,
                        'var_pointer_map': pointer_variable_mapping,
                        'instance_linearized_ali':node_index,
                        'edge_linearized_ali':edge_index,
                        'attribute_linearized_ali':attribute_index
                        }     
            with open(os.path.join(sentence_annotation_directory, use, f'meta_expressinos_{postfix}.json'), 'w') as f:
                json.dump({'videos': new_json_file}, f)
        
    
import re
def change_a2ds_linearized_alignment_index_key():
    # a2ds:
    files = ['/home/xhh/datasets/a2d_sentences/a2ds_test_change_root.json',
             '/home/xhh/datasets/a2d_sentences/a2ds_train_change_root.json']
    new_files = ['/home/xhh/datasets/a2d_sentences/a2ds_test_change_root_filterAli.json',
             '/home/xhh/datasets/a2d_sentences/a2ds_train_change_root_filterAli.json']
    
    alignment_pattern = re.compile(r'~(?:[a-z]\.?)?(?P<ali>[0-9]+)(?:,[0-9]+)*')
    
    for file, new_file in zip(files, new_files):
        with open(file, 'r') as f:
            instances = json.load(f) # list[dict]
            
        for idx, ins in enumerate(instances):
            linearization_dict = ins['amr_tree_string_linearization_dict']
            instance_ali, attribute_ali, edge_ali = linearization_dict['instance_linearized_ali'],\
                linearization_dict['attribute_linearized_ali'],\
                linearization_dict['edge_linearized_ali']
                
            new_instance_ali = {}
            for ins_key, ins_val in instance_ali.items():
                ins_key = re.sub(alignment_pattern, repl='', string=ins_key)
                new_instance_ali[ins_key] = ins_val
            
            new_attribute_ali = {}  
            for attr_key, attr_val in attribute_ali.items():
                attr_key = re.sub(alignment_pattern, repl='', string=attr_key)
                new_attribute_ali[attr_key] = attr_val
            
            new_edge_ali = {}  
            for edge_key, edge_val in edge_ali.items():
                edge_key = re.sub(alignment_pattern, repl='', string=edge_key)
                new_edge_ali[edge_key] = edge_val  
                    
            instances[idx]['amr_tree_string_linearization_dict']['instance_linearized_ali'] = new_instance_ali
            instances[idx]['amr_tree_string_linearization_dict']['attribute_linearized_ali'] = new_attribute_ali
            instances[idx]['amr_tree_string_linearization_dict']['edge_linearized_ali'] = new_edge_ali
        with open(new_file, 'w') as f:
            json.dump(instances, f)
    
    
    
    # youtube_rvos
    


if __name__ == '__main__':
    # save_a2ds_json_withamr_perAnnClip(root='/home/xhh/datasets/a2d_sentences',
    #                                   postfix='')

    # # 改变top之后
    # save_a2ds_json_withamr_perAnnClip(root='/home/xhh/datasets/a2d_sentences',
    #                                   amr_file='change_root_a2ds_tam_ann.txt',
    #                                   linearized_amr_file='change_root_a2ds_tam_ann_linearized.amr',
    #                                   sentence_annotation_file='text_annotations/a2ds_amr_root_stanford_corenlp_pos.txt',
    #                                   postfix='_change_root')

    # # 改变top之后, fix process
    # save_a2ds_json_withamr_perAnnClip(root='/home/xhh/datasets/a2d_sentences',
    #                                   amr_file='change_root_a2ds_tam_ann.txt',
    #                                   linearized_amr_file='change_root_a2ds_tam_ann_linearized_fix_preprocess.amr',
    #                                   sentence_annotation_file='text_annotations/a2ds_amr_root_stanford_corenlp_pos.txt',
    #                                   postfix='_change_root_fix_process')
    
    
    # 发现 用AMR-Process 预处理 transition amr有问题： “Americal"~3 变成了 "Americal" ~3
    # 重新写了Preprocess的代码
    # 还生成了graph的每个triple 和linearized penmen的alignment
    # save_a2ds_json_withamr_perAnnClip_linarized_file_is_json(root='/home/xhh/datasets/a2d_sentences',
    #                                   amr_file='change_root_a2ds_tam_ann.txt',
    #                                   linearized_amr_file='change_root_fix_preprocess_with_tripleAlign.json',
    #                                   sentence_annotation_file='text_annotations/a2ds_amr_root_stanford_corenlp_pos.txt',
    #                                   postfix='change_root_fix_preprocess_with_tripleAlign')
    
    
    # a2ds_augument_with_amr_string(root='/home/xhh/datasets/a2d_sentences',
    #                               postfix='change_root',
    #                               sentence_annotation_file='text_annotations/a2d_annotation.txt',
    #                               code_v = '2')
    
    
    
    # youtube_rvos_augument_with_amr_string(postfix='only_parsed')
    # youtube_rvos_augument_with_amr_string(postfix='changeRoot_Linearized', code_v='2')
    
    change_a2ds_linearized_alignment_index_key()