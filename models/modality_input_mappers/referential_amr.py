import os
import networkx as nx
from torch_geometric.data import Data
import torch
import json
from tqdm import tqdm
import logging
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
# models.aux_data 没有__init__函数, 对于每一个task, 都要在对应的data_schedule里进行实例化, 比如RAMR -> RVOS_RAMR

def text_pad_token_ids(token_ids, pad_id):
    # list[ni], batch -> batch, n_max
    batch_size = len(token_ids)
    n_max = max([len(t_ids) for t_ids in token_ids])
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool)
    
    for i in range(batch_size):
        num_tokens = len(token_ids[i])
        token_ids[i] = token_ids[i] + [pad_id] * (n_max - num_tokens)
        pad_mask[i][:num_tokens] = False

    token_ids = torch.tensor(token_ids, dtype=torch.long)
    return token_ids, pad_mask 

def text_padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    """
    features: list[dict{'input_ids': list[int], ..}]
    """
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return

def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return

# 和任务无关
@MODELITY_INPUT_MAPPER_REGISTRY.register()
class RAMR:
    def __init__(self,
                 configs,
                 ) -> None:
        
        dataset_path = os.getenv('DATASET_PATH')
        pt_path = os.getenv('PT_PATH')
        amr_aux_files = configs['amr_aux_files']
        tokenizer_path = configs['tokenizer_path']
        self.version = configs['version']
        text_to_amr = {}
        for aux_file in amr_aux_files:
            logging.debug(f'reading {aux_file}')
            assert os.path.exists(os.path.join(dataset_path, aux_file))
            with open(os.path.join(dataset_path, aux_file), 'r') as f:
                text_amrs = json.load(f)
            for text in tqdm(list(text_amrs.keys())):
                if text not in text_to_amr:
                    text_to_amr[text] = text_amrs[text]
        self.text_aux_by_auxid = text_to_amr
        self.pt_path = pt_path
    
        if self.version == 'ramr_without_variable' or \
            self.version == 'ramr_with_variable' or\
            self.version == 'linearized_ramr' or\
            self.version == 'amr_penman_graph':
            from models.utils.amr.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(self.pt_path, tokenizer_path))
            self.prefix = ""
            self.max_src_length = 256 
            self.label_pad_token_id = -100
            # from datasets.propbank_frames import PropBankFrames
            # self.pbframes = PropBankFrames('/home/xhh/workspace/rvos_encoder/datasets/propbank-frames/frames')
            # self.all_predicates = list(self.pbframes.rolesets_meaning.keys()) 
        else:
            raise ValueError()
            
    def mapper(self, text_auxid):
        if self.version == 'ramr_without_variable':
            amr_wv = self.text_aux_by_auxid[text_auxid]['inference_graph'] 
            tok_string = self.text_aux_by_auxid[text_auxid]['toknized_string']
            amr_tree_string = self.text_aux_by_auxid[text_auxid]['amr_tree_string']
            G : nx.DiGraph = nx.node_link_graph(amr_wv)
            top_var = G.graph['top']
            nodekey_to_token = {key:node_token for key, node_token in zip(G.nodes(), G.nodes())} # var和concept一样
            nodekey_to_segid = {key:G.nodes[key]['seg_id'] for key in G.nodes()}
            nodekey_to_alignment = {key:G.nodes[key]['alignment'] for key in G.nodes()}
            # 标号，过滤掉segid=2的节点, 第一个永远是top var
            nodekey_to_idx = {} 
            nodekey_to_idx[top_var] = 0
            cnt = 1
            for node_key in G.nodes():
                if nodekey_to_segid[node_key] == 2:
                    continue
                if node_key == top_var:
                    continue
                nodekey_to_idx[node_key] = cnt
                cnt += 1
            idx_to_nodekey = {value:key for key, value in nodekey_to_idx.items()}
            
            edge_index = []
            edge_seg_ids = []
            edge_tokens = []
            for i, (src, dst) in enumerate(G.edges()):
                edge_seg_id = G[src][dst]['seg_id']
                if edge_seg_id == -2:
                    # 把/边的dst的token改成src的token, dst的segid改成2, dst的alignment改成src的alignment
                    if not ((nodekey_to_segid[dst] == 1) and (nodekey_to_segid[src] == 2)):
                        nodekey_to_token[dst] = nodekey_to_token[src]
                        nodekey_to_segid[dst] = 2
                        nodekey_to_alignment[dst] = 0 # 瞎选的                       
                    else:
                        nodekey_to_token[dst] = nodekey_to_token[src]
                        nodekey_to_segid[dst] = 2
                        nodekey_to_alignment[dst] = nodekey_to_alignment[src]
                        # 忽略/边
                else:
                    # 有个concept既当concept又当value
                    if src not in nodekey_to_idx or dst not in nodekey_to_idx:
                        continue
                        # for edge_source, edge_target in G.edges():
                        #     if (src == edge_source) and (dst == edge_target) and \
                        #         G.edges[(edge_source, edge_target)]['seg_id'] == -3:
                    edge_index.append([nodekey_to_idx[src], nodekey_to_idx[dst]])
                    edge_seg_ids.append(edge_seg_id)
                    edge_tokens.append(G[src][dst]['role'])
            # 按照idx获得token序列, segid序列
            node_tokens = [nodekey_to_token[idx_to_nodekey[idx]] for idx in range(cnt)] 
            node_seg_ids = [nodekey_to_segid[idx_to_nodekey[idx]] for idx in range(cnt)] 
            node_alignments = [nodekey_to_alignment[idx_to_nodekey[idx]] for idx in range(cnt)]
            assert 1 not in node_seg_ids
            assert -2 not in edge_seg_ids
            assert -100 not in node_alignments
            edge_index = torch.tensor(edge_index)
            if edge_index.dim() == 1:
                edge_index = torch.empty([2, 0], dtype=torch.int64)
            else:
                edge_index = edge_index.permute(1, 0)
            amr = Data(edge_index=edge_index)
            amr.num_nodes = len(node_tokens)
            seg_ids = node_seg_ids + edge_seg_ids
            tokens = node_tokens + edge_tokens
            assert tokens[0] == nodekey_to_token[top_var]
            
            tokens_ids, meta_dict = self.tokenizer.tokenize_amr(tokens)
            text_tokens = self.tokenizer.tokenize(tok_string)
            text_token_ids = [self.tokenizer.encoder.get(tok, self.tokenizer.unk_token_id) for tok in text_tokens]
            text_token_splits = [] # list[int]
            cnt = 0
            for idx, tok in enumerate(text_tokens):
                if idx == 0:
                    assert tok.startswith('Ġ')
                    cnt += 1
                elif not tok.startswith('Ġ'):
                    cnt += 1
                else:
                    text_token_splits.append(cnt)
                    cnt = 1

                if idx == (len(text_tokens) - 1):
                    text_token_splits.append(cnt)

            token_splits = meta_dict['each_token_length']
            return {
                'amrs': amr, # Graph
                'seg_ids': seg_ids, # V+E
                'token_splits': token_splits, # V+E, sum=len(token_ids)
                'token_ids': tokens_ids, 
                'node_alignments': node_alignments,
                'text_token_ids': text_token_ids,
                'text_token_splits': text_token_splits,
                'amr_tree_string': amr_tree_string,
            }

        elif self.version == 'ramr_with_variable': 
            amr = self.text_aux_by_auxid[text_auxid]['inference_graph']
            G : nx.DiGraph = nx.node_link_graph(amr)
            node_keys = list(G.nodes())
            top_var = G.graph['top']
            if node_keys[0] != top_var:
                top_var_idx = [idx for idx in range(len(node_keys)) if node_keys[idx] == top_var]
                assert len(top_var_idx) ==0
                top_var_idx = top_var_idx[0]
                first_node = node_keys[0]
                node_keys[top_var_idx] = first_node
                node_keys[0] = top_var
            node_seg_ids = [G.nodes[node]['seg_id'] for node in node_keys]
            node_tokens = [node if seg_id != 1 else 'is_a_variable' for (seg_id, node) in zip(node_seg_ids, node_keys)]
            
            mapping = dict(zip(node_keys, range(len(node_keys))))
            edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
            edge_seg_ids = []
            edge_tokens = []
            for i, (src, dst) in enumerate(G.edges()):
                edge_index[0, i] = mapping[src]
                edge_index[1, i] = mapping[dst]
                edge_seg_id = G[src][dst]['seg_id']
                edge_seg_ids.append(edge_seg_id)
                if edge_seg_id == -2:
                    edge_tokens.append('is_an_instance')
                else:
                    edge_tokens.append(G[src][dst]['role'])

            amr = Data(edge_index=edge_index)
            amr.num_nodes = len(node_tokens)
            
            seg_ids = node_seg_ids + edge_seg_ids
            tokens = node_tokens + edge_tokens
            
            tokens_ids, meta_dict = self.tokenizer.tokenize_amr(tokens)
            token_splits = meta_dict['each_token_length']

            return {'amrs': amr,
                    'seg_ids': seg_ids,
                    'token_ids': tokens_ids,
                    'token_splits': token_splits}


        elif self.version == 'linearized_ramr':
            amrs =  [self.text_aux_by_auxid[text_auxid]['amr_tree_string_linearization_dict']['amr_tree_string_linearized']]
            sents = [self.prefix + text_auxid]
            # tokenize句子
            # <s> text </s>  # 0 ... 2
            # dict["input_ids": list[list[int] ], "attention_masks" ]
            model_inputs = self.tokenizer(sents, max_length=self.max_src_length, padding=False, truncation=True)
            for input_id in model_inputs['input_ids']: # [list[int]]
                if len(input_id) > self.max_src_length - 10:
                    raise ValueError()
                
            # tokenize AMR
            amr_ids = []
            amr_tree_string_linearized_each_token_length = []
            for itm in amrs:
                amr_id, meta_dict = self.tokenizer.tokenize_amr(itm.split())
                if len(amr_id) > self.max_src_length - 10:
                    raise ValueError()
                amr_ids.append(amr_id[:self.max_src_length - 1] + [self.tokenizer.amr_eos_token_id])
                amr_tree_string_linearized_each_token_length.append(meta_dict['each_token_length'])
            # amr </g> # 53227 [list[int]]
            model_inputs["labels"] = amr_ids
            
            joint_ids = [
                srci + [self.tokenizer.amr_bos_token_id] + tgti
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ] # <s> text </s> <g> amr </g>
            
            max_src_length = min(self.max_src_length * 2, 512)
            joint_ids = [
                itm[:max_src_length - 1] + [self.tokenizer.amr_eos_token_id]
                if len(itm) > max_src_length
                else itm
                for itm in joint_ids
            ]
            seg_ids = [
                [0 for _ in range(len(srci))] + [1 for _ in range(len(tgti) + 1)]
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # 0代表text部分, 1代表是amr部分
            seg_ids = [itm[:max_src_length] for itm in seg_ids]
            model_inputs["joint_ids"] = joint_ids # 
            model_inputs["seg_ids"] = seg_ids 
            srcEtgt_ids = [  
                srci[: self.max_src_length - 4]
                + [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.amr_eos_token_id,
                ]
                if len(srci) > self.max_src_length - 3
                else srci 
                + [
                    self.tokenizer.amr_bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.amr_eos_token_id,
                ]
                for srci in model_inputs["input_ids"]
            ]  # <s> text </s> <g> <MASK> </g>
            Esrctgt_ids = [ 
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti 
                if len(tgti) <= self.max_src_length - 4
                else
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti[: self.max_src_length - 5]
                + [self.tokenizer.amr_eos_token_id]
                for tgti in model_inputs["labels"]
            ]  # <s> <MASK> </s> <g> AMR </g>

            Esrctgt_segids = [
                [0 for _ in range(3)] + [1 for _ in range(len(itm) - 3)]
                for itm in Esrctgt_ids
            ]
            srcEtgt_segids = [
                [0 for _ in range(len(itm) - 3)] + [1 for _ in range(3)]
                for itm in srcEtgt_ids
            ]
            model_inputs["srcEtgt_ids"] = srcEtgt_ids[0]
            model_inputs["srcEtgt_segids"] = srcEtgt_segids[0]
            model_inputs["Esrctgt_ids"] = Esrctgt_ids[0]
            model_inputs["Esrctgt_segids"] = Esrctgt_segids[0]
            model_inputs['input_ids'] = model_inputs['input_ids'][0]
            model_inputs['joint_ids'] = model_inputs['joint_ids'][0]
            model_inputs['seg_ids'] = model_inputs['seg_ids'][0]
            model_inputs['labels'] = model_inputs['labels'][0]
            
            model_inputs['attention_mask'] = model_inputs['attention_mask'][0]
            
            
            return {'model_inputs': model_inputs}
        

        elif self.version == 'amr_penman_graph':
            # refering的amr
            amr_wv = self.text_aux_by_auxid[text_auxid]['inference_graph'] 
            G : nx.DiGraph = nx.node_link_graph(amr_wv)
            top_var = G.graph['top']
            nodekey_to_token = {key:node_token for key, node_token in zip(G.nodes(), G.nodes())}
            nodekey_to_segid = {key:G.nodes[key]['seg_id'] for key in G.nodes()}
            # 标号，过滤掉segid=2的节点, 第一个永远是top var
            nodekey_to_idx = {} 
            nodekey_to_idx[top_var] = 0
            cnt = 1
            for node_key in G.nodes():
                if nodekey_to_segid[node_key] == 2:
                    continue
                if node_key == top_var:
                    continue
                nodekey_to_idx[node_key] = cnt
                cnt += 1
            idx_to_nodekey = {value:key for key, value in nodekey_to_idx.items()}
            
            edge_index = []
            edge_seg_ids = []
            edge_tokens = []
            for i, (src, dst) in enumerate(G.edges()):
                edge_seg_id = G[src][dst]['seg_id']
                if edge_seg_id == -2:
                    # 把/边的dst的token改成src的token, dst的segid改成2
                    assert nodekey_to_segid[dst] == 1 and nodekey_to_segid[src] == 2
                    nodekey_to_token[dst] = nodekey_to_token[src]
                    nodekey_to_segid[dst] = 2
                    # 忽略/边
                else:
                    edge_index.append([nodekey_to_idx[src], nodekey_to_idx[dst]])
                    edge_seg_ids.append(edge_seg_id)
                    edge_tokens.append(G[src][dst]['role'])
            # 按照idx获得token序列, segid序列
            node_tokens = [nodekey_to_token[idx_to_nodekey[idx]] for idx in range(cnt)] 
            node_seg_ids = [nodekey_to_segid[idx_to_nodekey[idx]] for idx in range(cnt)] 
            assert 1 not in node_seg_ids
            assert -2 not in edge_seg_ids
            edge_index = torch.tensor(edge_index).permute(1, 0)
            amr = Data(edge_index=edge_index)
            amr.num_nodes = len(node_tokens)
            seg_ids = node_seg_ids + edge_seg_ids
            tokens = node_tokens + edge_tokens
            assert tokens[0] == nodekey_to_token[top_var]
            
            tokens_ids, meta_dict = self.tokenizer.tokenize_amr(tokens)
            token_splits = meta_dict['each_token_length']
            output = {'amrs': amr,
                        'seg_ids': seg_ids,
                        'token_ids': tokens_ids,
                        'token_splits': token_splits}

  
            # linamr generation aux
            linamrs_aux = {}
            all_aux_ids = [] # 每个exist query对应的obj idx
            obj_id_identifier = []
            for obj_idx, obj_text_queries in enumerate(queries_by_objid):
                obj_id_identifier.extend([obj_idx] * len(obj_text_queries))
                all_aux_ids.extend(obj_text_queries)
            linamrs_aux['obj_id_identifier'] = obj_id_identifier # list[int, 这个text对应的obj_idx], n_text
    
            linamrs =  [self.text_aux_by_auxid[auxid]['amr_tree_string_linearization_dict']['amr_tree_string_linearized'] for auxid in all_aux_ids]
            linamr_ids = []
            for itm in linamrs:
                linamr_id, _ = self.tokenizer.tokenize_amr(itm.split())
                if len(linamr_id) > self.max_src_length - 10:
                    raise ValueError()
                linamr_ids.append(linamr_id[:self.max_src_length - 1] + [self.tokenizer.amr_eos_token_id])
            # amr </g> # 53227 [list[int]]
            linamrs_aux["labels"] = linamr_ids # list[list[int], AMR </g>], n_text
            
            max_src_length = min(self.max_src_length * 2, 512)
            Esrctgt_ids = [ 
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti 
                if len(tgti) <= self.max_src_length - 4
                else
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti[: self.max_src_length - 5]
                + [self.tokenizer.amr_eos_token_id]
                for tgti in linamrs_aux["labels"]
            ]  # <s> <MASK> </s> <g> AMR </g>
            linamrs_aux["Esrctgt_ids"] = Esrctgt_ids # # list[list[int], <s> MASK </s> <g> AMR </g>], n_text
            # labels: AMR </g>
            output['linamrs_aux'] = linamrs_aux
            return output

        else:
            raise ValueError()
                    
    def collate(self, auxiliary):
        if self.version == 'ramr_without_variable': # amr without variable
            amrs = [s_dic['amrs'] for s_dic in auxiliary]
            seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
            token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
            token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]
            node_alignments = [s_dic['node_alignments'] for s_dic in auxiliary]
            text_token_ids = [s_dic['text_token_ids'] for s_dic in auxiliary] # list[list[int]], batch
            text_token_splits = [s_dic['text_token_splits'] for s_dic in auxiliary]
            amr_tree_strings = [s_dic['amr_tree_string'] for s_dic in auxiliary]
            text_token_ids, _ = text_pad_token_ids(text_token_ids, self.tokenizer.pad_token_id)
            if 'all_concept_roles' in auxiliary[0]:
                all_concept_roles = [s_dic['all_concept_roles'] for s_dic in auxiliary]
                all_concept_roles, all_concept_roles_pad = text_pad_token_ids(all_concept_roles, self.tokenizer.pad_token_id) # b max
            else:
                all_concept_roles, all_concept_roles_pad = None, None
            return {
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
                'node_alignments': node_alignments,
                'text_token_ids': text_token_ids,
                'text_token_splits': text_token_splits,
                'all_concept_roles': all_concept_roles,
                'all_concept_roles_pad': all_concept_roles_pad,
                'amr_tree_strings': amr_tree_strings
            }
        
        elif self.version == 'ramr_with_variable':
            amrs = [s_dic['amrs'] for s_dic in auxiliary]
            seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
            token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
            token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]

            return {
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
            } 
              
        elif self.version == 'linearized_ramr':
            model_inputs = [aux['model_inputs'] for aux in auxiliary] # list[dict{'input_ids': [list[int]], ..}]
            self.v3_pad_model_inputs(model_inputs)
            model_inputs = self.tokenizer.pad(model_inputs,
                                            padding=True,
                                            max_length=None,
                                            pad_to_multiple_of=None,
                                            return_tensors='pt')
            return {
                'model_inputs': model_inputs, 
            }

        elif self.version == 'amr_penman_graph':
            amrs = [s_dic['amrs'] for s_dic in auxiliary]
            seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
            token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
            token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]

            # list[dict{'labels', 'Esr': list[list[int]], ni }], batch
            linamrs_aux = [aux['linamrs_aux'] for aux in auxiliary]
            # list[list[int], n_text], batch
            linamrs_obj_id = [linamr_au.pop('obj_id_identifier') for linamr_au in linamrs_aux]
            # list[list[AMR </g>], n_text], batch
            linamrs_labels = [linamr_au.pop('labels') for linamr_au in linamrs_aux]
            linamrs_flat_labels = [] # list[AMR </g>], text_sigma
            for linamr_lb in linamrs_labels:
                linamrs_flat_labels.extend(linamr_lb)
             
            linamrs_esrctgts = [linamr_au.pop('Esrctgt_ids') for linamr_au in linamrs_aux]
            linamrs_flat_esrctgts = [] # list[<s> <MASK> </s> <g> AMR </g>], text_sigma
            for linamr_estg in linamrs_esrctgts:
                linamrs_flat_esrctgts.extend(linamr_estg)

            linamrs_aux = {'Esrctgt_ids': text_pad_token_ids(linamrs_flat_esrctgts,pad_id=self.tokenizer.pad_token_id)[0],
                                'labels': text_pad_token_ids(linamrs_flat_labels, pad_id=-100)[0],
                                'obj_ids': linamrs_obj_id}
            return {
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
                'linamrs_aux': linamrs_aux,
            } 

        else:
            raise ValueError()
        
    def v4_pad_model_inputs(self, features):
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels", # 对amr </g>进行Pading
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_segids",
        )

    def v3_pad_model_inputs(self, features):
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels", # 对amr </g>进行Pading
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="joint_ids", # 对<s> text </s> <g> amr </g>进行padding
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="seg_ids", # 对0是text, 1是amr进行padding(1)
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_segids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_segids",
        )


                
        

            
        # amr_graph = penman.decode(amr_tree_string)
            
        # node_seg_ids = [] # 2/3, concepts/constants, 没有variable
        # edge_seg_ids = [] # -1/-3, edge/value edge
        
        # node_list = [] # list[variable]
        # node_id_map = {} # variable:id
        # node_concept_map = {} # variable:concept
        
        # edge_tokens = [] 
        # connectivity = [] 
        
        # node_list += amr_graph.variables() # m / v / m2
        # if node_list[0] != amr_graph._top:
        #     # swap the first one
        #     top_var_idx = [idx for idx in range(len(node_list)) if node_list[idx] == amr_graph._top]
        #     assert len(top_var_idx) == 1
        #     top_var_idx = top_var_idx[0]
        #     swap_value = node_list[0]
        #     node_list[0] = amr_graph._top
        #     node_list[top_var_idx] = swap_value
            
        # node_id_map = {node:id for id, node in enumerate(node_list)}
        # node_seg_ids += [2] * len(node_list)  # concept
        
        # for instance in amr_graph.instances():
        #     variable, role, concept = instance.source, instance.role, instance.target
        #     assert variable in node_list and role == ':instance'
        #     if concept is None:
        #         # coreference
        #         continue
        #     node_concept_map[variable] = concept
        # assert len(node_concept_map.keys()) == len(node_list)
                
        # for attr in amr_graph.attributes():
        #     # var :value 3
        #     variable, role, constant = attr.source, attr.role, attr.target
        #     assert variable in node_list
        #     dummy_var = ''.join(node_list)
        #     assert dummy_var not in node_concept_map
            
        #     node_id_map[dummy_var] = len(node_list)
        #     node_list.append(dummy_var)
        #     node_seg_ids.append(3)
        #     node_concept_map[dummy_var] = constant
                
        #     connectivity.append([node_id_map[dummy_var], node_id_map[variable]])
        #     edge_tokens.append(role)
        #     edge_seg_ids.append(-3)
                    
        # for edge in amr_graph.edges():
        #     var1, role, var2 = edge.source, edge.role,edge.target 
        #     assert var1 in node_list and var2 in node_list
        #     if ":ARG" in role:
        #         if "-of" in role:
        #             predicate = node_concept_map[var2]
        #             connectivity.append([node_id_map[var2], node_id_map[var1]])
        #             role_without_of = role[:-3]
        #             edge_tokens.append(role_without_of)
        #         else:
        #             predicate = node_concept_map[var1]
        #             connectivity.append([node_id_map[var1], node_id_map[var2]])
        #             edge_tokens.append(role)
        #         assert predicate in self.all_predicates
        #     else:
        #         connectivity.append([node_id_map[var2], node_id_map[var1]])
        #         role_of = role[:-3] if '-of' in role else f'{role}-of'
        #         edge_tokens.append(role_of)
        #     edge_seg_ids.append(-1)  
            
        # connectivity = torch.tensor(connectivity, dtype=torch.long).transpose(0, 1) # 2 num_edges
        # node_tokens = [node_concept_map[var] for var in node_list]
        # graphs = Data(edge_index=connectivity, kwargs={'num_nodes': len(node_tokens)})
        # graphs.num_nodes = len(node_tokens)
        # tokens = node_tokens + edge_tokens
        # tokens_ids, meta_dict = self.tokenizer.tokenize_amr(tokens)
        # token_splits = meta_dict['each_token_length']
        # seg_ids = node_seg_ids + edge_seg_ids
        
        # return graphs, seg_ids, tokens_ids, token_splits


