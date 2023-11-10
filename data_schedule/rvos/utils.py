
from typing import Iterator, Optional, Sized
from util.misc import _max_by_axis, NestedTensor
import torch
import numpy as np
from torch.utils.data import Sampler
import logging

def generate_windows_of_video(all_frames, 
                                    window_size,
                                    window_step, 
                                    force_not_interleave=None, force_all_used=None,):
    all_frames = sorted(all_frames)
    if window_size is None:
        assert window_step == None
        sampled_windows = [all_frames]
        return sampled_windows
    else:
        if force_not_interleave:
            assert window_step >= window_size
        if force_all_used:
            assert window_step <= window_size

        sampled_windows = []
        for i in range(0, len(all_frames), window_step):
            sampled_windows.append(all_frames[i:i+window_size])
            if i + window_size >= (len(all_frames)-1):
                # 第一次超过
                break
        if force_not_interleave and force_all_used:
            assert sum([len(win) for win in sampled_windows]) == len(all_frames)
    
        return sampled_windows

class TrainRandomSampler_ByEpoch(Sampler[int]):
    def __init__(self, 
                 data_source,
                 seed,
                 ) -> None:
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.seed = seed
        self.epoch = None

    def __iter__(self):
        seed = self.seed + self.epoch
        print(f'generating a new indices permutations for this epoch using seed {seed}')
        n = len(self.data_source)
        g = torch.Generator()
        g.manual_seed(seed)
        
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=g).tolist()
        yield from torch.randperm(n, generator=g).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        
import math
import torch.distributed as dist
from typing import TypeVar, Optional, Iterator        
T_co = TypeVar('T_co', covariant=True)
from torch.utils.data.distributed import DistributedSampler
class TrainRandomSampler_ByEpoch_Distributed(Sampler[T_co]):
    def __init__(self, 
                 dataset, num_replicas,
                 rank,
                 seed: int = 0) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"" [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = None

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        seed = self.seed + self.epoch
        logging.info(f'generating a new indices permutations for this epoch using seed {seed}')
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size
        indices = indices[492:] + indices[0:492]
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.epoch = None # 必须每次都要调用set_epoch
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class Evaluate_Sampler_Distributed(Sampler[T_co]):
    def __init__(self, dataset, 
                 num_replicas,
                 rank) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"" [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas) 
        
        self.total_size = self.num_samples * self.num_replicas
        delta = self.total_size - len(self.dataset)
        if delta > 0:
            print(f'there will additional {delta} testing samples added to the whole evaluation dataset')

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples




def bounding_box_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax # y1, y2, x1, x2



def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id

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

import os
from torch.utils.data import Dataset
import networkx as nx
from torch_geometric.data import Data
class DatasetWithAux(Dataset):
    def __init__(self, 
                 pt_tokenizer_dir,
                 text_aux_version,
                 text_aux_by_auxid,
                 
                 video_aux_version,
                 video_aux_by_auxid,
                 ) -> None:
        self.text_aux_version = text_aux_version
        self.video_aux_version = video_aux_version
        self.text_aux_by_auxid = text_aux_by_auxid
        self.video_aux_by_auxid = video_aux_by_auxid
        
        if self.text_aux_version == 0:
            pass
        elif self.text_aux_version == 1: # amr without variable
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_tokenizer_dir, 'amr',
                                                                                              'AMRBART_pretrain'))
            self.prefix = ""
            self.max_src_length = 256

        elif self.text_aux_version == 2: # linamr sequence
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_tokenizer_dir,'amr',
                                                                                              'AMRBART_pretrain'))
            self.prefix = ""
            self.max_src_length = 256     
                   
        elif self.text_aux_version == 3: # amr_with_variable
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_tokenizer_dir,'amr',
                                                                                              'AMRBART_pretrain'))
        elif self.text_aux_version == 4: # amr_with_variable
            from models.amr_utils.tokenization_bart import AMRBartTokenizer
            self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained(os.path.join(pt_tokenizer_dir,'amr',
                                                                                              'AMRBART_pretrain'))
            self.prefix = ""
            self.max_src_length = 256  
            # from datasets.propbank_frames import PropBankFrames
            # self.pbframes = PropBankFrames('/home/xhh/workspace/rvos_encoder/datasets/propbank-frames/frames')
            # self.all_predicates = list(self.pbframes.rolesets_meaning.keys())
            
    def get_text_aux(self, text_auxid, queries_by_objid):
        # 0: not
        # 1: amr_wor
        # 2. lin_amr
        # 3. amr_wr
        # 4. amr_g
        if self.text_aux_version == 0:
            return {}
        
        elif self.text_aux_version == 1:
            amr_wv = self.text_aux_by_auxid[text_auxid]['inference_graph'] 
            tok_string = self.text_aux_by_auxid[text_auxid]['toknized_string']
            G : nx.DiGraph = nx.node_link_graph(amr_wv)
            top_var = G.graph['top']
            nodekey_to_token = {key:node_token for key, node_token in zip(G.nodes(), G.nodes())}
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
                    assert nodekey_to_segid[dst] == 1 and nodekey_to_segid[src] == 2
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
            edge_index = torch.tensor(edge_index).permute(1, 0)
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
                'text_token_splits': text_token_splits
            }
            
        elif self.text_aux_version == 2:
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
        
        elif self.text_aux_version == 3:  # amr with variable
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

        elif self.text_aux_version == 4:
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

    def get_video_aux(self, video_auxid):
        if self.video_aux_version == 0:
            return {}

def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return

class CollatorWithAux:
    def __init__(self, 
                 text_aux_version,
                 video_aux_version, **kwargs) -> None:
        self.text_aux_version = text_aux_version
        self.video_aux_version = video_aux_version
        if text_aux_version == 1:
            self.tokenizer = kwargs['tokenizer']
            self.label_pad_token_id = -100
        if text_aux_version == 2:
            self.tokenizer = kwargs['tokenizer']
            self.label_pad_token_id = -100
            
        elif text_aux_version == 3:
            self.tokenizer = kwargs['tokenizer']
            self.label_pad_token_id = -100
        elif text_aux_version == 4:
            self.tokenizer = kwargs['tokenizer']
            self.label_pad_token_id = -100
    def batching_aux(self, auxiliary):
        if self.text_aux_version == 0:
            return {
                'exist_queries':  [s['exist_queries'] for s in auxiliary],
                'sample_idx': [s['sample_idx'] for s in auxiliary]
            }
        
        elif self.text_aux_version == 1: # amr without variable
            amrs = [s_dic['amrs'] for s_dic in auxiliary]
            seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
            token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
            token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]
            node_alignments = [s_dic['node_alignments'] for s_dic in auxiliary]
            text_token_ids = [s_dic['text_token_ids'] for s_dic in auxiliary] # list[list[int]], batch
            text_token_splits = [s_dic['text_token_splits'] for s_dic in auxiliary]
            text_token_ids, _ = text_pad_token_ids(text_token_ids, self.tokenizer.pad_token_id)
            return {
                'exist_queries':  [s['exist_queries'] for s in auxiliary],
                'sample_idx': [s['sample_idx'] for s in auxiliary],
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
                'node_alignments': node_alignments,
                'text_token_ids': text_token_ids,
                'text_token_splits': text_token_splits
            }
        elif self.text_aux_version == 3:
            amrs = [s_dic['amrs'] for s_dic in auxiliary]
            seg_ids = [s_dic['seg_ids'] for s_dic in auxiliary]
            token_splits = [s_dic['token_splits'] for s_dic in auxiliary]
            token_ids =  [s_dic['token_ids'] for s_dic in auxiliary]

            return {
                'exist_queries':  [s['exist_queries'] for s in auxiliary],
                'sample_idx': [s['sample_idx'] for s in auxiliary],
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
            } 
              
        elif self.text_aux_version == 2:
            model_inputs = [aux['model_inputs'] for aux in auxiliary] # list[dict{'input_ids': [list[int]], ..}]
            self.v3_pad_model_inputs(model_inputs)
            model_inputs = self.tokenizer.pad(model_inputs,
                                            padding=True,
                                            max_length=None,
                                            pad_to_multiple_of=None,
                                            return_tensors='pt')
            return {
                'model_inputs': model_inputs, 
                'exist_queries':  [s['exist_queries'] for s in auxiliary],
                'sample_idx': [s['sample_idx'] for s in auxiliary]
            }

        elif self.text_aux_version == 4:
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
                'exist_queries':  [s['exist_queries'] for s in auxiliary],
                'sample_idx': [s['sample_idx'] for s in auxiliary],
                'amrs': amrs, 
                'seg_ids': text_pad_token_ids(seg_ids, 0)[0], # b (V+E)max
                'token_splits': token_splits, # list[list[int]]
                'token_ids': text_pad_token_ids(token_ids, self.tokenizer.pad_token_id)[0],  # b max
                'linamrs_aux': linamrs_aux,
            } 

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