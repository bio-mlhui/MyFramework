import torch.nn as nn
from torch_geometric.data import Data, Batch
import torch
from transformers import RobertaModel, RobertaTokenizerFast, AutoConfig, RobertaForMaskedLM, AutoConfig
from models.layers_unimodal_attention import FeatureResizer, FeatureResizer_MultiLayer
import penman 
import torch.nn.functional as F
import re
from typing import List
"""把penman 转换成 带特征(d_model)的geometric graph
1. 对于相同lemma, 不同的predicate是否有不同的embedding
"""
_amr_tokenizer_entrypoints = {}

def register_amr_tokenizer(fn):
    amr_tokenizer_name = fn.__name__
    _amr_tokenizer_entrypoints[amr_tokenizer_name] = fn

    return fn

def amr_tokenizer_entrypoints(amr_tokenizer_name):
    try:
        return _amr_tokenizer_entrypoints[amr_tokenizer_name]
    except KeyError as e:
        print(f'amr_tokenizer {amr_tokenizer_name} not found')

class Learned_ConceptRole_Embedding(nn.Module):
    """
    给出concept, role的vocab, 学习两个embedding
    """
    def __init__(self, 
                 concept_vocab_file,
                 role_vocab_file,
                 d_model,
                 initialization=None) -> None:
        super().__init__()
        
        with open(concept_vocab_file, 'r') as f:
            concepts = f.read().splitlines()
        
        with open(role_vocab_file, 'r') as f:
            roles = f.read().splitlines()
        
        self.concept_encoder = {concept:id for id, concept in enumerate(concepts)}
        self.role_encoder = {role:id for id, role in enumerate(roles)}
        
        self.concept_embeddings = nn.Embedding(len(concepts), d_model)
        self.role_embeddings = nn.Embedding(len(roles), d_model)
    
    def forward(self, penmans, device) -> Batch:
        """
        penmans: list[penman.Graph]
        # every penman graph is a list of tuples, with two types:
        # edge: (node_id, role, node_id,)
        # node attribute: (node_id, :instance constant)

        # .top: root node的 node_id
        # .triplets: list[Triples]
        # .epidata - a mapping of triples to epigraphical markers
        # .instances list[Node (triplet)]
        # .edges(source) list[Edge]
        # attributes: list[]
        # .variables: list[node_id]
        # .reentrancies: {node_id:re-entrancy count}
        """
        graphs = []
        for amr_string in penmans:
            amr = penman.decode(amr_string)
            # 先转换
            num_nodes = len(amr.variables())
            num_edges = len(amr.edges())
            nodes_id = {var:id for id, var in enumerate(amr.variables())} 
            
            nodes = torch.zeros([num_nodes], device=device, dtype=torch.long)
            edges = torch.zeros([num_edges], device=device, dtype=torch.long)
            connectivity = []
            for edge_id, edge in enumerate(amr.edges()):
                connectivity.append([nodes_id[edge.source], nodes_id[edge.target]])
                edges[edge_id] = self.role_encoder[edge.role]
            connectivity = torch.tensor(connectivity, device=device, dtype=torch.long)
                
            for node in amr.instances():
                nodes[nodes_id[node.source]] = self.concept_encoder[node.target.split('-')[0]]
            
            g = Data(x=self.concept_embeddings(nodes), # n -> n c
                    edge_index=connectivity,  
                    edge_attr=self.role_embeddings(edges)) # # m -> m c
            # g.root = nodes_id[amr.top]
            graphs.append(g)
        graph_batch = Batch.from_data_list(graphs)
        return graph_batch

class PtEncoder_ConceptRole_Embedding(nn.Module):
    """
    在bert的embedding 的基础上增加一堆role embedding
    可选:
    1. bert的embedding是否更新
    2. 是否使用句子的bert输出, soft_attend_sentence
    3. 一个concept如果被分解成2个token, 如何聚合多个token, aggregate_multiple_tokens
    """
    def __init__(self,
                 d_model,
                 proj_configs,
                 encoder_configs,
                 role_vocab_file,
                 soft_attend_sentence=True,
                 aggregate_multiple_tokens='mean',
                 ) -> None:
        super().__init__()
        self.build_text_backbone(encoder_configs, d_model)
        self.build_proj(proj_configs, d_model)
        
        with open(role_vocab_file, 'r') as f:
            roles = f.read().splitlines()
        self.role_encoder = {role:id for id, role in enumerate(roles)}
        self.role_embeddings = nn.Embedding(len(roles), d_model)
        
        self.soft_attend_sentence = soft_attend_sentence
        self.aggregate_multiple_tokens = aggregate_multiple_tokens

    def build_text_backbone(self, configs, d_model):
        configs = vars(configs)
        if configs['name'] == 'pretrain_roberta_base':            
            self.tokenizer = RobertaTokenizerFast.from_pretrained('/home/xhh/pt/roberta-base')
            use_pretrained = configs['pretrained']
            freeze_encoder = configs['freeze_encoder']
            assert ((not use_pretrained) and (not freeze_encoder)) or (use_pretrained)
            
            if configs['pretrained']:
                text_backbone = RobertaModel.from_pretrained('/home/xhh/pt/roberta-base')
            else:
                config = AutoConfig.from_pretrained('/home/xhh/pt/roberta-base')
                text_backbone = RobertaModel(config=config)

            if configs['freeze_encoder']:
                for p in text_backbone.parameters():
                    p.requires_grad_(False) 
            self.text_backbone_vocab = text_backbone.embeddings
            self.text_backbone_encoder = text_backbone.encoder
            self.text_backbone_pooler = text_backbone.pooler
            self.hidden_size = text_backbone.config.hidden_size
        else:
            raise ValueError()
        
    def build_proj(self, proj_configs, d_model):
        configs = vars(proj_configs)
        if configs == {}:
            pass
        elif configs['name'] == 'resizer':
            self.text_proj = FeatureResizer(
                input_feat_size=self.hidden_size,
                output_feat_size=d_model,
                dropout=configs['dropout'],
                do_ln=configs['do_ln']
            )
        elif configs['name'] == 'resizer_multilayer':
            self.text_proj = FeatureResizer_MultiLayer(
                input_feat_size=self.hidden_size,
                hidden_size=d_model,
                output_feat_size=d_model,
                num_layers=configs['nlayers'],
                dropout=['dropout'],
                do_ln=configs['do_ln'],
            )
            
        else:
            raise NotImplementedError()
    
    def forward_sentence(self, texts, device):
        tokens = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        # b max
        token_padding_mask = tokens['attention_mask'].ne(1).bool() # b max, 1是padding的位置, 0是没有padding的位置
        # b max c, 最开始toknized后的特征
        tokenized_feats: torch.Tensor = self.text_backbone_vocab(input_ids=tokens['input_ids']) # b max d
        
        self_attn_mask = token_padding_mask.float()[:, None, None, :] # b 1 1 max
        self_attn_mask = self_attn_mask * torch.finfo(tokenized_feats.dtype).min
        encoder_outputs = self.text_backbone_encoder(
            tokenized_feats,
            attention_mask=self_attn_mask
        )
        sequence_output = encoder_outputs[0]  # b max c

        return sequence_output, token_padding_mask
        
    def get_concept_embeds(self, 
                           concepts, 
                           device,
                           sentence_feat=None,):
        """
        concepts: list[str], c
        sentence_feats: not_pad c
        """
        if self.soft_attend_sentence:
            assert sentence_feat is not None
        # c max
        tokens = self.tokenizer.batch_encode_plus(concepts, padding="longest", return_tensors="pt").to(device)
        token_padding_mask = tokens['attention_mask'].ne(1).bool() # c max, 1是padding的位置, 0是没有padding的位置
        tokenized_feats: torch.Tensor = self.text_backbone_vocab(input_ids=tokens['input_ids']) # c max d  # 还有start, end
        num_tokenized = (1.0 - token_padding_mask.float()).sum(-1) # c
        # c max 1 * c max d  -> c d / c 1 -> c d
        feats = ((1.0 - token_padding_mask.float()).unsqueeze(-1) * tokenized_feats).sum(dim=1) / (num_tokenized.unsqueeze(-1))
        
        if self.soft_attend_sentence:
            # 每个concept的 soft span
            # c n
            soft_span = F.softmax(torch.einsum('cd,nd->cn', feats, sentence_feat), dim=-1) 
            # c n, n d -> c d
            feats = torch.einsum('cn,nd->cd', soft_span, sentence_feat)
        feats = self.text_proj(feats) 
        return feats
        
    def forward(self, penmans: List[penman.Graph], text_query,  device) -> Batch:
        """
        text_query: list[str]
        penmans: list[penman.Graph]
        # every penman graph is a list of tuples, with two types:
        # edge: (node_id, role, node_id,)
        # node attribute: (node_id, :instance constant)

        # .top: root node的 node_id
        # .triplets: list[Triples]
        # .epidata - a mapping of triples to epigraphical markers
        # .instances list[Node (triplet)]
        # .edges(source) list[Edge]
        # attributes: list[]
        # .variables: list[node_id]
        # .reentrancies: {node_id:re-entrancy count}
        """
        # b max c, b max
        if self.soft_attend_sentence:
            sentence_features, pad_masks = self.forward_sentence(text_query, device=device)
        
        graphs = []
        for batch_id, amr in enumerate(penmans):
            variables = {} # var:id
            concepts = [] # list[str]
            edges = []  # list[role]
            
            for node_id, node in enumerate(amr.instances()):
                assert node.role == ':instance'
                variables[node.source] = node_id
                split_result = re.split('-|~', node.target)[0]
                concepts.append(split_result)
    
            connectivity = []        
            for edge_id, edge in enumerate(amr.edges()):
                connectivity.append([variables[edge.source], variables[edge.target]])
                edges.append(self.role_encoder[edge.role])
            if len(connectivity) == 0: # 没有edge, 只有node
                connectivity = torch.zeros([2, 0], device=device, dtype=torch.long)
            else:
                connectivity = torch.tensor(connectivity, device=device, dtype=torch.long).permute(1,0)
            edges = torch.tensor(edges, device=device, dtype=torch.long)
            edges_feats = self.role_embeddings(edges)

            # c d
            feats = self.get_concept_embeds(concepts,
                                            # max d [max] -> not_pad d
                                            sentence_feat=sentence_features[batch_id][~(pad_masks[batch_id].bool())] \
                                                if self.soft_attend_sentence else None,
                                            device=device)
            assert connectivity.shape[1] == edges_feats.shape[0]
            g = Data(x=feats, 
                    edge_index=connectivity,  
                    edge_attr=edges_feats) 
            graphs.append(g)
        graph_batch = Batch.from_data_list(graphs)
        return graph_batch


@register_amr_tokenizer
def learned(configs, d_model):
    return Learned_ConceptRole_Embedding(
        concept_vocab_file=configs.concept_vocab_file,
        role_vocab_file=configs.role_vocab_file,
        d_model=d_model,
        initialization=configs.initialization
    )
@register_amr_tokenizer
def pt_encoder(configs, d_model):
    return PtEncoder_ConceptRole_Embedding(
        d_model=d_model,
        proj_configs=configs.proj,
        encoder_configs=configs.encoder,
        role_vocab_file=configs.role_vocab_file,
        soft_attend_sentence=configs.soft_attend_sentence,
        aggregate_multiple_tokens=configs.aggregate_multiple_tokens
    )