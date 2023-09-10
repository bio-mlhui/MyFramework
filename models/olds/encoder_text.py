import torch
import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, Size
import torchvision
from torch import Tensor, nn
from .layers_unimodal_attention import FeatureResizer, FeatureResizer_MultiLayer, MLP
from .transformer import _get_clones
from transformers import RobertaModel, RobertaTokenizerFast, AutoConfig, RobertaForMaskedLM, AutoConfig
from typing import Dict, List, Optional, Any, Union
import random
import numpy as np
from .encoder_multiscale import multiscale_encoder_entrypoints
from util.misc import find_scales_from_multiscales, find_scale_from_multiscales
from einops import rearrange, repeat, reduce
from torch_geometric.data import Data, Batch
from models.amr_utils.tokenize_amr_penman import amr_tokenizer_entrypoints
from .layers_graph import graph_layer_entrypoints

class LoadPercentageRobertaModel(RobertaModel):
    # 每个module 都要
    def _init_weights_with_sparsity(self, module):
        if getattr(module, "_pretrained_have_sparsified", False):
            return
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            sparsify_normal_from_pretrained_(module.weight.data, 
                                             std=self.config.initializer_range, 
                                             sparsity=self.sparsify_pretrained)
            if module.bias is not None:
                sparsify_zero_from_pretrained_(module.bias.data, sparsity=self.sparsify_pretrained)
        elif isinstance(module, nn.Embedding):
            sparsify_normal_from_pretrained_(module.weight.data, 
                                             std=self.config.initializer_range,
                                             sparsity=self.sparsify_pretrained)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        module._pretrained_have_sparsified = True
    
    def init_weights_with_sparsity(self):
        self.apply(self._init_weights_with_sparsity)

def sparsify_normal_from_pretrained_(weight, std, sparsity):
    prob_m = torch.ones(weight.shape) * sparsity # shape
    chosen_indices = torch.bernoulli(prob_m).bool() # True/False
    normal_init_weights = torch.normal(0.0, std, weight.shape).to(weight)
    with torch.no_grad():
        weight[chosen_indices] = normal_init_weights[chosen_indices]

def sparsify_zero_from_pretrained_(weight, sparsity):
    prob_m = torch.ones(weight.shape) * sparsity # shape
    chosen_indices = torch.bernoulli(prob_m).bool() # True/False
    with torch.no_grad():
        weight.masked_fill_(chosen_indices, 0.)  

def pad_1d_feats(feat_list):
    # list[ni c] -> b nmax c
    feat_len = [len(feat) for feat in feat_list]
    n_max = max(feat_len) 
    batch_size = len(feat_list)
    pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=feat_list[0].device)
    for i in range(batch_size):
        feat_list[i] = F.pad(feat_list[i].clone(), pad=[0, 0, 0, n_max-feat_len[i]])
        pad_mask[i, :feat_len[i]] = False
    feat_list = torch.stack(feat_list, dim=0) # b nmax c
    return feat_list, pad_mask

_text_encoder_entrypoints = {}

def register_text_encoder(fn):
    text_encoder_name = fn.__name__
    _text_encoder_entrypoints[text_encoder_name] = fn

    return fn

def text_encoder_entrypoints(text_encoder_name):
    try:
        return _text_encoder_entrypoints[text_encoder_name]
    except KeyError as e:
        print(f'text_encoder {text_encoder_name} not found')


class TextSeq2Seq(nn.Module):                                                                                                                                     
    def __init__(self, 
                 backbone_configs,
                 proj_configs,
                 d_model,        
                 mlmhead_configs) -> None:                
        super().__init__()

        self.build_text_backbone(backbone_configs, d_model=d_model)
        self.build_proj(proj_configs, d_model)
        self.build_mlmhead(mlmhead_configs, d_model=d_model)

    @torch.no_grad()
    def _torch_mask_tokens(self, inputs: Any, 
                           special_tokens_mask: Optional[Any] = None,
                           masked_indices=None, get_masking_info=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone() # b num_token_max
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if self.mask_sampling == 'random':
            assert masked_indices is None
            probability_matrix = torch.full(labels.shape, self.mask_probability) # b num_token_max
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool) # b num_token_max [bool]
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0) # 不mask [CLS], [SEP], [PAD]
            masked_indices = torch.bernoulli(probability_matrix).bool() # 一句话里每个位置都有0.15的概率被masked掉, 并且至少有一个 # b max
            # b num_token_max
            for btc in range(len(masked_indices)):
                # 如果没有一个选择到，那么随机从token里选一个成为True
                if not masked_indices[btc, :].any():
                    non_special_indices = (~special_tokens_mask)[btc, :].nonzero(as_tuple=True)[0]
                    # select one to mask/replace/no_change
                    random_index = random.randint(0, len(non_special_indices)-1)
                    random_element = non_special_indices[random_index]
                    masked_indices[btc, random_element] = True  
                if masked_indices[btc, :].all():
                    pass # 如果都选择到了？？

        elif self.mask_sampling == 'gradient':
            assert 'grad' in get_masking_info and masked_indices is None
            grad_data = get_masking_info['grad'] # b max c
            grad_norm: torch.Tensor = torch.norm(grad_data,p=2.0, dim=-1) # b max
            masked_indices = torch.zeros(labels.shape).bool() # b max
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool) # b num_token_max [bool]
            else:
                special_tokens_mask = special_tokens_mask.bool()  
            
            for btc_idx, (grad, irspecial_mask) in enumerate(zip(grad_norm, ~special_tokens_mask)):
                ord_value, ord_idx = grad.sort(descending=True) 
                ord_idx = ord_idx[irspecial_mask[ord_idx.cpu()]] # max -> sm
                num = len(ord_idx)
                if num * self.mask_probability < 1:
                    chosex_num = 1
                else:
                    chosex_num = int(num * self.mask_probability)
                chose_idx = ord_idx[:chosex_num]
                masked_indices[btc_idx][chose_idx] = True
        else:
            raise ValueError()
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        if self.mask_onlymask:
            inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        else:
            # 在选中的token中再选80% 成为[MASK]
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices 
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 在选中的token中再选20%*50%=10% 成为随机另一个词语
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
            inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices
   
       
    def build_text_backbone(self, configs, d_model):
        configs = vars(configs)
        if configs['name'] == 'pretrain_roberta_base':            
            self.tokenizer = RobertaTokenizerFast.from_pretrained('/home/xhh/pt/roberta-base')
            use_pretrained = configs['pretrained']
            freeze_encoder = configs['freeze_encoder']
            assert ((not use_pretrained) and (not freeze_encoder)) or (use_pretrained)
            
            if configs['pretrained']:
                if 'pretrained_sparsity' in configs:
                    sparsity = configs['pretrained_sparsity']
                    assert sparsity < 1. and sparsity > 0.
                    text_backbone = LoadPercentageRobertaModel.from_pretrained('/home/xhh/pt/roberta-base')
                    text_backbone.sparsify_pretrained = sparsity
                    text_backbone.init_weights_with_sparsity()
                else:
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
            self.mask_probability = configs['mask_probability']  # 0.3
            self.mask_onlymask = configs['mask_onlymask']   # False
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
    
    def build_mlmhead(self, configs, d_model):
        configs = vars(configs)
        if configs == {}:
            pass
        elif configs['name'] == 'pretrain_roberta_base_mlmhead':
            self.mlm_proj = FeatureResizer(
                input_feat_size=d_model,
                output_feat_size=self.hidden_size,
                dropout=configs['proj_dropout'],
                do_ln=configs['proj_do_ln']
            ) 
            use_pretrained = configs['pretrained']
            freeze_mlmhead = configs['freeze_mlmhead']
            assert ((not use_pretrained) and (not freeze_mlmhead)) or (use_pretrained)
            if configs['pretrained']:
                self.mlm_head = RobertaForMaskedLM.from_pretrained('/home/xhh/pt/roberta-base').lm_head
            else:
                config = AutoConfig.from_pretrained('/home/xhh/pt/roberta-base')
                self.mlm_head = RobertaForMaskedLM(config=config).lm_head
            if configs['freeze_mlmhead']:
                for p in self.mlm_head.parameters():
                    p.requires_grad_(False) 
            self.head_type = 'pretrain'
            
        elif configs['name'] == 'tie_pretrain_roberta-base':
            self.mlm_bias = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=True) 
            self.mlm_norm = nn.LayerNorm(self.hidden_size)
            
            self.mlm_head = RobertaForMaskedLM.from_pretrained('/home/xhh/pt/roberta-base').lm_head
            if configs['freeze_mlmhead']:
                for p in self.mlm_head.parameters():
                    p.requires_grad_(False) 
            self.head_type = 'tie_pretrain'

        else:
            raise ValueError()
    
    def get_words_embeds(self, words, device):
        """
        words: list[word_token_id]
        """
        words = torch.tensor(words, device=device).long()
        token_feats = self.text_backbone_vocab.word_embeddings(words)
        return self.text_proj(token_feats)
    
    def forward(self, texts, text_auxiliary, device, 
                mask_sentence=False, masked_indices=None, get_masking_info=None):
        tokens = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        # b max
        token_padding_mask = tokens['attention_mask'].ne(1).bool() # b max, 1是padding的位置, 0是没有padding的位置
        tokenized_result = []
        for batch_idx in range(len(token_padding_mask)):
            pad = token_padding_mask[batch_idx, :] # max
            tk_ids = tokens['input_ids'][batch_idx] # max
            tks = np.array(self.tokenizer.convert_ids_to_tokens(tk_ids))[~pad.cpu()].tolist()
            tokenized_result.append(tks)

        if mask_sentence:
            tokens['input_ids'], token_gt_labels, token_masked_bool = self._torch_mask_tokens(inputs=tokens['input_ids'],
                                                                                              masked_indices=masked_indices,
                                                                                              get_masking_info=get_masking_info)
            masked_sentences = []
            for batch_idx in range(len(token_padding_mask)):
                pad = token_padding_mask[batch_idx, :] # max
                tk_ids = tokens['input_ids'][batch_idx] # max
                tks = np.array(self.tokenizer.convert_ids_to_tokens(tk_ids))[~pad.cpu()].tolist()
                masked_sentences.append(' '.join(tks))

        # b max c, 最开始toknized后的特征
        tokenized_feats: torch.Tensor = self.text_backbone_vocab(input_ids=tokens['input_ids']) # b max d
        if not mask_sentence:
            tokenized_feats.requires_grad_(True)
        # tokenized_feats.requires_grad_(True)  # 虽然text backbone是没有在训练的, 但是中间的text特征会有梯度, 所以可以得到
 
        self_attn_mask = token_padding_mask.float()[:, None, None, :] # b 1 1 max
        self_attn_mask = self_attn_mask * torch.finfo(tokenized_feats.dtype).min
        encoder_outputs = self.text_backbone_encoder(
            tokenized_feats,
            attention_mask=self_attn_mask
        )
        sequence_output = encoder_outputs[0]  # b max c
        # b c
        pooled_output = self.text_backbone_pooler(sequence_output) if self.text_backbone_pooler is not None else None

        text_features = self.text_proj(sequence_output) 
        text_sentence_features = self.text_proj(pooled_output) 
        
        # b token_max c, b token_max, b c
        if mask_sentence:
            return {
                'tokenized_result': tokenized_result,
                'tokenized_feats': tokenized_feats,
                'masked_sentences': masked_sentences,
                
                'token_feats': text_features,
                'token_pad_masks': token_padding_mask,
                'token_sentence_feats': text_sentence_features,
                
                'mlm_text_gt': {
                    'token_gt_labels': token_gt_labels,
                    'token_masked_bool':token_masked_bool
                }
            }
            
        else:
            return {
                'tokenized_result': tokenized_result,
                'tokenized_feats': tokenized_feats,
                'token_feats': text_features,
                'token_pad_masks': token_padding_mask,
                'token_sentence_feats': text_sentence_features,
            }
            
    def forward_mlm(self, text_feats, masked_gt): 
        """
        text_feats: b s c
        token_gt_labels: b s; token_masked_bool: b s; tokens_gt_ids: b s
        """  
        token_gt_labels, token_masked_bool = masked_gt['token_gt_labels'], masked_gt['token_masked_bool']
        
        logits = self.get_pred_logits(text_feats)  # b vocab max
        mlm_loss = F.cross_entropy(logits, token_gt_labels, ignore_index=-100)

        out = []
        for batch_idx in range(len(logits)):
            sentence_out = []
            masked_idxs = token_masked_bool[batch_idx].nonzero(as_tuple=True)[0] 
            for idx in masked_idxs:
                scores, ids = torch.topk(logits[batch_idx, :, idx].detach().softmax(-1), k=30, largest=True, sorted=True) # 30
                tokens = self.tokenizer.convert_ids_to_tokens(ids) # 30
                sentence_out.append(list(zip(tokens, scores)))
            out.append(sentence_out)
        # 由于有些可能是非常low-level或者非常high-level的, 所以只最小化最小的loss
        return out, {"loss_mlm": mlm_loss}
        
    def get_pred_logits(self, text_feats):
        # b s c
        if self.head_type == 'pretrain':
            logits = self.mlm_head(self.mlm_proj(text_feats)).permute(0, 2, 1) # b s vocab -> b vocab max
            return logits
        elif self.head_type == 'tie_pretrain':
            weight = self.mlm_proj.fc.weight.T[None, None, ...] # 1, 1, 256 768
            logits = text_feats[:, :, None, :] @ weight + self.mlm_bias[:, :, :, None] 
            # b s 1 c; 1, 1, c 768 -> b s 1 768
            logits = logits.squeeze(2) # b s 768
            logits = self.mlm_norm(logits) # b s 768
            logits = self.mlm_head(logits).permute(0, 2, 1)
        else:
            raise ValueError()
    

class LinAMR_Seq2seq(nn.Module):
 
    def __init__(self, 
                 d_model,
                 freeze_text_encoder=True,
                 how_to_get_word_embedding='amr_encoder amr_decoder'
                 ) -> None:
        super().__init__()
        from .amr_utils.utils import BartForConditionalGeneration
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        if freeze_text_encoder:
            for p in self.model.parameters():
                p.requires_grad_(False) 
        self.hidden_size = self.model.config.d_model
        self.text_proj = FeatureResizer(
                input_feat_size=self.hidden_size,
                output_feat_size=d_model,
                dropout=0.,
                do_ln=True
        )
        self.how_to_get_word_embedding = how_to_get_word_embedding
        
    def get_words_embeds(self, words, device):
        """
        words: list[word_token_id]
        """
        words = torch.tensor(words, device=device).long()
        token_feats = self.model.model.shared(words) # k c
        return self.text_proj(token_feats)
    
    def get_linearized_token_embedding(self, model_inputs, device):
 
        # input_ids: <s> text </s>
        # srcEtgt_ids: <s> text </s> <g> <MASK> </g>
        # Esrctgt_ids: <s> <MASK> </s> <g> amr </g>
        # labels: amr </g>
        # joint_ids: <s> text </s> <g> amr </g>

        if self.how_to_get_word_embedding == 'amr_encoder':
            # Esrctgt, label
            pass
        elif self.how_to_get_word_embedding == 'amr_encoder amr_decoder':
            # <s> <MASK> </s> <g> amr </g> pad
            bart_input = model_inputs["Esrctgt_ids"] # b max
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()      
            # amr </g> pad pad
            labels = model_inputs["labels"] # b max
            
            dec_input = labels.new_zeros(labels.size(0), labels.size(1))
            # <g> amr </g> pad -> amr </g> pad pad
            dec_input[:, 1:] = labels[:, :-1].clone()
            dec_input[:, 0] = self.tokenizer.amr_bos_token_id 
 
            decoder_input_pad_mask = (dec_input == -100) 
            dec_input.masked_fill_(decoder_input_pad_mask, self.tokenizer.pad_token_id)
            
            bart_input = bart_input.to(device) # <s> <MASK> </s> <g> amr </g> pad
            attention_mask = attention_mask.to(device) # 0代表padding的位置
            labels = labels.to(device) # amr </g> -100
            dec_input = dec_input.to(device) # <g> amr </g> pad
            # self.tokenizer.decode([self.model.lm_head(decoder_output[0][i]).argmax().item() for i in range(len(decoder_output[0]))])
            # amr </g> pad
            amr_embeds = self.model(input_ids=bart_input,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=dec_input,
                                    labels=labels)
            # list[num_token c], amr </g>
            amr_embeds = [dec_out[~dec_pad][:-1] for dec_out, dec_pad in zip(amr_embeds, decoder_input_pad_mask)]
            return amr_embeds, None, None
        
        elif self.how_to_get_word_embedding == 'amr+text_encoder amr_decoder':
            # joint, label
            pass
        elif self.how_to_get_word_embedding == 'amr+text_encoder amr+text_decoder':
            bart_input = model_inputs["joint_ids"]
            seg_ids = model_inputs['seg_ids'] # 0: text, 1: graph
            labels = model_inputs["joint_ids"].clone()
            labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)
            labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
            dec_input = model_inputs["joint_ids"].clone()
            dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()          # attention mask
            
            # text </s> <g> amr </g>
            decoder_output = self.model(input_ids=bart_input,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=dec_input,
                                        labels=labels).decoder_hidden_states
            decoder_output = self.text_proj(decoder_output)
            text_feat = decoder_output
            
            return decoder_output, meta_dict['each_token_length'], text_feat, None
     
    def forward(self, graph, text, device):
        model_inputs = graph['model_inputs']  # dict["input_ids", "labels", "joint_ids"]
        
        lin_amr_embeds, atext_feats, text_pad_mask \
            = self.get_linearized_token_embedding(model_inputs, device=device)
            
        lin_amr_embeds, amr_lin_pad_mask = pad_1d_feats(lin_amr_embeds)    
        lin_amr_embeds = self.text_proj(lin_amr_embeds)
        return {
            'token_feats': lin_amr_embeds,
            'token_pad_masks': amr_lin_pad_mask,
            'token_sentence_feats': None,
        }
            

class AMRWordembedding(nn.Module):
    def __init__(self, 
                 d_model,
                 how_to_get_tokk_embed_configs,
                 from_scratch=False, # 是否从新开始初始化
                 freeze_vocabulary=False,
                 model_name='amrbart'
                 ) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name == 'amrbart':
            from transformers import BartModel
            from .amr_utils.tokenization_bart import AMRBartTokenizer
            amr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            self.d_model = d_model
            self.need_proj = False
            amr2text_model = BartModel.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            pretrained_dim = amr2text_model.config.d_model
            self.word_embedding = amr2text_model.shared
            if freeze_vocabulary:
                for p in self.word_embedding.parameters():
                    p.requires_grad_(False) 
            if pretrained_dim != d_model:
                self.need_proj=True
                self.text_proj = FeatureResizer(input_feat_size=pretrained_dim,
                                                output_feat_size=d_model,
                                                dropout=0.,
                                                do_ln=True)
            
            if how_to_get_tokk_embed_configs.name == 1:
                self.fuse_subtoken = how_to_get_tokk_embed_configs.fuse_subtoken
            else:
                raise ValueError()
            self.how_to_get_tokk_embed = how_to_get_tokk_embed_configs.name
            
        elif model_name == 'clip':
            from transformers import CLIPModel
            clip = CLIPModel.from_pretrained("/home/xhh/pt/clip")
            self.word_embedding = clip.text_model.embeddings
            self.text_projection = clip.text_projection
            pretrained_dim = clip.projection_dim
            if freeze_vocabulary:
                for p in self.word_embedding.parameters():
                    p.requires_grad_(False) 
                    
            if pretrained_dim != d_model:
                self.need_proj=True
                self.text_proj = FeatureResizer(input_feat_size=pretrained_dim,
                                                output_feat_size=d_model,
                                                dropout=0.,
                                                do_ln=True)
        elif model_name == 'from_scratch':
            self.word_embedding = nn.Embedding(amr_tokenizer.vocab_size, d_model, amr_tokenizer.pad_token_id,)
                       
    def get_words_embeds(self, words, device):
        """
        words: list[word_token_id]
        """
        words = torch.tensor(words, device=device).long()
        token_feats = self.word_embedding(words) # k c
        if self.need_proj:
            return self.text_proj(token_feats)
        return token_feats
               
    def forward(self, graph, text, device):
        graphs = graph['graphs'] # list[T(2 E_i)]
        seg_ids = graph['seg_ids'].to(device)  # b (V+E)max
        token_splits = graph['token_splits'] # list[list[int]]
        tokens_ids = graph['tokens_ids'].to(device) # b max 
        
        token_feats = self.word_embedding(tokens_ids) # b max c
        if self.need_proj:
            token_feats = self.text_proj(token_feats)
            
        # list[list[ti c]] -> list[Vi+Ei c]
        token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(token_feats, token_splits)]
        for batch_idx in range(len(token_feats)):
            token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in token_feats[batch_idx]], dim=0)
            
        token_feats = pad_1d_feats(token_feats)[0] # b (V+E)max c
        assert token_feats.shape[1] == seg_ids.shape[1]
        return {
            'all_feats': token_feats,
            'all_seg_ids': seg_ids,
            'graph': graphs
        }
  


@register_text_encoder
def text_encoder_mlm(configs, d_model):
    return TextSeq2Seq(
        backbone_configs=configs.text_backbone,
        proj_configs=configs.proj,
        d_model=d_model,
        mlmhead_configs=configs.mlmhead_configs)


@register_text_encoder
def linearized_seq2seq_outSequence(configs, d_model):
    return LinAMR_Seq2seq(
        d_model=d_model,
        freeze_text_encoder=configs.freeze_text_encoder,

        how_to_get_word_embedding=configs.how_to_get_word_embedding
    )


@register_text_encoder
def inputpenmangraph_wordembeddingencoder_outputgraph(configs, d_model):
    return AMRWordembedding(
        d_model=d_model,
        how_to_get_tokk_embed_configs=configs.how_to_get_tokk_embed,
        from_scratch=configs.from_scratch,
        freeze_vocabulary=configs.freeze_vocabulary
    )

























class AMRBart_Seq2Seq_OutGraph(nn.Module):
    def __init__(self, 
                 d_model,
                 freeze_text_encoder,
                 proj_configs,
                 how_to_get_word_embedding,
                 fuse_subtoken_feats,
                 fuse_multiple_alignment,
                 ) -> None:
        super().__init__()

        from .amr_utils.tokenization_bart import AMRBartTokenizer        
        from .amr_utils.utils import BartForConditionalGeneration
        self.model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        if freeze_text_encoder:
            for p in self.model.parameters():
                p.requires_grad_(False) 
        self.hidden_size = self.model.config.d_model
        self.build_proj(proj_configs, d_model)
        # vocab.json + additional.json (free-form English words + roles + predicates)
        # 1. 所有的variable name换成<pointer:x>/<Rx>; (因为一个variable name很可能被一个nlp tokenizer变成多个tokens)
        # 2. variable/concept 被换成 <pointer:x> concept
        # 3. 括号不变, 为了表示深度
        self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.is_a_embedding =nn.Embedding(1, self.hidden_size)
        self.how_to_get_word_embedding = how_to_get_word_embedding
        self.fuse_subtoken_feats = fuse_subtoken_feats
        
        self.prefix = ""
        self.max_src_length = 256
        self.label_pad_token_id = -100
        self.fuse_multiple_alignment = fuse_multiple_alignment
        
   
    def build_proj(self, proj_configs, d_model):
        configs = vars(proj_configs)
        if configs == {}:
            self.text_proj = nn.Sequential()
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
        
    def fuse_subtoken(self, token_feats):
        """
        token_feats: list[ni c] -> list[c]
        """   
        if self.fuse_subtoken_feats == 'mean':
            return [t_feat.mean(0) for t_feat in token_feats]
        else:
            raise ValueError()
        
    def fuse_multiple_alignment_feat(self, node_list, node_feats):
        """
        node_feats: {node: list[c]} ->
        """
        if self.fuse_multiple_alignment == 'mean':
            graph_node_feats = []
            for node in node_list:
                if node is None:
                    raise ValueError()
                graph_node_feats.append(torch.stack(node_feats[node], dim=0).mean(0))
            graph_node_feats = torch.stack(graph_node_feats, dim=0) # num_nodes c
            return graph_node_feats
        
        else:
            raise ValueError()

    def get_linearized_token_embedding(self, lin_amr_tree_strings, text_query, device):
        """
        list[str], list[text_query]
        """
        def tokenize_function(examples):
            # Remove empty lines
            amrs = examples["amr"]           # AMR tokens
            sents = examples["text"]          # text tokens
            sents = [self.prefix + inp for inp in sents]

            model_inputs = self.tokenizer(sents, max_length=self.max_src_length, padding=False, truncation=True)
            for input_id in model_inputs['input_ids']:
                if len(input_id) > self.max_src_length - 10:
                    raise ValueError()
                
            amr_ids = []
            amr_tree_string_linearized_each_token_length = []
            for itm in amrs:
                amr_id, meta_dict = self.tokenizer.tokenize_amr(itm.split())
                if len(amr_id) > self.max_src_length - 10:
                    raise ValueError()
                amr_ids.append(amr_id[:self.max_src_length - 1] + [self.tokenizer.amr_eos_token_id])
                amr_tree_string_linearized_each_token_length.append(meta_dict['each_token_length'])
            model_inputs["labels"] = amr_ids
            
            joint_ids = [
                srci + [self.tokenizer.amr_bos_token_id] + tgti
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # [<s> x1,x2...,xn </s> <AMR> y1,y2,...ym </AMR>]
            
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
            ]  # [0,0,...,0,1,1,...1]
            seg_ids = [itm[:max_src_length] for itm in seg_ids]
            model_inputs["joint_ids"] = joint_ids
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
            ]  # [<s> x1,x2...,xn <\s> <AMR> [mask] </AMR>]
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
            ]  # [<s> [mask] <\s> <AMR> y1,y2...,yn </AMR>]

            Esrctgt_segids = [
                [0 for _ in range(3)] + [1 for _ in range(len(itm) - 3)]
                for itm in Esrctgt_ids
            ]
            srcEtgt_segids = [
                [0 for _ in range(len(itm) - 3)] + [1 for _ in range(3)]
                for itm in srcEtgt_ids
            ]
            model_inputs["srcEtgt_ids"] = srcEtgt_ids
            model_inputs["srcEtgt_segids"] = srcEtgt_segids
            model_inputs["Esrctgt_ids"] = Esrctgt_ids
            model_inputs["Esrctgt_segids"] = Esrctgt_segids
            return [model_inputs], amr_tree_string_linearized_each_token_length

        def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
            assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
            max_label_length = max(len(feature[key]) for feature in features)
            for feature in features:
                remainder = [pad_token_id] * (max_label_length - len(feature[key]))
                feature[key] = (
                    feature[key] + remainder if padding_side == "right" else remainder + feature[key]
                )
            return

        def pad_model_inputs(features):
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.label_pad_token_id,
                key="labels",
            )
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
                key="joint_ids",
            )
            padding_func(
                features,
                padding_side=self.tokenizer.padding_side,
                pad_token_id=self.tokenizer.pad_token_id,
                key="seg_ids",
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

        # input_ids: <s> text </s>
        # srcEtgt_ids: <s> text </s> <g> <MASK> </g>
        # Esrctgt_ids: <s> <MASK> </s> <g> amr </g>
        # labels: amr </g>
        # joint_ids: <s> text </s> <g> amr </g>
        model_inputs, amr_tree_string_linearized_each_token_length = tokenize_function({"amr": lin_amr_tree_strings,
                                                    "text": text_query})
        pad_model_inputs(model_inputs)
        model_inputs = self.tokenizer.pad(model_inputs[0],
                                          padding=True,
                                          max_length=None,
                                          pad_to_multiple_of=None,
                                          return_tensors='pt')
        
        if self.how_to_get_word_embedding == 'amr_encoder':
            # Esrctgt, label
            pass
        elif self.how_to_get_word_embedding == 'amr_encoder amr_decoder':
            # <s> <MASK> </s> <g> amr </g>
            # 0     0      0  1...
            bart_input = model_inputs["Esrctgt_ids"] # b max
            seg_ids = model_inputs["Esrctgt_segids"] # b max 0/1/pad_token
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()      # attention mask
            # amr </g> pad pad
            labels = model_inputs["labels"] # b max
            dec_input = labels.new_zeros(labels.size(0), labels.size(1))
            # <g> amr </g> pad
            dec_input[:, 1:] = labels[:, :-1].clone()
            dec_input[:, 0] = self.tokenizer.amr_bos_token_id                        # <s> w1 w2, ..., wn
            #  0   0    0   1
            decoder_input_pad_mask = (dec_input == -100) 
            dec_input.masked_fill_(decoder_input_pad_mask, self.tokenizer.pad_token_id)
            
            bart_input = bart_input.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            dec_input = dec_input.to(device)
            # self.tokenizer.decode([self.model.lm_head(decoder_output[0][i]).argmax().item() for i in range(len(decoder_output[0]))])
            # amr </g> pad
            decoder_output = self.model(input_ids=bart_input,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=dec_input,
                                        labels=labels)
            # list[num_token c]
            decoder_output = [dec_out[~dec_pad][:-1] for dec_out, dec_pad in zip(decoder_output, decoder_input_pad_mask)]
            return decoder_output, amr_tree_string_linearized_each_token_length, None, None
        
        elif self.how_to_get_word_embedding == 'amr+text_encoder amr_decoder':
            # joint, label
            pass
        elif self.how_to_get_word_embedding == 'amr+text_encoder amr+text_decoder':
            bart_input = model_inputs["joint_ids"]
            seg_ids = model_inputs['seg_ids'] # 0: text, 1: graph
            labels = model_inputs["joint_ids"].clone()
            labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)
            labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
            dec_input = model_inputs["joint_ids"].clone()
            dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
            attention_mask = bart_input.ne(self.tokenizer.pad_token_id).int()          # attention mask
            
            # text </s> <g> amr </g>
            decoder_output = self.model(input_ids=bart_input,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=dec_input,
                                        labels=labels).decoder_hidden_states
            decoder_output = self.text_proj(decoder_output)
            text_feat = decoder_output
            
            return decoder_output, meta_dict['each_token_length'], text_feat, None
     
       
    def forward(self, graph, text, device):
        """
        graph: {'amr_tree_string': list[string], 'lineaerized_penman': List[dict]}
        text: 
        """
        var_pointer_map_by_batch = []
        instance_linearized_ali_by_batch = []
        edge_linearized_ali_by_batch = []
        attribute_linearized_ali_by_batch = []
        lin_amr_tree_strings = []
        for amr_dict in graph['amr_tree_string_linearization_dict']:
            var_pointer_map_by_batch.append(amr_dict['var_pointer_map'])
            instance_linearized_ali_by_batch.append(amr_dict['instance_linearized_ali'])
            edge_linearized_ali_by_batch.append(amr_dict['edge_linearized_ali'])
            attribute_linearized_ali_by_batch.append(amr_dict['attribute_linearized_ali'])
            lin_amr_tree_strings.append(amr_dict['amr_tree_string_linearized'])
        
        # list[num_tok c], list[list[int]], None/list[num_tok_text c]
        amr_linearized_feats, amr_linearized_each_token_length, text_feats, text_pad_masks\
            = self.get_linearized_token_embedding(lin_amr_tree_strings, text, device)
        
        graphs = []
        node_identifiers = []
        edge_identifiers = []
        for batch_idx, (linearized_feat, each_tok_len, 
                        var_pointer_map, instance_ali, edge_ali, attribute_ali, 
                        amr_tree_string, amr_tree_string_linearized) in \
            enumerate(zip(amr_linearized_feats, amr_linearized_each_token_length,
                          var_pointer_map_by_batch,instance_linearized_ali_by_batch,edge_linearized_ali_by_batch, attribute_linearized_ali_by_batch,
                          graph['amr_tree_string'], lin_amr_tree_strings)):
            assert len(each_tok_len) == len(amr_tree_string_linearized.split())
            assert len(linearized_feat) == sum(each_tok_len)
            amr_graph = penman.decode(amr_tree_string)
            # n c -> list[ni c] -> list[c]
            linearized_feat = self.fuse_subtoken(torch.split(linearized_feat, each_tok_len))
            node_list = [] # list[var/concept_value/constant]
            node_id_map = {} # node:id
            node_feats = {} # node:list[c]
            node_identifier = [] # 0:variable, 1:concept_val, 2: constant
            connectivity = [] # list[2]
            edge_feats = [] # list[c], num_edge
            edge_identifier = [] # # 0: regular, 1: /, 2: constant edge
            
            node_list += amr_graph.variables() # m / v / m2
            if node_list[0] != amr_graph._top:
                # swap the first one
                top_var_idx = [idx for idx in range(len(node_list)) if node_list[idx] == amr_graph._top]
                assert len(top_var_idx) == 1
                top_var_idx = top_var_idx[0]
                swap_value = node_list[0]
                node_list[0] = amr_graph._top
                node_list[top_var_idx] = swap_value
            
            node_id_map = {node:id for id, node in enumerate(node_list)}
            pointer_var_map = {pointer:var for var, pointer in var_pointer_map.items()}
            node_feats = {var:[] for var in var_pointer_map.keys()}
            for id, tok in enumerate(amr_tree_string_linearized.split()):
                if tok.startswith('<pointer:') and tok.endswith('>'):
                    node_feats[pointer_var_map[tok]].append(linearized_feat[id])
            node_identifier = [0] * len(node_list)
            
            # variable / concept_value
            for instance in amr_graph.instances():
                variable = instance.source
                concept = instance.target
                assert variable in node_list
                assert instance.role == ':instance'
                if concept is None:
                    # coreference, 没有node也没有edge
                    continue  
                var_ali, concept_ali = instance_ali[f'{variable}, :instance, {concept}'] 
                if concept in node_list:  # 比如一个graph里有m / m2 都是man
                    node_feats[concept].append(linearized_feat[concept_ali])
                else:  
                    # 得到concept的embedding
                    node_id_map[concept] = len(node_list)
                    node_list.append(concept)
                    node_identifier.append(1)
                    node_feats[concept] = [linearized_feat[concept_ali]]
                    
                connectivity.append([node_id_map[variable], node_id_map[concept]])
                edge_feats.append(self.is_a_embedding.weight.squeeze(0))
                edge_identifier.append(1)
                             
            for attr in amr_graph.attributes():
                variable, role, constant = attr.source, attr.role, attr.target
                variable_ali, role_ali, constant_ali = attribute_ali[f'{variable}, {role}, {constant}']
                assert variable in node_list
                if constant in node_list:
                    node_feats[constant].append(linearized_feat[constant_ali])
                else:
                    # 得到constant的embedding
                    node_id_map[constant] = len(node_list)
                    node_list.append(constant)
                    node_feats[constant] = [linearized_feat[constant_ali]]
                    node_identifier.append(2)
                connectivity.append([node_id_map[variable], node_id_map[constant]])
                edge_feats.append(linearized_feat[role_ali])
                edge_identifier.append(2)
                     
            for edge in amr_graph.edges():
                var1, role, var2 = edge.source, edge.role,edge.target 
                if f'{var1}, {role}, {var2}' not in edge_ali:
                    if '-of' in role:
                        var2_ali, role_ali, var1_ali = edge_ali[f'{var2}, {role[:-3]}, {var1}'] # linearized里是去掉了-of, 树上是反着连的
                    else:
                        var2_ali, role_ali, var1_ali = edge_ali[f'{var2}, {role}-of, {var1}'] # linearized里是-of, 树上是反着-of连的
                    assert var1 in node_list and var2 in node_list
                    connectivity.append([node_id_map[var2], node_id_map[var1]])
                else:
                    var1_ali, role_ali, var2_ali = edge_ali[f'{var1}, {role}, {var2}'] # 书上就是这么连接的
                    assert var1 in node_list and var2 in node_list
                    connectivity.append([node_id_map[var1], node_id_map[var2]])
                edge_feats.append(linearized_feat[role_ali])
                edge_identifier.append(0)
            
            graph_node_feats = self.fuse_multiple_alignment_feat(node_list=node_list,
                                                                 node_feats=node_feats)

            graph_edge_feats = torch.stack(edge_feats, dim=0) # num_edges c
            assert len(graph_edge_feats) == len(connectivity)
            # connectivity必须符合tree结构, 因为bart见到的就是tree
            connectivity = torch.tensor(connectivity, device=device, dtype=torch.long).transpose(0, 1) # 2 num_edges
            
            graph_node_feats = self.text_proj(graph_node_feats)
            graph_edge_feats = self.text_proj(graph_edge_feats)
            graphs.append(Data(
                    x = graph_node_feats,
                    edge_index = connectivity,
                    edge_attr=graph_edge_feats
            ))
            node_identifiers.append(torch.tensor(node_identifier,dtype=torch.long, device=device))
            edge_identifiers.append(torch.tensor(edge_identifier,dtype=torch.long, device=device))

        return {'graphs': graphs,
                'node_identifiers':node_identifiers,
                'edge_identifiers': edge_identifiers,
                'text_feats': text_feats,
                'text_pad_masks': text_pad_masks}
                
    @classmethod
    def pad_token_ids(cls, token_ids, pad_id, device):
        # list[list[int], ni], batch -> T(batch, n_max)
        batch_size = len(token_ids)
        n_max = max(len(t_ids) for t_ids in token_ids)
        pad_mask = torch.ones([batch_size, n_max], dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            num_tokens = len(token_ids[i])
            token_ids[i] = token_ids[i] + [pad_id] * (n_max - num_tokens)
            pad_mask[i][:num_tokens] = False

        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        return token_ids, pad_mask

@register_text_encoder
def inputlinearized_amrbartseq2seq_outputgraph(configs, d_model):
    return AMRBart_Seq2Seq_OutGraph(
        d_model=d_model,
        freeze_text_encoder=configs.freeze_text_encoder,
        proj_configs=configs.proj,
        how_to_get_word_embedding=configs.how_to_get_word_embedding,
        fuse_subtoken_feats=configs.fuse_subtoken_feats,
        fuse_multiple_alignment=configs.fuse_multiple_alignment
    )


class InputPenmanGraph_WordEmbeddingEncoder_OutputGraph_addSelfEncodeInside(nn.Module):
    def __init__(self, 
                 d_model,
                 how_to_get_tokk_embed_configs,
                 from_scratch=False, # 是否从新开始初始化
                 freeze_vocabulary=False,
                 graph_layer_configs=None,
                 ) -> None:
        super().__init__()
        from transformers import BartModel
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        amr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.d_model = d_model
        self.need_proj = False
        if from_scratch:
            assert False
            self.word_embedding = nn.Embedding(amr_tokenizer.vocab_size, d_model, amr_tokenizer.pad_token_id,)
        else:
            amr2text_model = BartModel.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            pretrained_dim = amr2text_model.config.d_model
            self.word_embedding = amr2text_model.shared
            if freeze_vocabulary:
                for p in self.word_embedding.parameters():
                    p.requires_grad_(False) 
            if pretrained_dim != d_model:
                self.need_proj=True
                self.text_proj = FeatureResizer(input_feat_size=pretrained_dim,
                                                output_feat_size=d_model,
                                                dropout=0.,
                                                do_ln=True)
        
        if how_to_get_tokk_embed_configs.name == 1:
       
            self.fuse_subtoken = how_to_get_tokk_embed_configs.fuse_subtoken
        else:
            raise ValueError()
        self.how_to_get_tokk_embed = how_to_get_tokk_embed_configs.name
        

        from .layers_graph import graph_layer_entrypoints
        from .transformer import _get_clones
        create_graph_layer  = graph_layer_entrypoints(graph_layer_configs.name)
        graph_layer = create_graph_layer(graph_layer_configs, pretrained_dim)
        self.text_graph_layers = _get_clones(graph_layer, graph_layer_configs.nlayers)

  
    def text_self_graph_encoder(self, all_feats, all_seg_ids, graphs):
        device = all_feats.device
        edge_index = graphs.edge_index.to(device)
        num_nodes = [(seg_ids>0).int().sum().item() for seg_ids in all_seg_ids]
        num_edges = [(seg_ids<0).int().sum().item() for seg_ids in all_seg_ids]
        nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(all_feats, all_seg_ids)], dim=0)
        edge_feats = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(all_feats, all_seg_ids)], dim=0)
        
        for graph_layer in self.text_graph_layers:
            nodes_feats, edge_feats = graph_layer(nodes_feats, edge_index, edge_feats)
            
        assert sum(num_nodes) == len(nodes_feats)
        assert sum(num_edges) == len(edge_feats)
        batch_node_feats = torch.split(nodes_feats, num_nodes)
        batch_edge_feats = torch.split(edge_feats, num_edges)
        for batch_idx, seg_ids in enumerate(all_seg_ids):
            all_feats[batch_idx, seg_ids > 0] = batch_node_feats[batch_idx]
            all_feats[batch_idx, seg_ids < 0] = batch_edge_feats[batch_idx] 
               
        return all_feats    
     

    def get_words_embeds(self, words, device):
        """
        words: list[word_token_id]
        """
        words = torch.tensor(words, device=device).long()
        token_feats = self.word_embedding(words) # k c
        if self.need_proj:
            return self.text_proj(token_feats)
        return token_feats
               
    def forward(self, graph, text, device):
        graphs = graph['graphs'] # list[T(2 E_i)]
        seg_ids = graph['seg_ids'].to(device)  # b (V+E)max
        token_splits = graph['token_splits'] # list[list[int]]
        tokens_ids = graph['tokens_ids'].to(device) # b max 
        
        token_feats = self.word_embedding(tokens_ids) # b max c
        
        # list[list[ti c]] -> list[Vi+Ei c]
        token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(token_feats, token_splits)]
        for batch_idx in range(len(token_feats)):
            token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in token_feats[batch_idx]], dim=0)
            
        token_feats = pad_1d_feats(token_feats)[0] # b (V+E)max c
        assert token_feats.shape[1] == seg_ids.shape[1]
        
        token_feats = self.text_self_graph_encoder(token_feats, seg_ids, graphs)

        if self.need_proj:
            token_feats = self.text_proj(token_feats)
        
        return {
            'all_feats': token_feats,
            'all_seg_ids': seg_ids,
            'graph': graphs
        }
  
@register_text_encoder
def inputpenmangraph_wordembeddingencoder_outputgraph_addSelfGraphInside(configs, d_model):
    return InputPenmanGraph_WordEmbeddingEncoder_OutputGraph_addSelfEncodeInside(
        d_model=d_model,
        how_to_get_tokk_embed_configs=configs.how_to_get_tokk_embed,
        from_scratch=configs.from_scratch,
        freeze_vocabulary=configs.freeze_vocabulary,
        graph_layer_configs=configs.graph_layer if hasattr(configs, 'graph_layer') else None
    )



"""
text已经被parse成一个amr graph
使用tokenizer把AMR graph变成geometric.graph的形式 (tokenizer_configs)

encoder是多层相同的graph_self_attention layer (layer_configs)
"""
class Graph_SelfEncoder(nn.Module):
    def __init__(self, 
                 d_model,
                 tokenizer_configs,
                 num_layers,
                 layer_configs,) -> None:
        """
        """
        super().__init__()
        
        create_amr_tokenizer = amr_tokenizer_entrypoints(tokenizer_configs.name)
        self.tokenizer = create_amr_tokenizer(tokenizer_configs, d_model)

        create_graph_layer  = graph_layer_entrypoints(layer_configs.name)
        graph_layer = create_graph_layer(layer_configs, d_model)
        self.encoder = _get_clones(graph_layer, N=num_layers)
    
    def forward(self, graph, text, device):
        """A stack of self-attention
        graph: list[penman]
        """
        # data.Batch
        batch_graph :Data = self.tokenizer(graph['amr_tree_string'], text, device)
        
        for layer in self.encoder:
            x = layer(x=batch_graph.x,
                    edge_index=batch_graph.edge_index,
                    edge_attr=batch_graph.edge_attr
                    )
            batch_graph.update({'x': x})
        
        list_graph = batch_graph.to_data_list()
        
        return {'graphs':list_graph}
   
@register_text_encoder
def graph_encoder(configs, d_model):
    return GCN_graph2graph(
        d_model=d_model,
        tokenizer_configs=configs.tokenizer,
        num_layers=configs.num_layers,
        layer_configs=configs.layer
    )



# 改masking的策略
# 没有了 256->768 projections
class TextEncoderDecoder_MLM(nn.Module):
    def __init__(self, backbone_configs, proj_configs, d_model,
                 fused_scale,
                 task_conditioning_form) -> None:
        super().__init__()
        self.build_text_backbone(backbone_configs, d_model=d_model)
        self.build_proj(proj_configs, d_model)
        self.fused_scale=fused_scale

        self.task_conditioning_form = task_conditioning_form

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
    
    def build_text_backbone(self, configs, d_model):
        configs = vars(configs)
        if configs['name'] == 'pretrain_roberta_base_with_decoder':            
            self.tokenizer = RobertaTokenizerFast.from_pretrained('/home/xhh/pt/roberta-base')
            use_pretrained = configs['pretrained']
            freeze_encoder = configs['freeze_encoder']
            assert ((not use_pretrained) and (not freeze_encoder)) or (use_pretrained)

            config = AutoConfig.from_pretrained('/home/xhh/pt/roberta-base')
            config.add_cross_attention = True
            config.is_encoder_decoder = True
            config.tie_encoder_decoder = True           
            if configs['pretrained']:
                text_backbone = RobertaForMaskedLM.from_pretrained('/home/xhh/pt/roberta-base', config=config)
            else:
                text_backbone = RobertaForMaskedLM(config=config)

            if configs['freeze_encoder']:
                for p in text_backbone.parameters():
                    p.requires_grad_(False)
            self.text_backbone_vocab = text_backbone.embeddings
            self.text_backbone_encoder = text_backbone.encoder
            self.text_backbone_pooler = text_backbone.pooler
            self.text_backbone_mlmhead = text_backbone.lm_head
            self.hidden_size = text_backbone.config.hidden_size
            self.mask_probability = configs['mask_probability']  # 0.3
            self.mask_onlymask = configs['mask_onlymask']   # False
        else:
            raise ValueError()
    
    def forward(self, texts, text_auxiliary, device, 
                # used as decoder when doing mlm tasks
                # if not mlm, just extract the fatures of the sentence
                mlm=False, tokenized_feats_requires_grad=False,
                masked_indices=None, get_masking_info=None,
                mlm_video_args=None):
        
        tokens = self.tokenizer.batch_encode_plus(texts, padding="longest", return_tensors="pt").to(device)
        # b max
        token_padding_mask = tokens['attention_mask'].ne(1).bool() # b max, 1是padding的位置, 0是没有padding的位置
        tokenized_result = []
        for batch_idx in range(len(token_padding_mask)):
            pad = token_padding_mask[batch_idx, :] # max
            tk_ids = tokens['input_ids'][batch_idx] # max
            tks = np.array(self.tokenizer.convert_ids_to_tokens(tk_ids))[~pad.cpu()].tolist()
            tokenized_result.append(tks)

        if mlm:
            assert mlm_video_args is not None
            # masking 
            tokens['input_ids'], token_gt_labels, token_masked_bool = self._torch_mask_tokens(inputs=tokens['input_ids'],
                                                                                              masked_indices=masked_indices,
                                                                                              get_masking_info=get_masking_info)
            masked_sentences = []
            for batch_idx in range(len(token_padding_mask)):
                pad = token_padding_mask[batch_idx, :] # max
                tk_ids = tokens['input_ids'][batch_idx] # max
                tks = np.array(self.tokenizer.convert_ids_to_tokens(tk_ids))[~pad.cpu()].tolist()
                masked_sentences.append(' '.join(tks))

            tokenized_feats: torch.Tensor = self.text_backbone_vocab(input_ids=tokens['input_ids']) # b max d

            # multi-scale features
            multiscales, multiscales_pad_masks, multiscales_poses, descs = mlm_video_args
            idx = find_scale_from_multiscales(descs, self.fused_scale)

            self_attn_mask = token_padding_mask.float()[:, None, None, :] # b 1 1 max
            self_attn_mask = self_attn_mask * torch.finfo(tokenized_feats.dtype).min
            encoder_outputs = self.text_backbone_encoder(
                tokenized_feats,
                attention_mask=self_attn_mask,
                encoder_hidden_stats= rearrange(multiscales[idx], 'b t c h w -> b (t h w) c' ) +\
                      rearrange(multiscales_poses[idx], 'b t c h w -> b (t h w) c'),
                encoder_attention_mask = rearrange(multiscales_pad_masks[idx].float(), 'b t h w -> b (t h w)') *\
                      torch.finfo(tokenized_feats.dtype).min 
            )
            sequence_output = encoder_outputs[0]  # b max c

            mlm_pred, loss_dict = self.forward_mlm(sequence_output, (token_gt_labels, token_masked_bool))
            return  tokenized_result, token_padding_mask, mlm_pred, loss_dict

        else:
            # just extract the features
            assert mlm_video_args is None and masked_indices is None and get_masking_info is None

            # b max c, 最开始toknized后的特征
            tokenized_feats: torch.Tensor = self.text_backbone_vocab(input_ids=tokens['input_ids']) # b max d
            # tokenized_feats.requires_grad_(True)  # 虽然text backbone是没有在训练的, 但是中间的text特征会有梯度, 所以可以得到
            if tokenized_feats_requires_grad:
                tokenized_feats.requires_grad_(True)
        
            self_attn_mask = token_padding_mask.float()[:, None, None, :] # b 1 1 max
            self_attn_mask = self_attn_mask * torch.finfo(tokenized_feats.dtype).min
            encoder_outputs = self.text_backbone_encoder(
                tokenized_feats,
                attention_mask=self_attn_mask
            )
            sequence_output = encoder_outputs[0]  # b max c
            # b c
            pooled_output = self.text_backbone_pooler(sequence_output) if self.text_backbone_pooler is not None else None
            
            text_features = self.text_proj(sequence_output) 
            text_sentence_features = self.text_proj(pooled_output) 
            return (tokenized_result,                   tokenized_feats), (text_features, token_padding_mask, text_sentence_features), None

    def forward_mlm(self, text_feats, masked_gt): 
        """
        text_feats: b s c
        token_gt_labels: b s; token_masked_bool: b s; tokens_gt_ids: b s
        """  
        token_gt_labels, token_masked_bool = masked_gt
        
        logits = self.text_backbone_mlmhead(text_feats).permute(0, 2, 1)  # b vocab max
        mlm_loss = F.cross_entropy(logits, token_gt_labels, ignore_index=-100)

        out = []
        for batch_idx in range(len(logits)):
            sentence_out = []
            masked_idxs = token_masked_bool[batch_idx].nonzero(as_tuple=True)[0] 
            for idx in masked_idxs:
                scores, ids = torch.topk(logits[batch_idx, :, idx].detach().softmax(-1), k=30, largest=True, sorted=True) # 30
                tokens = self.tokenizer.convert_ids_to_tokens(ids) # 30
                sentence_out.append(list(zip(tokens, scores)))
            out.append(sentence_out)
        # 由于有些可能是非常low-level或者非常high-level的, 所以只最小化最小的loss
        return out, {"loss_mlm": mlm_loss}
   
@register_text_encoder
def text_encoderdecoder_mlm(configs, d_model):
    return TextEncoderDecoder_MLM(
        backbone_configs=configs.text_backbone,
        proj_configs=configs.proj,
        fused_scale=configs.fused_scale,
        task_conditioning_form=configs.task_conditioning_form,
        d_model=d_model,)
 

class InputPenmanGraph_WordEmbeddingEncoder_OutputGraph_v0(nn.Module):
    def __init__(self, 
                 d_model,
                 how_to_get_tokk_embed_configs,
                 from_scratch=False, # 是否从新开始初始化
                 freeze_vocabulary=False,
                 ) -> None:
        super().__init__()
        from transformers import BartModel
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.amr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.d_model = d_model
        self.need_proj = False
        if from_scratch:
            self.word_embedding = nn.Embedding(self.amr_tokenizer.vocab_size, d_model, self.amr_tokenizer.pad_token_id)
        else:
            amr2text_model = BartModel.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
            pretrained_dim = amr2text_model.config.d_model
            self.word_embedding = amr2text_model.shared
            if freeze_vocabulary:
                for p in self.word_embedding.parameters():
                    p.requires_grad_(False) 
            if pretrained_dim != d_model:
                self.need_proj=True
                self.text_proj = FeatureResizer(input_feat_size=pretrained_dim,
                                                output_feat_size=d_model,
                                                dropout=0.,
                                                do_ln=True)
        
        if how_to_get_tokk_embed_configs.name == 1:
            self.is_a_embedding = nn.Embedding(1, d_model)
            self.fuse_subtoken = how_to_get_tokk_embed_configs.fuse_subtoken
        else:
            raise ValueError()
        self.how_to_get_tokk_embed = how_to_get_tokk_embed_configs.name

    def get_tokk_embed(self, tokk, type, device):
        # tokk: string
        # type: 
        if self.how_to_get_tokk_embed == 1:
            if tokk == ':instance' and type == 'is_a':
                return self.is_a_embedding.weight.squeeze(0) # c
            
            alignment_pattern = re.compile(r'~(?:[a-z]\.?)?(?P<ali>[0-9]+)(?:,[0-9]+)*')
            tokk = re.sub(alignment_pattern, repl='', string=tokk)                
            is_in_enc = (self.amr_tokenizer.INIT + tokk) in self.amr_tokenizer.encoder   # Ġ<pointer:0>
            is_rel = tokk.startswith(':') and len(tokk) > 1 
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):                 # dealing with examples like "The_United_Kingdom_of_xxx"
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.amr_tokenizer.INIT + "<lit>"]
                bpe_toks += self.amr_tokenizer._tok_bpe(tokk)
                bpe_toks.append(self.amr_tokenizer.INIT + "</lit>")

            elif (is_rel or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.amr_tokenizer.INIT + tokk]
                elif is_frame: # 没有在vocabulary中的frame
                    bpe_toks = self.amr_tokenizer._tok_bpe(tokk[:-3]) + [tokk[-3:]]
                elif is_of: # 是of
                    rel = tokk[:-3]
                    if self.amr_tokenizer.INIT + rel in self.amr_tokenizer.encoder:
                        bpe_toks = [self.amr_tokenizer.INIT + rel, '-of'] # 是vocabulary中的role-of
                    else: # 不是vocabulary中的role-of
                        bpe_toks = [self.amr_tokenizer.INIT + ':'] + self.amr_tokenizer._tok_bpe(rel[1:]) + ['-of']
                elif is_rel: # 没在vocabulary中的role
                    bpe_toks = [self.amr_tokenizer.INIT + ':'] + self.amr_tokenizer._tok_bpe(tokk[1:])
                else:
                    print("tok:", tokk)
                    print(f"is_rel:{is_rel}, is_frame:{is_frame}, is_of:{is_of}")
                    raise ValueError()
            else: # free-form English
                if is_in_enc:
                    bpe_toks = [self.amr_tokenizer.INIT + tokk]
                else:
                    bpe_toks = self.amr_tokenizer._tok_bpe(tokk) 

            bpe_token_ids = torch.tensor([self.amr_tokenizer.encoder.get(b, self.amr_tokenizer.unk_token_id) for b in bpe_toks], device=device)
            concept_embed = self.word_embedding(bpe_token_ids) # ni c

            if self.fuse_subtoken == 'mean':
                concept_embed =  concept_embed.mean(dim=0)
            elif self.fuse_subtoken == 'sum':
                concept_embed = concept_embed.sum(dim=0)
            elif self.fuse_subtoken == 'dot_product':
                init = concept_embed[0]
                for i in range(1, len(concept_embed)):
                    init = init * concept_embed[i]
                concept_embed = init
            else:
                raise ValueError()
            
            if self.need_proj:
                return self.text_proj(concept_embed)
            else:
                return concept_embed
        else:
            raise ValueError()
         
            
    def forward(self, graph, text, device):
        graphs = []
        node_identifiers = []
        edge_identifiers = []
        
        role_links_by_batch = []
        for amr_dict in graph['amr_tree_string_linearization_dict']:
            role_links_by_batch.append(amr_dict['edge_linearized_ali'])
        
        for batch_idx, (role_links, amr_tree_string) in enumerate(zip(role_links_by_batch, graph['amr_tree_string'])):
            amr_graph = penman.decode(amr_tree_string) # graph
            
            node_list = [] # list[var/concept_value/constant]
            node_id_map = {} # node:id
            node_feats = [] # list[c]
            node_identifier = [] # 0:variable, 1:concept_val, 2: constant
            connectivity = [] # list[2]
            edge_feats = [] # list[c], num_edge
            edge_identifier = [] # 0: regular, 1: /, 2: constant edge

            node_list += amr_graph.variables() # m / v / m2
            if node_list[0] != amr_graph._top:
                # swap the first one
                top_var_idx = [idx for idx in range(len(node_list)) if node_list[idx] == amr_graph._top]
                assert len(top_var_idx) == 1
                top_var_idx = top_var_idx[0]
                swap_value = node_list[0]
                node_list[0] = amr_graph._top
                node_list[top_var_idx] = swap_value
            
            node_id_map = {node:id for id, node in enumerate(node_list)}
            node_feats.append(torch.zeros([len(node_list), self.d_model], dtype=torch.float, device=device))
            node_identifier = [0] * len(node_list)
            
            for instance in amr_graph.instances():
                variable, role, concept = instance.source, instance.role, instance.target
                assert variable in node_list and role == ':instance'
                if concept is None:
                    # coreference
                    continue
                if concept in node_list:  # 比如一个graph里有m / m2 都是man, 那么man只有一个node, m/m2都连接到他那里
                    pass 
                else:
                    # 得到concept的embedding
                    node_id_map[concept] = len(node_list)
                    node_list.append(concept)
                    node_identifier.append(1)
                    node_feats.append(self.get_tokk_embed(concept, type='concept', device=device).unsqueeze(0))
                    
                connectivity.append([node_id_map[concept], node_id_map[variable]])
                edge_feats.append(self.get_tokk_embed(':instance', type='is_a', device=device))
                edge_identifier.append(1)
                             
            for attr in amr_graph.attributes():
                variable, role, constant = attr.source, attr.role, attr.target
                assert variable in node_list
                if constant in node_list:
                    pass
                else:
                    # 得到constant的embedding
                    node_id_map[constant] = len(node_list)
                    node_list.append(constant)
                    node_feats.append(self.get_tokk_embed(constant, type='constant', device=device).unsqueeze(0))
                    node_identifier.append(2)
                connectivity.append([node_id_map[constant], node_id_map[variable]])
                if '-of' in role:
                    role = role[:-3]
                else:
                    role = f'{role}-of'
                edge_feats.append(self.get_tokk_embed(role, type='role', device=device))
                edge_identifier.append(2)
                     
            for edge in amr_graph.edges():
                var1, role, var2 = edge.source, edge.role,edge.target 
                assert var1 in node_list and var2 in node_list
                key = f'{var1}, {role}, {var2}'
                if key not in role_links:
                    # tree上是var2 -role-of-> var1
                    key = f'{var2}, {role[:-3]}, {var1}' if '-of' in role else f'{var2}, {role}-of, {var1}'
                    assert key in role_links
                    # 加上 var2 <--role-- var1
                    connectivity.append([node_id_map[var1], node_id_map[var2]])
                    edge_feats.append(self.get_tokk_embed(role, type='role', device=device))
                else:
                    # tree上是var1--role-->var2
                    # 加上 var1 <--role-of-- var2
                    connectivity.append([node_id_map[var2], node_id_map[var1]]) # 变成var2->var1, 向上走
                    role_of = role[:-3] if '-of' in role else f'{role}-of'
                    edge_feats.append(self.get_tokk_embed(role_of, type='role', device=device))
                edge_identifier.append(0)

            node_feats = torch.cat(node_feats, dim=0) # num_nodes c
            assert len(node_feats) == len(node_list)
            edge_feats = torch.stack(edge_feats, dim=0) # num_edges c
            assert len(edge_feats) == len(connectivity)
            connectivity = torch.tensor(connectivity, device=device, dtype=torch.long).transpose(0, 1) # 2 num_edges
            
            graphs.append(Data(
                    x = node_feats,
                    edge_index = connectivity,
                    edge_attr=edge_feats
            ))
            node_identifiers.append(torch.tensor(node_identifier,dtype=torch.long, device=device))
            edge_identifiers.append(torch.tensor(edge_identifier,dtype=torch.long, device=device))

        return {'graphs': graphs,
                'node_identifiers':node_identifiers,
                'edge_identifiers': edge_identifiers}


"""

使用预训练的text2amr seq2seq模型中的encoder+decoder 得到amr的encoding
"""
class BartText2AMR_Seq2Seq_GraphEncoder(nn.Module):
    def __init__(self, 
                 d_model,
                 freeze_text_encoder,
                 proj_configs,
                 # add_top_positional_embedding=False,
                 ) -> None:
        super().__init__()
        from transformers import BartForConditionalGeneration
        from .amr_utils.tokenization_bart import AMRBartTokenizer
        self.encoder_decoder = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/amr_parse')
        if freeze_text_encoder:
            for p in self.encoder_decoder.parameters():
                p.requires_grad_(False) 
        self.hidden_size = self.encoder_decoder.config.d_model
        self.build_proj(proj_configs, d_model)
        # vocab.json + additional.json (free-form English words + roles + predicates)
        # 1. 所有的variable name换成<pointer:x>/<Rx>; (因为一个variable name很可能被一个nlp tokenizer变成多个tokens)
        # 2. variable/concept 被换成 <pointer:x> concept
        # 3. 括号不变, 为了表示深度
        self.tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
        # if add_top_positional_embedding:
        #     self.top_embedding = nn.Embedding(1, d_model)
        # self.add_top_positional_embedding = add_top_positional_embedding

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
     

    def forward(self, graph, text, device):
        """
        graph: list[a linearized amr graph]
        text: 
        """
        batch_size = len(text)
        
        tokens = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
        token_padding_mask = tokens['attention_mask'].ne(1).bool()
        self.encoder_decoder.generate()
        encoder_outputs = self.encoder_decoder.forward(
            input_ids=tokens['input_ids'],
            attention_mask=token_padding_mask
        )
        sequence_output = encoder_outputs[0]  # b max c
        return {
            'token_feats': sequence_output,
            'token_pad_masks': torch.zeros([batch_size, sequence]),
            'token_sentence_feats': None,
            }


class TemplateDistribution_WordsSoftDistribution(nn.Module):
    def __init__(self, dim, maximum_time_steps,
                 num_templates,) -> None:
        super().__init__()
        self.maxmum_T = maximum_time_steps
        self.W1 = _get_clones(nn.Linear(dim, dim), maximum_time_steps)
        self.W2 = nn.Linear(2*dim, dim)
        self.template_distribution_mlp = MLP(dim, dim, num_templates,  num_layers=3)
        self.words_soft_attention = nn.Linear(dim, 1)

    def forward(self, text_feats, text_sentence_feats, textual_query, time_step,):
        '''For each time_step, using the corresponding time-dependent transformations to get a distribution over
        all templates and soft attention over the words
        Inputs:
            text_feats: b s c
            text_sentence_feats: b c
            textual_query: b c
            time_step: b
        Outputs:
            template_distribution: b #templates
            words_soft_distribution: b s
            textual query: b c
        '''
        x = torch.stack([self.W1[time_lay_idx](feat) \
                                for (time_lay_idx, feat) in zip(time_step, text_sentence_feats)], dim=0) # b c
        x = torch.cat([x, textual_query], dim=-1) # b 2c

        x = self.W2(x) # b c

        #
        template_distribution = F.softmax(self.template_distribution_mlp(x), dim=-1) # b #template
        #
        x_w = x.unsqueeze(1) * text_feats # b s c
        words_soft_distribution = F.softmax(self.words_soft_attention(x_w).squeeze(-1), dim=-1) # b s
        #
        c_t = (words_soft_distribution.unsqueeze(-1) * text_feats).sum(1) # b s c -> b c

        return template_distribution, words_soft_distribution, c_t

# from penman.models.amr import model
# import penman
# import re
# from transition_amr_parser.parse import AMRParser
# # Download and save a model named AMR3.0 to cache


# _amr_parser_entrypoints = {}
# def register_amr_parser(fn):
#     amr_parser_name = fn.__name__
#     _amr_parser_entrypoints[amr_parser_name] = fn

#     return fn
# def amr_parser_entrypoints(amr_parser_name):
#     try:
#         return _amr_parser_entrypoints[amr_parser_name]
#     except KeyError as e:
#         print(f'amr_parser {amr_parser_name} not found')
        
# class AMR_off_the_shelf(nn.Module):
#     def __init__(self, ) -> None:
#         super().__init__()
#         self.parser = AMRParser.from_pretrained('AMR3-structbart-L')
        
        
#     def forward(self, text_query):
        
#         with torch.no_grad():
#             tokens, positions = self.parser.tokenize('The girl travels and visits places')

#             # Use parse_sentence() for single sentences or parse_sentences() for a batch
#             annotations, machines = self.parser.parse_sentence(tokens)
#             amr = machines.get_amr()
#             tree = amr.to_penman(jamr=False, isi=True)

#         return tree

#     def transform_rott(self, tree):
#         return penman.transform(tree)


# class AMRTokens:
#     START, END = '<', '>'
#     _TEMPL = START + '{}' + END
#     BOS_N   = _TEMPL.format('s')  #<s>
#     EOS_N   = _TEMPL.format('/s')  #</s>
#     START_N = _TEMPL.format('start') #<start>
#     STOP_N  = _TEMPL.format('stop') #<stop>
    
#     BOS_E   = _TEMPL.format('s')
#     EOS_E   = _TEMPL.format('/s')
#     START_E = _TEMPL.format('start')
#     STOP_E  = _TEMPL.format('stop')
    
#     PNTR_N  = _TEMPL.format('pointer') #<pointer>
#     BACKR_SRC_N = _TEMPL.format('backr:src:XXX')  #<backr:src:XXX>
#     BACKR_TRG_N = _TEMPL.format('backr:trg:XXX')  #<backr:trg:XXX>
    
#     LIT_START = _TEMPL.format( 'lit') #<lit>
#     LIT_END   = _TEMPL.format('/lit')  #</lit>
    
#     _FIXED_SPECIAL_TOKENS_N = {
#         BOS_N, EOS_N, START_N, STOP_N}
#     _FIXED_SPECIAL_TOKENS_E = {
#         BOS_E, EOS_E, START_E, STOP_E}
    
#     _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E #&交集 |并集 -差集 ^对称差集
    
#     # match and read backreferences
#     _re_BACKR_SRC_N = re.compile(BACKR_SRC_N.replace('XXX', r'([0-9]+)'))  # an re object
#     _re_BACKR_TRG_N = re.compile(BACKR_TRG_N.replace('XXX', r'([0-9]+)'))  # an re object
    

#     @classmethod
#     def is_node(cls, string: str) -> bool:
#         if isinstance(string, str) and string.startswith(':'):
#             return False  # role :arg
#         elif string in cls._FIXED_SPECIAL_TOKENS_E:
#             return False  # 特殊token  <>
#         return True  

#     @classmethod
#     def read_backr(cls, string: str):
#         m_src = cls._re_BACKR_SRC_N.search(string)
#         if m_src is not None:
#             return m_src
#         m_trg = cls._re_BACKR_TRG_N.search(string)
#         if m_trg is not None:
#             return m_trg
#         return None
  
# from transformers import BartTokenizer
# # sentence -> word_tokens; graph -> amr_tokens
# class AMRBartTokenizer(BartTokenizer):
#     INIT = 'Ġ'
    
#     ADDITIONAL = [
#         AMRTokens.PNTR_N,  # <pointer>
#         AMRTokens.STOP_N,  #<stop>
#         AMRTokens.LIT_START, # <lit>
#         AMRTokens.LIT_END,  #</lit>
#         AMRTokens.BACKR_SRC_N, # <backr:src:XXX>
#         AMRTokens.BACKR_TRG_N,] # backr:src:XXX    

#     def _tokenize(self, text):
#         """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
#         bpe_tokens = []
#         for tok_span in text.lstrip().split(' '):  # 一句话取消前面的white space, 然后按空格分成单词
#             tok_span = tok_span.strip()  #单词的前后的white space都去掉
#             recats = tok_span.rsplit('_', 1)
#             if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
#                 bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
#             else:
#                 for token in re.findall(self.pat, ' ' + tok_span):
#                     token = "".join(
#                         self.byte_encoder[b] for b in token.encode("utf-8")
#                     )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
#                     bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

#         return bpe_tokens
# @register_amr_parser
# def off_the_shelf_amr_parser(self, configs):
#     return AMR_off_the_shelf()

# # a text-amr parser
# class BottomUp_AMR_Parser(nn.Module):
#     INIT = 'Ġ'
    
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
        
        
    
#     def forward(self, text_query):
#         pass
    
#     def forward_loss(self, pred_amr, gt_scene_graph):
        
#         pass


        
        
    