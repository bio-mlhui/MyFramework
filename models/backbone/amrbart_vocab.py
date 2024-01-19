

from typing import Any
from models.utils.amr.amrbart import BartForConditionalGeneration
import os
import torch.nn as nn

import torch
from models.layers.utils import pad_1d_feats
from models.utils.amr.amr_data import AMRData
from models.layers.anyc_trans import Linear_NormAct
from detectron2.modeling import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
class AMRBART_VocabEmbed(nn.Module):
    def __init__(self, 
                 configs,):
        super().__init__()
        model_path = configs['path']
        random_initialize = configs['random_initialize']
        freeze = configs['freeze']
        
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(os.getenv('PT_PATH'), model_path))
        self.text_backbone = AMRBart.model.shared # N d
        self.text_dim = self.text_backbone.weight.shape[-1]
        
        if random_initialize:
            assert not freeze
            for p in self.text_backbone.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        if freeze:
            for p in self.text_backbone.parameters():
                p.requires_grad_(False) 

     
    @property
    def device(self):
        return self.text_backbone.weight.device
    
    def forward(self, text_dict):
        amrs = text_dict['amrs'] # list[Graph]
        batch_size = len(amrs)
        text_tokens = text_dict['text_token_ids'] # b smax
        text_tok_splits = text_dict['text_token_splits'] # list[list[int]], batch
        text_feats = self.text_backbone(text_tokens) # b smax c
        text_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(text_feats, text_tok_splits)]
        for batch_idx in range(batch_size):
            text_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in text_feats[batch_idx]], dim=0) 
        text_feats, text_pad_masks = pad_1d_feats(text_feats)       

        amr_token_seg_ids = text_dict['seg_ids'].to(self.device)  # b (V+E)max
        amr_token_splits = text_dict['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_dict['token_ids'].to(self.device)  # b max_tok+pad
        amr_token_feats = self.text_backbone(amr_token_ids) 
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0


        return AMRData(amr=amrs, 
                       amr_seg_ids=amr_token_seg_ids, 
                       amr_feats=amr_token_feats, 
                       text_feats=text_feats, 
                       text_pad_masks=text_pad_masks)

    



