import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_unimodal_attention import CrossSelfFFN_Module
from einops import rearrange, reduce, repeat

class Video_Semantic_Role_Grounding(nn.Module):
    def __init__(self, 
                 event_backbone,
                 object_detector,
                 self_encoder,
                 caption_gpt,
                 hidden_dim,
                 
                 ) -> None:
        super().__init__()
        self.event_backbone = event_backbone
        self.obj_detector = object_detector
        self.self_encoder = self_encoder
        
        # 在这个clip中, event是什么？
        self.event_classifier = nn.Linear(hidden_dim, 1560)
        
        # 在这个clip中, event有哪些roles?
        self.role_classifier = nn.Linear(hidden_dim, 9)
        
        self.role_embeddings = nn.Embedding(hidden_dim, 9)
        self.role_decoder = CrossSelfFFN_Module()
        self.num_roles = 9
        
        self.caption_gpt = caption_gpt
    
    def forward(self, video):
        """
        video: b 10 3 h w
        """
        batch_size, nf, *_ = video.shape
        ####### 每个clip提取成一个event vector
        # b 5 c, 每个clip为2f, 但是每个clip的feature是3帧的函数
        # 每个clip的event feature
        event_feats = self.event_backbone(video)
        nf = event_feats.shape[1]
        event_feats = rearrange(event_feats, 'b t c -> t b c')
        
        ####### 提取出每帧的objects
        # (b n t_sampled c, b n t_sampled 4)
        obj_feats = self.obj_detector(video)
        obj_feats = rearrange(obj_feats, 'b n t_sampled c -> (t_sampled n) b c')
        
        ####### 让每个clip的event vector看到整个视频的所有Objects
        # [5 + (t_sampled n)] b c
        memory = torch.cat([event_feats, obj_feats])
        memory = self.self_encoder(memory)
        
        event_feats = rearrange(memory[:event_feats.shape[0]], 't b c -> b t c')
        obj_feats = rearrange(memory[event_feats.shape[0]:], '(t_sampled n) b c -> b t_sampled n c') 
        
        ###### 询问模型: 每个clip都是什么event呀?
        event_logits = self.event_classifier(event_feats) # b t c -> b t 1560
        event_probs = F.softmax(event_logits, -1)  # ---> gt_event
        
        ###### 询问模型: 每个clip都有什么role?
        role_logits = self.role_classifier(event_feats) # b t c -> b t 9
        role_probs = F.sigmoid(role_logits)       # ----> gt_role
        pred_role_indices = role_probs > 0.5 #[arg0, arg1, ...]
        
        # b 5 9 c
        output = repeat(self.role_embeddings.weight, 'r c -> b t r c',t=nf,b=batch_size)
        ###### 告诉每个role 他们的verb sense
        event_feats = repeat(event_feats, 'b t c -> b t r c',r=self.num_roles)
        output = output + event_feats
        
        ###### 去寻找每个role吧！
        output = rearrange(output, 'b t r c -> (t r) b c')
        obj_feats = rearrange(obj_feats, 'b t_sampled n c -> (t_sampled n) b c')
        # (t r) all_objects
        output, cross_attention_scores = self.role_decoder(output,
                                   obj_feats)
        
        
        ###### 每个role用一句话描述一下吧
        output = rearrange(output, '(t r) b c -> b t r c') # b t r c
        captions = self.caption_gpt(output)  # list[tokens] b t r   ------> cross entropy loss
        
        ##### 每个role的bounding box是啥?
        obj_idx = torch.argmax(cross_attention_scores, dim=-1) # (t r)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        