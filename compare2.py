@META_ARCH_REGISTRY.register()
class Video_MaskedAttn_MultiscaleMaskDecoder_v3(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.video_nqueries = configs['video_nqueries']
        self.nlayers = configs['nlayers']
        self.memory_scales = configs['memory_scales']
        self.mask_scale = configs['mask_scale']
        self.mask_spatial_stride = multiscale_shapes[self.mask_scale].spatial_stride
        num_classes = configs['num_classes']

        inputs_projs = configs['inputs_projs']
        self.inputs_projs = nn.Sequential()
        if inputs_projs is not None:
            self.inputs_projs = META_ARCH_REGISTRY.get(inputs_projs['name'])(inputs_projs, 
                                                                             multiscale_shapes=multiscale_shapes,
                                                                             out_dim=d_model)
        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)
        assert self.nlayers % len(self.memory_scales) == 0
        self.cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                        self.nlayers)
        self.self_layers = _get_clones(SelfAttentionLayer(d_model=d_model,
                                                          nhead=attn_configs['nheads'],
                                                          dropout=0.0,
                                                          normalize_before=attn_configs['normalize_before']),
                                       self.nlayers)  
        self.ffn_layers = _get_clones(FFNLayer(d_model=d_model,
                                               dim_feedforward=attn_configs['dim_feedforward'],
                                               dropout=0.0,
                                               normalize_before=attn_configs['normalize_before']),
                                      self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.temporal_query_poses = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_feats = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_norm = nn.LayerNorm(d_model)
        self.pos_3d = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d') # b t c h w
        
        self.head_outputs = configs['head_outputs']
        assert 'mask' in self.head_outputs
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.head_outputs:
            self.query_class = nn.Linear(d_model, num_classes + 1)    

        self.loss_module = Video_SetMatchingLoss(loss_config=configs['loss'], num_classes=num_classes)
        
    @property
    def device(self,):
        return self.temporal_query_feats.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c t h w 
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_3d(mem.permute(0, 2, 1,3, 4)).permute(0, 2, 1, 3, 4) for mem in memories]  # b c t h w
        memories = [rearrange(mem, 'b c t h w -> (t h w) b c').contiguous() for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c t h w -> (t h w) b c').contiguous() for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c t h w
        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales, # b c t h w
                video_aux_dict=None
                ):
        multiscales = multiscales[0]
        multiscales = self.inputs_projs(multiscales)
        # thw b c; b c t h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(multiscales)
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        batch_size, _, nf, *_ = mask_features.shape
        
        # nq b c
        temporal_query_poses = self.temporal_query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporal_query_feats = self.temporal_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)

        vid_ret = []
        # b nq class, b nq t h w; b*head nq thw
        vid_class, vid_mask, attn_mask = \
            self.forward_heads(temporal_query_feats=temporal_query_feats,
                               mask_features=mask_features, attn_mask_target_size=size_list[0]) # first sight you re not human
        vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})

        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # 全masked掉的 全注意, 比如有padding

            temporal_query_feats = self.cross_layers[i](
                tgt=temporal_query_feats, # nq b c
                memory=memories[level_index],  # thw b c
                memory_mask=attn_mask,  # b*h nq thw
                memory_key_padding_mask=None,
                pos=memories_poses[level_index], # thw b c
                query_pos=temporal_query_poses, # nq b c
            )
            temporal_query_feats = self.self_layers[i](
                temporal_query_feats,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=temporal_query_poses,
            )
            temporal_query_feats = self.ffn_layers[i](
                temporal_query_feats 
            )
            # b nq class, b nq t h w
            vid_class, vid_mask, attn_mask = \
                self.forward_heads(temporal_query_feats=temporal_query_feats,
                                  mask_features=mask_features, attn_mask_target_size=size_list[(i + 1) % len(self.memory_scales)]) # first sight you re not human
            vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})
        
        return vid_ret

    def forward_heads(self, temporal_query_feats,  mask_features, attn_mask_target_size): # nq b c; b c t h w
        batch_size, _, nf, *_ = mask_features.shape

        temporal_query_feats = self.temporal_query_norm(temporal_query_feats) # nq b c
        temporal_query_feats = temporal_query_feats.transpose(0, 1).contiguous() # b nq c

        class_logits = self.query_class(temporal_query_feats) if 'class' in self.head_outputs else None # b n class+1
        mask_embeds = self.query_mask(temporal_query_feats)  # b n c
        mask_logits = torch.einsum("bqc,bcthw->bqthw", mask_embeds, mask_features) 
        batch_size, nq, nf = mask_logits.shape[:3]
        mask_logits = F.interpolate(mask_logits.flatten(0, 1), scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)
        mask_logits = rearrange(mask_logits, '(b n) t h w -> b t n h w',b=batch_size, n=nq)

        # bt nq h w
        attn_mask = mask_logits.detach().clone().flatten(0, 1)
        attn_mask = (F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) < 0.5).bool()
        attn_mask = repeat(attn_mask, '(b t) nq h w -> (b head) nq (t h w)', b=batch_size, t=nf, head=self.nheads)
        
        if self.training:
            return class_logits, mask_logits, attn_mask
        else:
            return class_logits.softmax(-1).unsqueeze(1).repeat(1, nf, 1, 1) if class_logits is not None else None, mask_logits, attn_mask
    
    def compute_loss(self, outputs, targets, video_aux_dict, **kwargs):
        assert self.training
        return self.loss_module.compute_loss(model_outs=outputs,
                                             targets=targets,
                                             video_aux_dict=video_aux_dict)