class Video_MaskedAttn_MultiscaleMaskDecoder(nn.Module):
    def __init__(self,
                 configs,
                 multiscale_shapes):
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.frame_nqueries = configs['frame_nqueries'] # 20
        self.video_nqueries = configs['video_nqueries'] # 10
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
        self.frame_query_poses =  nn.Embedding(self.frame_nqueries, d_model)
        self.frame_query_feats = nn.Embedding(self.frame_nqueries, d_model)
        self.temporal_query_poses = nn.Embedding(self.video_nqueries, d_model)
        self.temporal_query_feats = nn.Embedding(self.video_nqueries, d_model)

        self.level_embeds = nn.Embedding(len(self.memory_scales), d_model)
        assert self.nlayers % len(self.memory_scales) == 0

        self.frame_cross_layers = _get_clones(CrossAttentionLayer(d_model=d_model,
                                                            nhead=attn_configs['nheads'],
                                                            dropout=0.0,
                                                            normalize_before=attn_configs['normalize_before']),
                                              self.nlayers)
        
        temporal_self_layer = META_ARCH_REGISTRY.get(configs['temporal_self_layer']['name'])(configs['temporal_self_layer'])
        self.temporal_self_layers = _get_clones(temporal_self_layer, self.nlayers)

        temporal_cross_layer = META_ARCH_REGISTRY.get(configs['temporal_cross_layer']['name'])(configs['temporal_cross_layer'])
        self.temporal_cross_layers = _get_clones(temporal_cross_layer, self.nlayers) 
                  
        self.nheads = attn_configs['nheads']
        self.frame_query_norm = nn.LayerNorm(d_model)
        self.temporal_query_norm = nn.LayerNorm(d_model)
        self.pos_3d = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d') # b t c h w
        
        self.head_outputs = configs['head_outputs']
        assert 'mask' in self.head_outputs
        self.query_mask = MLP(d_model, d_model, d_model, 3)
        if 'class' in self.head_outputs:
            self.query_class = nn.Linear(d_model, num_classes + 1)    

        self.vid_loss_module = Video_SetMatchingLoss(loss_config=configs['temporal_loss'], num_classes=num_classes)

    @property
    def device(self,):
        return self.frame_query_poses.weight.device

    def get_memories_and_mask_features(self, multiscales):
        # b c t h w 
        memories = [multiscales[scale] for scale in self.memory_scales]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories_poses = [self.pos_3d(mem.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) for mem in memories]
        memories = [rearrange(mem, 'b c t h w -> (h w) (b t) c').contiguous() for mem in memories]
        memories_poses = [rearrange(mem_pos, 'b c t h w -> (h w) (b t) c').contiguous() for mem_pos in memories_poses]
        mask_features = multiscales[self.mask_scale] # b c t h w

        return memories, memories_poses, mask_features, size_list

    def forward(self, 
                multiscales, # b c t h w
                ):
        multiscales = self.inputs_projs(multiscales)
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]

        # hw bt c; bt c h w
        memories, memories_poses, mask_features, size_list = self.get_memories_and_mask_features(multiscales)
        memories = [mem_feat + self.level_embeds.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]

        # nqf bt c
        frame_query_poses = self.frame_query_poses.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)
        frame_query_feats = self.frame_query_feats.weight.unsqueeze(1).repeat(1, batch_size * nf, 1)

        # nq b c
        temporal_query_poses = self.temporal_query_poses.weight.unsqueeze(1).repeat(1, batch_size, 1)
        temporal_query_feats = self.temporal_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)


        frame_ret = []
        vid_ret = []
        # bt nqf class, bt nqf 4, bt nqf h w, bt*head nqf hw
        # b nq class, b nq h w
        frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
            self.forward_heads(frame_query_feats=frame_query_feats, 
                               temporal_query_feats=temporal_query_feats,
                               mask_features=mask_features, attn_mask_target_size=size_list[0]) # first sight you re not human
        frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
        vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})

        for i in range(self.nlayers):
            level_index = i % len(self.memory_scales)
            frame_cross_attn_mask[torch.where(frame_cross_attn_mask.sum(-1) == frame_cross_attn_mask.shape[-1])] = False 

            frame_query_feats = self.frame_cross_layers[i](
                tgt=frame_query_feats, # nqf bt c
                memory=memories[level_index], # hw bt c
                memory_mask=frame_cross_attn_mask, 
                memory_key_padding_mask=None,
                pos=memories_poses[level_index], 
                query_pos=frame_query_poses,
            )
            frame_query_feats = self.temporal_self_layers[i](
                frame_query_feats=frame_query_feats, 
                frame_query_poses=frame_query_poses,
            )

            temporal_query_feats = self.temporal_cross_layers[i](
                temporal_query_feats=temporal_query_feats, 
                temporal_query_poses=temporal_query_poses,

                frame_query_feats=frame_query_feats,
                frame_query_poses=frame_query_poses,
            )

            frame_class, frame_box, frame_mask, frame_cross_attn_mask, vid_class, vid_mask = \
                self.forward_heads(frame_query_feats=frame_query_feats, 
                                    temporal_query_feats=temporal_query_feats,
                                    mask_features=mask_features, attn_mask_target_size=size_list[0])
            frame_ret.append({'pred_class':frame_class, 'pred_masks': frame_mask, 'pred_boxes': frame_box})
            vid_ret.append({'pred_class':vid_class, 'pred_masks': vid_mask})
    
        if self.training:
            return frame_ret, vid_ret
        else:
            return vid_ret

    def forward_heads(self, frame_query_feats, temporal_query_feats,  mask_features, attn_mask_target_size):
        frame_query_feats = self.frame_query_norm(frame_query_feats) # nqf bt c
        frame_query_feats = frame_query_feats.transpose(0, 1).contiguous() # bt nqf c
        frame_mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeds, mask_features.permute(0, 2,1,3,4).flatten(0,1))  #bt nq h w

        # b nq h w
        attn_mask = frame_mask_logits.detach().clone() 
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # b*head nq hw
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        

        temporal_query_feats = self.temporal_query_norm(temporal_query_feats) # nq b c
        temporal_query_feats = frame_query_feats.transpose(0, 1).contiguous() # b nq c

        class_logits = self.query_class(temporal_query_feats) if 'class' in self.head_outputs else None # b n class+1
        mask_embeds = self.query_mask(temporal_query_feats)  # b n c
        mask_logits = torch.einsum("bqc,bcthw->bqthw", mask_embeds, mask_features) 

        mask_logits = F.interpolate(mask_logits, scale_factor=self.mask_spatial_stride, mode='bilinear', align_corners=False)

        if self.training:
            return class_logits, mask_logits, attn_mask
        else:
            return class_logits.softmax(-1) if class_logits is not None else None, mask_logits, attn_mask
    

    def compute_loss(self, outputs, targets):
        assert self.training
        return self.loss_module.compute_loss(outputs, targets)
