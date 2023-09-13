class AMR_v0_detectObj_whole(nn.Module):
    def __init__(self, 
                 current_task = 'vis', # vis, rvos+vis
                 d_model=256,
                 max_stride=64,
                 pt_dir='/home/xhh/pt',
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 3, 'stride':2, 'padding': 1, \
                        'bias':True,}],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32], [1,64]],

                # amrtext
                amrbart_wordEmbedding_freeze=True,
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0,
                    'do_ln':True},
                objseg_class_token_ids=None,
                fusion={
                    'name': 'VisionLanguageFusionModule',
                    'd_model':256,
                    'nheads': 8,
                    'dropout':0.},
                parsing_encoder={
                    'name':'deform_video_2d_fpn',
                    'd_ffn': 2048,
                    'dropout':0.,
                    'activation': 'relu',
                    'nheads': 8,
                    'fused_scales':[[1,8],[1,16],[1,32],[1,64]],
                    'fpn_strides': [[1,4],[1,8]],
                    'npoints':4,
                    'obj_seg_nlayer':3,
                    'ref_seg_nlayer':3
                    },

                loss_weight={'refdecoder_mask': 5,
                             'refdecoder_dice': 5,
                             'refdecoder_giou': 0,
                             'refdecoder_bbox': 0,
                 },
                tasks = {'refdecoder_refseg': {'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                                },
                },
                refdecoder={
                    'nlayers': 9,
                    'amr_cross_video_layer':{
                        'name': 'cross_attention',
                        'amr_cross': ['只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3','只有2/3',],
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'amr_self_layer':{
                        'name': 'graph_layer_v1', # 只更新node
                        'd_model': 256,
                        'flow': 'source_to_target',
                        'aggr': 'min'
                    },
                    # add ffn layer
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'choose_who': '第一个'
                    },
                    
                objdecoder={ 
                    'num_classes': 7,
                    'nqueries': 100,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    },) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks
        self.pt_dir = pt_dir
        self.max_stride = max_stride
        self.current_task = current_task
        # video encoder
        from .video_swin import VideoSwinTransformer
        self.video_swint = VideoSwinTransformer(backbone_pretrained=True,
                                                backbone_pretrained_path=os.path.join(pt_dir, swint_pretrained_path),
                                                running_mode=swint_runnning_mode)
        
        if swint_freeze:
            for p in self.video_swint.parameters():
                p.requires_grad_(False) 
                 
        assert len(video_projs) == len(video_feat_scales)
        self.video_feat_scales = video_feat_scales
        backbone_channels, backbone_scales = self.video_swint.get_desc()
        assert len(backbone_channels) == len(backbone_scales)
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        # amr encoder
        from .amr_utils.utils import BartForConditionalGeneration
        AMRBart = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, 'amr', 'AMRBART_pretrain'))
        self.amrbart_wordEmbedding = AMRBart.model.shared
        if amrbart_wordEmbedding_freeze:
            for p in self.amrbart_wordEmbedding.parameters():
                p.requires_grad_(False) 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        
        self.objseg_class_token_ids = objseg_class_token_ids
        assert fusion.pop('name') == 'VisionLanguageFusionModule'
        self.fusion_amr_who_cross = fusion.pop('amr_cross')
        self.cross_product = VisionLanguageFusionModule(**fusion)

        assert parsing_encoder.pop('name') == 'deform_video_2d_fpn'
        self.parsing_encoder_obj_seg_nlayer = parsing_encoder.pop('obj_seg_nlayer')
        self.parsing_encoder_ref_seg_nlayer = parsing_encoder.pop('ref_seg_nlayer')
        self.obj_parsing_encoder = DeformVideo2D_with_FPN(**parsing_encoder,
                                                                nlayers=self.parsing_encoder_obj_seg_nlayer)
        self.ref_parsing_encoder = DeformVideo2D_with_FPN(**parsing_encoder,
                                                                  nlayers=self.parsing_encoder_ref_seg_nlayer)
        # ref decoder
        self.decoder_used_scales = refdecoder['used_scales']
        self.decoder_conved_scale = refdecoder['conved_scale']
        self.decoder_nlayers = refdecoder['nlayers']
        self.decoder_level_embed = nn.Embedding(len(self.decoder_used_scales), d_model)
        # amr_cross_video layer
        amr_cross_video_layer = refdecoder['amr_cross_video_layer']
        assert amr_cross_video_layer.pop('name') == 'cross_attention'
        self.decoder_amr_who_cross = amr_cross_video_layer.pop('amr_cross')
        self.decoder_amr_cross_video_layers = _get_clones(CrossAttentionLayer(**amr_cross_video_layer),
                                                                   self.decoder_nlayers)
        self.decoder_nheads = amr_cross_video_layer['nhead']
        # amr self layer
        amr_self_layer = refdecoder['amr_self_layer']
        amr_self_layer_name = amr_self_layer.pop('name')
        from .layer_graph import graphLayer_entrypoint
        create_graph_layer = graphLayer_entrypoint(amr_self_layer_name)
        graph_layer = create_graph_layer(amr_self_layer)
        self.decoder_amr_self_layers = _get_clones(graph_layer, self.decoder_nlayers)
        ffn_layer = refdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                        self.decoder_nlayers)
        # norm, mask out, box, mask
        self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3)
        self.decoder_mask_out_stride = refdecoder['mask_out_stride'] 
        self.decoder_mask_threshold = refdecoder['mask_threshold']
        self.decoder_choose_who = refdecoder['choose_who']
        
        # obj decoder
        self.obj_decoder_query_embed = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_query_feats = zero_module(nn.Embedding(objdecoder['nqueries'], d_model))
        self.obj_decoder_used_scales = objdecoder['used_scales']
        self.obj_decoder_conved_scale = objdecoder['conved_scale']
        self.obj_decoder_nlayers = objdecoder['nlayers']
        self.obj_decoder_nqueries = objdecoder['nqueries']
        self.obj_decoder_level_embed = nn.Embedding(len(self.obj_decoder_used_scales), d_model)
        cross_layer = objdecoder['cross_layer']
        assert cross_layer.pop('name') == 'cross_attention'
        self.obj_decoder_cross_video_layers = _get_clones(CrossAttentionLayer(**cross_layer),
                                                                   self.obj_decoder_nlayers)
        self.obj_decoder_nheads = cross_layer['nhead']
        self_layer = objdecoder['self_layer']
        assert self_layer.pop('name') == 'self_attention'
        self.obj_decoder_self_layers = _get_clones(SelfAttentionLayer(**self_layer),
                                                            self.obj_decoder_nlayers)  
        ffn_layer = objdecoder['ffn_layer']
        assert ffn_layer.pop('name') == 'ffn'
        self.obj_decoder_ffn_layers = _get_clones(FFNLayer(**ffn_layer),
                                                            self.obj_decoder_nlayers) 
        # norm, mask out, box, cls, mask
        self.obj_decoder_class_embed = nn.Linear(d_model, objdecoder['num_classes']+1)
        self.obj_decoder_nclasses = objdecoder['num_classes'] + 1
        self.obj_decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.obj_decoder_mask_out_stride = objdecoder['mask_out_stride']
        self.obj_decoder_mask_threshold = objdecoder['mask_threshold'] 

    def set_grad(self):
        if self.current_task == 'vis':
            pass
        else:
            pass

    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.obj_decoder_norm(output) # n bt c
        decoder_output = decoder_output.transpose(0, 1)   # bt n c
        
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # bt n c
        outputs_box = self.obj_decoder_box_embed(decoder_output) # bt n 4
        mask_embed = self.obj_decoder_mask_embed(decoder_output)  # bt n d
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # bt n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
        # bt n h w -> bt 1 n hw -> bt head n hw -> bt*head n hw
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.obj_decoder_nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_classes, outputs_mask, outputs_box, attn_mask
    
    def get_objdecoder_memories(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.obj_decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.obj_decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx]
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs]
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]
        return memories, memories_poses, memories_pad_masks, conved_features

    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_objdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        bt = memories[0].shape[0]
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, bt, 1)
        decoder_layer_preds = {}
        out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=memories[level_index], # hw bt  c
                memory_mask=attn_mask, # bt*head n hw
                memory_key_padding_mask=memories_pad_masks[level_index], # bt hw
                pos=memories_poses[level_index],  # hw bt  c
                query_pos=query_embed, # n bt  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n bt c
            )
            # bt n c
            out_class, out_mask, out_box, attn_mask = self.forward_objdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.obj_decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        # bt n
        output_mask = torch.zeros_like(output.detach())[..., 0].permute(1,0).bool()
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), output_mask, decoder_layer_preds # n bt c

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """ text_auxiliary
        'amrs': list[T(2 E_i)]
        'seg_ids': b (V+E)max
        'token_splits': list[list[int]]
        'tokens_ids': b max
        """
        # 你想visualize的东西
        check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)
        
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=amr_fusion_tokens, 
                                                    memory_key_padding_mask=amr_fusion_pad_masks,
                                                    pos=None, query_pos=poses)
            check_visualize[f'scale{lvl} attention weights'] = attn_weight
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
            
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> bT c h w -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b s c -> bT s c, bT s -> bt s c, bt s
        amr_token_feats = repeat(amr_token_feats, 'b s c -> (b t) s c', t=nf)[perFrame_has_ann]
        amr_token_seg_ids = repeat(amr_token_seg_ids, 'b s -> (b t) s', t=nf)[perFrame_has_ann] 
        repeated_amrs = [] # bT -> bt
        for idx in range(batch_size):
            for _ in range(nf):
                repeated_amrs.append(copy.deepcopy(amrs[idx]))
        filtered_amrs = []
        for idx, hsnn in enumerate(perFrame_has_ann):
            if hsnn:
                filtered_amrs.append(repeated_amrs[idx])
        assert len(filtered_amrs) != 0
        amrs = filtered_amrs
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer

        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        
        
        # 准备decoder的东西  thw b c
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=multiscales[0].device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # bt (V+E)max h w, bt*head (V+E)max hw
        out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.decoder_nlayers):
            scale_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max bt c
                memory=torch.cat([memories[scale_index], obj_queries], dim=0), # hw bt c
                memory_mask=F.pad(attn_mask.float(), pad=(0, len(obj_queries))).bool(),  # bt*head (V+E)max hw
                memory_key_padding_mask=torch.cat([memories_pad_masks[scale_index], obj_queries_mask], dim=1), 
                pos=torch.cat([memories_poses[scale_index], torch.zeros_like(obj_queries)],dim=0),  # thw b c
                query_pos=None,
            )
            # bt (V+E)max
            amr_who_cross_video = self.get_refdecoder_amr_cross(amr_token_seg_ids.clone(), layer_idx=i)
            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            # output = output2 * (1 - amr_who_cross_video.permute(1,0).float().unsqueeze(-1)) +\
            #     output * (amr_who_cross_video.permute(1,0).float().unsqueeze(-1))
            graph_self_memory = self.build_multimodal_features_along_edge(torch.cat([memories[scale_index], obj_queries], dim=0), 
                                                                          torch.cat([memories_poses[scale_index], torch.zeros_like(obj_queries)],dim=0),
                                                                          torch.cat([memories_pad_masks[scale_index], obj_queries_mask], dim=1),
                                                                          num_edges_by_batch)
            batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone().permute(1,0,2), amr_token_seg_ids.clone())], dim=0)
            assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
            batched_nodes_feats, batched_edge_feats = self.decoder_amr_self_layers[i](batched_nodes_feats, 
                                                                  batched_edge_index, 
                                                                  batched_edge_feats,
                                                                  memory=graph_self_memory,
                                                                  batch_id=graph_batch_id)
            batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids > 0, batch_idx] = batch_node_feats[batch_idx]
            batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
            for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
                amr_token_feats[seg_ids < 0, batch_idx] = batched_edge_feats[batch_idx] 
            amr_token_feats = self.decoder_ffn_layers[i](
                amr_token_feats
            )
            out_mask, out_box, attn_mask = self.forward_refdecoder_heads(amr_token_feats, conved_features, 
                                                                         attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        # choose who
        
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

    def obj_decoder_targets_handler(self, targets):
        # list[n h w], bt
        # list[n t h w] -> list[n h w], bt
        batch_size = len(targets)
        target_masks = []
        for bth_idx in range(batch_size):
            # n t h w
            t_m = targets[bth_idx]["masks"].split(1, dim=1) # list[n 1 h w], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n h w]
            target_masks.extend(t_m)
            
        for idx in range(len(target_masks)):
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = target_masks[idx].shape[-2:]
            target_masks[idx] = target_masks[idx][:, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert target_masks[idx].size(1) * self.obj_decoder_mask_out_stride == im_h
            assert target_masks[idx].size(2) * self.obj_decoder_mask_out_stride == im_w
        
        # list[n 4], bt
        target_boxes = []
        for bth_idx in range(batch_size):
            # n t 4
            t_m = targets[bth_idx]["boxes"].split(1, dim=1) # list[n 1 4], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n 4], t
            target_boxes.extend(t_m)

        # list[n], bt
        target_classes = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].bool() # n t
            bth_labels = targets[bth_idx]['class_labels'].unsqueeze(-1).repeat(1, bth_valids.shape[1]) # n t
            bth_labels = torch.where(bth_valids, bth_labels, self.obj_decoder_nclasses-1)
            t_m = bth_labels.split(1, dim=1) # list[n 1], t
            t_m = [tm.squeeze(1) for tm in t_m] # list[n], t
            target_classes.extend(t_m)    

        # list[n], bt
        is_valid = []
        for bth_idx in range(batch_size):
            bth_valids = targets[bth_idx]['valid'].split(1, dim=1) # n t -> list[n 1], t
            bth_valids = [bv.squeeze(1) for bv in bth_valids] # list[n], t
            is_valid.extend(bth_valids)
        
        return {
            'masks': target_masks,
            'boxes': target_boxes,
            'class_labels': target_classes,
            'is_valid': is_valid
        }
                        
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        
        # bt n H W, 
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 
        refseg_src = self.get_decoder_preds(model_outs)
        objseg_src = model_outs['objdecoder_objseg']
        for idx in range(batch_size):
            h, w = targets[idx]['masks'].shape[-2:]
            # n t h w -> n t H W
            targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
        
        obj_decoder_targets = self.obj_decoder_targets_handler(targets)
        loss_value_dict.update(self.obj_decoder_objseg_loss(objseg_src, obj_decoder_targets))  
                  
        loss = sum((loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss)
            sys.exit(1)
        loss.backward()
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w -> bt' h w
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)
        decoder_layer_preds = self.get_decoder_preds(decoder_layer_preds)
        last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        out_masks_logits =  last_layer_preds['pred_mask_logits'] 
        # bt 1 h w
        query_pred_masks = F.interpolate(out_masks_logits.unsqueeze(1), 
                                         scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False)
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' 1
        query_pred_is_referred_prob = torch.ones([len(query_pred_masks)]).unsqueeze(1).to(query_pred_masks.device).float()
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:].tolist()]*ann_number_by_batch[bth_idx])
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w] 
            if (orig_size[0] != after_aug_size[0]) or (orig_size[1] != after_aug_size[1]):
                # n h w -> 1 n h w -> n h w
                f_pred_masks = F.interpolate(f_pred_masks.unsqueeze(0).float(), size=orig_size, mode="nearest")[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt -> list[n t'], b
        query_pred_is_referred_prob = query_pred_is_referred_prob.split(1, dim=0)
        query_pred_is_referred_prob = [qq.squeeze(0) for qq in query_pred_is_referred_prob]
        for bth_idx in range(batch_size):
            by_batch_preds.append(torch.stack(processed_pred_masks[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            by_batch_preds_probs.append(torch.stack(query_pred_is_referred_prob[cnt:(cnt+ann_number_by_batch[bth_idx])], dim=1))
            cnt += ann_number_by_batch[bth_idx]
        assert cnt == len(processed_pred_masks)
        return {
            'query_pred_masks': by_batch_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }

    # task loss
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n h w], bt
        tgt_masks = targets['masks']
        num_boxes = sum([t.flatten(1).any(-1).int().sum() for t in tgt_masks])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=tgt_masks[0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'objdecoder_mask': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=tgt_masks[0].device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_dice': torch.tensor(0, device=tgt_masks[0].device).float(),
                      'objdecoder_class': torch.tensor(0, device=tgt_masks[0].device).float(),}

        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] # bT n H W
                layer_matching_indices = AMR_v0_detectObj.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.decoder_mask_out_stride)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detectObj.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = AMR_v0_detectObj.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detectObj.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]
        return loss_value         

    @staticmethod
    def obj_decoder_class_loss(outputs, targets, indices, class_weight):
        """
        indices: [[], []], bt
        """

        src_logits = outputs["pred_class_logits"] # bt nq c

        # list[n], bt
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets['class_labels'], indices)]) # btn_sigma
    
        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], len(class_weight)-1, dtype=torch.int64, device=src_logits.device
        ) # bt n
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.tensor(class_weight).to(src_logits))
        losses = {"objdecoder_class": loss_ce}
        return losses

    
    @staticmethod
    def obj_decoder_boxes_loss(outputs, targets, indices, num_boxes): 
        src_idx = get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_box_logits'].sigmoid()[src_idx]  # bt nq 4 -> btn_sigma 4
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n 4], bt -> btn_sigma 4
        target_boxes = torch.cat([t[J] for t, (_, J) in zip(targets['boxes'], indices)]).to(src_boxes)
            

        src_boxes = src_boxes[is_consistent]  # btn_sigma 4
        target_boxes = target_boxes[is_consistent] # btn_sigma 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['objdecoder_bbox'] = loss_bbox.sum() / num_boxes
        # bt
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['objdecoder_giou'] = loss_giou.sum() / num_boxes
        return losses
    

    @staticmethod
    def obj_decoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        src_idx = get_src_permutation_idx(indices)
        src_masks = outputs["pred_mask_logits"][src_idx]  # bt nq h w -> btn_sigma h w
        
        # list[n], bt -> btn_simga
        is_consistent = torch.cat([t[J] for t, (_, J) in zip(targets['is_valid'], indices)]).bool() 
        # list[n h w], bt -> btn_sigma h w
        target_masks = torch.cat([t[J] for t, (_, J) in zip(targets['masks'], indices)]).to(src_masks)
        
        
        src_masks = src_masks[is_consistent].flatten(1) # btn_sigma hw
        target_masks = target_masks[is_consistent].flatten(1) # btn_sigma hw
        
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "objdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    @staticmethod
    @torch.no_grad()
    def obj_decoder_matching(outputs, targets, matching_costs, class_weight, decoder_mask_out_stride):
        src_class_prob = outputs["pred_class_logits"].softmax(dim=-1) # bt n c
        src_boxes = outputs["pred_box_logits"].sigmoid()   # bt n 4
        src_masks_logits = outputs["pred_mask_logits"]  # bt n h w
        bt, nq, h, w = src_masks_logits.shape 
        
        target_boxes = targets['boxes'] # [n 4], bt
        target_masks = targets['masks'] # n h w, bt
        target_classes = targets['class_labels'] # n, bt

        indices = [] 
        for i in range(bt):
            out_prob = src_class_prob[i] # nq c
            out_bbox = src_boxes[i]  # nq 4
            out_mask = src_masks_logits[i]  # nq h w

            tgt_bbox = target_boxes[i].to(out_bbox)# n 4
            tgt_mask = target_masks[i].to(out_mask)# n h w
            tgt_cls = target_classes[i] # n

            cost_class = -out_prob[:, tgt_cls] # nq n

            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),
                                                box_ops.box_cxcywh_to_xyxy(tgt_bbox)) # n 4
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq hw : n hw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1))

            C = matching_costs['class'] * cost_class +\
                matching_costs['bbox'] * cost_bbox + \
                matching_costs['giou'] * cost_giou + \
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
