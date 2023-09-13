# representation上: 加上一个obj_decoder, obj_queries作为memory加到ref_decoder
class Text_V0_V1(Text_V0):
    def __init__(self, 
                 d_model=256, 
                 max_stride=64,
                 pt_dir='/home/xhh/pt', 
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth', 
                 swint_freeze=True, 
                 swint_runnning_mode='train', 
                 video_projs=[{ 'name': 'conv2d','in_channels': 96,'out_channels': 256,'kernel_size': 3,'padding': 1,'bias': True }, 
                              { 'name': 'conv2d','in_channels': 192,'out_channels': 256,'kernel_size': 1,'bias': True },
                              { 'name': 'conv2d','in_channels': 384,'out_channels': 256,'kernel_size': 1,'bias': True }, 
                              { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 1,'bias': True }, 
                              { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 3,'stride': 2,'padding': 1,'bias': True }],
                 video_feat_scales=[[1, 4], [1, 8], [1, 16], [1, 32], [1, 64]], 
                 roberta_freeze=True, 
                 text_proj={ 'name': 'FeatureResizer',
                            'input_feat_size': 768,
                            'output_feat_size': 256,
                            'dropout': 0.1,
                            'do_ln': True }, 
                 fusion={ 'name': 'VisionLanguageFusionModule',
                         'd_model': 256,
                         'nheads': 8,
                         'dropout': 0 }, 
                 parsing_encoder={ 'name': 'deform_video_2d_fpn',
                                  'd_ffn': 2048,
                                  'dropout': 0,
                                  'activation': 'relu',
                                  'nheads': 8,
                                  'fused_scales': [[1, 8], [1, 16], [1, 32], [1, 64]],
                                  'fpn_strides': [[1, 4], [1, 8]],
                                  'npoints': 4,
                                  'nlayers': 6 }, 
                 
                 loss_weight={ 'refdecoder_mask': 5,
                              'refdecoder_dice': 5,
                              'refdecoder_refer': 2,
                              'refdecoder_giou': 0,
                              'refdecoder_bbox': 0,
                              
                              'objdecoder_mask': 5,
                              'objdecoder_dice': 5,
                              'objdecoder_class': 2,
                              'objdecoder_giou': 0,
                              'objdecoder_bbox': 0,}, 
                 tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },
                                               'refer_class_weight': [1, 0.1],
                                               'matching_costs': { 'refer': 2,'mask': 5,'dice': 5,'box': 0,'giou': 0 } },
                        'objdecoder_objseg': {'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },
                                               'class_weight': [1, 1, 1, 1, 1, 1, 1, 0.1],
                                               'matching_costs': { 'class': 2,'mask': 5,'dice': 5,'box': 0,'giou': 0 }}}, 
                 refdecoder={ 'nqueries': 5,
                             'nlayers': 9,
                             'cross_layer': { 'name': 'cross_attention',
                                             'd_model': 256,
                                             'nhead': 8,
                                             'dropout': 0 },
                             'self_layer': { 'name': 'self_attention',
                                            'd_model': 256,
                                            'd_model': 256,
                                            'nhead': 8,
                                            'dropout': 0 },
                             'ffn_layer': { 'name': 'ffn',
                                           'd_model': 256 },
                             'used_scales': [[1, 32], [1, 16], [1, 8]],
                             'conved_scale': [1, 4],
                             'mask_out_stride': 4,
                             'mask_threshold': 0.5 },
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
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)

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
        pred_classes_by_query = out_class.argmax(dim=-1, keepdim=False)
        # bt n
        output_mask = pred_classes_by_query == (self.obj_decoder_nclasses - 1)
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return output, output_mask, decoder_layer_preds # n bt c
    
    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """
        samples: t b c h w, t b h w
        frame_has_ann_by_batch: list[t, True/False], b
        """
        check_visualize = {} 
        device = samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # 抽文本的特征 max b c,  b max, b c 
        token_feats, token_pad_masks, token_sentence_feats = self.encode_text(text_queries, device)
        token_pos = self.text_pos_embed(token_pad_masks, hidden_dim=token_feats.shape[-1]).permute(2, 0, 1)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=token_feats, 
                                                    memory_key_padding_mask=token_pad_masks,
                                                    pos=token_pos, 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} fusion attention weights'] = attn_weight # b thw s, float, 0, 1
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> list[t c h w] -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b c -> bT c -> bt c
        token_sentence_feats = repeat(token_sentence_feats, 'b c -> (b t) c', t=nf)[perFrame_has_ann]
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer
        
        # n bt c, bt n,
        obj_queries, obj_queries_mask, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = repeat(token_sentence_feats, 'bt c -> n bt c', n=self.decoder_nqueries,) # n bt c

        decoder_layer_preds = {}
        out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.decoder_nlayers):
            level_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=torch.cat([memories[level_index], obj_queries], dim=0), # hw+n bt  c
                memory_mask=F.pad(attn_mask.float(), pad=(0, len(obj_queries))).bool(), # bt*head n hw+n
                memory_key_padding_mask=torch.cat([memories_pad_masks[level_index], obj_queries_mask], dim=1), # bt hw+n
                pos=torch.cat([memories_poses[level_index],torch.zeros_like(obj_queries)], dim=0),  # hw+n bt c
                query_pos=query_embed, # n bt  c
            )
            output = self.decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.decoder_ffn_layers[i](
                output # n bt c
            )
            out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }

        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                'objdecoder_objseg': objdecoder_layer_preds,} 
    
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
        
        refseg_src = model_outs['refdecoder_refseg']
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
                layer_matching_indices = Text_V0_V1.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.decoder_mask_out_stride)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = Text_V0_V1.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_bbox'] != 0 or loss_weight['objdecoder_giou'] !=0:
                    boxes_losses = Text_V0_V1.obj_decoder_boxes_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in boxes_losses.keys():
                        loss_value[k] += layer_weight * boxes_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = Text_V0_V1.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
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

class Text_V0_V1_no_objmask(Text_V0_V1):
    def __init__(self, d_model=256, max_stride=64, pt_dir='/home/xhh/pt', swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth', swint_freeze=True, swint_runnning_mode='train', video_projs=[{ 'name': 'conv2d','in_channels': 96,'out_channels': 256,'kernel_size': 3,'padding': 1,'bias': True }, { 'name': 'conv2d','in_channels': 192,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 384,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 3,'stride': 2,'padding': 1,'bias': True }], video_feat_scales=[[1, 4], [1, 8], [1, 16], [1, 32], [1, 64]], roberta_freeze=True, text_proj={ 'name': 'FeatureResizer','input_feat_size': 768,'output_feat_size': 256,'dropout': 0.1,'do_ln': True }, fusion={ 'name': 'VisionLanguageFusionModule','d_model': 256,'nheads': 8,'dropout': 0 }, parsing_encoder={ 'name': 'deform_video_2d_fpn','d_ffn': 2048,'dropout': 0,'activation': 'relu','nheads': 8,'fused_scales': [[1, 8], [1, 16], [1, 32], [1, 64]],'fpn_strides': [[1, 4], [1, 8]],'npoints': 4,'nlayers': 6 }, loss_weight={ 'refdecoder_mask': 5,'refdecoder_dice': 5,'refdecoder_refer': 2,'refdecoder_giou': 0,'refdecoder_bbox': 0,'objdecoder_mask': 5,'objdecoder_dice': 5,'objdecoder_class': 2,'objdecoder_giou': 0,'objdecoder_bbox': 0 }, tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },'refer_class_weight': [1, 0.1],'matching_costs': { 'refer': 2,'mask': 5,'dice': 5,'box': 0,'giou': 0 } },'objdecoder_objseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },'class_weight': [1, 1, 1, 1, 1, 1, 1, 0.1],'matching_costs': { 'class': 2,'mask': 5,'dice': 5,'box': 0,'giou': 0 } } }, refdecoder={ 'nqueries': 5,'nlayers': 9,'cross_layer': { 'name': 'cross_attention','d_model': 256,'nhead': 8,'dropout': 0 },'self_layer': { 'name': 'self_attention','d_model': 256,'d_model': 256,'nhead': 8,'dropout': 0 },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'mask_out_stride': 4,'mask_threshold': 0.5 }, objdecoder={ 'num_classes': 7,'nqueries': 100,'nlayers': 9,'cross_layer': { 'name': 'cross_attention','d_model': 256,'nhead': 8,'dropout': 0 },'self_layer': { 'name': 'self_attention','d_model': 256,'d_model': 256,'nhead': 8,'dropout': 0 },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'mask_out_stride': 4,'mask_threshold': 0.5 }) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder, objdecoder)
    
    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """
        samples: t b c h w, t b h w
        frame_has_ann_by_batch: list[t, True/False], b
        """
        check_visualize = {} 
        device = samples.tensors.device
        # 抽视频的特征 b t c h w
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # 抽文本的特征 max b c,  b max, b c 
        token_feats, token_pad_masks, token_sentence_feats = self.encode_text(text_queries, device)
        token_pos = self.text_pos_embed(token_pad_masks, hidden_dim=token_feats.shape[-1]).permute(2, 0, 1)
        
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=token_feats, 
                                                    memory_key_padding_mask=token_pad_masks,
                                                    pos=token_pos, 
                                                    query_pos=poses)
            check_visualize[f'scale{lvl} fusion attention weights'] = attn_weight # b thw s, float, 0, 1
            multiscales[lvl] = rearrange(feat, '(t h w) b c -> b t c h w',t=nf, h=h,w=w)
        
        # 从这里开始变成2d， 只关注每一帧
        nf = multiscales[0].shape[1]
        # b T c h w -> list[t c h w] -> bt c h w
        for idx, scale_feat in enumerate(multiscales):
            multiscales[idx] = scale_feat.flatten(0, 1)[perFrame_has_ann]
        for idx, scale_pad in enumerate(multiscales_pad_masks):
            multiscales_pad_masks[idx] = scale_pad.flatten(0,1)[perFrame_has_ann]
        for idx, scale_pos in enumerate(multiscales_poses):
            multiscales_poses[idx] = scale_pos.flatten(0,1)[perFrame_has_ann]
        bt = multiscales[0].shape[0]
        # b c -> bT c -> bt c
        token_sentence_feats = repeat(token_sentence_feats, 'b c -> (b t) c', t=nf)[perFrame_has_ann]
        
        # 多模态特征进一步parsing  # bt hw_sigma head num_scale num_point 2
        multiscales, sampling_locations_by_layer, attention_weights_by_layer\
            = self.deform_multiscale_2dencoder(multiscales, multiscales_pad_masks, multiscales_poses, self.video_feat_scales)
        check_visualize['deform parsing encoder sampling_locations_by_layer'] = sampling_locations_by_layer
        check_visualize['deform parsing encoder attention_weights_by_layer'] = attention_weights_by_layer
        
        # n bt c, bt n,
        obj_queries, _, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                                         [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                                         [scale_pos.clone() for scale_pos in multiscales_poses])
        
        # bt c h w -> hw bt c,  cross attention里video不用padding mask
        memories, memories_poses, memories_pad_masks, conved_features = self.get_refdecoder_memories(multiscales, multiscales_pad_masks, multiscales_poses)
        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [rearrange(mem_feat, 'bt c h w -> (h w) bt c') for mem_feat in memories]
        memories_poses = [rearrange(mem_pos, 'bt c h w -> (h w) bt c') for mem_pos in memories_poses]
        memories_pad_masks = [rearrange(mem_pad, 'bt h w -> bt (h w)') for mem_pad in memories_pad_masks]
        memories = [mem_feat + self.decoder_level_embed.weight[i][None, None, :] for i, mem_feat in enumerate(memories)]
        query_embed = self.decoder_query_embed.weight.unsqueeze(1).repeat(1, bt, 1) # n bt c
        output = repeat(token_sentence_feats, 'bt c -> n bt c', n=self.decoder_nqueries,) # n bt c

        decoder_layer_preds = {}
        out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }
        for i in range(self.decoder_nlayers):
            level_index = i % len(self.decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.decoder_cross_video_layers[i](
                tgt=output,  # n bt c
                memory=torch.cat([memories[level_index], obj_queries], dim=0), # hw+n bt  c
                memory_mask=F.pad(attn_mask.float(), pad=(0, len(obj_queries))).bool(), # bt*head n hw+n
                memory_key_padding_mask=F.pad(memories_pad_masks[level_index].float(), 
                                              pad=(0, len(obj_queries))).bool(), # bt hw+n
                pos=torch.cat([memories_poses[level_index],torch.zeros_like(obj_queries)], dim=0),  # hw+n bt c
                query_pos=query_embed, # n bt  c
            )
            output = self.decoder_self_layers[i](
                output, # n bt  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n bt c
            )
            output = self.decoder_ffn_layers[i](
                output # n bt c
            )
            out_refer, out_mask, out_box, attn_mask = self.forward_refdecoder_heads(output, conved_features, 
                                                                                attn_mask_target_size=size_list[(i + 1) % len(self.decoder_used_scales)])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_refer_logits':out_refer, 'pred_mask_logits': out_mask, 'pred_box_logits': out_box }

        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return {'refdecoder_refseg': decoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                'objdecoder_objseg': objdecoder_layer_preds,} 
     

@register_model
def text_v0_v1(device, configs):
    model = Text_V0_V1(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("roberta" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 

@register_model
def text_v0_v1_no_objmask(device, configs):
    model = Text_V0_V1_no_objmask(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        roberta_freeze=configs['roberta_freeze'],
        text_proj=configs['text_proj'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder'],
        objdecoder=configs['objdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("roberta" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("roberta" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 