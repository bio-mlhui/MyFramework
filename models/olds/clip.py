
###########################################################################
# 
###########################################################################
# @register_model
# def clip_v0(device, model_configs):
    
#     configs = model_configs
#     model =  CLIP_v0(
#         d_model=configs.d_model,
#         object_classes=configs.object_classes,
#         weight_dict=vars(configs.weight_dict),
#         object_decoder_configs=configs.object_decoder,
#         referent_decoder_configs=configs.referent_decoder,
#     )
#     model.to(device)

#     optmization_configs = configs.optimization
#     param_dicts = [
#         {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and p.requires_grad]},
#         {"params": [p for n, p in model.named_parameters() if ("vid_backbone" in n) and p.requires_grad],
#         "lr": optmization_configs.vid_backbone_lr},
#         {"params": [p for n, p in model.named_parameters() if ("text_backbone" in n) and p.requires_grad],
#         "lr": optmization_configs.text_backbone_lr}, 
#     ] # CHECK params dict every run
#     optimizer = get_optimizer(param_dicts=param_dicts, configs=optmization_configs.optimizer)

#     return model, optimizer 
# class CLIP_v0(nn.Module):
#     def __init__(self, 
#                  d_model,
#                  weight_dict,
#                  object_classes,
                 
#                  object_decoder_configs,
#                  referent_decoder_configs) -> None:
#         super().__init__()
#         self.weight_dict = weight_dict
#         self.d_model = d_model
        
#         self.object_classes = object_classes
        
#         from .encoder_fusion import VisionLanguageFusionModule
#         from .encoder_multiscale import multiscale_encoder_entrypoints
#         self.cross_module = VisionLanguageFusionModule(d_model=d_model, nhead=8)
#         from .transformer import TransformerEncoder, TransformerEncoderLayer
#         self.self_parser = TransformerEncoder(TransformerEncoderLayer(d_model=d_model,
#                                                                         nheads=8,
#                                                                         dim_feedforward=2048,
#                                                                         dropout=0.1,
#                                                                         activation='relu',
#                                                                         normalize_before=False), 6)

#         self.object_decoder = object_detector(object_decoder_configs, d_model=d_model) 
        
#         self.referent_decoder = referent_decoder_forSequenceText(referent_decoder_configs, d_model=d_model)
#         from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
#         clip_model = CLIPModel.from_pretrained("/home/xhh/pt/clip_base")
#         for p in clip_model.parameters():
#             p.requires_grad_(False)
#         self.clip_video_encoder = clip_model.vision_model
#         self.clip_visual_projection = clip_model.visual_projection
#         self.clip_text_encoder = clip_model.text_model
#         self.clip_text_projection = clip_model.text_projection
#         self.clip_processor = CLIPProcessor.from_pretrained("/home/xhh/pt/clip_base")
#         vocab = self.clip_processor.tokenizer.get_vocab()
#         self.object_class_token_ids = [vocab[w] for w in object_classes]
#         self.vid_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='3d')
#         self.text_pos_embed = build_position_encoding(hidden_dim=d_model, position_embedding_name='1d')
    
#     def proj_video(self, samples):
#         nf, batch_size, *_ = samples.tensors.shape
#         vid_frames = rearrange(samples.tensors, 't b c h w -> (b t) c h w')
        
#         # .last_hidden_state b s c    # 24 50 512
#         # .pooler-output: b c  # 24 1024
#         # .hidden_states: (b s c)
#         vision_outputs = self.clip_video_encoder(
#             pixel_values=vid_frames,
#             output_attentions=True,
#             output_hidden_states=True,
#             return_dict=True,
#         )
        
#         video_feat = vision_outputs.last_hidden_state[:, 1:] # bt seq c

#         video_feat = self.clip_visual_projection(video_feat)
        
#         video_feat = video_feat / video_feat.norm(p=2, dim=-1, keepdim=True)
        
#         video_feat = rearrange(video_feat, '(b t) (h w) c -> b t c h w', b=batch_size,t=nf, h=7, w=7)
#         pos = self.vid_pos_embed(video_feat, None) # b t c h w
#         orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # t b h w -> b t h w
#         pad_mask = F.interpolate(orig_pad_mask.float(), size=video_feat.shape[-2:]).to(torch.bool)
#         return video_feat, pad_mask, pos, [[1,32]]  
    
#     def proj_text(self, text_queries, device):
#         text_encoding = self.clip_processor(text=text_queries,
#                                             return_tensors='pt', padding=True)
#         input_ids = text_encoding['input_ids'].to(device) # b max
#         attention_mask = text_encoding['attention_mask'].to(device) # 0代表没有Padding
#         text_encoder_output = self.clip_text_encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True
#         )
#         text_pad_mask = attention_mask.ne(1).bool() # b max 
        
#         text_feats = text_encoder_output.last_hidden_state
#         text_sentence_feats = text_encoder_output.pooler_output
        
#         text_feats = self.clip_text_projection(text_feats)
#         text_sentence_feats = self.clip_text_projection(text_sentence_feats)
        
#         text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
#         text_sentence_feats = text_sentence_feats / text_sentence_feats.norm(p=2, dim=-1, keepdim=True)
        
#         return {
#             'token_feats': text_feats,
#             'token_pad_masks': text_pad_mask,
#             'token_sentence_feats': text_sentence_feats
#         }               
    
#     @torch.no_grad()
#     def sample(self, samples, valid_indices, text_queries, text_auxiliary, visualize=False, targets=None, saved_path=None):
#         nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device

#         video_feat, video_pad_mask, video_pos, decs = self.proj_video(samples=samples) 
#         text_encoder_output = self.proj_text(text_queries=text_queries, device=device)
#         # b max c, b max
#         crossed_text_feats, crossed_text_pad_mask = text_encoder_output['token_feats'].clone(),\
#             text_encoder_output['token_pad_masks'].clone()
#         object_class_embeds = self.get_word_embeds(self.object_class_token_ids, device=device)  
#         object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
#         crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
#         crossed_text_pos = self.text_pos_embed(crossed_text_pad_mask, hidden_dim=crossed_text_feats.shape[-1]).permute(2, 0, 1) # b c max -> max b c
#         crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
#         crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
#         crossed_text_pos = torch.cat([crossed_text_pos, torch.zeros_like(object_class_embeds)], dim=0)
#         video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
#         video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
#         video_feat = self.cross_module(tgt=video_feat,
#                                 memory=crossed_text_feats,
#                                 memory_key_padding_mask=crossed_text_pad_mask,
#                                 pos=crossed_text_pos,
#                                 query_pos=video_pos) # 6 * 49
#         video_feat = self.self_parser(src=video_feat,
#                                        mask=None, src_key_padding_mask=None,
#                                        pos=video_pos)
#         video_feat = rearrange(video_feat, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         video_pos = rearrange(video_pos, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         decoder_video_input = {
#             'multiscales': [video_feat],
#             'multiscale_pad_masks': [video_pad_mask],
#             'multiscale_poses': [video_pos],
#             'multiscale_des': decs
#         }
#         # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
#         object_decoder_output, _  = self.object_decoder(decoder_video_input, 
#                                                                       return_loss=False,
#                                                                       targets=None,
#                                                                       valid_indices=valid_indices)
        
#         out, _ = self.referent_decoder(decoder_video_input, 
#                                                             object_decoder_output, 
#                                                             text_encoder_output,
#                                                             return_loss=False, 
#                                                             targets=None, 
#                                                             matching_results=None,
#                                                             valid_indices=valid_indices)                                                          
#         # pred_logits: b t n classes, real
#         # pred_boxes: b t n 4, [0, 1]
#         # pred_masks: b t n h w, real
#         # aux_outputs: list[{pred_logits:, pred_boxes: pred_masks}]
#         # final_token_feats: 
#         output = {}
#         if len(out['pred_masks'].shape) == 5:
#             output['pred_masks'] = rearrange(out['pred_masks'], 'b t n h w -> t b n h w')
#             output['pred_is_referred'] = repeat(out['pred_logits'], 'b n c -> t b n c',t=nf)
#         else:
#             assert len(out['pred_masks'].shape) == 4
#             output['pred_masks'] = rearrange(out['pred_masks'], 'b t h w -> t b 1 h w')
#             nf, batch_size, *_ = output['pred_masks'].shape
#             pred_is_referred = torch.ones([nf, batch_size, 1, 2], dtype=torch.float, device=out['pred_masks'].device) * 100
#             pred_is_referred[..., 1] = -100
#             output['pred_is_referred'] = pred_is_referred

#         return output
    

#     def get_word_embeds(self, token_ids, device):
#         if type(token_ids[0]) == str:
#             vocab = self.clip_processor.tokenizer.get_vocab()
#             token_ids = [vocab[w] for w in token_ids]
#         token_ids = torch.tensor(token_ids, device=device)
#         token_embeds = self.clip_text_encoder.embeddings.token_embedding(token_ids) # 7 512
#         token_embeds = self.clip_text_projection(token_embeds)
        
#         token_embeds = token_embeds / token_embeds.norm(p=2, dim=-1, keepdim=True)
#         return token_embeds
    
#     # get the loss, and the model has gradients;
#     def forward(self, samples : NestedTensor, valid_indices : Tensor, text_queries, targets, text_auxiliary,
#                 visualize=False, saved_path=None):
#         """
#         'graphs': list[T(2 E_i)]
#         'seg_ids': b (V+E)max
#         'token_splits': list[list[int]]
#         'tokens_ids': b max
#         """
        
#         losses = {}
        
#         nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device

#         video_feat, video_pad_mask, video_pos, decs = self.proj_video(samples=samples) 
#         text_encoder_output = self.proj_text(text_queries=text_queries, device=device)
#         # b max c, b max
#         crossed_text_feats, crossed_text_pad_mask = text_encoder_output['token_feats'].clone(),\
#             text_encoder_output['token_pad_masks'].clone()
#         object_class_embeds = self.get_word_embeds(self.object_class_token_ids, device=device)  
#         object_class_embeds = repeat(object_class_embeds, 'k c -> k b c', b=batch_size)
#         crossed_text_feats = crossed_text_feats.permute(1, 0, 2) # max b c
#         crossed_text_pos = self.text_pos_embed(crossed_text_pad_mask, hidden_dim=crossed_text_feats.shape[-1]).permute(2, 0, 1) # b c max -> max b c
#         crossed_text_feats = torch.cat([crossed_text_feats, object_class_embeds], dim=0)
#         crossed_text_pad_mask = F.pad(crossed_text_pad_mask.float(), [0, len(object_class_embeds), 0, 0], value=0).bool()
#         crossed_text_pos = torch.cat([crossed_text_pos, torch.zeros_like(object_class_embeds)], dim=0)
#         video_feat = rearrange(video_feat, 'b t c h w -> (t h w) b c')
#         video_pos = rearrange(video_pos, 'b t c h w -> (t h w) b c')
#         video_feat = self.cross_module(tgt=video_feat,
#                                 memory=crossed_text_feats,
#                                 memory_key_padding_mask=crossed_text_pad_mask,
#                                 pos=crossed_text_pos,
#                                 query_pos=video_pos) # 6 * 49
#         video_feat = self.self_parser(src=video_feat,
#                                        mask=None, src_key_padding_mask=None,
#                                        pos=video_pos)
#         video_feat = rearrange(video_feat, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         video_pos = rearrange(video_pos, '(t h w) b c -> b t c h w',t=nf, h=7,w=7)
#         decoder_video_input = {
#             'multiscales': [video_feat],
#             'multiscale_pad_masks': [video_pad_mask],
#             'multiscale_poses': [video_pos],
#             'multiscale_des': decs
#         }
#         all_instance_targets = targets_all_instance_handler(targets, mask_size=samples.tensors.shape[-2:], 
#                                                             class_token_id_map={idx:tok_id for idx, tok_id in enumerate(self.object_class_token_ids)})
#         # {'object_embeds': b n c, 'object_box_diff':..} {'loss_mask', 'matching_results'}
#         object_decoder_output, object_loss_dict = self.object_decoder(decoder_video_input, 
#                                                                       return_loss=True,
#                                                                       targets=all_instance_targets,
#                                                                       valid_indices=valid_indices)
#         matching_results = object_loss_dict.pop('matching_results')
#         losses.update(object_loss_dict)
        
#         # 进行referent 推断
#         # referent应该和objects的matching结果一致, 
#         referent_targets = targets_refer_handler(targets, mask_size=samples.tensors.shape[-2:])
#         refer_pred, refer_loss_dict = self.referent_decoder(decoder_video_input, 
#                                                             object_decoder_output, 
#                                                             text_encoder_output,
#                                                             return_loss=True, 
#                                                             targets=referent_targets, 
#                                                             matching_results=matching_results,
#                                                             valid_indices=valid_indices)
#         losses.update(refer_loss_dict)
        
#         assert set(losses.keys()).issubset(self.weight_dict.keys())

#         loss = sum((losses[k] * self.weight_dict[k] for k in losses.keys()))
#         if not math.isfinite(loss.item()):
#             print("Loss is {}, stopping training".format(loss.item()))
#             print(losses)
#             sys.exit(1)
#         loss.backward()

#         loss_dict_unscaled = {f'{k}_unscaled': v for k, v in losses.items()}
#         loss_dict_scaled = {k: v * self.weight_dict[k] for k, v in losses.items()}    
#         grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
            
#         return loss_dict_unscaled, loss_dict_scaled, grad_total_norm

# class Referent_Decoder_forSequenceText(nn.Module):
#     def __init__(
#         self, # decoder               
#         in_channels,
#         hidden_dim: int,
        
#         nheads: int,
#         pre_norm: bool,
#         mask_dim: int,
#         enforce_input_project: bool,
#         dim_feedforward,

#         # important
#         nqueries,
#         dec_layers: int,
#         used_scales,
#         conved_scale,
#         matching_configs,
#         aux_loss,
   
#     ):
#         super().__init__()
#         self.query_pos = nn.Embedding(nqueries, hidden_dim)
#         self.nqueries = nqueries
#         self.hidden_dim = hidden_dim
#         self.num_feature_levels = len(used_scales)
#         self.used_scales = used_scales
#         assert dec_layers % self.num_feature_levels == 0
#         self.conved_scale = conved_scale
#         if self.num_feature_levels == 1:
#             self.level_embed = None
#         else:
#             self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
#         self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
#         for _ in range(self.num_feature_levels):
#             if in_channels != hidden_dim or enforce_input_project:
#                 # should be 
#                 raise NotImplementedError()
#             else:
#                 self.input_proj.append(nn.Sequential())  
                     
#         self.num_heads = nheads
#         self.num_layers = dec_layers

#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()

#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )        

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.aux_loss = aux_loss
        
#         self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
#         self.class_embed = nn.Linear(hidden_dim, 2)     
#         create_criterion = matching_entrypoints(matching_configs.name)
#         self.criterion = create_criterion(matching_configs)


#     def forward(self, 
#                 video_features_args,
#                 object_args,
#                 text_args,
#                 return_loss=False,
#                 targets=None,
#                 matching_results=None,
#                 valid_indices=None):
#         # b t c h w
#         multiscales, multiscale_masks, multiscale_poses, multiscale_dec \
#             = video_features_args['multiscales'], video_features_args['multiscale_pad_masks'], \
#                 video_features_args['multiscale_poses'], video_features_args['multiscale_des']
                
#         # n b c
#         objects_queries = object_args['object_embeds'].permute(1,0,2)
#         num_objects, batch_size, _ = objects_queries.shape
        
#         used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
#         used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
#         used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
#         conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
#         mask_features = multiscales[conved_feat_idx]
#         batch_size, nf, *_, device = *mask_features.shape, mask_features.device
        
#         cross_memories_by_scale = []
#         cross_memory_poses_by_scale = []
#         size_list = []
#         for i in range(self.num_feature_levels):
#             # 32x 
#             size_list.append(used_video_feats[i].shape[-2:])
#             scale_feats = used_video_feats[i]
#             scale_feats = self.input_proj[i](scale_feats) # b t c h w
#             scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
#             if self.num_feature_levels != 1:
#                 scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
                
#             memory = torch.cat([scale_feats, objects_queries], dim=0) # (thw + n) b c
#             pos = torch.cat([rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'), torch.zeros_like(objects_queries)], dim=0)
#             cross_memories_by_scale.append(memory) # thw+n b c
#             cross_memory_poses_by_scale.append(pos) # thw+n b c
       

#         token_sentence_feats = text_args['token_sentence_feats'] # b c
#         output = repeat(token_sentence_feats, 'b c -> n b c', n=self.nqueries)
#         # output_pos = repeat(self.text_pos.weight, '1 c -> n b c',n=output.shape[0], b=batch_size)
#         output_pos = repeat(self.query_pos.weight, 'n c -> n b c', b=batch_size)
        
        
#         predictions_mask = [] # list[b t n H/4 W/4],
#         predictions_class = [] # b n 2
#         attn_mask_size = size_list[0] 
#         # b t n h w, b*h n thw+num_objects
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size,
#                                                  num_objects=len(objects_queries))
#         predictions_class.append(outputs_class)
#         if valid_indices is not None:
#             outputs_mask = outputs_mask.index_select(dim=1, index=valid_indices)
#         predictions_mask.append(outputs_mask)
        
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels 
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
#             output = self.transformer_cross_attention_layers[i](
#                 tgt=output,  # max b c
#                 memory=cross_memories_by_scale[level_index], # thw b c
#                 memory_mask=attn_mask, # 
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=cross_memory_poses_by_scale[level_index],  # thw b c
#                 query_pos=output_pos, # max b c
#             )

#             output = self.transformer_self_attention_layers[i](
#                 output, # n b c
#                 tgt_mask=None,
#                 tgt_key_padding_mask=None, # b n 
#                 query_pos=output_pos, # n b c
#             )
            
#             output = self.transformer_ffn_layers[i](
#                 output # n b c
#             )
                
#             attn_mask_size = size_list[(i + 1) % self.num_feature_levels]

#             outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size,
#                                                  num_objects=len(objects_queries))
#             predictions_class.append(outputs_class)
#             if valid_indices is not None:
#                 outputs_mask = outputs_mask.index_select(dim=1, index=valid_indices)
#             predictions_mask.append(outputs_mask)
            
#         assert len(predictions_mask) == self.num_layers + 1
#         outputs = {
#             'pred_logits': predictions_class[-1], # b nq 2
#             'pred_masks': predictions_mask[-1], # b t nq H W
#             'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
#         }

#         if return_loss:
#             assert targets is not None and matching_results is not None
#             losses = self.forward_refer_loss(outputs, targets, matching_results)
#             return outputs, losses
#         else:
#             assert targets is None
#             return outputs, None

#     def forward_refer_loss(self, out, targets, matching_results):
#         """
#         Params:
#             targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
#         """
#         losses = {}
        
#         outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
        
#         indices = self.criterion.matching(outputs_without_aux, targets)
        
#         losses = self.criterion(out, targets, indices)
#         if self.aux_loss:
#             for i, aux_outputs in enumerate(out['aux_outputs']):
#                 indices_i = self.criterion.matching(aux_outputs, targets)
#                 l_dict_i = self.criterion(aux_outputs, targets, indices_i)
                
#                 for k in l_dict_i.keys():
#                     assert k in losses
#                     losses[k] += l_dict_i[k]  
#         return losses
    
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         return [
#             {"pred_logits": a, "pred_masks": b}
#             for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#         ]  
    
#     def forward_prediction_heads(self, output, mask_features, 
#                                  attn_mask_target_size=None, 
#                                  return_attn_mask=True,
#                                  num_objects=None):
#         bs, nf, *_= mask_features.shape # b t c h w
#         decoder_output = self.decoder_norm(output)  # n b c
#         decoder_output = decoder_output.transpose(0, 1)  # b n c
        
#         mask_embed = self.mask_embed(decoder_output)  # b n c
#         mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
#         outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w
        
#         attn_mask = None
#         if return_attn_mask:
#             assert attn_mask_target_size is not None and num_objects is not None
#             attn_mask = outputs_mask.detach().flatten(0,1) # bt n h w
#             attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w, real
#             attn_mask = repeat(attn_mask, '(b t) n h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads) # b*h n (t h w)
#             attn_mask = (attn_mask.sigmoid() < 0.5).bool()  
            
#             pad_objects_cross = F.pad(attn_mask.float(), pad=[0, num_objects, 0, 0], value=0.).bool() # b*h n thw+num_objects
            
#         outputs_class = self.class_embed(decoder_output) 
           
#         return outputs_class, outputs_mask, pad_objects_cross

# def referent_decoder_forSequenceText(decoder_configs, d_model):
#     configs = vars(decoder_configs)
#     return Referent_Decoder_forSequenceText(
#                             in_channels=d_model,
#                             hidden_dim=d_model,
#                             nheads=configs['nheads'],
#                             pre_norm=configs['pre_norm'],
#                             mask_dim=configs['mask_dim'],
#                             enforce_input_project=configs['enforce_proj_input'],
#                             dim_feedforward=configs['dff'],
#                             # important
#                             nqueries=configs['nqueries'],
#                             dec_layers=configs['nlayers'],
#                             used_scales=configs['used_scales'],
#                             conved_scale=configs['conved_scale'],
#                             matching_configs=decoder_configs.matching,
#                             aux_loss=configs['aux_loss'],)

# class ObjectDetector(nn.Module):
#     def __init__(
#         self, # decoder 
#         num_classes,
#         in_channels,
#         hidden_dim: int,
#         nheads: int,
#         dim_feedforward: int,
#         pre_norm: bool,
#         mask_dim: int,
#         enforce_input_project: bool,

#         # important
#         num_queries: int,
#         dec_layers: int,
#         used_scales,
#         conved_scale,
#         matching_configs,
#         aux_loss,
   
#     ):
#         super().__init__()
#         assert num_queries > 10
#         self.query_feat = nn.Embedding(num_queries, hidden_dim)
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.num_queries = num_queries

#         self.hidden_dim = hidden_dim
        
#         self.num_feature_levels = len(used_scales)
#         self.used_scales = used_scales
#         assert dec_layers % self.num_feature_levels == 0
#         self.conved_scale = conved_scale
#         if self.num_feature_levels == 1:
#             self.level_embed = None
#         else:
#             self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
#         self.input_proj = nn.ModuleList() # 同一层的input proj使用相同的proj
#         for _ in range(self.num_feature_levels):
#             if in_channels != hidden_dim or enforce_input_project:
#                 # should be 
#                 raise NotImplementedError()
#             else:
#                 self.input_proj.append(nn.Sequential())  
                     
#         self.num_heads = nheads
#         self.num_layers = dec_layers
#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()

#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.aux_loss = aux_loss

#         self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
#         self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

#         create_criterion = matching_entrypoints(matching_configs.name)
#         self.criterion = create_criterion(matching_configs, num_classes=num_classes)

#     def forward(self, video_args,return_loss=False, targets=None, valid_indices=None):
#         """
#         query_feats: n b c
#         video: b t c h w
#         text: b s c
#         """
#         # make sure that the video features are fused with the text features before
#         multiscales = [scale_feat.clone() for scale_feat in video_args['multiscales']]
#         multiscale_masks = [pad_mask.clone() for pad_mask in video_args['multiscale_pad_masks']]
#         multiscale_poses = [pos.clone() for pos in video_args['multiscale_poses']]
#         multiscale_dec = copy.deepcopy(video_args['multiscale_des'])
        
#         used_feat_idxs = find_scales_from_multiscales(multiscale_dec, self.used_scales)
#         used_video_feats = [multiscales[idx] for idx in used_feat_idxs]
#         used_video_poses = [multiscale_poses[idx] for idx in used_feat_idxs]
#         conved_feat_idx = find_scale_from_multiscales(multiscale_dec, self.conved_scale)
#         mask_features = multiscales[conved_feat_idx]

#         batch_size, nf, *_, device = *mask_features.shape, mask_features.device

#         query_feats = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
#         query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

#         output = query_feats
        
#         srcs = []
#         poses = []
#         size_list = []
#         for i in range(self.num_feature_levels):
#             # 32x -> 16x -> 8x
#             size_list.append(used_video_feats[i].shape[-2:])
#             scale_feats = used_video_feats[i]
#             scale_feats = self.input_proj[i](scale_feats) # b t c h w
#             scale_feats = rearrange(scale_feats, 'b t c h w -> (t h w) b c')
#             if self.num_feature_levels != 1:
#                 scale_feats += self.level_embed.weight[i][None, None, :] # thw b c
#             srcs.append(scale_feats) # thw b c
#             poses.append(rearrange(used_video_poses[i], 'b t c h w -> (t h w) b c'))
            
#         predictions_class = [] # list[b nq k+1], init -> 32x -> 16x -> 8x
#         predictions_mask = [] # list[b nq t H/4 W/4], 
#         attn_mask_size = size_list[0]
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
#                                                  output, mask_features, attn_mask_target_size=attn_mask_size)
#         if valid_indices is not None: # [3]
#             outputs_mask = outputs_mask.index_select(index=valid_indices, dim=2)
#         predictions_class.append(outputs_class)
#         predictions_mask.append(outputs_mask)
        
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels
#             # b*h n thw
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            
#             output = self.transformer_cross_attention_layers[i](
#                 tgt=output,  # n b c
#                 memory=srcs[level_index], # thw b c
#                 memory_mask=attn_mask, # bh n thw
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=poses[level_index],  # thw b c
#                 query_pos=query_pos, # n b c
#             )

#             output = self.transformer_self_attention_layers[i](
#                 output, # n b c
#                 tgt_mask=None,
#                 tgt_key_padding_mask=None, # b n 
#                 query_pos=query_pos, # n b c
#             )
#             output = self.transformer_ffn_layers[i](
#                 output # n b c
#             )
            
#             attn_mask_size = size_list[(i + 1) % self.num_feature_levels]
#             # (b nq 2, real), (b nq t H W, real), bh n thw
#             outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=attn_mask_size)
#             predictions_class.append(outputs_class)
#             if valid_indices is not None: # [3]
#                 outputs_mask = outputs_mask.index_select(index=valid_indices, dim=2)
#             predictions_mask.append(outputs_mask)

#         assert len(predictions_class) == self.num_layers + 1
#         outputs = {
#             'object_embeds': output.permute(1, 0, 2), # b n c
#             'pred_logits': predictions_class[-1], # b nq k+1
#             'pred_masks': predictions_mask[-1], # b nq t H W
 
#             'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask)
#         } # {'object_embeds': b n c, 'object_box_diff':..}

#         if return_loss:
#             assert targets is not None
#             losses, indices = self.forward_object_loss(outputs, targets)
#             losses.update({'matching_results': indices})
#             return outputs, losses
#         else:
#             assert targets is None
#             return outputs, None
    
#     def forward_object_loss(self, out, targets):
#         """
#         Params:
#             targets: list[{'masks': t h w, 'labels':int(0/1), 'valid':t, }]
#         """
#         losses = {}
        
#         outputs_without_aux = {k: v for k, v in out.items() if k != "aux_outputs"}
        
#         indices = self.criterion.matching(outputs_without_aux, targets)
        
#         losses = self.criterion(out, targets, indices)
#         if self.aux_loss:
#             for i, aux_outputs in enumerate(out['aux_outputs']):
#                 indices_i = self.criterion.matching(aux_outputs, targets)
#                 l_dict_i = self.criterion(aux_outputs, targets, indices_i)
                
#                 for k in l_dict_i.keys():
#                     assert k in losses
#                     losses[k] += l_dict_i[k]  
#         return losses, indices
    
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         """
#         Input:
#             - output_class:
#                 list[T(tb n classes)]
#             - outputs_seg_masks:
#                 list[T(tb n H W)]
#             - outputs_boxes:
#                 list[T(tb n 4)]
#         """
#         return [
#             {"pred_logits": a, "pred_masks": b}
#             for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#         ]
        
#     def forward_prediction_heads(self, output, mask_features, 
#                                  attn_mask_target_size=None, 
#                                  return_cls=True, return_attn_mask=True, return_box=False):
#         bs, nf, *_= mask_features.shape # b t c h w
#         decoder_output = self.decoder_norm(output)  # n b c
#         decoder_output = decoder_output.transpose(0, 1)  # b n c
        
#         mask_embed = self.mask_embed(decoder_output)  # b n c
#         mask_embed = repeat(mask_embed, 'b n c -> b t n c',t=nf)
#         outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features).permute(0, 2, 1, 3, 4)  # b n t h w
        
#         attn_mask = None
#         outputs_class = None
#         if return_attn_mask:
#             assert attn_mask_target_size is not None
#             attn_mask = outputs_mask.detach().flatten(0,1) # bn t h w
#             attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False) # bn t h w, real
#             attn_mask = repeat(attn_mask, '(b n) t h w -> (b head) n (t h w)',t=nf,b=bs,head=self.num_heads,
#                                n=self.num_queries) # b*h n (t h w)
#             attn_mask = (attn_mask.sigmoid() < 0.5).bool()   
            
#         if return_cls:
#             outputs_class = self.class_embed(decoder_output)  # b n k+1
            
#         return outputs_class, outputs_mask, attn_mask

# def object_detector(decoder_configs, d_model):
#     configs = vars(decoder_configs)
#     return ObjectDetector(
#         num_classes=configs['num_classes'],
#         in_channels=d_model,
#         hidden_dim=d_model,
#         nheads=configs['nheads'],
#         dim_feedforward=configs['dff'],
#         pre_norm=configs['pre_norm'],
#         mask_dim=configs['mask_dim'],
#         enforce_input_project=configs['enforce_proj_input'],
#         # important
#         num_queries=configs['num_queries'],
#         dec_layers=configs['nlayers'],
#         used_scales=configs['used_scales'],
#         conved_scale=configs['conved_scale'],
#         matching_configs=decoder_configs.matching,
#         aux_loss=configs['aux_loss'],)

