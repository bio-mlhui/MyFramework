
from transformers import Blip2Model
class AMR_v0_detectObj_onlyobj_fusionAsLoss_video(nn.Module):
    def __init__(self, 
                 d_model=256,
                 max_stride=32,
                 pt_dir='/home/xhh/pt',
                loss_weight={'objdecoder_mask': 5,
                             'objdecoder_dice': 5,
                             'objdecoder_class': 2,

                             'objdecoder_vtc':2,
                             'objdecoder_vtg':2,
                             'objdecoder_vtm':0,

                             'refdecoder_mask':5,
                             'refdecoder_dice':5
                 },
                tasks = {'objdecoder_objseg': {
                        'layer_weights': {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                        'class_weight': [1, 1,1,1,1,1,1,0.1],
                        'matching_costs': {'class': 2, 'mask': 5, 'dice': 5, 'box': 0, 'giou': 0 },
                    },
                    'refdecoder_seg':{
                        'layer_weights':{-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},}
                },
                 # video encoder
                 swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                 swint_freeze=True,
                 swint_runnning_mode='train',
                 video_projs = [
                    {'name': 'conv2d', 'in_channels': 96,  'out_channels': 256, 'kernel_size': 3, 'padding':1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 192, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},
                    {'name': 'conv2d', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'bias':True,},],
                video_feat_scales=[[1,4],[1,8],[1,16],[1,32]],

                # amrtext
                amrbart_freeze=True,
                amrbart_pt_dir='amr/AMRBART_pretrain',
                amrtext_wordEmbedding_proj = {
                    'name': 'FeatureResizer',
                    'input_feat_size': 1024,
                    'output_feat_size': 256,
                    'dropout':0.1,
                    'do_ln':True},

                objdecoder={ 
                    'num_classes': 7,
                    'nqueries': 32,
                    'nlayers': 9,
                    'cross_layer':{
                        'name': 'cross_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.1,
                    },
                    'self_layer':{
                        'name': 'self_attention',
                        'd_model': 256,
                        'nhead': 8,
                        'dropout': 0.1,
                    },
                    'ffn_layer':{
                        'name': 'ffn',
                        'd_model': 256,
                    },
                    'used_scales': [[1,32],[1,16],[1,8]],
                    'conved_scale': [1,4],
                    'mask_out_stride': 4,
                    'mask_threshold': 0.5,
                    'proj_to_amrbart':{
                        'name': 'FeatureResizer',
                        'input_feat_size': 256,
                        'output_feat_size': 1024,
                        'dropout':0.1,
                        'do_ln':True},
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
                        'name': 'graph_layer_inferfullstep_obj', # 只更新node
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
                    
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.loss_weight = loss_weight
        self.tasks = tasks # objdecoder_objseg, refdecoder_seg
        self.pt_dir = pt_dir
        self.max_stride = max_stride
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
        self.video_proj = nn.ModuleList()
        for proj_config in video_projs:
            assert proj_config.pop('name') == 'conv2d'
            self.video_proj.append(nn.Sequential(nn.Conv2d(**proj_config),
                                                 nn.GroupNorm(32, d_model)))
        self.video_3d_pos = build_position_encoding(position_embedding_name='3d',
                                                    hidden_dim=d_model)
        
        from .amr_utils.utils import BartForConditionalGeneration
        self.amrbart_model = BartForConditionalGeneration.from_pretrained(os.path.join(self.pt_dir, amrbart_pt_dir))
        if amrbart_freeze:
            for p in self.amrbart_model.parameters():
                p.requires_grad_(False)
        self.amrbart_wordEmbedding = self.amrbart_model.model.shared 
        assert amrtext_wordEmbedding_proj.pop('name') == 'FeatureResizer'
        self.amrtext_wordEmbedding_proj = FeatureResizer(**amrtext_wordEmbedding_proj)
        self.obj_decoder_refer_pos = nn.Embedding(1, self.amrbart_model.config.hidden_size)

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
        self.obj_decoder_norm = nn.LayerNorm(d_model)
        self.obj_decoder_class_embed = nn.Linear(d_model, objdecoder['num_classes']+1)
        self.obj_decoder_nclasses = objdecoder['num_classes'] + 1
        self.obj_decoder_mask_embed = MLP(d_model, d_model, d_model, 3) 
        self.obj_decoder_mask_out_stride = objdecoder['mask_out_stride']
        self.obj_decoder_mask_threshold = objdecoder['mask_threshold'] 
        proj_to_amrbart = objdecoder['proj_to_amrbart']
        assert proj_to_amrbart.pop('name') == 'FeatureResizer'
        self.obj_decoder_proj_to_amrbart = FeatureResizer(**proj_to_amrbart)
        

        self.decoder_used_scales = refdecoder['used_scales']
        self.decoder_conved_scale = refdecoder['conved_scale']
        self.decoder_nlayers = refdecoder['nlayers']
        self.decoder_level_embed = nn.Embedding(len(self.decoder_used_scales), d_model)
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
        # self.decoder_box_embed = MLP(d_model, d_model, 4, 3)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_mask_embed = MLP(d_model, d_model, d_model, 3)
        self.decoder_mask_out_stride = refdecoder['mask_out_stride'] 
        self.decoder_mask_threshold = refdecoder['mask_threshold']
        self.decoder_choose_who = refdecoder['choose_who']


    def init_parameters(self,): 
        for proj in self.video_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
    
    def encode_video(self, samples):
        bb_out = self.video_swint(samples)  
        nf, batch_size, *_ = bb_out[0].tensors.shape
        orig_pad_mask = samples.mask.permute(1, 0, 2, 3) # b t h w
        for layer_out in bb_out:
            layer_out.tensors = rearrange(layer_out.tensors, 't b c h w -> (b t) c h w')
            layer_out.mask = rearrange(layer_out.mask, 't b h w -> b t h w')
        multiscales = []
        multiscales_pad_masks = []
        multiscales_poses = []
        for lvl, feat in enumerate(bb_out): 
            src, pad_mask = feat.decompose() 
            src_proj_l = self.video_proj[lvl](src.clone())
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
            multiscales.append(src_proj_l)
            multiscales_pad_masks.append(pad_mask)
            multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
            if lvl == (len(bb_out) - 1):
                for idx in range(lvl+1, len(self.video_proj)):
                    src_proj_l = self.video_proj[idx](src.clone())
                    src_proj_l = rearrange(src_proj_l, '(b t) c h w -> b t c h w',t=nf,b=batch_size)
                    pad_mask = F.interpolate(orig_pad_mask.float(),
                                             size=src_proj_l.shape[-2:],mode='nearest') > 0.5
                    multiscales.append(src_proj_l)
                    multiscales_pad_masks.append(pad_mask)
                    multiscales_poses.append(self.video_3d_pos(src_proj_l, None))
        return multiscales, multiscales_pad_masks, multiscales_poses
    
    def encode_amr(self,  text_query, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        return amrs, amr_token_feats, amr_token_seg_ids

    
    def encode_linamr(self, text_query, text_auxiliary, device):
        amrs = text_auxiliary['amrs'] # list[Graph]
        batch_size = len(amrs)
        amr_token_seg_ids = text_auxiliary['seg_ids'].to(device)  # b (V+E)max
        amr_token_splits = text_auxiliary['token_splits'] # list[list[int]]; max() = max_tok
        amr_token_ids = text_auxiliary['token_ids'].to(device)  # b max_tok+pad
        amr_token_feats = self.amrbart_wordEmbedding(amr_token_ids) 
        amr_token_feats = self.amrtext_wordEmbedding_proj(amr_token_feats) # b max c
            
        # list[list[ti c]] -> list[Vi+Ei c]
        amr_token_feats = [torch.split(tok_feat[:sum(tok_spli)], tok_spli, dim=0) for tok_feat, tok_spli in zip(amr_token_feats, amr_token_splits)]
        for batch_idx in range(batch_size):
            amr_token_feats[batch_idx] = torch.stack([t_f.mean(dim=0) for t_f in amr_token_feats[batch_idx]], dim=0)
            
        amr_token_feats = pad_1d_feats(amr_token_feats)[0] # b (V+E)max c
        assert amr_token_feats.shape[1] == amr_token_seg_ids.shape[1]
        assert (amr_token_feats.flatten(0, 1)[amr_token_seg_ids.flatten()==0]).sum() == 0
        return amrs, amr_token_feats, amr_token_seg_ids

        
    def forward_objdecoder_heads(self, output, mask_features, attn_mask_target_size):
        batch_size, T, *_ = mask_features.shape
        decoder_output = self.obj_decoder_norm(output) # n b c
        decoder_output = decoder_output.transpose(0, 1)   # b n c
        
        prompt = self.obj_decoder_proj_to_amrbart(decoder_output) # b n 1024
        outputs_classes = self.obj_decoder_class_embed(decoder_output)  # b n c
        mask_embed = self.obj_decoder_mask_embed(decoder_output).unsqueeze(1).repeat(1, T, 1, 1)  # b t n d
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)  # b t n h w

        attn_mask = outputs_mask
        attn_mask = F.interpolate(attn_mask.flatten(0,1), size=attn_mask_target_size, mode="bilinear", align_corners=False) # bt n h w
        attn_mask = repeat(attn_mask, '(b t) n he w -> (b h) n (t he w)', b=batch_size, t=T, h=self.obj_decoder_nheads).sigmoid().detach()
        attn_mask = (attn_mask < 0.5).bool(0)
        
        return outputs_classes, outputs_mask, attn_mask, prompt

    def forward_refdecoder_heads(self, output, mask_features):
        batch_size, T, *_ = mask_features.shape
        decoder_output = self.decoder_norm(output) # n b c
        decoder_output = decoder_output.transpose(0, 1)   # b n c
        
        mask_embed = self.decoder_mask_embed(decoder_output).unsqueeze(1).repeat(1, T, 1, 1)  # b t n d
        outputs_mask = torch.einsum("btqc,btchw->bqhw", mask_embed, mask_features)  # b t n h w
        
        return outputs_mask

    def forward_obj_decoder(self, multiscales, multiscales_pad_masks, multiscales_poses):
        batch_size, device = multiscales[0], multiscales[0].device
        # b T c h w -> hw bt c
        used_feat_idxs = find_scales_from_multiscales(self.video_feat_scales, self.obj_decoder_used_scales)
        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.obj_decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx] # b T c h w
        memories = [multiscales[used_idx] for used_idx in used_feat_idxs] # b T c h w
        memories_pad_masks = [multiscales_pad_masks[used_idx] for used_idx in used_feat_idxs]
        memories_poses = [multiscales_poses[used_idx] for used_idx in used_feat_idxs]

        size_list = [mem_feat.shape[-2:] for mem_feat in memories]
        memories = [mem_feat + self.obj_decoder_level_embed.weight[i][None, None, :, None, None] for i, mem_feat in enumerate(memories)]
        query_embed = self.obj_decoder_query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # n b c
        output = self.obj_decoder_query_feats.weight.unsqueeze(1).repeat(1, batch_size, 1)
        decoder_layer_preds = {}

        # b n class, b t n h w, bh n thw, b n 1024
        out_class, out_mask, attn_mask, prompt = \
            self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[0])
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, \
                                                   'prompts':prompt, 'queries':self.obj_decoder_norm(output)}
        for i in range(self.obj_decoder_nlayers):
            level_index = i % len(self.obj_decoder_used_scales)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False 
            output = self.obj_decoder_cross_video_layers[i](
                tgt=output,  # n b c
                memory=rearrange(memories[level_index], 'b t c h w -> (t h w) b c'), 
                memory_mask=attn_mask, # b*head n hw
                memory_key_padding_mask=rearrange(memories_pad_masks[level_index], 'b t h w -> b (t h w)'),
                pos=rearrange(memories_poses[level_index], 'b t c h w -> (t h w) b c'), 
                query_pos=query_embed, # n b  c
            )
            output = self.obj_decoder_self_layers[i](
                output, # n b  c
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed, # n b c
            )
            output = self.obj_decoder_ffn_layers[i](
                output # n b c
            )
            out_class, out_mask, attn_mask, prompt = \
                self.forward_objdecoder_heads(output, conved_features, attn_mask_target_size=size_list[i])
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_class_logits':out_class, 'pred_mask_logits': out_mask, \
                                                   'prompt':prompt, 'queries':self.obj_decoder_norm(output)}
        assert len(decoder_layer_preds) == self.obj_decoder_nlayers + 1
        return self.obj_decoder_norm(output), query_embed, decoder_layer_preds # n b c

    def forward_ref_decoder(self, conved_features, memories, memories_pos, amrs, amr_token_feats, amr_token_seg_ids):
        device = memories.device
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        amr_token_feats = amr_token_feats.permute(1,0,2)
        decoder_layer_preds = {}
        # b t n h w
        out_mask  = self.forward_refdecoder_heads(amr_token_feats, conved_features)
        decoder_layer_preds[f'layer{-1}_preds'] = {'pred_mask_logits': out_mask}
        for i in range(self.decoder_nlayers):
            cross_output = self.decoder_amr_cross_video_layers[i](
                tgt=amr_token_feats.clone(),  # (V+E)max b c
                memory=memories, # nq b c
                memory_key_padding_mask=None,
                pos=memories_pos,
                query_pos=None,
            )
            # b (V+E)max
            if self.decoder_amr_who_cross[i] == '只有2/3':
                amr_who_cross_video =  torch.logical_or(amr_token_seg_ids==2, amr_token_seg_ids==3)
            elif self.decoder_amr_who_cross[i] == '所有':
                amr_who_cross_video = amr_token_seg_ids != 0

            amr_token_feats = torch.where(amr_who_cross_video.permute(1, 0).unsqueeze(-1), cross_output, amr_token_feats)
            memory_by_edge = []
            for bt_memory,  num_edges in zip(memories.permute(1,0,2), num_edges_by_batch):
                memory_by_edge.append(repeat(bt_memory, 'hw c -> E hw c', E=num_edges))
            graph_self_memory =  {'feat': torch.cat(memory_by_edge, dim=0)} # btE nq c
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
            out_mask  = self.forward_refdecoder_heads(amr_token_feats, conved_features)
            decoder_layer_preds[f'layer{i}_preds'] = {'pred_mask_logits': out_mask}
            
        assert len(decoder_layer_preds) == self.decoder_nlayers + 1
        return decoder_layer_preds

    def model_outputs(self, samples : NestedTensor, text_queries, auxiliary, perFrame_has_ann):
        """
        perFrame_has_ann: list[T], batch
        text_auxiliary
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

        # n b c, 
        obj_queries, obj_pos, objdecoder_layer_preds = self.forward_obj_decoder([scale_feat.clone() for scale_feat in multiscales],
                                                                    [scale_pad.clone() for scale_pad in multiscales_pad_masks],
                                                                    [scale_pos.clone() for scale_pos in multiscales_poses])
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_amr(text_queries, auxiliary, device)
                            
        memories = obj_queries # nq b c  
        memories_pos = obj_pos       

        conved_feat_idx = find_scale_from_multiscales(self.video_feat_scales, self.decoder_conved_scale)
        conved_features = multiscales[conved_feat_idx].clone()
        refdecoder_layer_preds = self.forward_ref_decoder(conved_features,
                                                           memories, 
                                                           memories_pos,
                                                           amrs, amr_token_feats, amr_token_seg_ids) 
        return {'refdecoder_refseg': refdecoder_layer_preds,
                # TODO: 添加一些其他可以加loss, 加postprocessing的东西, 输出的接口和trainer evaluation一致;, 输出的接口和task loss一致
                'check_visualze': check_visualize,
                 'objdecoder_objseg': objdecoder_layer_preds } 

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
    
                 
    def forward(self, samples : NestedTensor, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
            for idx in range(batch_size):
                orig_t, h, w = targets[idx]['masks'].shape[-3:]
                # n t' h w -> n t' H W
                targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
                # t -> T
                targets[idx]['has_ann'] = F.pad(targets[idx]['has_ann'].float(), pad=(0, T-len(orig_t))).bool() # T

        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [torch.tensor(t['has_ann']) for t in targets]
        perFrame_has_ann = torch.stack([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # b T
        
        model_outs = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann) 

        refseg_src = model_outs['refdecoder_refseg'] 
        objseg_src = model_outs['objdecoder_objseg'] # b n class, b T n h w, b n 1024
        if self.decoder_choose_who == '第一个':
            for i in range(-1, self.decoder_nlayers):
                layer_mask_pred = refseg_src[f'layer{i}_preds']['pred_mask_logits'] # b T n h w
                refseg_src[f'layer{i}_preds']['pred_mask_logits'] = layer_mask_pred[:, :, 0] # b T h w

        for idx in range(batch_size):
            target_masks = targets[idx]['masks'] # n t h w
            start = int(self.obj_decoder_mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[:, :, start::self.obj_decoder_mask_out_stride, start::self.obj_decoder_mask_out_stride] 
            assert target_masks.size(2) * self.obj_decoder_mask_out_stride == im_h
            assert target_masks.size(3) * self.obj_decoder_mask_out_stride == im_w
            targets[idx]['masks'] = target_masks # n t H/4 W/4

        loss_value_dict = self.refdecoder_refseg_loss(refseg_src, targets)
        loss_value_dict.update(self.obj_decoder_objseg_loss(objseg_src, targets, auxiliary)[0])  
                  
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


    def refdecoder_refseg_loss(self, decoder_layer_preds, targets):
        layer_weights = self.tasks['refdecoder_refseg']['layer_weights']
        loss_weight = self.loss_weight
        if 'mask_loss_type' in self.tasks['refdecoder_refseg']:
            mask_loss_type = self.tasks['refdecoder_refseg']['mask_loss_type']
        else:
            mask_loss_type = 'ce'
        
        # list[ni t] -> list[t] -> bt
        target_valid = torch.cat([t["valid"][t['referent_idx']] for t in targets], dim=0).long()
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        loss_value = {'refdecoder_mask': torch.tensor(0, device=target_valid.device).float(), 
                      'refdecoder_dice': torch.tensor(0, device=target_valid.device).float(), }

        for i in range(-1, self.decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds'] 
                if loss_weight['refdecoder_mask'] != 0 or loss_weight['refdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detectObj_onlyobj_fusionAsLoss.refdecoder_masks_loss(layer_pred, targets, num_boxes, self.decoder_mask_out_stride, mask_loss_type)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
        return loss_value         
    
    def obj_decoder_objseg_loss(self, decoder_layer_preds, targets, text_auxiliary):
        
        layer_weights = self.tasks['objdecoder_objseg']['layer_weights']
        class_weight = self.tasks['objdecoder_objseg']['class_weight']
        matching_costs = self.tasks['objdecoder_objseg']['matching_costs']
        loss_weight = self.loss_weight
        
        # list[n t] -> list[nt] -> bnt
        target_valid = torch.cat([t["valid"].flatten() for t in targets]).long()
        device = target_valid.device
        num_boxes = target_valid.sum().item() 
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=target_valid.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        linamr_token_feats, linamr_pad_masks = self.encode_linamr(text_auxiliary)
        loss_value = {'objdecoder_mask': torch.tensor(0, device=device).float(), 
                      'objdecoder_bbox': torch.tensor(0, device=device).float(), 
                      'objdecoder_giou': torch.tensor(0, device=device).float(),
                      'objdecoder_dice': torch.tensor(0, device=device).float(),
                      'objdecoder_class': torch.tensor(0, device=device).float(),
                      'objdecoder_vtc':torch.tensor(0, device=device).float(),
                    'objdecoder_vtg':torch.tensor(0, device=device).float(),
                    'objdecoder_vtm':torch.tensor(0, device=device).float(),}
        matching_indices_by_layer = []
        for i in range(-1, self.obj_decoder_nlayers):
            layer_weight = layer_weights[i] 
            if layer_weight != 0:
                layer_pred = decoder_layer_preds[f'layer{i}_preds']
                layer_matching_indices = AMR_v0_detectObj_onlyobj_fusionAsLoss.obj_decoder_matching(layer_pred, targets, matching_costs, class_weight, self.decoder_mask_out_stride)
                matching_indices_by_layer.append(layer_matching_indices)
                if loss_weight['objdecoder_mask'] != 0 or loss_weight['objdecoder_dice'] !=0:
                    masks_losses = AMR_v0_detectObj_onlyobj_fusionAsLoss.obj_decoder_masks_loss(layer_pred, targets, layer_matching_indices, num_boxes, self.decoder_mask_out_stride)
                    for k in masks_losses.keys():
                        loss_value[k] += layer_weight * masks_losses[k]
                if loss_weight['objdecoder_class'] != 0:
                    classes_losses = AMR_v0_detectObj_onlyobj_fusionAsLoss.obj_decoder_class_loss(layer_pred, targets, layer_matching_indices, class_weight)
                    for k in classes_losses.keys():
                        loss_value[k] += layer_weight * classes_losses[k]

                if loss_weight['objdecoder_vtc'] != 0:
                    vtc_losses = self.obj_decoder_vtc_loss(layer_pred, targets, text_auxiliary, layer_matching_indices, num_boxes)
                    for k in vtc_losses.keys():
                        loss_value[k] += layer_weight * vtc_losses[k]
                if loss_weight['objdecoder_vtg'] != 0:
                    vtg_losses = self.obj_decoder_vtg_loss(linamr_token_feats, linamr_pad_masks, text_auxiliary, layer_matching_indices, num_boxes)
                    for k in vtg_losses.keys():
                        loss_value[k] += layer_weight * vtg_losses[k]
                if loss_weight['objdecoder_vtm'] != 0:
                    vtm_losses = self.obj_decoder_vtm_loss(layer_pred, targets, layer_matching_indices, num_boxes)
                    for k in vtm_losses.keys():
                        loss_value[k] += layer_weight * vtm_losses[k]

        return loss_value,matching_indices_by_layer       

    @staticmethod
    @torch.no_grad()
    def obj_decoder_matching(outputs, targets, matching_costs, class_weight, decoder_mask_out_stride):
        # 对于所有帧都出现的, 正常
        # 有的帧出现，有的帧没出现, 把没出现的当作gt,
        # 保证是valid的
        src_class_prob = outputs["pred_class_logits"].softmax(dim=-1) # b nq c
        batch_size = len(targets)
        src_masks_logits = outputs["pred_mask_logits"].permute(0,2,1,3,4)  # b nq T h w

        # list[nq t' h w]
        src_masks_logits = [sml[:, t['has_ann']] for sml, t in zip(src_masks_logits, targets)]
        target_masks = [t['masks'] for t in targets] # list[n t' h w]
        target_classes = [t['class_labels'] for t in targets] # list[n]

        indices = [] 
        for i in range(batch_size):
            out_prob = src_class_prob[i] # nq c
            out_mask = src_masks_logits[i]  # nq t h w

            tgt_mask = target_masks[i].to(out_mask) # n t h w
            tgt_cls = target_classes[i] # n

            cost_class = -out_prob[:, tgt_cls] # nq n
            
            cost_mask = batch_sigmoid_ce_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq thw : n thw -> nq n
            cost_dice = batch_dice_loss(out_mask.flatten(1), tgt_mask.flatten(1)) # nq n

            C = matching_costs['class'] * cost_class +\
                matching_costs['mask'] * cost_mask + \
                matching_costs['dice'] * cost_dice 
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @staticmethod
    def obj_decoder_class_loss(outputs, targets, indices, class_weight):
        src_logits = outputs["pred_class_logits"] # b nq c

        # list[n], b
        target_classes_o = torch.cat([t['class_labels'][J] for t, (_, J) in zip(targets, indices)])
    
        idx = get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], len(class_weight)-1, dtype=torch.int64, device=src_logits.device
        ) # b nq
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.tensor(class_weight).to(src_logits))
        losses = {"objdecoder_class": loss_ce}
        return losses
   
    @staticmethod
    def obj_decoder_masks_loss(outputs, targets, indices, num_boxes, decoder_mask_out_stride):
        # b nq T h w
        src_masks = outputs["pred_mask_logits"].permute(0,2,1,3,4)
        # list[n T h w] -> list[n t' h w]
        src_masks = [sma[J][:, t['has_ann']] for sma, t, (J, _) in zip(src_masks, targets, indices)]

        # bnt' h w
        src_masks = torch.cat([src_m.flatten(0,1) for src_m in src_masks], dim=0)
        # list[ni t' h w]
        target_masks = torch.cat([t['masks'][J] for t, (_, J) in zip(targets, indices)]).to(src_masks)
        # bnt' h w
        target_masks = torch.cat([tgt_m.flatten(0,1) for tgt_m in target_masks], dim=0)

        src_masks = src_masks.flatten(1) # btn_sigma hw
        target_masks = target_masks.flatten(1) # btn_sigma hw
        
        losses = {
            "objdecoder_mask": ce_mask_loss(src_masks, target_masks, num_boxes),
            "objdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses    

    def obj_decoder_vtc_loss(self, outputs, text_auxiliary, indices, num_boxes):
        batch_size = len(indices)
        obj_queries = outputs["queries"].permute(1,0,2) # b nq c
        linamrs_aux = text_auxiliary['linamrs_aux'] # 'labels': text_sigma nmax, 'Esrctgts_ids': text_sigma nmax
        linamr_labels = linamrs_aux['labels']
        linamr_esrctgts = linamrs_aux['Esrctgts_ids']
        linamrs_obj_ids = text_auxiliary['linamrs_obj_id'] # list[list[int], 0-ni], batch

        # list[ni c], batch
        obj_queries = [obj_q[J] for obj_q, (J, _) in zip(obj_queries, indices)]

        tgt_idx_to_query_by_batch = []
        for bt_idx in range(batch_size):
            J = indices[bt_idx][1]
            bt_tgt_idx_to_query = {idx:query for idx, query in zip(J, obj_queries[bt_idx])}
            tgt_idx_to_query_by_batch.append(bt_tgt_idx_to_query)

        # text_sigma nmax c, <s> MASK </s> <g> AMR </g>
        linamr_feats = self.amrbart_model(linamr_esrctgts, decoder_inputs=None)
        obj_feats = []
        for bt_idx in range(batch_size):
            tgt_idx_to_query = tgt_idx_to_query_by_batch[bt_idx]
            for obj_id in range(len(linamrs_obj_ids[bt_idx])):
                obj_feats.append(tgt_idx_to_query[obj_id])
        obj_feats = torch.stack(obj_feats, dim=0) # text_sigma c

        amr_bos_feats = linamr_feats[:, 3] # text_sigma c
        contrastive_affinity = None # text_sigma text_sigma

        
        # <g> embed # list[c], n_sigma

        return

    def obj_decoder_vtg_loss(self, outputs, text_auxiliary, indices, num_boxes):
        batch_size = len(indices)
        linamrs_aux = text_auxiliary['linamrs_aux'] # 'labels': text_sigma nmax, 'Esrctgts_ids': text_sigma nmax
        linamr_labels = linamrs_aux['labels'] # text_sigma nmax
        decoder_input = torch.zeros_like(linamr_labels)
        decoder_input[:, 1:] = linamr_labels[:, :-1].clone()
        decoder_input[:, 0] = self.linamr_tokenier.amr_bos_token_id
        decoder_input_pad_mask = (decoder_input == -100)
        decoder_input.masked_fill_(decoder_input_pad_mask,self.linamr_tokenizer.pad_token_id)

        linamrs_obj_ids = text_auxiliary['linamrs_obj_id'] # list[list[int], 0-ni], batch

        # list[ni c], batch
        prompts = outputs["prompt"].permute(1,0,2) # b nq 1024

        prompt_by_text = []
        for batch_idx in range(batch_size):
            btc_prompt = prompts[batch_idx] # nq 1024
            btc_src_idx, btc_tgt_idx = indices[batch_idx]
            btc_obj_ids = linamrs_obj_ids[batch_idx] # list[int], 0-ni
            for obj_id in btc_obj_ids:
                in_idx = btc_tgt_idx.tolist().index(obj_id)
                src_idx = btc_src_idx[in_idx]
                txt_prompt = btc_prompt.clone()
                txt_prompt[src_idx] += self.obj_decoder_refer_pos.weight[0]
                prompt_by_text.append(txt_prompt)
        prompt_by_text = torch.stack(prompt_by_text, dim=0) # text_sigma nq 1024

        next_amr_token_loss = self.amrbart_model(hidden_states=prompt_by_text,
                                                attention_mask=None,
                                                decoder_input_ids=decoder_input,
                                                labels=linamr_labels)

        return {
            'obdecoder_vtg': next_amr_token_loss
        }

    def obj_decoder_vtm_loss(self, outputs, targets, indices, class_weight):
        """
        indices: [[], []], bt
        """
        raise NotImplementedError
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
    def refdecoder_masks_loss(outputs, targets, num_boxes, decoder_mask_out_stride, mask_loss_type):
        src_masks = outputs["pred_mask_logits"].flatten(0,1)  # b T h w
        # list[T h w] -> list[t' h w] -> bt' h w
        src_masks = torch.cat([src_m[t['has_ann']] for src_m, t in zip(src_masks, targets)], dim=0)

        # list[n t' h w] -> list[t' h w] -> bt' h w
        target_masks = torch.cat([t["masks"][t['referent_idx']] for t in targets], dim=0).to(src_masks)
        
        src_masks = src_masks.flatten(1) # bt hw
        target_masks = target_masks.flatten(1) # bt hw
        
        if mask_loss_type == 'ce':
            mask_loss = ce_mask_loss(src_masks, target_masks, num_boxes)
        elif mask_loss_type == 'focal':
            mask_loss = sigmoid_focal_loss(src_masks, target_masks, num_boxes)
        losses = {
            "refdecoder_mask": mask_loss,
            "refdecoder_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses         
