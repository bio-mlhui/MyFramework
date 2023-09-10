# text encoder加上几层self-encoder
class AMR_v0_selfEncode(AMR_v0):
    def __init__(self, 
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
                amrGraph_encoder={
                    'name': 'Graph_Layer_gatHead',
                    'd_model': 256,
                    'flow': 'source_to_target',
                    'aggr': 'min',
                    'nlayers': 6,
                    
                },
                
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
                    'nlayers': 6,},
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
                    
                ) -> None: 
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, amrbart_wordEmbedding_freeze, amrtext_wordEmbedding_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)
        from .layer_graph import graphLayer_entrypoint
        amrGraph_encoder_layer_name = amrGraph_encoder['name']
        amrGraph_encoder_nlayers = amrGraph_encoder['nlayers']
        create_graph_layer = graphLayer_entrypoint(amrGraph_encoder_layer_name)
        graph_layer = create_graph_layer(amrGraph_encoder)
        self.amr_tree_encoder = _get_clones(graph_layer, amrGraph_encoder_nlayers)
        ffn_layer = FFNLayer(d_model=d_model,
                             dim_feedforward=2048,
                             dropout=0.1)
        self.amr_tree_encoder_ffnlayer = _get_clones(ffn_layer, amrGraph_encoder_nlayers)

    def encode_text(self, text_auxiliary, device):
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
     
        graph_batch_id = []
        num_nodes_by_batch = [g.num_nodes for g in amrs]
        for bch_idx, nnode in enumerate(num_nodes_by_batch):
            graph_batch_id.extend([bch_idx] * nnode)
        num_edges_by_batch = [g.num_edges for g in amrs]
        for bch_idx, nedge in enumerate(num_edges_by_batch):
            graph_batch_id.extend([bch_idx] * nedge)
        graph_batch_id = torch.tensor(graph_batch_id, device=amr_token_feats.device)
        batched_amrs = Batch.from_data_list(amrs) # concate
        batched_edge_index = batched_amrs.edge_index.to(device)
        
        batched_nodes_feats = torch.cat([b_f[seg_ids>0] for b_f, seg_ids in zip(amr_token_feats.clone(), amr_token_seg_ids.clone())], dim=0)
        batched_edge_feats  = torch.cat([b_f[seg_ids<0] for b_f, seg_ids in zip(amr_token_feats.clone(), amr_token_seg_ids.clone())], dim=0)
        assert (sum(num_nodes_by_batch) == len(batched_nodes_feats)) and (sum(num_edges_by_batch) == len(batched_edge_feats))
        for layer, ffn_layer in zip(self.amr_tree_encoder,self.amr_tree_encoder_ffnlayer):
            batched_nodes_feats, batched_edge_feats = layer(batched_nodes_feats, 
                                                            batched_edge_index, 
                                                            batched_edge_feats,
                                                            memory=None,
                                                            batch_id=graph_batch_id)
            batched_nodes_feats = ffn_layer(batched_nodes_feats)
            batched_edge_feats = ffn_layer(batched_edge_feats)
            
        batch_node_feats = torch.split(batched_nodes_feats, num_nodes_by_batch)
        for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
            amr_token_feats[batch_idx, seg_ids > 0] = batch_node_feats[batch_idx]
        batched_edge_feats = torch.split(batched_edge_feats, num_edges_by_batch)
        for batch_idx, seg_ids in enumerate(amr_token_seg_ids):
            amr_token_feats[batch_idx, seg_ids < 0] = batched_edge_feats[batch_idx] 
        return amrs, amr_token_feats, amr_token_seg_ids

@register_model
def amr_v0_selfencode(device, configs):
    model = AMR_v0_selfEncode(
        d_model=configs['d_model'],
        pt_dir=configs['pt_dir'],
        max_stride=configs['max_stride'],
        swint_pretrained_path=configs['swint_pretrained_path'],
        swint_freeze=configs['swint_freeze'],
        swint_runnning_mode=configs['swint_runnning_mode'],
        video_projs=configs['video_projs'],
        video_feat_scales=configs['video_feat_scales'],
        amrbart_wordEmbedding_freeze=configs['amrbart_wordEmbedding_freeze'],
        amrtext_wordEmbedding_proj=configs['amrtext_wordEmbedding_proj'],
        amrGraph_encoder=configs['amrGraph_encoder'],
        fusion=configs['fusion'],
        parsing_encoder=configs['parsing_encoder'],
        loss_weight=configs['loss_weight'],
        tasks=configs['tasks'],
        refdecoder=configs['refdecoder']
        
    )
    model.to(device)


    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                    if (("video_swint" not in n) and ("amrbart_wordEmbedding" not in n) and p.requires_grad)]},
        {"params": [p for n, p in model.named_parameters() if ("video_swint" in n) and p.requires_grad],
            "lr": configs['optimization']['vid_backbone_lr']},
        {"params": [p for n, p in model.named_parameters() if ("amrbart_wordEmbedding" in n) and p.requires_grad],
            "lr": configs['optimization']['text_backbone_lr']}, 
    ] # CHECK params dict every run
    optimizer = get_optimizer(param_dicts=param_dicts, configs=configs['optimization']['optimizer'])

    return model, optimizer 
