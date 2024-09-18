=    def forward_backbone_features_v0(self, x):
        # list[b 3 h w]
        self.backbone.eval()
        x = [torchv_Normalize(foo, self.pixel_mean, self.pixel_std, False) for foo in x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)

        out_features = [] # list[b hi wi c]
        for end_idx in idx_crops:
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                _out = self.backbone(torch.cat(x[start_idx: end_idx])) # 2B 3 h w
            B, _, H, W = x[start_idx].shape
            if self.feat_type == 'feat':
                features = _out['features'][0] # b cls_hw c
                features = features[:, 1:, :].reshape(B*(end_idx-start_idx), H//self.patch_size, W//self.patch_size, -1)

            elif self.feat_type == 'key':
                features = _out['qkvs'][0] # 3 b head cls_hw head_dim
                features = features[1, :, :, 1:, :] # b head hw head_di
                features: torch.Tensor = rearrange(features, 'b head (h w) d -> b h w (head d)',h=H//self.patch_size, w=W//self.patch_size)
            else:
                raise ValueError()
            out_features.extend(features.chunk(end_idx-start_idx, dim=0))
            start_idx = end_idx        

        graph_node_features = torch.cat([foo.flatten(0, 2) for foo in image_features], dim=0) # list[bhw c], crop -> N c
        # list[list[ni c], threshold] crop_batch
        cluster_feats = aggo_whole_batch(nodes_feature=graph_node_features,
                                         edge_index=self.whole_graph_edge_index,
                                         node_batch_tensor=self.nodes_batch_ids,
                                         edge_batch_tensor=self.edges_batch_ids,
                                         node_num_patches=self.node_num_patches,)



