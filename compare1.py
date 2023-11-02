
# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly_fusionText(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 fusion_configs=None,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder_fusionText(encoder_layer, num_encoder_layers,
                                                                 fusion_configs=fusion_configs)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, text_feats=None, text_pad_masks=None, amr_feats=None, amr_pad_masks=None):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten,
                              text_feats=text_feats, text_pad_masks=text_pad_masks, 
                              amr_feats=amr_feats, amr_pad_masks=amr_pad_masks)

        return memory, spatial_shapes, level_start_index
    class MSDeformAttnTransformerEncoder_fusionText(nn.Module):
    def __init__(self, encoder_layer, num_layers, fusion_configs,):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        
        from models.layer_fusion import fusion_entrypoint
        create_fusion = fusion_entrypoint(fusion_configs['name'])
        fusion_module = create_fusion(fusion_configs)
        self.fusion_modules = _get_clones(fusion_module, num_layers)

        fusion_rel_self = fusion_configs['rel_self']
        self.fusion_rel_self = fusion_rel_self

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                text_feats=None, text_pad_masks=None,
                amr_feats=None, amr_pad_masks=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, (layer, fusion_module) in enumerate(zip(self.layers, self.fusion_modules)):
            if self.fusion_rel_self == 'before':
                if text_feats is not None:
                    output = fusion_module(flatten_multiscale=output, 
                                            flatten_multiscale_pos=pos,
                                            text_feats=text_feats, text_pad_masks=text_pad_masks,
                                            amr_feats=amr_feats, amr_pad_masks=amr_pad_masks)
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            elif self.fusion_rel_self == 'after':
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)                
                if text_feats is not None:
                   output = fusion_module(flatten_multiscale=output, 
                                            flatten_multiscale_pos=pos,
                                            text_feats=text_feats, text_pad_masks=text_pad_masks,
                                            amr_feats=amr_feats, amr_pad_masks=amr_pad_masks)
            else:
                raise ValueError()
        return output

