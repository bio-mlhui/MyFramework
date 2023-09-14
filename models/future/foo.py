 check_visualize = {} 
        nf, batch_size, *_, device = *samples.tensors.shape, samples.tensors.device
        # b T c H W
        multiscales, multiscales_pad_masks, multiscales_poses = self.encode_video(samples)
        # list[Graph], b (V+E)max c, b (V+E)max 
        amrs, amr_token_feats, amr_token_seg_ids = self.encode_text(auxiliary, device)

        # 1 7 c
        obj_class_tokens = self.amrbart_wordEmbedding(torch.tensor(self.obj_decoder_class_tokens).unsqueeze(0).long().to(amr_token_feats.device)) 
        obj_class_tokens = self.amrtext_wordEmbedding_proj(obj_class_tokens).permute(1,0,2).repeat(1, batch_size, 1)# 7 b c
        # chosen_max b c, b chosen_max
        amr_fusion_tokens, amr_fusion_pad_masks = self.get_fusion_amr_cross(amr_token_feats.clone(), amr_token_seg_ids.clone(),)
        for lvl, (feat, pad_mask, poses) in enumerate(zip(multiscales, multiscales_pad_masks, multiscales_poses)):
            bs, nf, _, h, w = feat.shape
            feat = rearrange(feat, 'b t c h w -> (t h w) b c')
            poses = rearrange(poses, 'b t c h w -> (t h w) b c')
            feat, attn_weight = self.cross_product(tgt=feat,
                                                    memory=torch.cat([amr_fusion_tokens, obj_class_tokens],dim=0), 
                                                    memory_key_padding_mask=F.pad(amr_fusion_pad_masks.float(), pad=[0, len(obj_class_tokens)]).bool(),
                                                    pos=None, query_pos=poses)