 def obj_decoder_vtc_loss(self, outputs, text_auxiliary, indices, num_boxes):
        bt = len(indices)
        obj_queries = outputs["queries"].permute(1,0,2) # bt nq c
        linamrs_aux = text_auxiliary['linamrs_aux'] # 
        linamr_esrctgts = linamrs_aux['Esrctgts_ids'] # text_sigma nmax
        linamrs_obj_ids = linamrs_aux['obj_ids'] # list[list[int], 0-ni], bt

        # list[ni c], bt
        obj_queries = [obj_q[J] for obj_q, (J, _) in zip(obj_queries, indices)]

        tgt_idx_to_query_by_batch = []
        for bt_idx in range(bt):
            J = indices[bt_idx][1]
            bt_tgt_idx_to_query = {idx.item():query for idx, query in zip(J, obj_queries[bt_idx])}
            tgt_idx_to_query_by_batch.append(bt_tgt_idx_to_query)

        # text_sigma nmax c, <s> MASK </s> <g> AMR </g>
        attention_mask = linamr_esrctgts.ne(self.amrbart_tokenizer.pad_token_id)
        linamr_feats = self.amrbart_model_encoder(input_ids = linamr_esrctgts, 
                                                  attention_mask = attention_mask).last_hidden_state
        linamr_feats = linamr_feats[:, 3] # text_sigma c
        obj_feats = []
        for bt_idx in range(bt):
            tgt_idx_to_query = tgt_idx_to_query_by_batch[bt_idx]
            for obj_id in linamrs_obj_ids[bt_idx]:
                obj_feats.append(tgt_idx_to_query[obj_id])
        obj_feats = torch.stack(obj_feats, dim=0) # text_sigma c
        
        linamr_feats = linamr_feats / linamr_feats.norm(dim=1, keepdim=True)
        obj_feats = obj_feats / obj_feats.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_linamr = logit_scale * (linamr_feats @ (obj_feats.t())) # text_sigma text_sigma
        logits_per_obj = logits_per_linamr.t()
        
        # 指向相同的obj具有相同的label
        labels = np.arange(len(logits_per_linamr))
        loss_i = F.cross_entropy(logits_per_linamr, labels, axis=0)
        loss_t = F.cross_entropy(logits_per_obj, labels, axis=1)
        loss = (loss_i + loss_t)/2
