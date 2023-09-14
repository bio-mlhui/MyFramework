out_mask_logits = obj_last_layer_preds['pred_mask_logits'] # bt nq h w
for idx in range(batch_size):
    h, w = targets[idx]['masks'].shape[-2:]
    # n t h w -> n t H W
    targets[idx]['masks'] = F.pad(targets[idx]['masks'].float(), pad=(0, W-w, 0, H-h)).bool()
obj_decoder_targets = self.obj_decoder_targets_handler(targets)
_, matching_result = self.obj_decoder_objseg_loss(objseg_preds, obj_decoder_targets)
matching_result = matching_result[-1] # list(tgt, src), bt
gt_referent_idx = obj_decoder_targets['referent_idx'] # list[int], bt
out_mask_logits = torch.stack([out_mask[tgt_idx[src_idx.tolist().index(gt_ref_idx)]] 
                                for out_mask, gt_ref_idx, (tgt_idx, src_idx) in zip(out_mask_logits, gt_referent_idx, matching_result)], dim=0)
