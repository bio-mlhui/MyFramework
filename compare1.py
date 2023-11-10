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
