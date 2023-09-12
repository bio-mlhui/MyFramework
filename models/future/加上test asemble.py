class Text_V0_nearestInterpolatePost(Text_V0):
    def __init__(self, d_model=256, max_stride=64, pt_dir='/home/xhh/pt', swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth', swint_freeze=True, swint_runnning_mode='train', video_projs=[{ 'name': 'conv2d','in_channels': 96,'out_channels': 256,'kernel_size': 3,'padding': 1,'bias': True }, { 'name': 'conv2d','in_channels': 192,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 384,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 3,'stride': 2,'padding': 1,'bias': True }], video_feat_scales=[[1, 4], [1, 8], [1, 16], [1, 32], [1, 64]], roberta_freeze=True, text_proj={ 'name': 'FeatureResizer','input_feat_size': 768,'output_feat_size': 256,'dropout': 0.1,'do_ln': True }, fusion={ 'name': 'VisionLanguageFusionModule','d_model': 256,'nheads': 8,'dropout': 0 }, parsing_encoder={ 'name': 'deform_video_2d_fpn','d_ffn': 2048,'dropout': 0,'activation': 'relu','nheads': 8,'fused_scales': [[1, 8], [1, 16], [1, 32], [1, 64]],'fpn_strides': [[1, 4], [1, 8]],'npoints': 4,'nlayers': 6 }, loss_weight={ 'refdecoder_mask': 5,'refdecoder_dice': 5,'refdecoder_refer': 2,'refdecoder_giou': 2,'refdecoder_bbox': 5 }, tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },'refer_class_weight': [1, 0.1],'matching_costs': { 'refer': 2,'mask': 5,'dice': 5,'box': 5,'giou': 2 } } }, refdecoder={ 'nqueries': 5,'nlayers': 9,'cross_layer': { 'name': 'cross_attention','d_model': 256,'nhead': 8,'dropout': 0 },'self_layer': { 'name': 'self_attention','d_model': 256,'d_model': 256,'nhead': 8,'dropout': 0 },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'mask_out_stride': 4,'mask_threshold': 0.5 }) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)

    @torch.no_grad()
    def sample(self, samples, text_queries, auxiliary, targets, visualize=False):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)
        T, batch_size, _, H, W = samples.tensors.shape
        perFrame_has_ann = [t['has_ann'] for t in targets] # bool, t
        ann_number_by_batch = [f.int().sum() for f in perFrame_has_ann]
        perFrame_has_ann = torch.cat([F.pad(t.float(), pad=(0, T-len(t))).bool() for t in perFrame_has_ann], dim=0) # bT
        # bt' n h w, bt' n
        decoder_layer_preds = self.model_outputs(samples, text_queries, auxiliary, perFrame_has_ann)['refdecoder_refseg']
        last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
        out_masks_logits =  last_layer_preds['pred_mask_logits'] 
        out_prob = last_layer_preds['pred_refer_logits'].softmax(dim=-1)
        # bt' n h w
        query_pred_masks = F.interpolate(out_masks_logits, scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False) 
        query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold) 
        # bt' n
        query_pred_is_referred_prob = out_prob[..., 0]
        
        size_original = [] #list[h,w], bt'
        size_after_aug = [] #list[h,w], bt'
        
        # 保证没有temporal增强
        for bth_idx in range(batch_size):
            size_original.extend([targets[bth_idx]['orig_size'][-2:]]*ann_number_by_batch[bth_idx])
            size_after_aug.extend([targets[bth_idx]['size'][-2:]]*ann_number_by_batch[bth_idx])
        # list[n h w], bt'
        processed_pred_masks = []
        for f_pred_masks, orig_size, after_aug_size, in zip(query_pred_masks, size_original, size_after_aug):
            f_mask_h, f_mask_w = after_aug_size  
            f_pred_masks = f_pred_masks[:, :f_mask_h, :f_mask_w]  # n h w, bool
            # n h w, float
            f_pred_masks = F.interpolate(f_pred_masks.float().unsqueeze(0), size=orig_size, mode='nearest')[0]
            processed_pred_masks.append(f_pred_masks) # n h w
            
        # list[n h w], bt -> list[n t' h w], b
        by_batch_preds = []
        by_batch_preds_probs = []
        cnt = 0
        # bt n -> list[n], bt'
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

class Text_V0_EvalAsemble(Text_V0):
    def __init__(self, d_model=256, max_stride=64, pt_dir='/home/xhh/pt', swint_pretrained_path='pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth', swint_freeze=True, swint_runnning_mode='train', video_projs=[{ 'name': 'conv2d','in_channels': 96,'out_channels': 256,'kernel_size': 3,'padding': 1,'bias': True }, { 'name': 'conv2d','in_channels': 192,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 384,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 1,'bias': True }, { 'name': 'conv2d','in_channels': 768,'out_channels': 256,'kernel_size': 3,'stride': 2,'padding': 1,'bias': True }], video_feat_scales=[[1, 4], [1, 8], [1, 16], [1, 32], [1, 64]], roberta_freeze=True, text_proj={ 'name': 'FeatureResizer','input_feat_size': 768,'output_feat_size': 256,'dropout': 0.1,'do_ln': True }, fusion={ 'name': 'VisionLanguageFusionModule','d_model': 256,'nheads': 8,'dropout': 0 }, parsing_encoder={ 'name': 'deform_video_2d_fpn','d_ffn': 2048,'dropout': 0,'activation': 'relu','nheads': 8,'fused_scales': [[1, 8], [1, 16], [1, 32], [1, 64]],'fpn_strides': [[1, 4], [1, 8]],'npoints': 4,'nlayers': 6 }, loss_weight={ 'refdecoder_mask': 5,'refdecoder_dice': 5,'refdecoder_refer': 2,'refdecoder_giou': 2,'refdecoder_bbox': 5 }, tasks={ 'refdecoder_refseg': { 'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },'refer_class_weight': [1, 0.1],'matching_costs': { 'refer': 2,'mask': 5,'dice': 5,'box': 5,'giou': 2 } } }, refdecoder={ 'nqueries': 5,'nlayers': 9,'cross_layer': { 'name': 'cross_attention','d_model': 256,'nhead': 8,'dropout': 0 },'self_layer': { 'name': 'self_attention','d_model': 256,'d_model': 256,'nhead': 8,'dropout': 0 },'ffn_layer': { 'name': 'ffn','d_model': 256 },'used_scales': [[1, 32], [1, 16], [1, 8]],'conved_scale': [1, 4],'mask_out_stride': 4,'mask_threshold': 0.5 }) -> None:
        super().__init__(d_model, max_stride, pt_dir, swint_pretrained_path, swint_freeze, swint_runnning_mode, video_projs, video_feat_scales, roberta_freeze, text_proj, fusion, parsing_encoder, loss_weight, tasks, refdecoder)

    @torch.no_grad()
    def sample(self, asembles, auxiliary, visualize=False):
        """
        asembles: list[list[vid, text, meta, callback], num_of_asemble], batch_size
        auxiliary: list[dict], batch_size
        """
        by_batch_mask_preds = []
        by_batch_preds_probs = []
        for s_asem, s_auxiliary in zip(asembles, auxiliary):
            # list[[vid, text, {has_ann}, callback]], num_asembles
            samples_without_padding, text_queries, asem_metas, callbacks = list(zip(*s_asem))
            num_asembles = len(samples_without_padding)
            sizes_without_pad = [torch.tensor(samples_without_padding[i].shape)[[0, 2, 3]].tolist() for i in range(num_asembles)]
            samples = nested_tensor_from_videos_list_with_stride(samples_without_padding, max_stride=self.max_stride)
            T, _, _, H, W, device = *samples.tensors.shape, samples.tensors.device
            
            perFrame_has_ann = [asm['has_ann'] for asm in asem_metas] # list[ti]
            
            # t -> T -> numAs T
            perFrame_has_ann = torch.stack([F.pad(pha.float(), pad=(0, T-len(pha))).bool() for pha in perFrame_has_ann], dim=0)
            
            decoder_layer_preds = self.model_outputs(samples, text_queries, s_auxiliary, perFrame_has_ann)['refdecoder_refseg']
            
            last_layer_preds = decoder_layer_preds[f'layer{self.decoder_nlayers-1}_preds']
            # bt' n H/4 W/4
            out_masks_logits =  last_layer_preds['pred_mask_logits'] 
            num_queries = out_masks_logits.shape[1]
            query_pred_refer_prob = last_layer_preds['pred_refer_logits'] # bt' n 2
            # bt' n H W
            query_pred_masks = F.interpolate(out_masks_logits, scale_factor=self.decoder_mask_out_stride, mode="bilinear", align_corners=False) 
            # query_pred_masks = (query_pred_masks.sigmoid() > self.decoder_mask_threshold)
            
            # b T n H W / b T n 2
            asems_mask_preds = torch.zeros([num_asembles, T, num_queries, H, W ], device=device).float().flatten(0, 1)
            asems_refer_prob_preds = torch.zeros([num_asembles, T, num_queries, query_pred_refer_prob.shape[-1]], device=device).float().flatten(0, 1)
            asems_mask_preds[perFrame_has_ann] = query_pred_masks
            asems_refer_prob_preds[perFrame_has_ann] = query_pred_refer_prob
            
            asems_mask_preds = rearrange(asems_mask_preds, '(b T) n H W -> b T n H W', b=num_asembles, T=T)
            asems_refer_prob_preds = rearrange(asems_refer_prob_preds, '(b T) n -> b T n', b=num_asembles, T=T)
            
            # unpad
            pred_masks_by_asem = [] # n t' h w, bool
            pred_probs_by_asem = [] # n t', float
            for asem_vid, asem_text, asem_pred_masks, asem_pred_probs, asem_has_ann, size_without_pad, asem_callback \
                in zip(samples_without_padding, text_queries, asems_mask_preds,
                       asems_refer_prob_preds, perFrame_has_ann, sizes_without_pad, callbacks):
                s_nf, s_h, s_w = size_without_pad
                # T n H W -> t n h w -> n t' h w(after aug)
                asem_pred_masks = asem_pred_masks[:s_nf, :, :s_h, :s_w][asem_has_ann]
                # T n -> t n -> n t'(after aug)
                asem_pred_probs = asem_pred_probs[:s_nf][asem_has_ann]
                asem_pred = {'masks': asem_pred_masks.permute(1, 0, 2, 3), 'refer_prob': asem_pred_probs.permute(1, 0)}
                for cb in asem_callback:
                    asem_vid, asem_text, asem_pred = cb(asem_vid, asem_text, asem_pred)
                pred_masks_by_asem.append(asem_pred['masks'])
                pred_probs_by_asem.append(asem_pred['refer_prob'])
            by_batch_mask_preds.append(
                torch.stack(pred_masks_by_asem, dim=0).mean(dim=0).sigmoid() > self.decoder_mask_threshold
            )
            # list[n t' 2] -> n t' 2 -> n t'
            by_batch_preds_probs.append(torch.stack(pred_probs_by_asem, dim=0).mean(dim=0).softmax(-1)[..., 0])
        # visualization function
        
        return {
            'query_pred_masks': by_batch_mask_preds, # [n t' h w], batch
            'query_pred_is_referred_prob': by_batch_preds_probs, # [n t'], batch
        }
