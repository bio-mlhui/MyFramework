class NotWeak(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride
        multiscale_shapes = self.video_backbone.multiscale_shapes
        feature_dimensions = [shape.dim for key, shape in multiscale_shapes.items()]

        fusion_version = configs['model']['fusion_version']
        if fusion_version == 0:
            self.fusion = Fusion(feature_dimensions)
        elif fusion_version == 1:
            self.fusion = Fusion_v1(feature_dimensions)
        self.linear = nn.Conv2d(64, 1, kernel_size=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    @property
    def device(self):
        return self.pixel_mean.device
    def model_preds(self, videos):
        # b 3 h w -> b 3 1 h w
        bbout = self.video_backbone(x=videos.unsqueeze(2).contiguous(),)
        res2, res3, res4, res5 = bbout['res2'], bbout['res3'], bbout['res4'], bbout['res5']
        
        # b c t h w
        res2 = res2.squeeze(2).contiguous() # b c h w
        res3 = res3.squeeze(2).contiguous() # b c h w
        res4 = res4.squeeze(2).contiguous() # b c h w
        res5 = res5.squeeze(2).contiguous() # b c h w   
        pred = self.fusion(res2, res3, res4, res5) 
        pred = self.linear(pred)    

        return pred

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        # 看成instance个数是1的instance segmentation
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[1] == 1, 'weak_poly只输入单帧训练'
        image = videos[:, 0] # b 3 h w
        masks = batch_dict['targets']['masks'] # list[n t' h w], b
        mask = torch.stack([mk.squeeze() for mk in masks], dim=0).float() # b h w
        # plt.imsave('./mask.png', mask[0][0].cpu().numpy())
        pred          = self.model_preds(image) # b h w
        pred          = F.interpolate(pred, size=352, mode='bilinear').squeeze(1) # b 1 h w
        ## loss_ce + loss_dice 
        loss_ce        = F.binary_cross_entropy_with_logits(pred, mask)
        pred           = torch.sigmoid(pred)
        inter          = (pred*mask).sum(dim=(1,2))
        union          = (pred+mask).sum(dim=(1,2))
        loss_dice      = 1-(2*inter/(union+1)).mean()
        loss           = loss_ce + loss_dice

        loss.backward()       
        if not math.isfinite(loss.item()):
            logging.debug("Loss is {}, stopping training".format(loss.item()))
            raise RuntimeError()
        
        loss_value_dict = {
            'loss_ce': loss_ce,
            'loss_dice': loss_dice,
        }
        loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
        loss_dict_scaled = {k: v for k, v in loss_value_dict.items()} 
        grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
        return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 
            
    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[1] == 1, 'weak_poly只输入单帧训练和测试' 
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        # videos = 
        pred_mask = self.model_preds(videos[:, 0].contiguous()) # b 1 h w
        pred_mask = F.interpolate(pred_mask, size=(H, W), mode='bilinear') > 0 # b 1 h w
        # h w
        pred_mask = pred_mask[0][0]
        # 2
        pred_class = torch.tensor([1, 0]).to(self.device)

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        pred_mask = pred_mask[:orig_h, :orig_w] # h w
        pred_masks = pred_mask.unsqueeze(0) # 1 h w
        pred_class = pred_class.unsqueeze(0) # 1 c

        orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])

        return {
            'video': [orig_video.cpu()], # [t 3 h w], 1
            'pred_masks': [[pred_masks.cpu()]], # [list[1 h w], t, bool], 1
            'pred_class': [[pred_class.cpu()]], # [list[1 c], t, probability], 1
        }

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['weight_decay']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'backbone':None, 'base':None}

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "video_backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * configs['optim']['backbone_lr_multiplier']    
                    if log_lr_group_idx['backbone'] is None:
                        log_lr_group_idx['backbone'] = len(params)

                else:
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                                     
                # pos_embed, norm, embedding的weight decay特殊对待
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    logging.debug(f'setting weight decay of {module_name}.{module_param_name} to zero')
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
   
        return params, log_lr_group_idx


    # def forward(self, batch_dict):
    #     assert self.training
    #     orig_videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
    #     # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
    #     videos = (orig_videos - self.pixel_mean) / self.pixel_std
        
    #     # b 3 t h w -> b 1 t h w
    #     unet_output = self.model_preds(videos.permute(0, 2, 1, 3, 4).contiguous())
    #     loss_value_dict = self.unet_loss(unet_output, targets=batch_dict['targets']) # b 1 t h w

    #     assert set(list(self.loss_weight.keys())).issubset(set(list(loss_value_dict.keys())))
    #     assert set(list(loss_value_dict.keys())).issubset(set(list(self.loss_weight.keys())))
    #     loss = sum([loss_value_dict[k] * self.loss_weight[k] for k in loss_value_dict.keys()])
    #     if not math.isfinite(loss.item()):
    #         logging.debug("Loss is {}, stopping training".format(loss.item()))
    #         raise RuntimeError()
    #     loss.backward()
    #     loss_dict_unscaled = {k: v for k, v in loss_value_dict.items()}
    #     loss_dict_scaled = {f'{k}_scaled': v * self.loss_weight[k] for k, v in loss_value_dict.items()}    
    #     grad_total_norm = get_total_grad_norm(self.parameters(), norm_type=2)
    #     # from models.backbone.swin import compute_mask
    #     # compute_mask.cache_clear()        
    #     return loss_dict_unscaled, loss_dict_scaled, grad_total_norm 

    # @torch.no_grad()
    # def sample(self, batch_dict):

    #     VIS_EvalAPI_clipped_video_request_ann
    #     assert not self.training
    #     videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
    #     batch_size, T, _, H, W = videos.shape
    #     assert batch_size == 1
    #     orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
    #     # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
    #     videos = (videos - self.pixel_mean) / self.pixel_std
        
    #     # b t 3 h w -> b 3 t h w -> b 1 t h w
    #     unet_output = self.unet_forward(videos.permute(0, 2, 1, 3, 4).contiguous())
    #     pred_masks = unet_output['pred_masks'][0] # 1 t H/4 W/4
    #     pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False) > 0 # 1 t h w
    #     # 2
    #     pred_class = torch.tensor([1, 0]).to(self.device) # 0类是前景, 1是后景

    #     VIS_Aug_CallbackAPI
    #     pred_masks = pred_masks[:, :, :orig_h, :orig_w] # 1 t h w
    #     pred_class = pred_class.unsqueeze(0).repeat(orig_t, 1).unsqueeze(1) # t 1 c

    #     orig_video = videos[0][:orig_t, :, :orig_h, :orig_w] # t 3 h w
    #     orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
    #     orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1])
    #     # plt.imsave('./frame.png', orig_video[0].permute(1,2,0).cpu().numpy())
    #     # 每帧多个预测
    #     pred_masks = pred_masks.transpose(0, 1).cpu().unbind(0) # list[1 h w], t
    #     pred_class = pred_class.cpu().unbind(0) # list[1 c], t
    #     return {
    #         'video': [orig_video.cpu()], # [t 3 h w], 1
    #         'pred_masks': [pred_masks], # list[[1 h w], t], 1
    #         'pred_class': [pred_class], # list[[1 c], t], 1
    #     }
    
@register_model
def notweak(configs, device):
    from .aux_mapper import AUXMapper_v1
    model = NotWeak(configs)
    model.to(device)
    params_group, log_lr_group_idx = NotWeak.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])

    return model, optimizer, scheduler, model_input_mapper.mapper,  partial(model_input_mapper.collate, max_stride=model.max_stride), \
        log_lr_group_idx

