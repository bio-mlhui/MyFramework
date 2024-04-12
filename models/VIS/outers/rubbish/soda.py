

class Shadow_Splittime_SS(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.loss_weight = configs['model']['loss_weight']
        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride
         
        self.decoder = META_ARCH_REGISTRY.get(configs['model']['decoder']['name'])(configs['model']['decoder'],
                                                                                   multiscale_shapes=self.video_backbone.multiscale_shapes)
        
        self.test_clip_size = configs['model']['test_clip_size']
        
    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict,):
        if (not self.training) and (self.test_clip_size is not None):
            nf = videos.shape[2]
            clip_outputs = [] # list[dict]
            for start_idx in range(0, nf, self.test_clip_size):
                multiscales = self.video_backbone(x=videos[:, :, start_idx:(start_idx + self.test_clip_size)]) # b c t h w
                clip_outputs.append(self.decoder(multiscales, video_aux_dict=video_aux_dict)[-1])  # b t nq h w
            return [{
                'pred_masks': torch.cat([haosen['pred_masks'] for haosen in clip_outputs], dim=1), # b t n h w
                'pred_class':  torch.cat([haosen['pred_class'] for haosen in clip_outputs], dim=1),
            }]
            
        # b 3 t h w -> b 3 t h w
        multiscales = self.video_backbone(x=videos) # b c t h w
        return self.decoder(multiscales, video_aux_dict=video_aux_dict)

    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        videos = batch_dict['video_dict']['videos'] # b T 3 h w, 0-1
        targets = batch_dict['targets']
        batch_size, nf = videos.shape[:2]
        # plt.imsave('./frame.png', videos[0][0].permute(1,2,0).cpu().numpy())
        videos = (videos - self.pixel_mean) / self.pixel_std
        # plt.imsave('./mask.png', mask[0][0].cpu().numpy())
        pred1          = self.model_preds(videos.permute(0, 2, 1, 3, 4), 
                                          video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        pred1_loss = self.decoder.compute_loss(pred1, targets=targets, frame_targets=batch_dict['frame_targets'],
                                               video_aux_dict=batch_dict['video_dict'])

        loss_value_dict = {key: pred1_loss[key] for key in list(self.loss_weight.keys())}
        # gradient_norm = get_total_grad_norm(self.model.parameters(), norm_type=2)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        VIS_EvalAPI_clipped_video_request_ann
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b 1 t h w}
        # 如果是List的话, 那么取最后一层
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear') > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['finetune_lr']
        defaults['weight_decay'] = configs['optim']['finetune_wd']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'base':None, 'finetune':None}

        base_param_names = []
        finetune_param_names = []
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                
                hyperparams = copy.copy(defaults)
                if "temporal_block" in module_name or 'segformer_head' in module_name or 'temporal_norm' in module_name or 'video_backbone.outnorm' in module_name:
                    hyperparams["lr"] = configs['optim']['base_lr']  
                    hyperparams["weight_decay"]  = configs['optim']['base_wd'] 
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                    base_param_names.append(f'{module_name}.{module_param_name}')                    
                else:
                    if log_lr_group_idx['finetune'] is None:
                        log_lr_group_idx['finetune'] = len(params)
                    finetune_param_names.append(f'{module_name}.{module_param_name}')
                                     
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
    
        logging.debug(f'List of Base Parameters: \n {base_param_names}')
        logging.debug(f'List of Finetune Parameters: \n {finetune_param_names}')
   
        return params, log_lr_group_idx

@register_model
def shadow_splittime_ss(configs, device):
    from .aux_mapper import AUXMapper_v1
    model = Shadow_Splittime_SS(configs)
    model.to(device)
    params_group, log_lr_group_idx = Shadow_Splittime_SS.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])
    

    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                    model_input_mapper.mapper, 
                                                                    partial(model_input_mapper.collate, max_stride=model.max_stride))

    # dataset_specific initialization

    return model, optimizer, scheduler,  train_samplers, train_loaders, log_lr_group_idx, eval_function


class Shadow_CombineTime_SS(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.loss_weight = configs['model']['loss_weight']
        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride
         
        self.decoder = META_ARCH_REGISTRY.get(configs['model']['decoder']['name'])(configs['model']['decoder'],
                                                                                   multiscale_shapes=self.video_backbone.multiscale_shapes)
        self.test_clip_size = configs['model']['test_clip_size']
        
    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict=None):
        if (not self.training) and (self.test_clip_size is not None):
            nf = videos.shape[2]
            clip_outputs = [] # list[dict]
            for start_idx in range(0, nf, self.test_clip_size):
                multiscales = self.video_backbone(x=videos[:, :, start_idx:(start_idx + self.test_clip_size)]) # b c t h w
                clip_outputs.append(self.decoder(multiscales, video_aux_dict=video_aux_dict)[-1])  # b t nq h w
            return [{
                'pred_masks': torch.cat([haosen['pred_masks'] for haosen in clip_outputs], dim=1), # b t n h w
                'pred_class':  torch.cat([haosen['pred_class'] for haosen in clip_outputs], dim=1),
            }]
            
        multiscales = self.video_backbone(x=videos) 
        return self.decoder(multiscales, video_aux_dict=video_aux_dict)

    # with scale consistency scan
    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        videos = batch_dict['video_dict']['videos'] # b T 3 h w, 0-1
        batch_size, nf = videos.shape[:2]
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = batch_dict['targets']
        size1          = np.random.choice([320, 352, 384, 416, 448, 512, 544, 576, 608])
        vid_1         = F.interpolate(videos.flatten(0, 1), size=size1, mode='bilinear')
        vid_1          = rearrange(vid_1, '(b T) c h w -> b c T h w',b=batch_size, T=nf)
        
        pred1          = self.model_preds(vid_1,  video_aux_dict=batch_dict['video_dict']) 
        pred1_loss = self.decoder.compute_loss(pred1, targets=targets, frame_targets=batch_dict['frame_targets'],
                                               video_aux_dict=batch_dict['video_dict'])

        loss_value_dict = {key: pred1_loss[key] for key in list(self.loss_weight.keys())}
        # gradient_norm = get_total_grad_norm(self.model.parameters(), norm_type=2)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b t n h w, b t n c}
        # 如果是List的话, 那么取最后一层
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False) > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']
        defaults = {}
        defaults['lr'] = configs['optim']['base_lr']
        defaults['weight_decay'] = configs['optim']['base_wd']
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'base':None}
        base_param_names = []
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if log_lr_group_idx['base'] is None:
                    log_lr_group_idx['base'] = len(params)
                base_param_names.append(f'{module_name}.{module_param_name}')                    
                                     
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
    
        logging.debug(f'List of Base Parameters: \n {base_param_names}')
   
        return params, log_lr_group_idx

@register_model
def shadow_combinetime_ss(configs, device):
    from .aux_mapper import AUXMapper_v1
    model = Shadow_CombineTime_SS(configs)
    model.to(device)
    params_group, log_lr_group_idx = Shadow_CombineTime_SS.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])
    

    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                    model_input_mapper.mapper, 
                                                                    partial(model_input_mapper.collate, max_stride=model.max_stride))

    return model, optimizer, scheduler,  train_samplers, train_loaders, log_lr_group_idx, eval_function


class Shadow_SOTA(nn.Module):
    def __init__(
        self,
        configs,
        pixel_mean = [0.485, 0.456, 0.406],
        pixel_std = [0.229, 0.224, 0.225],):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) # 3 1 1
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.loss_weight = configs['model']['loss_weight']
        video_backbone_configs = configs['model']['video_backbone'] 
        video_backbone_cls = BACKBONE_REGISTRY.get(video_backbone_configs['name'])
        self.video_backbone = video_backbone_cls(video_backbone_configs)
        self.max_stride = self.video_backbone.max_stride
        self.decoder = META_ARCH_REGISTRY.get(configs['model']['decoder']['name'])(configs['model']['decoder'],
                                                                                   multiscale_shapes=self.video_backbone.multiscale_shapes)
        
    @property
    def device(self):
        return self.pixel_mean.device
    
    def model_preds(self, videos, video_aux_dict,):
        # b 3 t h w -> b 3 t h w
        multiscales = self.video_backbone(x=videos) # b c t h w
        return self.decoder(multiscales, video_aux_dict=video_aux_dict)

    # with scale consistency scan
    def forward(self, batch_dict):
        assert self.training
        VIS_TrainAPI_clipped_video
        videos = batch_dict['video_dict']['videos'] # b T 3 h w, 0-1
        batch_size, nf = videos.shape[:2]
        videos = (videos - self.pixel_mean) / self.pixel_std
        targets = batch_dict['targets']
        size1          = np.random.choice([320, 352, 384, 416, 448, 512, 544, 576, 608])
        vid_1         = F.interpolate(videos.flatten(0, 1), size=size1, mode='bilinear')
        vid_1          = rearrange(vid_1, '(b T) c h w -> b c T h w',b=batch_size, T=nf)
        
        pred1          = self.model_preds(vid_1,  video_aux_dict=batch_dict['video_dict']) 
        pred1_loss = self.decoder.compute_loss(pred1, targets=targets, frame_targets=batch_dict['frame_targets'],
                                               video_aux_dict=batch_dict['video_dict'])

        loss_value_dict = {key: pred1_loss[key] for key in list(self.loss_weight.keys())}
        # gradient_norm = get_total_grad_norm(self.model.parameters(), norm_type=2)
        return loss_value_dict, self.loss_weight

    @torch.no_grad()
    def sample(self, batch_dict):
        assert not self.training
        videos = batch_dict['video_dict']['videos'] # b t 3 h w, 0-1
        orig_t, _, orig_h, orig_w = batch_dict['video_dict']['orig_sizes'][0]
        videos = (videos - self.pixel_mean) / self.pixel_std
        assert videos.shape[0] == 1
        batch_size, T, _, H, W = videos.shape
        videos = videos.permute(0, 2, 1,3,4) # b c t h w
        decoder_output = self.model_preds(videos, video_aux_dict=batch_dict['video_dict']) # {pred_masks: b t n h w, b t n c}
        # 如果是List的话, 那么取最后一层
        if isinstance(decoder_output, list):
            decoder_output = decoder_output[-1]
        pred_masks = decoder_output['pred_masks'][0] # T n h w
        pred_masks = F.interpolate(pred_masks, size=(H, W), mode='bilinear', align_corners=False) > 0 # T n h w
        pred_masks = pred_masks[:orig_t, :, :orig_h, :orig_w] # T n h w
        #
        pred_classes = decoder_output['pred_class'][0][:orig_t, :,:] # T n c, probability
        pred_classes = pred_classes.cpu().unbind(0) # list[n c], T
        pred_masks = pred_masks.cpu().unbind(0) # list[n h w], T

        VIS_Aug_CallbackAPI
        # 每一帧多个预测, 每个预测有box/mask, 每个预测的类别概率
        orig_video = videos[0][:, :orig_t, :orig_h, :orig_w].permute(1,0,2,3) # T 3 h w
        orig_video = Trans_F.normalize(orig_video, [0, 0, 0], 1 / self.pixel_std)
        orig_video = Trans_F.normalize(orig_video, -self.pixel_mean, [1, 1, 1]).cpu()

        return {
            'video': [orig_video], # [t 3 h w], 1
            'pred_masks': [pred_masks], # [list[n h w], t, bool], 1
            'pred_class': [pred_classes], # [list[n c], t, probability], 1
        }

    @staticmethod
    def get_optim_params_group(model, configs):
        weight_decay_norm = configs['optim']['weight_decay_norm']
        weight_decay_embed = configs['optim']['weight_decay_embed']

        defaults = {}
        defaults['lr'] = configs['optim']['finetune_lr']
        defaults['weight_decay'] = configs['optim']['finetune_wd']

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        # 除了temporal block, segformer_head.linear_c, segformer_head.classifier
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        log_lr_group_idx = {'base':None, 'finetune':None}

        base_param_names = []
        finetune_param_names = []
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                
                hyperparams = copy.copy(defaults)
                if 'segformer_head.classifier' in module_name:
                    hyperparams["lr"] = configs['optim']['base_lr']  
                    hyperparams["weight_decay"]  = configs['optim']['base_wd'] 
                    if log_lr_group_idx['base'] is None:
                        log_lr_group_idx['base'] = len(params)
                    base_param_names.append(f'{module_name}.{module_param_name}')                    
                else:
                    if log_lr_group_idx['finetune'] is None:
                        log_lr_group_idx['finetune'] = len(params)
                finetune_param_names.append(f'{module_name}.{module_param_name}')
                                     
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
    
        logging.debug(f'List of Base Parameters: \n {base_param_names}')
        logging.debug(f'List of Finetune Parameters: \n {finetune_param_names}')
   
        return params, log_lr_group_idx

@register_model
def shadown_sota(configs, device):
    from .aux_mapper import AUXMapper_v1
    model = Shadow_SOTA(configs)
    model.to(device)
    params_group, log_lr_group_idx = Shadow_SOTA.get_optim_params_group(model=model, configs=configs)
    to_train_num_parameters = len([n for n, p in model.named_parameters() if p.requires_grad])
    assert len(params_group) == to_train_num_parameters, \
        f'parames_group设计出错, 有{len(to_train_num_parameters) - len(params_group)}个参数没有列在params_group里'
    optimizer = get_optimizer(params_group, configs)

    scheduler = build_scheduler(configs=configs, optimizer=optimizer)
    model_input_mapper = AUXMapper_v1(configs['model']['input_aux'])
    

    train_samplers, train_loaders, eval_function = build_schedule(configs, 
                                                                    model_input_mapper.mapper, 
                                                                    partial(model_input_mapper.collate, max_stride=model.max_stride))

    # dataset_specific initialization

    return model, optimizer, scheduler,  train_samplers, train_loaders, log_lr_group_idx, eval_function
