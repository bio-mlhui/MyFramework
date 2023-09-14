
trainer_configs = { 
                   # out_dir
                   # trainer_ckpt
                   # wandb
                   # data_dir
                   # pt_dir
    "data":{
        "num_workers": 6,
        "validate_batch_size": 1,
        "name": "a2ds_schedule",
        
        # 但凡有一个不同, 就是一个不同的run
        # 训练样本的生成, 测试样本的生成
        "generate_trainvalidate_params":{
            "name":"61m61m",
            "train_window_size": 6,   # Union[int, None]
            "validate_window_size": 6, 
            "train_window_step": 1,
            "validate_window_step":1,
            "train_filter":"middle",
            "validate_filter": "middle"
        },
       
        
        # 训练时额外的数据    
        "amr_are_used": True,
        "text_aux_version": 1,
        "video_aux_version": 0,

        # 训练数据增强, 测试数据增强
        "train_augmentation":{
            "name":"hflip_fixsize", 
            "fixsize": [576, 320],
        },
        "validate_augmentation": { # 最多有resize
            "name": "fixsize",
            "fixsize": [576, 320]
        },
        #  "train_augmentation":{
        #     "name":"hflip_resize",  # 不应该使用crop
        #     "train_max_size": 640,
        #     "scales": [288, 320, 352, 392, 416],
        #     "crop_range": [384, 576],
        # },
        
        # 训练时的SGD的loading
        "training_seed": 2023,  # 每次epoch运行的seed 是seed + epoch
        "train_batch_size": 2,
        
        "validate_metrics": ["pFpE_mAP_Pat_IOU"]
    },
    "model": {
        "name": "amr_v0_detObjRefChoose",
        "d_model":256,
        "max_stride": 64,
        "swint_pretrained_path":"pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth",
        "swint_freeze":True,
        "swint_runnning_mode":"train",
        "video_projs" : [
            {"name": "conv2d", "in_channels": 96,  "out_channels": 256, "kernel_size": 3, "padding":1, "bias":True,},
            {"name": "conv2d", "in_channels": 192, "out_channels": 256, "kernel_size": 1, "bias":True,},
            {"name": "conv2d", "in_channels": 384, "out_channels": 256, "kernel_size": 1, "bias":True,},
            {"name": "conv2d", "in_channels": 768, "out_channels": 256, "kernel_size": 1, "bias":True,},
            {"name": "conv2d", "in_channels": 768, "out_channels": 256, "kernel_size": 3, "stride":2, "padding": 1, \
                "bias":True,}],
        "video_feat_scales":[[1,4],[1,8],[1,16],[1,32], [1,64]],

        # amrtext
        "amrbart_wordEmbedding_freeze":True,
        "amrtext_wordEmbedding_proj" : {
            "name": "FeatureResizer",
            "input_feat_size": 1024,
            "output_feat_size": 256,
            "dropout":0.1,
            "do_ln":True},
        
        "fusion":{
            "name": "VisionLanguageFusionModule",
            "d_model":256,
            "nhead": 8,
            "dropout":0.,
            "amr_cross":"所有"},
        "parsing_encoder":{
            "name":"deform_video_2d_fpn",
            "d_model": 256,
            "d_ffn": 2048,
            "dropout":0.,
            "activation": "relu",
            "nheads": 8,
            "fused_scales":[[1,8],[1,16],[1,32],[1,64]],
            "fpn_strides": [[1,4],[1,8]],
            "npoints":4,
            "nlayers": 6,},
    
        "loss_weight":{'objdecoder_mask': 0,
                        'objdecoder_dice': 0,
                        'objdecoder_class': 0,
                        'objdecoder_giou': 0,
                        'objdecoder_bbox': 0,
                        'refdecoder_choose':2}, 
        "tasks" : {"refdecoder_refseg": {"layer_weights": {-1:1., 0:1., 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1.,},
                                        },
                    'objdecoder_objseg': {'layer_weights': { -1: 1,0: 1,1: 1,2: 1,3: 1,4: 1,5: 1,6: 1,7: 1,8: 1 },
                                    'class_weight': [1, 1, 1, 1, 1, 1, 1, 0.1],
                                    'matching_costs': { 'class': 2,'mask': 5,'dice': 5,'bbox': 0,'giou': 0 }}},
        "refdecoder":{
            "nlayers": 9,
            "amr_cross_video_layer":{
                "name": "cross_attention",
                "d_model": 256,
                "nhead": 8,
                "dropout": 0.,
                "amr_cross": ["所有","所有","所有","所有","所有","所有","所有","所有","所有",],
            },
            "amr_self_layer":{
                "name": "graph_layer_inferfullstep",
                "d_model": 256,
                "flow": "source_to_target",
                "aggr": "min"
            },
            'ffn_layer':{
                'name': 'ffn',
                'd_model': 256,
            },
            "used_scales": [[1,32],[1,16],[1,8]],
            "conved_scale": [1,4],
            "choose_who": "第一个",
            "mask_out_stride": 4,
            "mask_threshold": 0.5,
    },
        "objdecoder":{ 
                'num_classes': 7,
                'nqueries': 100,
                'nlayers': 9,
                'cross_layer':{
                    'name': 'cross_attention',
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'self_layer':{
                    'name': 'self_attention',
                    'd_model': 256,
                    'd_model': 256,
                    'nhead': 8,
                    'dropout': 0.,
                },
                'ffn_layer':{
                    'name': 'ffn',
                    'd_model': 256,
                },
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'mask_out_stride': 4,
                'mask_threshold': 0.5,
                    },
        "optimization":{
            "optimizer":{
                "name": "AdamW",
                "wd": 5e-4,
                "lr": 1e-4,
            },
            "vid_backbone_lr" : 5e-5,
            "text_backbone_lr": 5e-5,
            "clip_max_norm": 0.1,
            "epochs": 100,
            "scheduler": {
                "name": "MultiStepLR",
                "gamma": 0.1,
                "milestones": [15, 18],
            },
            "enable_amp": False, # no used when using deformable
        },
    },
    

    # -1是trainset visual使用的seed
    "visualize": {
        "trainset_idxs": list(range(100)),
        "validateset_idxs": list(range(200)),
    }
    
}



