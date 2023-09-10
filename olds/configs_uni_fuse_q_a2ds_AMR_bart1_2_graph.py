import math
import yaml
import os
from decimal import Decimal

group_name = 'a2ds_rvos'
all_configs = {
    'algo_name': 'RVOS',
    'enable_amp': False, # no used when using deformable
    'data':{
        'name': 'a2ds_rvos',  # a2dds_rvos, jhmdb_rvos, youtube_rvos, davis_rvos
        'entrypoint': 'a2ds_rvos_perExp',
        'dataset_root': '/home/xhh/datasets/a2d_sentences',
        'gt_path':'/home/xhh/datasets/a2d_sentences/a2d_sentences_test_annotations_in_coco_format.json',
        'num_workers': 6,
        'max_stride': 16, # padding a batch 
        'eval_batch_size': 1,
        'amr_are_used': True,
        'train_window_size': 6,
        'batch_size': 1,
        'augmentation':{
            'name':'fixsize', 
            'fixsize':[576, 320],
        },
    },
    
    'model': {
        'name': 'unimodal_video_graphDecoder',
        'checkpoint_path': '',  

        'weight_dict':{
            'loss_mask': 5.,
            'loss_dice': 5.,
        },

        'd_model': 256,

        # entrypoint configs
        'video_encoder':{
            'name': 'video_encoder_mvm',   
            'video_backbone':{
                'name': 'video_swin_t',  # video_resnet
                'running_mode': 'train',
                'pretrained_path': '/home/xhh/pt/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                # important
                'train': False,
                'pretrained': True,
                # mask probabilities
            },
            "proj":{
                'name': 'conv2d',
                'local_kernel_size':3,
                # important
                'bias':True,
                'norm':'group_32',
                'out_scale_strides':[[1,4],[1,8],[1,16],[1,32]],
                'each_proj_types': ['local', 'linear', 'linear', 'linear'],
            },
            "mvmhead_configs":{
            },    
            
            'scale_before_fuse_configs':{
                'name':'no_scale_fusion',
            },
        },
        
        # entrypoint configs
        'text_encoder':{
            'name': 'inputpenmangraph_wordembeddingencoder_outputgraph',
            'how_to_get_word_embedding': 'amr_encoder amr_decoder',
        },
        # 
        'fusion_encoder':{
            'name': 'video_textgraph',
            'fusion_strategy': 0,
            'scale_encoder':{
                'name':'deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.1,
                'activation': 'relu',
                'nheads': 8,
                # important
                'fused_scales':[[1,8],[1,16],[1,32]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'nlayers': 6,
            },
        },
        'decoder':{
            'name': 'mask2_video_graph_cross_multiscale',
            
            'cross_nheads': 8,
            'cross_normalize_before': False,
            'cross_dropout': 0.0,
            'cross_activation': 'relu',
            'mask_dim': 256,
            'pre_norm':False,
            'enforce_proj_input': False, 
            
            'add_ffn_layers_after_self_attn': False,
            'ffn_dim_ffd': 2048,
            'ffn_activation': 'relu',
            'ffn_dropout': 0.0,
            'ffn_normalize_before': False,
            # important
            'nlayers': 9,
            'used_scales': [[1,32],[1,16],[1,8]],
            'conved_scale': [1,4],
            'matching':{
                'name': 'no_matching_video',
                'mask_out_stride':4,
                'losses': ['masks'],
            },
            'aux_loss': True,
            'graph_which_to_cross_strategy': '0层只有concept nodes, 之后所有nodes都用, edge不变',
            'graph_layer':{
                'name': 'graph_layer_v2',
                'reduce': 'min'
            },
            'share_graph_layers': False
        },
        'optimization':{
            'optimizer':{
                'name': 'AdamW',
                'wd': 5e-4,
                'lr': 1e-4,
            },
            'vid_backbone_lr' : 5e-5,
            'text_backbone_lr': 5e-5,
            
            'clip_max_norm': 0.1,
            'epochs': 100,
            'scheduler': {
                'name': 'MultiStepLR',
                'gamma': 0.1,
                'milestones': [30, 35],
                'verbose': True,
            },
        },

    },

}

config = f'[uni_fuse_q][finetue_refer255_mlm0_mvm0][AMR]_bart1_2_graph_EarlyFusion'


all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






