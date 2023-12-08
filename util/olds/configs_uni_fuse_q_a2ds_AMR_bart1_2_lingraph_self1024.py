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
        'num_workers': 16,
        'max_stride': 16, # padding a batch 
        'eval_batch_size': 4,
        'fix_spelling': False,
        'amr_are_used': True,

        'train_window_size': 5,
        'batch_size': 4,
        'augmentation':{
            'name':'fixsize', 
            'fixsize':[576, 320],
        },
        'file_postfix': '_change_root'
    },
    
    'model': {
        'name': 'unimodal_video_graphencoder',
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
        
        # entrypoint configs
        'text_encoder':{
            'name': 'bart_amr2text_seq2seq_graphencoder_only_vocab',
            'freeze_vocabulary':False,
            
        },
        # 
        'fusion_encoder':{
            'name': 'nofusion',
        },
        'decoder':{
            'name': 'mask2former_video_lingraph_Self1024',
            
            'mask_dim': 256,
            'nheads': 8,
            'pre_norm':False,
            'enforce_proj_input': False, 
            # important
            'used_scales': [[1,32],[1,16],[1,8]],
            'conved_scale': [1,4],
            'matching':{
                'name': 'no_matching_video',
                'mask_out_stride':4,
                'losses': ['masks'],
                'matching_costs': {'refer': 2, 'mask': 5,},
            },
            'aux_loss': True,
            'graph_which_to_cross': '只有node', # 只有node, 整个linearized graph, 只有node和edge
            'proj':{
                'name': 'linear',
            },
            'freeze_self_attention_layers': True
            
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

config = f'[uni_fuse_q][finetue_refer255_mlm0_mvm0][AMR]_bart1_2_lingraph_self1024'


all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






