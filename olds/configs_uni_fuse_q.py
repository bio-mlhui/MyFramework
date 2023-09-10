import math
import yaml
import os
from decimal import Decimal

group_name = 'youtube_rvos'
all_configs = {
    'algo_name': 'RVOS',
    'mode': 'train',
    'enable_amp': False, # no used when using deformable

    'data':{
        'name': 'youtube_rvos',  # a2dds_rvos, jhmdb_rvos, youtube_rvos, davis_rvos
        'entrypoint': 'youtube_rvos_perExp',
        'dataset_root': '/home/xhh/datasets/youtube_rvos',
        'num_workers': 16,
        'max_stride': 16, # padding a batch 
        'eval_batch_size': 1,
        'fix_spelling': False,

        'train_window_size': 5,
        'batch_size': 2,
        'augmentation':{
            'name':'hflip_fixsize',
            'fixsize':[640, 384],
        }
    },
    
    'model': {
        'name': 'unimodal_encoder',
        'checkpoint_path': '',  

        'do_refer':True,
        'do_mvm':False,
        'do_mlm':True,
        'weight_dict':{
            'loss_refer': 2., 
            # 'loss_classes': 2.,
            'loss_mask': 5.,
            'loss_dice': 5.,
            # 'loss_contrastive': 5,
            # 'loss_matching': 3,
            'loss_mlm': 5.,
            # 'loss_mvm': 0.,
        },
        
        'mode_name': 'joint',

        'mask_sampling': 'random',
        'clip_configs': {},

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
                'out_scale_strides':[[1,4],[1,8],[1,16],[1,32], [1,64]],
                'each_proj_types': ['local', 'linear', 'linear', 'linear', 'local'],
            },
            "mvmhead_configs":{
            },    
            'scale_before_fuse_configs':{
                'name':'no_scale_fusion',
            } 
        },
        
        # entrypoint configs
        'text_encoder':{
            'name': 'text_encoder_mlm',
                       
            'text_backbone':{
                'name':'pretrain_roberta_base',
                # important
                'freeze_encoder': True,
                # mask probabilities
                'mask_probability': 0.3,
                'mask_onlymask': True,
            },
            'proj':{
                'name': 'resizer',
                'dropout': 0.1,
                'do_ln': True,
            },

            'mlmhead_configs':{
                'name': 'pretrain_roberta_base_mlmhead', #tie_pretrain_roberta-base
                # multiple layers of FeatureResizer
                'proj_dropout': 0.1,
                'proj_do_ln': True,
                # important
                'freeze_mlmhead':True,
            },
        },
        # 
        'fusion_encoder':{
            'name': 'videotextfeats',
            # important
            'nqueries': 50,
            'query_feat': 'learn',
            'fused_scale': [1, 32],
            'task_conditioning_form': 'attn_mask',
            
            'mmattn_encoder':{
                'name': 'videotext_seperate_cross',
                'nhead':8,
                'dropout':0.0,
                # important
                'num_layers': 1,
            },

            'scale_encoder':{
                'name':'deform_video_2d_fpn',
                'd_ffn': 2048,
                'dropout':0.1,
                'activation': 'relu',
                'nheads': 8,
                # important
                'fused_scales':[[1,8],[1,16],[1,32], [1,64]],
                'fpn_strides': [[1,4],[1,8]],
                'npoints':4,
                'nlayers': 6,
            },

            'decoder':{
                'name': 'mask2former_video_refer',
                'pre_norm':False,
                'mask_dim': 256,
                'enforce_proj_input': False, 
                'nheads': 8,
                'dff': 2048,
                # important
                'concate_text': False,
                'nlayers': 9,
                'used_scales': [[1,32],[1,16],[1,8]],
                'conved_scale': [1,4],
                'matching':{
                    'name': 'refer_video',
                    'eos_coef':0.1,
                    'mask_out_stride':4,
                    # important
                    'losses': ['masks', 'refer'],
                    'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5},
                    'mask_is_point':False,
                    'num_points': 4,
                },
                'aux_loss': True,

            },
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

config = f'[uni_fuse_q][joint_refer255_mlm5_mvm0]'

if not all_configs['model']['do_mlm']:
    assert all_configs['model']['text_encoder']['mlmhead_configs'] == {}

if not all_configs['model']['do_mvm']:
    assert all_configs['model']['video_encoder']['mvmhead_configs'] == {}

all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






