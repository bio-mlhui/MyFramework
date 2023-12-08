import math
import yaml
import os
from decimal import Decimal

group_name = 'words'
all_configs = {
    'algo_name': 'RVOS',
    'mode': 'train',
    'enable_amp': False, # no used when using deformable

    'data':{
        'name': 'youtube_rvos',  # a2dds_rvos, jhmdb_rvos, youtube_rvos, davis_rvos
        'dataset_root': '/home/xhh/datasets/youtube_rvos',
        'window_size': 5,

        'eval_batch_size': 1,
        'batch_size': 4,
        'num_workers': 16,
        'augmentation':{
            'name':'hflip',
            # augmentations and resize
            'resize_and_crop_augmentations': False,
            'horizontal_flip_augmentations':True,
            'train_max_size': 640,
            'transform_normalize':True,
            'resize':[640, 384],
        }
    },
    
    'model': {
        'name': 'unimodal_encoder',
        'checkpoint_path': '',  

        'do_refer':True,
        'do_mvm':True,
        'do_mlm':False,
        'weight_dict':{
            'loss_refer': 2., 
            # 'loss_classes': 2.,
            'loss_mask': 5,
            'loss_dice': 5,
            # 'loss_contrastive': 5,
            # 'loss_matching': 3,
            'loss_mlm': 5.,
            'loss_mvm': 0.,
        },
        
        'mode_name': 'joint',

        'mask_sampling': 'random',
        'clip_configs': {

        },

        'd_model': 256,

        # entrypoint configs
        'video_encoder':{
            'name': 'video_encoder_mvm',
            "mvmhead_configs":{
            },       
            'video_backbone':{
                'name': 'video_swin_t',  # video_resnet
                'pretrained': True,
                'pretrained_path': '/home/xhh/pt/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth',
                'train': False,
                'running_mode': 'train',
                # mask probabilities
            },
            # 'scale_before_fuse_configs':{
            #     'name':'deform_video_2d',
            #     'nheads':8,
            #     'npoints':4,
            #     'nlayers':3,
            # },
            'scale_before_fuse_configs':{
                'name':'no_scale_fusion',
            } 
        },
        
        # entrypoint configs
        'text_encoder':{
            'name': 'text_encoder_mlm',
         
            'mlmhead_configs':{
                'name': 'pretrain_roberta_base_mlmhead', #tie_pretrain_roberta-base
                'freeze_mlmhead':True,
                'proj_dropout': 0.1,
                'proj_do_ln': True,
            },
                       
            'text_backbone':{
                'name':'pretrain_roberta_base',
                'freeze_encoder': True,
                # mask probabilities
                'mask_probability': 0.3,
                'mask_onlymask': True,
                'proj_dropout': 0.1,
                'proj_do_ln': True,
            },


        },
        
        # 
        'fusion_encoder':{
            'name': 'videotextfeats',

            'nqueries': 50,
            'query_feat': 'learn',
            
            'mmattn_encoder':{
                'attention_layer_name': 'videotext_seperate_cross',
                'nhead':8,
                'num_layers': 1,
            },

            'scale_encoder':{
                'name':'deform_video_2d_with_fpn4',
                'd_ffn': 2048,
                'dropout':0.1,
                'activation': 'relu',
                'nlevels':4,
                'nheads': 8,
                'npoints':4,
                'nlayers': 6,
            },

            'decoder':{
                'name': 'mask2former_video_refer', 
                'nheads': 8,
                'dff': 2048,
                'nlayers': 9,
                'pre_norm':False,
                'mask_dim': 256,
                'enforce_proj_input': False,
                'num_feature_levels': 3,
                'aux_loss': True,
                'concate_text': False,

                'matching':{
                    'name': 'refer_video',
                    'losses': ['masks', 'refer'],
                    'matching_costs': {'refer': 2, 'mask': 5, 'dice': 5},
                    'eos_coef':0.1,
                    'mask_out_stride':4,
                    'mask_is_point':False,
                    'num_points': 4,
                },

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
                'name': 'MultiStepR',
                'gamma': 0.1,
                'milestones': [70,100],
                'verbose': True,
            },
        },

    },

}

# {"_multiple{}".format(all_configs["model"]["num_instances"]) if all_configs["model"]["num_instances"] > 1 else ""}\
config = f'youtube\
_refer[{all_configs["model"]["main"]["lambda_refer"]}]\
mlm[{all_configs["model"]["main"]["lambda_mlm"]}]\
_fs[{all_configs["model"]["main"]["fusion_module"]["abbr"]}]'

all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






