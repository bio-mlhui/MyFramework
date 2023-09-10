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
        'num_workers': 8,
        'max_stride': 16, # padding a batch 
        'eval_batch_size': 1,
        'amr_are_used': False,
        'train_window_size': 6,
        'batch_size': 4,
        'augmentation':{
            'name':'hflip_fixsize', 
            'fixsize':[224, 224],
        },
        'version': 0,
    },
    
    'model': {
        'name': 'clip_text_detectObj',
        'checkpoint_path': '',  
        
        'd_model': 512,
        'object_classes': ['adult', 'baby', 'ball', 'bird', 'car', 'cat', 'dog'],
        'weight_dict':{
            'loss_object_mask': 5.,
            'loss_object_dice': 5.,
            'loss_object_ce': 2,
            'loss_mask': 5.,
            'loss_dice': 5.,
            'loss_refer': 2,
        },

        # entrypoint configs
        
        'object_decoder':{
            'name': 'object_detector',
            'num_classes': 7, # ytvos, davis, coco /65/78/91
            'nheads': 8,
            'dff': 2048,
            'mask_dim': 512,
            'pre_norm':False,
            'enforce_proj_input': False, 
            # important
            'num_queries': 100,
            'nlayers': 3,
            'used_scales': [[1,32]],
            'conved_scale': [1,32],
            'matching':{
                'name': 'allinstance_video',
                'losses': ['masks', 'labels'],
                'matching_costs': {'dice': 5, 'mask': 5, 'ce': 2},
                'eos': 0.1,
                'mask_out_stride':32,
                'mask_is_point': False,
            },
            'aux_loss': True,
        },
        
        'referent_decoder':{
            'name': 'referent_decoder_forSequenceText',
            'mask_dim': 512,
            'nheads': 8,
            'pre_norm':False,
            'enforce_proj_input': False, 
            'dff': 2048,
            # important
            'nqueries': 5,
            'nlayers': 3,
            'used_scales': [[1,32]],
            'conved_scale': [1,32],
            'matching':{
                'name': 'refer_video',
                'mask_out_stride':32,
                'losses': ['masks', 'refer'],
                'matching_costs': {'refer': 2, 'mask':5, 'dice': 5},
                'mask_is_point':False,
                'num_points': 4,
                'eos_coef': 0.1,
            },
            'aux_loss': True,    
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

config = f'text_clip_detecObj'


all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






