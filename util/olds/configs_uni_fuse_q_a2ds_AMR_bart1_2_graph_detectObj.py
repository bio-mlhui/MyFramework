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
        'batch_size': 4,
        'augmentation':{
            'name':'fixsize', 
            'fixsize':[576, 320],
        },
    },
    
    'model': {
        'name': 'two_decoders_v2',
        'checkpoint_path': '',  
        'object_classes': [4194, 1928, 1011, 5103, 512, 4758, 2335],  # Ġadult, Ġbaby, Ġball, Ġbird, Ġcar, Ġcat, Ġdog
        'weight_dict':{
            'loss_object_mask': 5.,
            'loss_object_dice': 5.,
            'loss_object_ce': 2,
            'loss_object_token': 5,
            'loss_mask': 5.,
            'loss_dice': 5.,
            # loss token classification
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
                'name': 'no_scale_fusion'
            },
        },
        
        'object_decoder':{
            'name': 'object_detector',
            'num_classes': 7, # ytvos, davis, coco /65/78/91
            'nheads': 8,
            'dff': 2048,
            'mask_dim': 256,
            'pre_norm':False,
            'enforce_proj_input': False, 
            # important
            'num_queries': 100,
            'nlayers': 9,
            'used_scales': [[1,32],[1,16],[1,8]],
            'conved_scale': [1,4],
            'matching':{
                'name': 'allinstance_video',
                'losses': ['masks', 'labels', 'token'],
                'matching_costs': {'dice': 5, 'mask': 5, 'ce': 2},
                'eos': 0.1,
                'mask_out_stride':4,
                'mask_is_point': False,
            },
            'aux_loss': True,
            'video_feat_proj':{
                'name': 'no_proj',
            },
            'scale_encoder':{
                'name':'no_scale_fusion',
            }
            
        },
        
        # entrypoint configs
        'text_encoder':{
            'name': 'inputpenmangraph_wordembeddingencoder_outputgraph',
            'how_to_get_tokk_embed': {
                'name': 1,
                'fuse_subtoken': 'mean',
            },
            'from_scratch': False,
            'freeze_vocabulary': False,
        },
        # 
        'fusion_encoder':{
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
        
        'referent_decoder':{
            'name': 'referent_decoder',
            
            'mask_dim': 256,
            'nheads': 8,
            'pre_norm':False,
            'enforce_proj_input': False, 
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
            # edges, nodes都去做cross attention
            # 0: 只有concept/constant nodes
            # 
            'graph_which_to_cross_strategy': '0层只有concept nodes, 之后所有nodes都用, edge不变', 
            'graph_layer':{
                'name': 'graph_layer_v1',
                'flow': 'source_to_target',
                'aggr': 'min'
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

config = f'[uni_fuse_q][finetue_refer255_[somuch]_graph_srcxtgt_detectObjv5'


all_configs['command_line'] = f'python main.py --config {config} --group_name {group_name}'

with open(os.path.join(f'./configs_{group_name}', f'{config}.yaml'), 'w') as file:
    yaml.dump(all_configs, file, width=2000)






