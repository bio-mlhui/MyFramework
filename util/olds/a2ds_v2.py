import torch.utils.data as torch_data
import torch

from datasets.registry import register_dataset
import os
import pandas
import h5py
from glob import glob
import json
import numpy as np
import torchvision.io as video_io
from pycocotools.mask import encode, area
import torchvision.transforms.functional as F
import torch.nn.functional as nn_F
import torch.distributed as dist
from util.misc import is_dist_avail_and_initialized
from .create_gt_in_coco_format import create_a2d_sentences_ground_truth_test_annotations
from .augmentation_videos import video_aug_entrypoints
from datasets.collate import Video_Collator
import torch.nn as nn
import pycocotools.mask as mask_util
from torch_geometric.data import Data
__all__ = ['a2ds_rvos_perExp']


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id

def get_class_label(idx):
        """idx是np.float64, 想要获得他的label
        """
        return int(str(int(idx))[0]) - 1

def get_action_label(idx):
        """idx是np.float64, 想要获得他的label
        """
        return int(str(int(idx))[1]) - 1
import re
import penman
class A2DS_RVOS_PerExp(torch_data.Dataset): 
    """
    For a (video, object index, text) pair, 根据NUM_frames和标注情况(A2ds的每个视频只标注了3帧), 生成一个sample
    """   
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def __init__(self, 
                 transforms,
                 collate_cls,
                 root, 
                 train,
                 window_size, 
                 dataset_coco_gt_format_path,) -> None:
        """
        amr, exist_queries都是 auxiliary,不管model是否使用, 都在a2ds_train/test.json文件里.
        """
        super().__init__()
        self.transforms = transforms
        
        
        self.root = root
        self.train = train
        self.window_size = window_size
        file_path = os.path.join(root, f"a2ds_{'train' if train else 'test'}.json")
        # file_path = os.path.join(root, f"a2ds_{'train' if train else 'test'}_change_root_filterAli.json")
        assert os.path.exists(file_path)
        with open(os.path.join(root, file_path), 'r') as f:
             self.instances_by_frame = json.load(f)
       
        self.video_dir = os.path.join(root, f'Release/clips320H')
        self.mask_dir = os.path.join(root, f'text_annotations/a2d_annotation_with_instances')
        distributed = is_dist_avail_and_initialized()
        if (not train) and (not os.path.exists(dataset_coco_gt_format_path)):
            if (distributed and dist.get_rank() == 0) or not distributed:
                create_a2d_sentences_ground_truth_test_annotations()
            if distributed:
                dist.barrier()
                
        from models.amr_utils.tokenization_bart import AMRBartTokenizer
        self.amr_tokenizer : AMRBartTokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART_pretrain')
        self.collator = collate_cls(self.amr_tokenizer.pad_token_id)
        
    def __getitem__(self, idx):
        video_id, frame_idx, instance_id, text_query, exist_queries, amr_tree_string,\
            first_noun, amr_tree_string_linearization_dict  = \
            self.instances_by_frame[idx]['video_id'], \
            self.instances_by_frame[idx]['frame_idx'],\
            self.instances_by_frame[idx]['instance_id'],\
            self.instances_by_frame[idx]['text_query'],\
            self.instances_by_frame[idx]['exist_queries'], \
            self.instances_by_frame[idx]['amr_tree_string'] ,\
            self.instances_by_frame[idx]['first_noun'],\
            self.instances_by_frame[idx]['amr_tree_string_linearization_dict']
            
        text_query = " ".join(text_query.lower().split())
        exist_queries = [q.join(q.lower().split()) for q in exist_queries]
        
        # the frames are pillow images
        vframes, _, _ = video_io.read_video(
            filename=os.path.join(self.video_dir, f'{video_id}.mp4'),
            pts_unit='sec',
            output_format='TCHW'
        )
        idx = np.arange(frame_idx - self.window_size//2, frame_idx + (self.window_size-1)//2 + 1) - 1
        idx = idx.clip(0, len(vframes)).tolist()
        vframes = vframes[idx]
        vframes = [F.to_pil_image(frame) for frame in vframes]

        frame_annotation = h5py.File(os.path.join(self.mask_dir, video_id,
                                     f'{frame_idx:05d}.h5'))
        # the masks are torch tensors 
        masks = torch.from_numpy(np.array(frame_annotation['reMask'])).transpose(-1, -2)
        # 有些帧只有一个instance
        masks = masks.unsqueeze(dim=0) if masks.dim() == 2 else masks
        
        class_ids = [get_class_label(idx) for idx in frame_annotation['id'][0]] # 0-6
        action_ids = [get_action_label(idx) for idx in frame_annotation['id'][0]]
        
        boxes = []
        valids = []
        for object_mask in masks:
            if (object_mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(object_mask.numpy())
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valids.append(1)
                boxes.append(box)
            else:
                box = torch.tensor([0,0,0,0]).to(torch.float)
                valids.append(0)
                boxes.append(box)
        boxes = torch.stack(boxes, dim=0) # n h w
        valids = torch.tensor(valids,dtype=torch.int32)
    
        instances = list(frame_annotation['instance'])
        referred_idx = instances.index(instance_id)
        
        assert len(masks) >= len(exist_queries) 
        assert len(exist_queries) == len(instances)
        
        frame_annotation.close()
        labels = torch.tensor(class_ids) # T(n, ) 0-6
        
        targets = self.window_size * [None]
        targets[self.window_size//2] =  {
            'masks': masks, # T(n h w)
            'orig_size': masks.shape[-2:],  # original frame shape without any augmentations
            'size': masks.shape[-2:], # changed at RandomResize transform
            'referred_instance_idx': referred_idx,  # idx in 'masks' of the text referred instance
            'iscrowd': torch.zeros(len(masks)),  # for compatibility with DETR COCO transforms
            'image_id': get_image_id(video_id, frame_idx, instance_id),
            'text_query': text_query, # str
            'exist_queries': exist_queries, # list[str]
            'valid': valids, # n
            'boxes': boxes, # n 4
            'labels': labels # n
        }
        vframes, targets = self.transforms(vframes, targets)
        text_query = targets[self.window_size//2].pop('text_query')
        exist_queries = targets[self.window_size//2].pop('exist_queries')
        

        
        return vframes, targets, text_query, graphs, seg_ids, tokens_ids, token_splits, {
            'exist_queries': exist_queries,
            'amr_tree_string': amr_tree_string,
            'first_noun': first_noun,
            'amr_tree_string_linearization_dict': amr_tree_string_linearization_dict,
        }
        # treeObj
    
    def __len__(self):
        return len(self.instances_by_frame)

        
class A2DSentencesPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self):
        super(A2DSentencesPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size, resized_sample_sizes, orig_sample_sizes):
        """ Perform the computation
        Parameters:
            outputs:
                pred_masks: t_target*b num_queries H/4 W/4
                pred_is_referred: t_target*b num_queries 2
            resized_padded_sample_size: (H, W) after nested_tensor, in a batch
            resized_sample_sizes: list[(Resized_H Resized_W)], t_valid*b
            orig_sample_sizes: list[Original_H, Original_W], t_valid*b
        """
        # t_target*b num_queries 2(positive, negative)
        pred_is_referred = outputs['pred_is_referred']
        if pred_is_referred.shape[-1] == 2:
            prob = nn_F.softmax(pred_is_referred, dim=-1)
            # t_target*b num_queries
            scores = prob[..., 0]
        else:
            prob = pred_is_referred.sigmoid() # tb nq 1
            scores = prob.squeeze(-1) # tb nq
        
        # t_target*b num_queries H W
        pred_masks = outputs['pred_masks']
        pred_masks = nn_F.interpolate(pred_masks, size=resized_padded_sample_size, mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        
        
        # t_target*b
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  
            
            # crop: n H W --> n 1 Resized_H Resized_W
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  
            # resize: n 1 Resized_H Resized_W --> n 1 Original_H, Original_W
            f_pred_masks_processed = nn_F.interpolate(f_pred_masks_no_pad.float(), size=orig_size, mode="nearest-exact")
            
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        # t_target*b
        predictions = [{'scores': s, # postive scores: (num_queries,)
                        'masks': m,  # num_queries Original_H Original W
                        'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions

@register_dataset
def a2ds_rvos_perExp(data_configs):
    configs = vars(data_configs)
    #TODO: 如果要实现支持text augmentation, 必须把amr parser加到模型里
    amr_are_used = configs['amr_are_used'] if "amr_are_used" in configs else False
    if amr_are_used:
        assert configs['augmentation'].name == 'fixsize'
        
    create_aug = video_aug_entrypoints(data_configs.augmentation.name)
    train_aug, eval_aug = create_aug(data_configs.augmentation)

    train_dataset = A2DS_RVOS_PerExp(transforms=train_aug,
                         collate_cls=Video_Collator,
                         root=configs['dataset_root'],
                         train=True,
                         window_size=configs['train_window_size'],
                         dataset_coco_gt_format_path=configs['gt_path'])
    
    test_dataset = A2DS_RVOS_PerExp(transforms=eval_aug,
                         collate_cls=Video_Collator,
                         root=configs['dataset_root'],
                         train=False,
                         window_size=configs['train_window_size'],
                         dataset_coco_gt_format_path=configs['gt_path'],)
    
    postprocessor = A2DSentencesPostProcess()
    def my_dataset_function():
        return [{}]
    from detectron2.data import DatasetCatalog
    DatasetCatalog.register('a2ds_rvos', my_dataset_function)
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('a2ds_rvos').thing_classes = ['r', 'nr']
    MetadataCatalog.get('a2ds_rvos').thing_colors = [(255., 140., 0.), (0., 255., 0.)]

    return train_dataset, test_dataset, postprocessor




            
        
    