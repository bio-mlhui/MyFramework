import json
import torch
import torch.utils.data as torch_data
import torch.distributed as dist
from os import path
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image
import numpy as np
from einops import rearrange
import copy
from datasets.registry import register_dataset
from .augmentation_videos import video_aug_entrypoints

from util.misc import is_dist_avail_and_initialized

import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from detectron2.data import DatasetCatalog

__all__ = ['youtube_rvos_perExp']

all_categories = {'truck': 0, 'leopard': 1, 'knife': 2, 'sedan': 3, 'parrot': 4, 'snail': 5, 'snowboard': 6, 'sign': 7, 'rabbit': 8, 'bus': 9, 'crocodile': 10, 'penguin': 11, 'skateboard': 12, 'eagle': 13, 'dolphin': 14, 'lion': 15, 'others': 16, 'surfboard': 17, 'earless_seal': 18, 'cow': 19, 'hat': 20, 'lizard': 21, 'duck': 22, 'dog': 23, 'hedgehog': 24, 'zebra': 25, 'bird': 26, 'turtle': 27, 'hand': 28, 'elephant': 29, 'motorbike': 30, 'bike': 31, 'bear': 32, 'tiger': 33, 'ape': 34, 'fish': 35, 'deer': 36, 'frisbee': 37, 'snake': 38, 'horse': 39, 'bucket': 40, 'train': 41, 'owl': 42, 'parachute': 43, 'monkey': 44, 'airplane': 45, 'person': 46, 'paddle': 47, 'plant': 48, 'mouse': 49, 'camel': 50, 'shark': 51, 'raccoon': 52, 'squirrel': 53, 'giraffe': 54, 'boat': 55, 'sheep': 56, 'giant_panda': 57, 'whale': 58, 'tennis_racket': 59, 'toilet': 60, 'umbrella': 61, 'frog': 62, 'cat': 63, 'fox': 64}

class ReferYoutubeVOSPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """
    def __init__(self):
        super(ReferYoutubeVOSPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, videos_metadata, samples_shape_with_padding):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            videos_metadata: a dictionary with each video's metadata.
            samples_shape_with_padding: size of the batch frames with padding.
        """
        pred_is_referred = outputs['pred_is_referred']  # t b nq 2
        if pred_is_referred.shape[-1] == 2:
            prob_is_referred = F.softmax(pred_is_referred, dim=-1)
            # note we average on the temporal dim to compute score per trajectory:
            trajectory_scores = prob_is_referred[..., 0].mean(dim=0)  # b nq
        else:
            raise ValueError() #prob_is_referred = pred_is_referred.sigmoid()
            trajectory_scores = prob_is_referred.squeeze(-1).mean(dim=0)
        pred_trajectory_indices = torch.argmax(trajectory_scores, dim=-1)  # b
        pred_masks = rearrange(outputs['pred_masks'], 't b nq h w -> b t nq h w')
        # keep only the masks of the chosen trajectories:
        b = pred_masks.shape[0]
        pred_masks = pred_masks[torch.arange(b), :, pred_trajectory_indices] # b t h w
        # resize the predicted masks to the size of the model input (which might include padding)
        pred_masks = F.interpolate(pred_masks, size=samples_shape_with_padding, mode="bilinear", align_corners=False)
        # apply a threshold to create binary masks:
        pred_masks = (pred_masks.sigmoid() > 0.5)
        # remove the padding per video (as videos might have different resolutions and thus different padding):
        preds_by_video = []
        for video_pred_masks, video_metadata in zip(pred_masks, videos_metadata):  # each sample
            # size of the model input batch frames without padding:
            resized_h, resized_w = video_metadata['resized_frame_size']  # t h w
            video_pred_masks = video_pred_masks[:, :resized_h, :resized_w].unsqueeze(1)  # remove the padding
            # resize the masks back to their original frames dataset size for evaluation:
            original_frames_size = video_metadata['original_frame_size']  # t 1 h w
            video_pred_masks = F.interpolate(video_pred_masks.float(), size=original_frames_size, mode="nearest-exact")
            video_pred_masks = video_pred_masks.to(torch.uint8).cpu()
            # combine the predicted masks and the video metadata to create a final predictions dict:
            video_pred = {**video_metadata, **{'pred_masks': video_pred_masks}}
            preds_by_video.append(video_pred)
        return preds_by_video



class Youtube_RVOS_PerExpression(torch_data.Dataset):
    @staticmethod
    def generate_train_test_samples(root):
        videos = pandas.read_csv(os.path.join(root, 'Release/videoset.csv'), header=None)
        videos.columns = ['video_id', '', '', '', '', '','', '', 'split']
        with open(os.path.join(root, 'a2d_missed_videos.txt'), 'r') as f:
            unused_ids = f.read().splitlines()
        unused_ids.append('e7Kjy13woXg')
        videos = videos[~videos.video_id.isin(unused_ids)]
        
        for split, train_or_test, in { 0: 'train', 1:'test'}.items():
            samples = []            
            split_videos = videos[videos.split==split]
            for video_id in list(split_videos['video_id']): 
                video_frame_files = sorted(glob((os.path.join(root, f'a2d_annotation_with_instances/{video_id}', '*.h5'))))
                for frame_file in video_frame_files:
                    frame_idx = int(frame_file.split('/')[-1].split('.')[0])
                    f = h5py.File(frame_file)
                    appear_instances = [int(ins_id) for ins_id in f['instance']]
                    if video_id == 'EadxBPmQvtg' and frame_idx == 25:
                        assert appear_instances == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
                        appear_instances = appear_instances[:-1] # 0,1,....,9,1
                    assert len(torch.tensor(appear_instances).unique()) == len(appear_instances)
                    for appear_instance in appear_instances:
                        samples.append({
                            'video_id':video_id,
                            'frame_idx':frame_idx, 
                            'instance_id': appear_instance, 
                        })
            with open(os.path.join(root, f'a2ds_{train_or_test}_samples.json'), 'w') as f:
                json.dump(samples, f)
            if split == 0:
                assert len(samples) == 15741
            else:
                assert len(samples) == 3800
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the full
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """
    def __init__(self, 
                 transforms,
                 collator,
                 # configs
                 subset_type,
                 dataset_path,
                 window_size,
                 fix_spelling,
                 device=None):
        super().__init__()
        assert subset_type in ['train', 'test']
        if subset_type == 'test':
            subset_type = 'valid'  # Refer-Youtube-VOS is tested on its 'validation' subset (see description above)
        self.subset_type = subset_type
        self.window_size = window_size   # 测试集没有window size
        num_videos_by_subset = {'train': 3471, 'valid': 202}
        self.videos_dir = path.join(dataset_path, subset_type, 'JPEGImages')
        assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
            f'error: {subset_type} subset is missing one or more frame samples' # JPEG目录下的目录数量等于视频的个数
        
        if subset_type == 'train':
            self.mask_annotations_dir = path.join(dataset_path, subset_type, 'Annotations')  # only available for train
            assert len(glob(path.join(self.mask_annotations_dir, '*'))) == num_videos_by_subset[subset_type], \
                f'error: {subset_type} subset is missing one or more mask annotations'
        else:
            self.mask_annotations_dir = None
        self.device = device if device is not None else torch.device('cpu')
        # 得到所有的sample (#clips * #exps)
        self.samples_list = self.generate_samples_metadata(dataset_path, subset_type, window_size)
        
        self.transforms = transforms
        self.collator = collator
        if fix_spelling:
            self.fix_spell_model = pipeline("text2text-generation",model="/home/xhh/pt/spelling-correction-english-base")
        self.fix_spelling = fix_spelling

    def generate_samples_metadata(self, dataset_path, subset_type, window_size):
        is_distributed = is_dist_avail_and_initialized()
        if subset_type == 'train':
            metadata_file_path = f'{dataset_path}/train_samples_metadata_win_size_{window_size}_with_labels.json'
        else:  # validation
            metadata_file_path = f'{dataset_path}/valid_samples_metadata.json'
        if path.exists(metadata_file_path):
            print(f'loading {subset_type} subset samples metadata...')
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable = (is_distributed and dist.get_rank() != 0))]
                return samples_list
        elif (is_distributed and dist.get_rank() == 0) or (not is_distributed):
            print(f'creating {subset_type} subset samples metadata...')
            subset_expressions_file_path = path.join(dataset_path, 'meta_expressions', subset_type, 'meta_expressions.json')
            subset_labels_file_path = path.join(dataset_path, subset_type,'meta.json')
            with open(subset_expressions_file_path, 'r') as f:
                subset_expressions_by_video = json.load(f)['videos']
            with open(subset_labels_file_path, 'r') as f:
                subset_objects_by_video = json.load(f)['videos']

            if subset_type == 'train':
                # generate video samples in parallel (this is required in 'train' mode to avoid long processing times):
                vid_extra_params = (window_size, subset_type, self.mask_annotations_dir, self.device)
                params_by_vid = [(vid_id, vid_data, subset_objects_by_video[vid_id]['objects'], 
                                  *vid_extra_params) for vid_id, vid_data in subset_expressions_by_video.items()]
                n_jobs = min(multiprocessing.cpu_count(), 12)
                samples_lists = Parallel(n_jobs)(delayed(self.generate_train_video_samples)(*p) for p in tqdm(params_by_vid))
                # samples_lists = [self.generate_train_video_samples(*p) for p in tqdm(params_by_vid)]
                samples_list = [s for l in samples_lists for s in l]  # flatten the jobs results lists
            else:  # validation
                # for some reasons the competition's validation expressions dict contains both the validation & test
                # videos. so we simply load the test expressions dict and use it to filter out the test videos from
                # the validation expressions dict:
                test_expressions_file_path = path.join(dataset_path, 'meta_expressions', 'test', 'meta_expressions.json')
                with open(test_expressions_file_path, 'r') as f:
                    test_expressions_by_video = json.load(f)['videos']
                test_videos = set(test_expressions_by_video.keys())
                valid_plus_test_videos = set(subset_expressions_by_video.keys())
                valid_videos = valid_plus_test_videos - test_videos
                subset_expressions_by_video = {k: subset_expressions_by_video[k] for k in valid_videos}
                assert len(subset_expressions_by_video) == 202, 'error: incorrect number of validation expressions'

                samples_list = []
                for vid_id, data in tqdm(subset_expressions_by_video.items()):
                    vid_frames_indices = sorted(data['frames'])
                    for exp_id, exp_dict in data['expressions'].items():
                        exp_dict['exp_id'] = exp_id
                        samples_list.append((vid_id, vid_frames_indices, exp_dict, None, None))

            with open(metadata_file_path, 'w') as f:
                json.dump(samples_list, f)
        if is_distributed:
            dist.barrier()
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable= (is_distributed and dist.get_rank() != 0))]
        return samples_list

    @staticmethod
    def generate_train_video_samples(vid_id, vid_data, vid_objects, window_size, subset_type, mask_annotations_dir, device):
        """
        vid_id: 003234408d,
        vid_data: 
            {
                "expressions": {
                    "0": {
                        "exp": "a penguin is on the left in the front with many others on the hill",
                        "obj_id": "1"
                    },
                    "1": {
                        "exp": "a penguin on the left side of the screen next to another penguin facing it",
                        "obj_id": "1"
                    },
                    "2": {
                        "exp": "a black and white penguin in the front looking down",
                        "obj_id": "2"
                    },
                    "3": {
                        "exp": "a penguin on the front looking down at the rocks",
                        "obj_id": "2"
                    },
                },
                "frames":["00000","00005","00010","00015","00020","00025","00030","00035",]
            }
        """
        vid_frames = sorted(vid_data['frames'])
        vid_windows = [vid_frames[i:i + window_size] for i in range(0, len(vid_frames), window_size)]
        # replace last window with a full window if it is too short:
        if len(vid_windows[-1]) < window_size:
            if len(vid_frames) >= window_size:  # there are enough frames to complete to a full window
                vid_windows[-1] = vid_frames[-window_size:]
            else:  # otherwise, just duplicate the last frame as necessary to complete to a full window
                num_missing_frames = window_size - len(vid_windows[-1])
                missing_frames = num_missing_frames * [vid_windows[-1][-1]]
                vid_windows[-1] = vid_windows[-1] + missing_frames
        samples_list = []
        # {"1":,"2":, "3":,}
        vid_objects = {int(key): object_meta['category'] for key, object_meta in vid_objects.items()}
        exist_queries = copy.deepcopy(vid_data['expressions'])
        for exp_id, exp_dict in vid_data['expressions'].items():
            exp_dict['exp_id'] = exp_id
            # {exp_id: "0", exp: "", obj_id: "2"}
            for window in vid_windows:
                if subset_type == 'train':
                    # if train subset, make sure that the referred object appears in the window, else skip:
                    annotation_paths = [path.join(mask_annotations_dir, vid_id, f'{idx}.png') for idx in window] # clip的annotation
                    mask_annotations = [torch.tensor(np.array(Image.open(p)), device=device) for p in annotation_paths]  
                    all_object_indices = set().union(*[m.unique().tolist() for m in mask_annotations]) # 这个clip的所有帧中出现的object id的并集
                    if int(exp_dict['obj_id']) not in all_object_indices: #如果所有帧里都没出现这个object, 则放弃这个window
                        continue
                samples_list.append((vid_id, window, exp_dict, vid_objects, exist_queries))
        return samples_list
    
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
    
    def __getitem__(self, idx):
        video_id, frame_indices, text_query_dict, video_objects, exist_queries = self.samples_list[idx]            
        # video_id: 00f8
        # frame_indices: "00000", "00005", "00010"
        # text_query_dict: {"exp_id": "0/1/2/3...", "exp": "", "obj_id": 0/0/1/1
        text_query = text_query_dict['exp']
        if self.fix_spelling:
            text_query = self.fix_spell_model(text_query,max_length=2048)[0]['generated_text']
        text_query = " ".join(text_query.lower().split())  # clean up the text query
        # read the source window frames:
        frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        source_frames = [Image.open(p) for p in frame_paths]
        original_frame_size = source_frames[0].size[::-1]

        if self.subset_type == 'train':
            # read the instance masks:
            annotation_paths = [path.join(self.mask_annotations_dir, video_id, f'{idx}.png') for idx in frame_indices] # 每帧都有annotation
            mask_annotations = [torch.tensor(np.array(Image.open(p))) for p in annotation_paths] # list[T(h w)], nf
            all_object_indices = set().union(*[m.unique().tolist() for m in mask_annotations])  # 当前抽到的clip中的存在的objects的合集, 0, 1, 2, 3, 5
            # 一个video的不同object的annotations是放在一张图片上的, 不同的object使用不一样的颜色, 0是背景
            all_object_indices.remove(0)  # remove the background index 1，2，3，5
            all_object_indices = sorted(list(all_object_indices)) # 1，2，3, 5
            mask_annotations_by_object = [] 
            labels_by_object = []
            exist_queries_by_object = [] # list[list[text], #ann], n
            for obj_id in all_object_indices:
                obj_id_mask_annotations = torch.stack([(m == obj_id).to(torch.uint8) for m in mask_annotations]) # t h w
                mask_annotations_by_object.append(obj_id_mask_annotations)
                labels_by_object.append(all_categories[video_objects[obj_id]])
                obj_exps = [a['exp'] for _, a in exist_queries.items() if int(a['obj_id']) == obj_id]
                if len(obj_exps) == 0:
                    print('there are some objects that in the video has no expressions')
                exist_queries_by_object.append(obj_exps)
            mask_annotations_by_object = torch.stack(mask_annotations_by_object) # T(n t h w)
            mask_annotations_by_frame = rearrange(mask_annotations_by_object, 'o t h w -> t o h w')  # T(t n h w)
            labels_by_object = torch.tensor(labels_by_object).long() # T(n)
            # next we get the referred instance index in the list of all the object ids:
            ref_obj_idx = torch.tensor(all_object_indices.index(int(text_query_dict['obj_id'])), dtype=torch.long) # 在1, 2, 3, 5中的下标

            # list[dict], nf个
            targets = []  
            for frame_masks in mask_annotations_by_frame: # (n h w) in (t n h w)
                boxes = []  # ni
                valids = [] # ni
                for object_mask in frame_masks:
                    if (object_mask > 0).any():
                        y1, y2, x1, x2 = self.bounding_box(object_mask.numpy())
                        box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                        valids.append(1)
                        boxes.append(box)
                    else:
                        box = torch.tensor([0,0,0,0]).to(torch.float)
                        valids.append(0)
                        boxes.append(box)
                boxes = torch.stack(boxes, dim=0) # T(ni 4)
                valids = torch.tensor(valids, dtype=torch.int32) # T(ni, )
                # TODO: labels, 获得每个instance的数据集label, T(ni, ) long
                
                target = {
                    'masks': frame_masks, # ni h w
                    'orig_size': frame_masks.shape[-2:],  # original frame shape without any augmentations
                    'size': frame_masks.shape[-2:],
                    'referred_instance_idx': ref_obj_idx, # int
                    'iscrowd': torch.zeros(len(frame_masks)),  # for compatibility with DETR COCO transforms
                    'text_query': text_query,
                    'exist_queries': [exp for obj_exps in exist_queries_by_object for exp in obj_exps], # compatible with the transforms
                    'boxes': boxes, # T(n, 4) x1y1x2y2
                    'valid': valids,  # T(n, ) 
                    'labels': labels_by_object, # T(n)
                }
                targets.append(target)
        
            source_frames, targets= self.transforms(source_frames, targets)  # T(t 3 h w), list[dict]
            text_query = [targets[i].pop('text_query') for i in range(len(targets))][0]
            exist_queries = [targets[i].pop('exist_queries') for i in range(len(targets))][0]
            split_exist_queries = []
            cnt = 0
            for obj_exps in exist_queries_by_object:
                split_exist_queries.append(exist_queries[cnt:(cnt+len(obj_exps))])
                cnt += len(obj_exps)
            assert cnt == len(exist_queries_by_object)
            return source_frames, targets, text_query, split_exist_queries
        
        else: # validation:
            # validation subset has no annotations, so create dummy targets:
            targets = len(source_frames) * [None]
            source_frames, targets = self.transforms(source_frames, targets)  # T(t 3 h w), list[none]
            video_metadata = {'video_id': video_id,
                              'frame_indices': frame_indices,
                              'resized_frame_size': source_frames.shape[-2:],
                              'original_frame_size': original_frame_size,
                              'exp_id': text_query_dict['exp_id']}
            return source_frames, video_metadata, text_query

    def __len__(self):
        return len(self.samples_list)


class Collator:
    def __init__(self, subset_type, max_stride):
        self.subset_type = subset_type
        self.max_stride = max_stride

    def __call__(self, batch):
        if self.subset_type == 'train':
            samples, targets, text_queries, exist_queries = list(zip(*batch))
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)  # list[t c h w] -> NT(t b c h w)
            # list[list[dict], t], b -> list[list[dict], b] t
            targets = list(zip(*targets))
            batch_dict = {
                'samples': samples,  # NT(t b 3 h w)
                'targets': targets,  # list[list[dict], batch], nf
                'text_queries': text_queries,  # list[str], b
                'exist_queries': exist_queries, # list[list[list[str], #ann], ni], b
            }
            return batch_dict
        else:  # validation:
            samples, videos_metadata, text_queries = list(zip(*batch))
            samples = nested_tensor_from_videos_list_with_stride(samples, max_stride=self.max_stride)  # list[t c h w] -> NT(t b c h w)
            batch_dict = {
                'samples': samples, # NT(t b 3 h w)
                'videos_metadata': videos_metadata,
                'text_queries': text_queries # # list[str], b
            }
            return batch_dict

@register_dataset
def youtube_rvos_perExp(data_configs):
    configs = vars(data_configs)
    create_aug = video_aug_entrypoints(data_configs.augmentation.name)
    train_aug, eval_aug = create_aug(data_configs.augmentation)
    train_collator = Collator('train', max_stride=configs['max_stride'])
    eval_collator = Collator('test', max_stride=configs['max_stride'])
    
    train_dataset = Youtube_RVOS_PerExpression(transforms=train_aug,
                                 collator=train_collator,
                                 subset_type='train',
                                 dataset_path=configs['dataset_root'],
                                 window_size=configs['train_window_size'],
                                 fix_spelling=configs['fix_spelling'])
    
    test_dataset = Youtube_RVOS_PerExpression(transforms=eval_aug,
                                 collator=eval_collator,
                                 subset_type='test',
                                 dataset_path=configs['dataset_root'],
                                 window_size=None,
                                 fix_spelling=configs['fix_spelling'])
    def my_dataset_function():
        return [{}]
    DatasetCatalog.register('youtube_rvos', my_dataset_function)
    postprocessor = ReferYoutubeVOSPostProcess()
    from detectron2.data import MetadataCatalog
    MetadataCatalog.get('youtube_rvos').thing_classes = ['r', 'nr']
    MetadataCatalog.get('youtube_rvos').thing_colors = [(255., 140., 0.), (0., 255., 0.)]
    return train_dataset, test_dataset, postprocessor


@register_dataset
def youtube_rvos_perClip(data_configs):
    pass