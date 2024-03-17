


# a2ds video.mp4 -> frames.jpg


# import os
# import cv2
# import numpy as np

# def preprocess(path_src, split_name):
#     print('process', path_src)
#     path_dst = path_src.replace(f'/{split_name}/', f'/{split_name}/uterus_myoma_WeakPolyP_{split_name}/')
#     for folder in os.listdir(path_src+'/Frame'):
#         print(folder)
#         for name in os.listdir(path_src+'/Frame/'+folder):
#             image    = cv2.imread(path_src+'/Frame/'+folder+'/'+name)
#             image    = cv2.resize(image, (352,352), interpolation=cv2.INTER_LINEAR)
#             mask     = cv2.imread(path_src+'/GT/'+folder+'/'+name.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
#             mask     = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)
#             contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#             box      = np.zeros_like(mask)
#             for contour in contours:
#                 x,y,w,h = cv2.boundingRect(contour)
#                 box[y:y+h, x:x+w] = 255
            
#             os.makedirs(path_dst+'/Frame/'+folder, exist_ok=True)
#             cv2.imwrite(path_dst+'/Frame/'+folder+'/'+name, image)
#             os.makedirs(path_dst+'/GT/'   +folder, exist_ok=True)
#             cv2.imwrite(path_dst+'/GT/'   +folder+'/'+name.replace('.jpg', '.png'), mask)
#             os.makedirs(path_dst+'/Box/'  +folder, exist_ok=True)
#             cv2.imwrite(path_dst+'/Box/'  +folder+'/'+name.replace('.jpg', '.png'), box)

# split_name = 'temp8'
# dst_dir = f'{split_name}/uterus_myoma_WeakPolyP_{split_name}'

# root = '/home/xuhuihui/datasets/uterus_myoma/Dataset'

# if not os.path.exists(os.path.join(root, dst_dir)):
#     preprocess(os.path.join(root, f'{split_name}/test'), split_name=split_name)
#     preprocess(os.path.join(root, f'{split_name}/train'), split_name=split_name)
    

# images/video; gt/video -> GT/video; images/video
# import os
# import shutil


# SET_NAME = [
#         'train',
#         'test',
# ]

# SET_NAME_TO_DIR = {
#     'train': 'train',
#     'test': 'test',}

# SET_NAME_TO_NUM_VIDEOS = {
#     'train': 50,
#     'test': 70,    
# }


# SET_NAME_TO_PREFIX = {
#     'train': 'visha_train',
#     'test': 'visha_test',
# }

# base_path = '/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/VPS-TrainSet/CVC-ColonDB-300/Train'
# video_ids = os.listdir(base_path)

# frame_path = os.path.join(base_path, 'Frame')
# gt_path = os.path.join(base_path, 'GT')
# os.makedirs(frame_path, exist_ok=True)
# os.makedirs(gt_path, exist_ok=True)

# # Iterate through each video ID directory
# for vid in video_ids:
#     shutil.copytree(os.path.join(base_path, vid, 'Frame'), os.path.join(frame_path, vid))
#     shutil.copytree(os.path.join(base_path, vid, 'GT'), os.path.join(gt_path, vid))
    
# # delete
# all_images = os.listdir('/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/IVPS-TrainSet/Frame_all')

# os.makedirs(os.path.join('/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/IVPS-TrainSet', 'Frame'),exist_ok=True)
# os.makedirs(os.path.join('/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/IVPS-TrainSet', 'GT'),exist_ok=True)

# ka_images = [b for b in all_images if b.startswith('K')]
# assert len(ka_images) == 1000
# for image_id in ka_images:
#     shutil.copy(os.path.join('/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/IVPS-TrainSet', 'Frame', f'{image_id}.jpg'),
#                 os.path.join('/home/xuhuihui/datasets/SUN/SUN-SEG2/MICCAI-VPS-dataset/IVPS-TrainSet', 'Frame', f'{image_id}.jpg'),)
    


# remove non-mask frames
# SET_NAME = [
#         'Kvasir-train',
#         'Mayo-train',
#         '300-train',
#         '612-train',
#          ]

# SET_NAME_TO_DIR = {
#     'Kvasir-train': 'MICCAI-VPS-dataset/Kvasir-SEG',
#     'Mayo-train': 'MICCAI-VPS-dataset/VPS-TrainSet/ASU-Mayo_Clinic/Train',
#     '300-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ColonDB-300/Train',
#     '612-train': 'MICCAI-VPS-dataset/VPS-TrainSet/CVC-ClinicDB-612/Train',
# }

# SET_NAME_TO_NUM_VIDEOS = {
#     'Kvasir-train': 1,
#     'Mayo-train': 10,
#     '300-train': 6,
#     '612-train': 18,
#     '300-tv': 6,
#     '612-test': 5,
#     '612-val': 5      
# }


# SET_NAME_TO_PREFIX = {
#     'Kvasir-train': 'Kvasir-train',
#     'Mayo-train': 'Mayo-train',
#     '300-train': '300-train',
#     '612-train': '612-train',
# }

# _root = os.getenv('DATASET_PATH')
# root = os.path.join(_root, 'SUN/SUN-SEG2')

# from PIL import Image
# import numpy as np
# import torch
# def get_frames_mask(mask_path, video_id, frames):
#     # masks = [cv2.imread(os.path.join(mask_path, video_id, f'{f}.jpg')) for f in frames]
#     if os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.png')):
#         masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.png')).convert('L') for f in frames]
#     elif os.path.exists(os.path.join(mask_path, video_id, f'{frames[0]}.jpg')):
#         masks = [Image.open(os.path.join(mask_path, video_id, f'{f}.jpg')).convert('L') for f in frames]
#     else:
#         raise ValueError()
#     masks = [np.array(mk) for mk in masks]
#     masks = torch.stack([torch.from_numpy(mk) for mk in masks], dim=0) # t h w
#     # assert set(masks.unique().tolist()) == set([0, 255]), f'{masks.unique().tolist()}'
#     masks = (masks > 0).int()
#     return masks, torch.ones(len(frames)).bool()
# num_delted_frames = 0
# for train_set_name in SET_NAME:
#     set_dir = SET_NAME_TO_DIR[train_set_name]
#     frames_dir = os.path.join(root, set_dir, 'Frame')
#     mask_dir = os.path.join(root, set_dir, 'GT')

#     video_ids = os.listdir(frames_dir)
#     for vid in video_ids:
#         frames = [haosen[:-4] for haosen in os.listdir(os.path.join(frames_dir, vid))]
#         frame_has_fore = [get_frames_mask(mask_dir, vid, [haosen])[0].any() for haosen in frames] # list[t]
#         assert len(frame_has_fore) == len(frames)
#         num_delted_frames += (~ torch.tensor(frame_has_fore)).int().sum()
#         for haosen, frame_name in zip(frame_has_fore, frames):
#             if not haosen:
#                 os.remove(os.path.join(frames_dir, vid, f'{frame_name}.jpg'))

#                 if os.path.exists(os.path.join(mask_dir, vid, f'{frame_name}.jpg')):
#                     os.remove(os.path.join(mask_dir, vid, f'{frame_name}.jpg'))
#                 elif os.path.exists(os.path.join(mask_dir, vid, f'{frame_name}.png')):
#                     os.remove(os.path.join(mask_dir, vid, f'{frame_name}.png')) 
#                 else:
#                     raise ValueError()

# print(num_delted_frames) # 1546帧




