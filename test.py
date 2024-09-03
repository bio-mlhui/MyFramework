# from transformers import pipeline
# from detectron2.utils.visualizer import Visualizer
# from detectron2.utils.video_visualizer import VideoVisualizer
# fix_spelling = pipeline("text2text-generation",model="/home/xhh/pt/spelling-correction-english-base")

# print(fix_spelling(["a baby giant panda near a large giant panda"],max_length=2048))

# import albumentations as A
# import numpy as np
# import cv2
# import torch
# hflip = A.ReplayCompose(
#     [A.HorizontalFlip(p=0.5)],
# )
# image = cv2.imread('/home/xuhuihui/datasets/SUN-SEG/TrainDataset/Frame/case2_1/case_M_20181003094031_0U62363100354631_1_001_002-1_a1_ayy_image0002.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# h, w, c = image.shape
# masks = (torch.randn([h, w]) > 0).numpy().astype(np.uint8)
# ret = hflip(image=image, mask=masks)
# ret['image']
# ret['mask']
      
# # 测试infinite stream 
# import torch
# import math
# def _infinite_indices(seed, dataset_length, shuffle=True,):
#     g = torch.Generator()
#     g.manual_seed(seed)
#     while True:
#         if shuffle:
#             yield from torch.randperm(dataset_length, generator=g).tolist()
#         else:
#             yield from torch.arange(dataset_length).tolist()

# # 生成一个无限的infinite stream, 保证每次运行返回的都相同
# def infinite_indices(seed, 
#                      dataset_length, 
#                      batch_sizes, splits, 
#                      one_batch_two_epoch,
#                      shuffle=True): # 'abandon', 'just_use', 'pad'
#     g = torch.Generator()
#     g.manual_seed(seed)

#     split_ranges = list(zip(splits[:-1], splits[1:]))
#     assert len(split_ranges) == (len(batch_sizes))
#     stream = _infinite_indices(seed, dataset_length=dataset_length, shuffle=shuffle)

#     stream_throw_cnt = 0
#     cnt = 0
#     for (range_start, range_end), btch_size in zip(split_ranges, batch_sizes):
#         assert cnt == range_start
#         if range_end == None:
#             range_end = math.inf
#         # stream_throw_cnt = 5996, stream_throw_cnt + infinite_btch_sizz = 6000(下一个batch的第一个sample的index), epoch_milestone是6000, 不会抽到6000
#         while cnt < range_end:
#             epoch_milestone = ((stream_throw_cnt // dataset_length) + 1 ) * dataset_length
#             if (stream_throw_cnt < epoch_milestone) and (stream_throw_cnt + btch_size > epoch_milestone):
#                 if one_batch_two_epoch == 'just_use':
#                     for _ in range(btch_size):
#                         cnt += 1
#                         stream_throw_cnt += 1
#                         yield next(stream)

#                 elif one_batch_two_epoch == 'abandon':
#                     print(f'abandon start {stream_throw_cnt} end {stream_throw_cnt + btch_size}')
#                     for _ in range(epoch_milestone - stream_throw_cnt):
#                         abandon = next(stream)
#                         stream_throw_cnt += 1

#                 elif one_batch_two_epoch == 'pad':
#                     print(f'pad start {stream_throw_cnt} end {stream_throw_cnt + btch_size}')
#                     diff = stream_throw_cnt + btch_size - epoch_milestone
#                     num_throw = btch_size - diff
#                     rand_idxs = torch.randperm(dataset_length, generator=g)[:diff].tolist()
#                     print(f'pad idxs {rand_idxs}')
#                     for _ in range(num_throw):
#                         cnt += 1
#                         stream_throw_cnt += 1
#                         yield next(stream)
#                     for idx in rand_idxs:
#                         cnt += 1
#                         yield idx
#                 else:
#                     raise ValueError()
#             else:
#                 for _ in range(btch_size):
#                     cnt += 1
#                     stream_throw_cnt += 1
#                     yield next(stream)  


# infinite_stream = infinite_indices(2024,
#                                    100,
#                                    [4, 8, 3],
#                                    [0, 20, 44, None],
#                                    'pad',
#                                    False)
# i = 0
# while i < 300:
#     i += 1
#     print(next(infinite_stream))

####################################medical image preview#############################

# import SimpleITK as sitk

# labelImg=sitk.ReadImage('/home/xuhuihui/datasets/anatomy_recon/multiclass_anatomy_recon/train/complete/s0224_full.nii.gz')
# labelNpy=sitk.GetArrayFromImage(labelImg)
# labelNpy_resized=zoom(labelNpy,(128/labelNpy.shape[0],128/labelNpy.shape[1],128/labelNpy.shape[2]),order=0, mode='constant')
# labelNpy_resized=np.expand_dims(np.expand_dims(labelNpy_resized,axis=0),axis=4) 
# name=train_label_list[j][-len('_full.nii.gz')-len('s0556'):-len('_full.nii.gz')]
# import matplotlib.pyplot as plt
# plt.imsave()

#####################################video clip#############################
# from models.videoclip.models.mmfusion import MMPTModel
# from models.videoclip.models import MMFusionSeparate
# import yaml
# import argparse
# import torch
# import torchvision.io as video_io

# def dict2namespace(config):
#     namespace = argparse.Namespace()
#     for key, value in config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace


# with open ('./models/videoclip/how2.yaml', 'r') as f:
#     configs = yaml.safe_load(f)
# configs = dict2namespace(configs)

# model = MMFusionSeparate(config=configs,
#                          checkpoint_path=configs.eval.save_path)

# from models.videoclip.processors.models.s3dg import S3D
# video_encoder = S3D('/home/xhh/workspace/fairseq/examples/MMPT/pretrained_models/s3d_dict.npy', 512)
# video_encoder.load_state_dict(
#     torch.load('/home/xhh/workspace/fairseq/examples/MMPT/pretrained_models/s3d_howto100m.pth'))
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     configs.dataset.bert_name, use_fast=False
# )
# from models.videoclip.processors import Aligner
# aligner = Aligner(configs.dataset)

# videoclip = MMPTModel(configs, model, video_encoder)


# videoclip.eval()

# from einops import rearrange
# # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
# fps=30
# video_frames = video_io.read_video(
#                 filename='./dog.mp4', pts_unit='sec', output_format='TCHW')[0].unsqueeze(0).float() / 255.0
# video_frames = rearrange(video_frames, 'b (t fps) c h w -> b t fps h w c',fps=fps)
# caps, cmasks = aligner._build_text_seq(
#     tokenizer("a girl standing with her mother", add_special_tokens=False)["input_ids"]
# )

# caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

# with torch.no_grad():
#     output = videoclip(video_frames, caps, cmasks, return_score=True)
# print(output["score"])  # dot-product


# #################################################clip###############################
# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# import matplotlib.pyplot as plt
# import torch
# from einops import rearrange, repeat
# import torch.nn.functional as F
# nf = 5
# h = 384
# w = 640

# pad_bottom = 10
# pad_right = 20
# font_size = 20
# linespacing = 2
# title = 'this is a test of long long long long long sentence'
# mlm_masked_title = 'this is a test of long long long long long mlm_masekd_title'
# mlm_pred_title = 'this is a test of long long long long long mlm_pred_title'
# font_length = len(mlm_pred_title)

# vid_frames = torch.rand([nf, 3, h, w])# t 3 h w
# vid_frames = F.pad(vid_frames, [0, pad_right, 0, pad_bottom]) # 
# vid_frames = rearrange(vid_frames, 't c h w -> (t h) w c')

# gt_referent = torch.randint(0, 2, [nf, h, w]) # t h w
# gt_referent = F.pad(gt_referent, [0, pad_right, 0, pad_bottom])
# gt_referent = rearrange(gt_referent, 't h w -> (t h) w')
# gt_referent = repeat(gt_referent, 'th w -> th w c',c=3)

# mvm_masked = vid_frames
# mvm_pred = vid_frames

# refer_pred = gt_referent # t h w

# whole_image = torch.hstack([vid_frames, gt_referent, refer_pred, mvm_masked, mvm_pred])
# print(whole_image.shape[1], whole_image.shape[0])

# fig_with = max(whole_image.shape[1], (font_size*font_length))

# fig, axs = plt.subplots(figsize=(fig_with/100, (whole_image.shape[0]+((3+linespacing*2)*font_size))/100,))
# axs.xaxis.set_visible(False)
# axs.yaxis.set_visible(False)
# axs.imshow(whole_image)
# # axs['video'].imshow(vid_frames)
# # axs['refer_pred'].imshow(refer_pred)
# # axs['mvm_masked'].imshow(mvm_masked)
# # axs['mvm_pred'].imshow(mvm_pred)

# fig.suptitle(f'{title}\n{mlm_masked_title}\n{mlm_pred_title}', fontsize=font_size, linespacing=2.)
# fig.savefig(f'./test.png')
# plt.close()

# canvas = Visualizer()


    
# anon_map = {}
# attributes = []
# for src, role, tgt in g.attributes():
#     if constant.type(tgt) in (constant.INTEGER, constant.FLOAT):
#         anon_val = f'number_{len(anon_map)}'
#         anon_map[anon_val] = tgt
#         tgt = anon_val
#     attributes.append((src, role, tgt))
# g2 = penman.Graph(g.instances() + g.edges() + attributes)
# print(penman.encode(g2))
# anon_map

# from transformers import BartForConditionalGeneration
# from models.amr_utils.tokenization_bart import AMRBartTokenizer
# amr2text_model = BartForConditionalGeneration.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')
# encoder = amr2text_model.get_encoder()
# amr_tokenizer = AMRBartTokenizer.from_pretrained('/home/xhh/pt/amr/AMRBART-large-finetuned-AMR3.0-AMR2Text-v2')

# linearized_amr = '( <pointer:0> hand~0 :ARG1-of ( <pointer:1> move-01~5 ) :part-of ( <pointer:2> man~3 ) )'

# tokenized_amr, meta_dict = amr_tokenizer.tokenize_amr_with_only_each_token_length(linearized_amr.split())
            
# class TrainRandomSampler_ByEpoch:
#     def __init__(self, 
#                  data_source,
#                  seed,
#                  ) -> None:
#         self.data_source = data_source
#         self.num_samples = len(self.data_source)
#         self.seed = seed
#         self.epoch = None

#     def __iter__(self):
#         seed = self.seed + self.epoch
#         print(f'generating a new indices permutations for this epoch using seed {seed}')
#         n = len(self.data_source)
#         generator = torch.Generator()
#         generator.manual_seed(seed)
        
#         for _ in range(self.num_samples // n):
#             yield from torch.randperm(n, generator=generator).tolist()
#         yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

#     def __len__(self) -> int:
#         return self.num_samples

#     def set_epoch(self, epoch: int) -> None:
#         r"""

#         Args:
#             epoch (int): Epoch number.
#         """
#         self.epoch = epoch

# datasource = torch.arange(100)

# sampler = TrainRandomSampler_StartEpoch(datasource, seed=1999)
# epoch5_top10 = []
# for epoch in range(10):
#     sampler.set_epoch(epoch)
#     for item in sampler:
#         if epoch == 5:
#             if len(epoch5_top10) < 10:
#                 epoch5_top10.append(item)
#         pass
    
# new_epoch_top10 = []       
# sampler = TrainRandomSampler_StartEpoch(datasource, seed=1999)
# for epoch in range(2, 10):
#     sampler.set_epoch(epoch)
#     for item in sampler:
#         if epoch == 5:
#             if len(new_epoch_top10) < 10:
#                 new_epoch_top10.append(item)
#         pass

# print(epoch5_top10)
# print(new_epoch_top10)


# from models.layers.gilbert.gilbert2d import gilbert2d_widthBigger, generate2d
# import matplotlib.pyplot as plt

# def draw_lines(height, width, coordinates, prefix):
#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Set the limits of the plot based on height and width
#     ax.set_xlim(0, width)
#     ax.set_ylim(0, height)

#     # Hide the axes
#     ax.axis('off')

#     # Plot the lighter 2D grid
#     ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

#     # Plot the lines by connecting consecutive coordinates
#     for i in range(len(coordinates) - 1):
#         x1, y1 = coordinates[i]
#         x2, y2 = coordinates[i + 1]
#         ax.plot([x1, x2], [y1, y2],  color='black', linewidth=2, markersize=8)  # Adjust line appearance
#     plt.savefig(f'./{prefix}.png')

# import torch
# # def generate_hilbert_curve2D(height, width, device):
# #     hilbert_curve = list(gilbert2d(width=width, height=height,))
# #     hilbert_curve = torch.tensor(hilbert_curve).long().to(device)
# #     sequence_indices = hilbert_curve[:, 1] * width + hilbert_curve[:, 0] # tensor
    
# #     hilbert_curve2 = list(gilbert2d(width=height, height=width,)) # hw 的下标
# #     hilbert_curve2 = torch.tensor(hilbert_curve2).long().to(device)
# #     sequence_indices2 = hilbert_curve2[:, 0] * width + hilbert_curve2[:, 1] # tensor

# #     return sequence_indices, sequence_indices2

# # def gilbert2d(x, y, ax, ay, bx, by): # major = width
# #     """
# #     Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
# #     2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
# #     of size (width x height).
# #     """

# #     # if width >= height:
# #     yield from generate2d(x, y, ax, ay, bx, by)

# W = 56
# H = 56
# curve_1 = torch.tensor(list(generate2d(0, 0, ax=W, ay=0, bx=0, by=H)))
# curve_2 = torch.tensor(list(generate2d(W-1, 0, ax=0, ay=H, bx=-W, by=0)))
# curve_3 = torch.tensor(list(generate2d(W-1, H-1, ax=-W, ay=0, bx=0, by=-H)))
# curve_4 = torch.tensor(list(generate2d(0, H-1, ax=0, ay=-H, bx=W, by=0, )))

# draw_lines(H, W, curve_1, '1')
# draw_lines(H, W, curve_2, '1_rev')
# draw_lines(H, W, curve_3, '2')
# draw_lines(H, W, curve_4, '2_rev')

# # def generate_hilbert_curve2D(height, width, device):
# #     # 从[0, 0,]横向切入, major=width
# #     hil1 = list(gilbert2d(width=width, height=height,)) # x, y 平着走, 
# #     # 从[W, H]横向切入
# #     hil1_rev = [[width - haosen[0] - 1, height-haosen[1] - 1] for haosen in hil1]

# #     # 从[0,0]纵向切入, major=height
# #     hil2_o = list(gilbert2d(width=height, height=width,)) # y, x 竖着走
# #     # 从(W, 0)纵向切入
# #     hil2 = [[haosen[0], width - haosen[1] - 1 ] for haosen in hil2_o]
# #     # 从(0, H)纵向切入
# #     hil2_rev = [[height - haosen[0] - 1, haosen[1]] for haosen in hil2_o]

# #     return hil1, hil1_rev, hil2, hil2_rev

# #     # hil1 = torch.tensor(hil1).long().to(device)
# #     # hil2 = torch.tenosr(hil2).long().to(device)

# #     # hil1_hw_ind = hil1[:, 1] * width

# #     # hilbert_curve = torch.tensor(hilbert_curve).long().to(device)
# #     # sequence_indices = hilbert_curve[:, 1] * width + hilbert_curve[:, 0] # tensor
    
# #     # hilbert_curve2 = torch.tensor(hilbert_curve2).long().to(device)
# #     # sequence_indices2 = hilbert_curve2[:, 0] * width + hilbert_curve2[:, 1] # tensor

# #     # return sequence_indices, sequence_indices2

# # def gilbert2d(width, height): # major = width
# #     """
# #     Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
# #     2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
# #     of size (width x height).
# #     """

# #     # if width >= height:
# #     yield from generate2d(0, 0, width, 0, 0, height)
# #     # else:
# #     #     yield from generate2d(0, 0, 0, height, width, 0)

# # hil1, hil1_rev, hil2, hil2_rev = generate_hilbert_curve2D(56, 56, device='cpu')

# # draw_lines(H, W, hil1, '1_2')
# # draw_lines(H, W, hil1_rev, '1_rev_2')
# # draw_lines(H, W, hil2, '2_2')
# # draw_lines(H, W, hil2_rev, '2_rev_2')



# # generate miccai

# # temp8 test set

# import os
# from PIL import Image
# import torch
# videos_dir = '/home/xuhuihui/datasets/visha/train/Frame'
# videos = os.listdir(videos_dir)
# sizes = []
# for vid in videos:
#     first_frame = Image.open(os.path.join(videos_dir, vid, '00000001.jpg'))
#     width, height = first_frame.size
#     sizes.append(torch.tensor([width, height]))

# sizes = torch.stack(sizes, dim=0) # L 2

# import cv2
# import pydicom
# import numpy as np
# import torch
# from torchvision.io import write_video

# # Function to read DICOM file and extract frames
# def read_dicom(file_path):
#     ds = pydicom.dcmread(file_path)
#     if 'NumberOfFrames' in ds:
#         frames = ds.pixel_array
#     else:
#         frames = np.expand_dims(ds.pixel_array, axis=0)
#     return frames, ds

# # Function to save frames as a video
# def save_video(frames, output_path, frame_rate=30):
#     grayscale_frames = []
#     for frame in frames:
#         grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         grayscale_frames.append(grayscale_frame)
#     grayscale_frames = np.array(grayscale_frames)
#     grayscale_frames = (grayscale_frames - grayscale_frames.min()) / (grayscale_frames.max() - grayscale_frames.min())
#     # t h w 1, 0-1
#     grayscale_frames = torch.from_numpy(grayscale_frames).float()
#     grayscale_frames = (grayscale_frames * 255).clamp(min=0, max=255).to(torch.uint8)
#     write_video(output_path, video_array=grayscale_frames.unsqueeze(-1).repeat(1,1,1,3), fps=frame_rate)

# # Main function
# def dicom_to_video(dicom_path, output_video_path, frame_rate=30):
#     frames, ds = read_dicom(dicom_path)
#     save_video(frames, output_video_path, frame_rate)
#     print(f"Video saved to {output_video_path}")

# # Example usage
# dicom_path = '/home/xuhuihui/workspace/rvos_encoder/miccai_generate_Dir/folder/202405281648120003VAS.DCM'
# output_video_path = '/home/xuhuihui/workspace/rvos_encoder/miccai_generate_Dir/folder/202405281648120003VAS.mp4'
# dicom_to_video(dicom_path, output_video_path, frame_rate=30)
    

import torch
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    
    

