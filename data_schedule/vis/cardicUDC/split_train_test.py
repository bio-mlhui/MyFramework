import os
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d
import torch
from PIL import Image
root = '/home/xuhuihui/datasets/cardiacUDC/transformed'

all_labeled_videos = os.listdir(os.path.join(root, 'labeled', 'Frame'))

num_test = int(0.2 * len(all_labeled_videos))
perm_indexs = torch.randperm(len(all_labeled_videos))
test_videos = perm_indexs[:num_test]
train_videos = perm_indexs[num_test:]
test_videos = [all_labeled_videos[f] for f in test_videos]
train_videos = [all_labeled_videos[f] for f in train_videos]


os.makedirs(os.path.join(root, 'labeled_train', 'Frame'), exist_ok=True)
os.makedirs(os.path.join(root, 'labeled_train', 'GT'), exist_ok=True)
os.makedirs(os.path.join(root, 'labeled_test', 'Frame'), exist_ok=True)
os.makedirs(os.path.join(root, 'labeled_test', 'GT'), exist_ok=True)

import shutil
for test_vid in test_videos:
    shutil.copytree(os.path.join(root, 'labeled', 'Frame', test_vid),
                    os.path.join(root, 'labeled_test', 'Frame', test_vid),)
    shutil.copytree(os.path.join(root, 'labeled', 'GT', test_vid),
                    os.path.join(root, 'labeled_test', 'GT', test_vid),)

for test_vid in train_videos:
    shutil.copytree(os.path.join(root, 'labeled', 'Frame', test_vid),
                    os.path.join(root, 'labeled_train', 'Frame', test_vid),)
    shutil.copytree(os.path.join(root, 'labeled', 'GT', test_vid),
                    os.path.join(root, 'labeled_train', 'GT', test_vid),)
