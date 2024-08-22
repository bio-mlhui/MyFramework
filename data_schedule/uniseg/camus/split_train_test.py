# 6个类别，每个类别有500个video, 450个用于训练，50个用于测试
class_names = ['camus_2ch_1', 'camus_2ch_2', 'camus_2ch_3', 'camus_4ch_1', 'camus_4ch_2', 'camus_4ch_3']
# 同一个病人不能同时出现在训练和测试集中，
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import PIL
import SimpleITK as sitk
from PIL.Image import Resampling
from skimage.measure import find_contours
import torch

dataset_root = Path("../uni_med_video/CAMUS")

ret = {} # class: 'train_videos': [], 'test_videos': []


