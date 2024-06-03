import torch
from copy import deepcopy as dcopy
import cv2 as cv
from data_schedule.utils.segmentation import bounding_box_from_mask
from data_schedule.registry import Mapper
import copy

class VIDVID_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,
                 **kwargs) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.get_frames_fn = dataset_meta.get('get_frames_fn')


# TODO: video inpaint learn
class Learn_TrainMapper(VIDVID_Mapper):
    pass

