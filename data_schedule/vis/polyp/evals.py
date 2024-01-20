from data_schedule.vis.evaluator_utils import register_vis_metric
from data_schedule.vis.apis import VIS_Evaluator_OutAPI_EvalFn_API
import os
from glob import glob
from tqdm import tqdm
import shutil
from functools import partial
from PIL import Image
import numpy as np
import torch
import detectron2.utils.comm as comm
import logging
import pycocotools.mask as mask_util
from pycocotools.mask import decode as decode_rle
import subprocess
import data_schedule.vis.polyp.metrics as metrics

@register_vis_metric
def polyp_vps_evaluator(output_dir,
                  dataset_meta,
                  **kwargs):
    assert comm.is_main_process()
    root = dataset_meta.get('root')
    eval_set_name = dataset_meta.get('eval_set_name')
    pred_path = os.path.join(output_dir, 'web')
    assert os.path.exists(pred_path)
    logging.info(subprocess.call(f'python -u vps_evaluator.py \
                 --gt_root {root}\
                 --pred_root {pred_path} \
                 --data_lst {eval_set_name}  \
                 --txt_name ', 
                 shell=True))
    
    # 提取最后的结果
    return {}

