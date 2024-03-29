from typing import Optional, Union
import json
import os
from functools import partial
import numpy as np
import torch
import logging
from tqdm import tqdm
import copy
from collections import defaultdict
from detectron2.data import DatasetCatalog, MetadataCatalog
import logging

scene_to_texts = {
    'text1': 'trump playing ping-pong with xijinping',
    'text2': 'justin biber kissing selena golmez',
    'text3': 'taylor swift slapping kanye west'
}

for scene_name, scene_text in scene_to_texts.items():    
    register_train_name = f'{scene_name}_train'
    register_test_name = f'{scene_name}_test'

    # train
    DatasetCatalog.register(register_train_name, lambda : [{'text': scene_text}])  
    MetadataCatalog.get(register_train_name).set(
        text=scene_text,
    )
    # test
    DatasetCatalog.register(register_test_name,  lambda : [{'text': scene_text}])
    MetadataCatalog.get(register_test_name).set(
        text=scene_text,
    )
    
    # pass
    # MetadataCatalog.get(register_train_name).set()   
        





