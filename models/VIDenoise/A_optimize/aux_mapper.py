
import torch

from torch.nn import functional as F
from models.registry import register_model
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY


class AuxMapper:
    def __init__(self, aux_configs):
        pass
    def mapper(self, data_dict, mode,):           
        data_dict['meta_idxs'] = [data_dict['meta_idx']]
        data_dict['visualize'] = [data_dict['visualize']]
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict[0]
