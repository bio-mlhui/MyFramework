
import torch

from torch.nn import functional as F
from models.registry import register_model
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.render.apis import Text_3D_Mapper


class AuxMapper:
    def __init__(self, aux_configs):
        pass
        # 假设:
        # optimize的batch_size是1

    def mapper(self, data_dict, mode,):      
        Text_3D_Mapper     
        data_dict['meta_idxs'] = [data_dict['meta_idx']]
        data_dict['visualize'] = [data_dict['visualize']]
        return data_dict

    def collate(self, batch_dict, mode):
        # 因为batch_size就是1
        return batch_dict[0]
