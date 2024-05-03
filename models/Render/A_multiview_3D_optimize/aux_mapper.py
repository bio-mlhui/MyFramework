
import torch

from torch.nn import functional as F
from models.registry import register_model
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.render.apis import Scene_Meta, Multiview3D_Optimize_Mapper
# 3D视角, 3Dimage;
class Image_3DGS_Optimize_AuxMapper:
    def __init__(self, aux_configs):
        view_auxes = aux_configs['3dview_auxes']
        view_auxes_names = [config['name'] for config in view_auxes]
        assert len(list(set(view_auxes_names))) == len(view_auxes_names), '每个aux的名字必须不一样'
        self.view_auxes_names = view_auxes_names
        self.view_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in view_auxes]


        rendering_auxes = aux_configs['3drendering_auxes']
        rendering_auxes_names = [config['name'] for config in rendering_auxes]
        assert len(list(set(rendering_auxes_names))) == len(rendering_auxes_names), '每个aux的名字必须不一样'
        self.rendering_auxes_names = rendering_auxes_names
        self.rendering_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in rendering_auxes]

    def mapper(self, data_dict, mode,):      
        Multiview3D_Optimize_Mapper     
        data_dict['meta_idxs'] = [data_dict['meta_idx']]
        data_dict['visualize'] = [data_dict['visualize']]
        return data_dict

    def collate(self, batch_dict, mode):
        # 因为batch_size就是1
        return batch_dict[0]
