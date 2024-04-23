
import torch

from torch.nn import functional as F
from models.registry import register_model
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY

class Text_3DGS_AuxMapper:
    def __init__(self, aux_configs):
        render_auxes = aux_configs['render_auxes']
        render_auxes_names = [config['name'] for config in render_auxes]
        assert len(list(set(render_auxes_names))) == len(render_auxes_names), '每个aux的名字必须不一样'
        self.render_auxes_names = render_auxes_names
        self.render_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in render_auxes]

    def mapper(self, data_dict, mode,):           
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict

class Text_4DGS_AuxMapper:
    def __init__(self, aux_configs):
        render_auxes = aux_configs['render_auxes']
        render_auxes_names = [config['name'] for config in render_auxes]
        assert len(list(set(render_auxes_names))) == len(render_auxes_names), '每个aux的名字必须不一样'
        self.render_auxes_names = render_auxes_names
        self.render_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in render_auxes]

    def mapper(self, data_dict, mode,):           
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict


class Video_4DGS_Optimize_AuxMapper:
    def __init__(self, aux_configs):
        render_auxes = aux_configs['render_auxes']
        render_auxes_names = [config['name'] for config in render_auxes]
        assert len(list(set(render_auxes_names))) == len(render_auxes_names), '每个aux的名字必须不一样'
        self.render_auxes_names = render_auxes_names
        self.render_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in render_auxes]

    def mapper(self, data_dict, mode,):           
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict
    

# 3D视角, 3D image;
class Image_3DGS_Optimize_AuxMapper:
    def __init__(self, aux_configs):
        render_auxes = aux_configs['render_auxes']
        render_auxes_names = [config['name'] for config in render_auxes]
        assert len(list(set(render_auxes_names))) == len(render_auxes_names), '每个aux的名字必须不一样'
        self.render_auxes_names = render_auxes_names
        self.render_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in render_auxes]

    def mapper(self, data_dict, mode,):           
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict
    
# 4D视角, 4D image
class Video_4DGS_Learning_AuxMapper:
    def __init__(self, aux_configs):
        render_auxes = aux_configs['render_auxes']
        render_auxes_names = [config['name'] for config in render_auxes]
        assert len(list(set(render_auxes_names))) == len(render_auxes_names), '每个aux的名字必须不一样'
        self.render_auxes_names = render_auxes_names
        self.render_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in render_auxes]

    def mapper(self, data_dict, mode,):           
        return data_dict

    def collate(self, batch_dict, mode):
        return batch_dict