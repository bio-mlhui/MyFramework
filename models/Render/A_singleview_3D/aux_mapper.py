
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
        Scene_MappMultiview3D_Optimize_Mapperer     
        data_dict['meta_idxs'] = [data_dict['meta_idx']]
        data_dict['visualize'] = [data_dict['visualize']]
        return data_dict

    def collate(self, batch_dict, mode):
        # 因为batch_size就是1
        return batch_dict[0]
    
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