
import torch

from torch.nn import functional as F
from models.registry import register_model
from models.registry import MODELITY_INPUT_MAPPER_REGISTRY
from data_schedule.render.apis import Scene_Meta, SingleView_3D_Mapper


class AuxMapper:
    def __init__(self, aux_configs):
        view_auxes = aux_configs['view_auxes']
        view_auxes_names = [config['name'] for config in view_auxes]
        assert len(list(set(view_auxes_names))) == len(view_auxes_names), '每个aux的名字必须不一样'
        self.view_auxes_names = view_auxes_names
        self.view_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in view_auxes]
        self.view_auxes_on_inview =  [config['do_in'] for config in view_auxes]
        self.view_auxes_on_outview =  [config['do_out'] for config in view_auxes]


        rendering_auxes = aux_configs['rendering_auxes']
        rendering_auxes_names = [config['name'] for config in rendering_auxes]
        assert len(list(set(rendering_auxes_names))) == len(rendering_auxes_names), '每个aux的名字必须不一样'
        self.rendering_auxes_names = rendering_auxes_names
        self.rendering_auxes = [MODELITY_INPUT_MAPPER_REGISTRY.get(config['name'])(config) for config in rendering_auxes]
        self.rendering_auxes_on_inview =  [config['do_in'] for config in rendering_auxes]
        self.rendering_auxes_on_outview =  [config['do_out'] for config in rendering_auxes]


    def mapper(self, data_dict, mode,):      
        SingleView_3D_Mapper
        for idx, view_aux in enumerate(self.view_auxes):
            if self.view_auxes_on_inview[idx]:
                data_dict['inviews_dict'] = view_aux(data_dict['inviews_dict'])
            if self.view_auxes_on_outview[idx]:
                data_dict['outviews_dict'] = view_aux(data_dict['outviews_dict'])
        
        for idx, render_aux in enumerate(self.rendering_auxes):
            if self.rendering_auxes_on_inview[idx]:
                data_dict['inviews_dict'] = render_aux(data_dict['inviews_dict'])
            if self.rendering_auxes_on_outview[idx]:
                data_dict['outviews_dict'] = render_aux(data_dict['outviews_dict'])

        return data_dict

    def collate(self, batch_dict, mode):
        # 因为batch_size就是1
        return batch_dict[0]
    