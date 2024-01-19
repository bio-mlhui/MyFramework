from .rvos_aug import RVOS_AUG_REGISTRY
from data_schedule.registry import Mapper
class RVOS_Mapper(Mapper):
    def __init__(self, 
                 meta_idx_shift,
                 dataset_meta,) -> None:
        super().__init__(meta_idx_shift=meta_idx_shift, dataset_meta=dataset_meta)
        self.category_to_id = dataset_meta.get('category_to_ids')
        self.normalize_text_fn = dataset_meta.get('normalize_text_fn') # yrvos/a2ds提供一个normalize_text_fn
        self.get_frames_fn = dataset_meta.get('get_frames_fn') # yrvos/a2ds提供一个get_frames_fn

class RVOS_TrainMapper(RVOS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        self.get_frames_mask_fn = dataset_meta.get('get_frames_mask_fn')    
    
        assert mapper_config['augmentation']['name'] in ['Hflip_RandomResize']

        self.augmentation = RVOS_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])

class RVOS_EvalMapper(RVOS_Mapper):
    def __init__(self, 
                 meta_idx_shift, 
                 dataset_meta,
                 mapper_config) -> None:
        super().__init__(meta_idx_shift, dataset_meta)
        assert mapper_config['augmentation']['name'] in ['RandomResize']
        self.augmentation = RVOS_AUG_REGISTRY.get(mapper_config['augmentation']['name'])(mapper_config['augmentation'])
        


