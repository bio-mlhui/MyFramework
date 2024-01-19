from detectron2.utils.registry import Registry


EVALUATOR_REGISTRY = Registry('EVALUATOR')
MAPPER_REGISTRY = Registry('MAPPER')

class Mapper:
    def __init__(self, 
                meta_idx_shift,
                dataset_meta,) -> None:
        self.meta_idx_shift = meta_idx_shift # 用于logging
        self.visualized_meta_idxs = dataset_meta.get('visualize_meta_idxs') # dataset里 需要可视化的meta_idx的

    def _call(self, data_dict):
        pass

    def __call__(self, data_dict):
        meta_idx = data_dict['meta_idx']
        ret = self._call(data_dict)
        ret['meta_idx'] = meta_idx + self.meta_idx_shift
        if meta_idx in self.visualized_meta_idxs:
            ret['visualize'] = True
        else:
            ret['visualize'] = False
        return ret
    

"""  
1. 对于一个task来说, 
    为了教会模型,  可以对data按照不同方式 进行 解构和认知; 
        这些不同的解构和认知 对应多个不同的 train api

    测试模型只有一个eval_api, 就是用户的输入
        比如RVOS_offline就是video, refer, masks, has_ann
        比如RVOS_online就是video, refer, masks, request_ann

    一个task只实现一个augmentation, 提供aug_api
        aug_api 需要融合 train_api_mapper.aug, eval_api_mapper.aug, model.sample.output_api 三者

每个model的输入是不同的, 但是可能对data的解构和认知 是一样的,
所以每个api 类都可以通过一个model_aux进行类实例化
api<model_aux>, 
    每个api有默认的keys, 也可以根据model_aux添加新的kys

2. 每个api都要实现
    meta2api_mapper: 从meta到 api的映射
    aux_mapper-collator: 
        aux_mapper 从api到api<model_aux>的映射, 
            作用: 
                1. 有些model里的操作, 为了速度更快, 转到 data_loader 里计算
                2. 对于两个具有不同输入的model, 

        collator 最model的input
"""






