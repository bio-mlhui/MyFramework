
import os
from collections import defaultdict

if os.getenv('CURRENT_TASK') == "RVOS":
    from . import rvos
elif os.getenv('CURRENT_TASK') == 'VIS':
    from . import vis
elif os.getenv('CURRENT_TASK') == 'RENDER':
    from . import render
elif os.getenv('CURRENT_TASK') == 'PSC':
    from . import psc
elif os.getenv('CURRENT_TASK') == 'VIDVID':
    from . import vidvid
else:
    raise ValueError()
# 合并多个训练集成一个train set，每次eval对每个eval dataset进行测试
def build_schedule(configs, model_input_mapper, model_input_collate_fn):
    import logging
    from functools import partial
    import detectron2.utils.comm as comm
    from torch.utils.data import DataLoader, ConcatDataset
    from .registry import MAPPER_REGISTRY, EVALUATOR_REGISTRY
    from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset, MetadataCatalog
    from data_schedule.utils.sampler import Evaluate_ExactSampler_Distributed, Train_InfiniteSampler_Distributed
    DatasetCatalog.register('global_dataset', func=lambda: [])
    datasets = {'train': [], 'evaluate': []}
    meta_idx_shift = 0 # train, eval都对meta_idx进行shift, 每个set的idx都在独立的范围内
    for mode in ['train', 'evaluate']:
        for dataset_name in configs['data'][mode].keys():
            dataset_assume_mode = MetadataCatalog.get(dataset_name).get('mode')
            # 注册的时候的mode只是一个预先的定义, meta(数据集), mapper(任务), model(任务)
            # 由于train/test只和任务有关, meta不对meta有强制的要求
            if (dataset_assume_mode != mode) and (dataset_assume_mode != 'all'): 
                logging.warning(f'{dataset_name} 的预设是用于 {dataset_assume_mode} 而非{mode}')
            dataset_dicts = DatasetFromList(DatasetCatalog.get(dataset_name), 
                                            copy=configs['data'][mode][dataset_name].pop('dcopy', True), 
                                            serialize=configs['data'][mode][dataset_name].pop('serialize', True))
            mapper = MAPPER_REGISTRY.get(configs['data'][mode][dataset_name]['mapper']['name'])(mode=mode,
                                                                                                dataset_name=dataset_name, 
                                                                                                configs=configs,
                                                                                                meta_idx_shift=meta_idx_shift if mode == 'train' else 0)
            meta_idx_shift += len(dataset_dicts)
            dataset = MapDataset(dataset_dicts, partial(composition, mappers=[mapper, 
                                                                              partial(model_input_mapper, mode=mode)]))
            if mode == 'train':
                datasets[mode].append(dataset)
            else:
                datasets[mode].append((dataset_name, dataset))
    MetadataCatalog.get('global_dataset').set(subset_list=list(configs['data'][mode].keys()))
    train_dataset = ConcatDataset(datasets['train'])
    logging.debug(f'Total number of training meta: {len(train_dataset)}')

    # 把inifnite stream 给到每个sampler里
    train_loader_splits = configs['optim']['splits'] # 每个loader分割的idx
    batch_sizes = configs['optim']['batch_sizes']
    splits = list(zip(train_loader_splits[:-1], train_loader_splits[1:]))
    assert len(splits) == (len(batch_sizes))
    inf_stream_fn = partial(infinite_indices,
                            seed=configs['stream_idx_seed'], # 每个进程相同的seed
                            batch_sizes=configs['optim']['batch_sizes'],
                            splits=configs['optim']['splits'],
                            one_batch_two_epoch=configs['optim']['one_batch_two_epoch'],
                            dataset_length=len(train_dataset),
                            shuffle=True) # 每次调用应该是相同的generator
    # 0, 200, 400,
    # 16, 8, 4
    # split假设是一个无限的sample index
    # 最后一个batch_size没有限制
    # 保证split_t - split_t-1 整除 batch_size_t 
    train_samplers = []
    train_loaders = []
    pin_memory = configs['data'].pop('pin_memory', True)
    for btch_size, (range_start, range_end) in zip(batch_sizes, splits):
        if range_end is not None:
            assert (range_end - range_start) % btch_size == 0, '要保证每个split的长度可以被当时的batch_size整除'
        assert btch_size % comm.get_world_size() == 0, '每个batch_size必须被gpu数量整除'
        each_process_batch_size = int(btch_size / comm.get_world_size())
        loader_sampler = Train_InfiniteSampler_Distributed(inf_stream_fn=inf_stream_fn,
                                                           start_idx=range_start,
                                                           end_idx=range_end,)
        train_samplers.append(loader_sampler)
        train_loaders.append(DataLoader(train_dataset,
                                        batch_size=each_process_batch_size,
                                        sampler=loader_sampler,
                                        collate_fn=partial(model_input_collate_fn, mode='train'), 
                                        num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
                                        pin_memory=pin_memory,
                                        persistent_workers=True if int(os.getenv('TORCH_NUM_WORKERS')) > 0 else False))

    evaluators = []
    for eval_dataset_name, eval_dataset in datasets['evaluate']:
        logging.debug(f'Number of evaluate meta in {eval_dataset_name}: {len(eval_dataset)}')
        loader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            sampler=Evaluate_ExactSampler_Distributed(eval_dataset),
                            collate_fn=partial(model_input_collate_fn, mode='evaluate'),
                            num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
                            pin_memory=pin_memory,
                            persistent_workers=True if int(os.getenv('TORCH_NUM_WORKERS')) > 0 else False)
        
        evaluator = EVALUATOR_REGISTRY.get(configs['data']['evaluate'][eval_dataset_name]['evaluator']['name'])(configs=configs,
                                                                                                                dataset_name=eval_dataset_name,
                                                                                                                data_loader=loader)
        evaluators.append((eval_dataset_name, evaluator))

    return train_samplers, train_loaders, partial(evaluate_call, evaluators=evaluators)

def composition(data_dict, mappers):
    for mappper in mappers:
        data_dict = mappper(data_dict)
        if data_dict is None:
            return None
    return data_dict

def evaluate_call(evaluators, model, output_dir):
    import detectron2.utils.comm as comm
    ret = {}
    for eval_dataset_name, evaluator in evaluators:
        metric_dict = evaluator(model=model,output_dir=output_dir)
        if comm.is_main_process():
            for key, value in metric_dict.items():
                assert f'{key}_{eval_dataset_name}' not in ret
                ret[f'{key}_{eval_dataset_name}'] = value
        comm.synchronize()
    return ret


def _infinite_indices(seed, dataset_length, shuffle=True,):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    while True:
        if shuffle:
            yield from torch.randperm(dataset_length, generator=g).tolist()
        else:
            yield from torch.arange(dataset_length).tolist()

def infinite_indices(seed, 
                     dataset_length, 
                     batch_sizes, 
                     splits, 
                     one_batch_two_epoch='just_use',
                     shuffle=True): # 'abandon', 'just_use', 'pad'
    # 生成一个无限的infinite stream, 保证每次运行返回的都相同
    import torch
    import math
    g = torch.Generator()
    g.manual_seed(seed)

    split_ranges = list(zip(splits[:-1], splits[1:]))
    assert len(split_ranges) == (len(batch_sizes))
    stream = _infinite_indices(seed, dataset_length=dataset_length, shuffle=shuffle)

    stream_throw_cnt = 0
    cnt = 0
    for (range_start, range_end), btch_size in zip(split_ranges, batch_sizes):
        assert cnt == range_start
        if range_end == None:
            range_end = math.inf
        
        # stream_throw_cnt = 5996, stream_throw_cnt + infinite_btch_size = 6000(下一个batch的第一个sample的index), epoch_milestone是6000, 不会抽到6000
        while cnt < range_end:
            epoch_milestone = ((stream_throw_cnt // dataset_length) + 1 ) * dataset_length
            if (stream_throw_cnt < epoch_milestone) and (stream_throw_cnt + btch_size > epoch_milestone) and (one_batch_two_epoch != 'just_use'):
                if one_batch_two_epoch == 'abandon':
                    for _ in range(epoch_milestone - stream_throw_cnt):
                        abandon = next(stream)
                        stream_throw_cnt += 1

                elif one_batch_two_epoch == 'pad':
                    diff = stream_throw_cnt + btch_size - epoch_milestone
                    num_throw = btch_size - diff
                    rand_idxs = torch.randperm(dataset_length, generator=g)[:diff].tolist()
                    for _ in range(num_throw):
                        cnt += 1
                        stream_throw_cnt += 1
                        yield next(stream)
                    for idx in rand_idxs:
                        cnt += 1
                        yield idx
                else:
                    raise ValueError()
            else:
                for _ in range(btch_size):
                    cnt += 1
                    stream_throw_cnt += 1
                    yield next(stream)  

        # cnt永远是跟着batch走的, 由于range_end-range_start 可以被batch整除, 所以不会出现cnt > range_end的情况 
        assert cnt == range_end


# 每个scene是一个训练集，有它自己的eval训练集, 调用每个train/test
# def build_render_schedule(configs, model_input_mapper, model_input_collate_fn):
#     import logging
#     from functools import partial
#     from torch.utils.data import DataLoader, ConcatDataset
#     from .registry import MAPPER_REGISTRY, EVALUATOR_REGISTRY
#     from detectron2.data import DatasetCatalog, DatasetFromList, MapDataset, MetadataCatalog
#     from data_schedule.utils.sampler import Evaluate_ExactSampler_Distributed, Train_InfiniteSampler_Distributed
#     train_datasets = [] 
#     # text3d/text4d: {'text':}; {'text':}
#     # video4d/image3d: list[{},{},{},{}]
#     eval_datasets = []
#     # text3d/text4d: 
#     # video4d/image3d: 
#     for mode in ['train', 'evaluate']:
#         for dataset_name in configs['data'][mode].keys():
#             dataset_assume_mode = MetadataCatalog.get(dataset_name).get('mode')
#             if dataset_assume_mode != mode:
#                 logging.warning(f'{dataset_name} 的预设是用于 {dataset_assume_mode} 而非{mode}')
#             dataset_dicts = DatasetFromList(DatasetCatalog.get(dataset_name), copy=True, serialize=True)
#             mapper = MAPPER_REGISTRY.get(configs['data'][mode][dataset_name]['mapper']['name'])(mode=mode,
#                                                                                                 dataset_name=dataset_name, 
#                                                                                                 configs=configs,
#                                                                                                 meta_idx_shift=0)
#             dataset = MapDataset(dataset_dicts, partial(composition, mappers=[mapper, partial(model_input_mapper, mode=mode)]))
#             if mode == 'train':
#                 train_datasets.append(dataset)
#             else:
#                 eval_datasets.append(dataset)

#     num_scenes = len(train_datasets)
#     iters_by_scene = configs['optim']['iters_by_scene'] 
#     if type(iters_by_scene) == int:
#         iters_by_scene = [iters_by_scene] * num_scenes
    
#     if ckpts_by_scene == 'last':
#         ckpts_by_scene = [[haosen] for haosen in iters_by_scene]
#     elif type(ckpts_by_scene) == int:
#         assert ckpts_by_scene > 0
#         ckpts_by_scene = [list(range(ckpts_by_scene, haosen, ckpts_by_scene))  for haosen in iters_by_scene]
#     else:
#         raise ValueError()

#     ckpts_by_scene = configs['optim']['ckpts_by_scene']

#     bch_by_scene = configs['optim']['batch_sizes']
#     if type(bch_by_scene) == int:
#         bch_by_scene = [bch_by_scene] * num_scenes
#     for bch in bch_by_scene:
#         assert bch == 1



#     train_loaders = []
#     for btch_size, ckpt_list, scene_iter in zip(bch_by_scene, ckpts_by_scene, iters_by_scene):
#         each_process_batch_size = int(btch_size / comm.get_world_size())
#         loader_sampler = Train_InfiniteSampler_Distributed(inf_stream_fn=inf_stream_fn,
#                                                            start_idx=range_start,
#                                                            end_idx=range_end,)
#         train_samplers.append(loader_sampler)
#         train_loaders.append(DataLoader(train_dataset,
#                                         batch_size=each_process_batch_size,
#                                         sampler=loader_sampler,
#                                         collate_fn=partial(model_input_collate_fn, mode='train'), 
#                                         num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
#                                         pin_memory=True,
#                                         persistent_workers=True))

#     evaluators = []
#     for eval_dataset_name, eval_dataset in datasets['evaluate']:
#         logging.debug(f'Number of evaluate meta in {eval_dataset_name}: {len(eval_dataset)}')
#         loader = DataLoader(eval_dataset, 
#                             batch_size=1, 
#                             sampler=Evaluate_ExactSampler_Distributed(eval_dataset),
#                             collate_fn=partial(model_input_collate_fn, mode='evaluate'),
#                             num_workers=int(os.getenv('TORCH_NUM_WORKERS')),
#                             pin_memory=True,
#                             persistent_workers=True)
        
#         evaluator = EVALUATOR_REGISTRY.get(configs['data']['evaluate'][eval_dataset_name]['evaluator']['name'])(configs=configs,
#                                                                                                                 dataset_name=eval_dataset_name,
#                                                                                                                 data_loader=loader)
#         evaluators.append((eval_dataset_name, evaluator))

#     return train_samplers, train_loaders, partial(evaluate_call, evaluators=evaluators)

# build_learning_render_schedule 
        
# 每个scene当成一个test sample, 然后只进行evaluate


