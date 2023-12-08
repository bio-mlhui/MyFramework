import os
import argparse
from argparse import Namespace
import logging
import shutil
import wandb
import torch
import importlib
import copy   
from trainers import task_entrypoint

import torch.distributed as dist
from util.misc import setup_for_distributed

def namespace_to_dict(namespace):
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Namespace):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def fuse_args_configs(args, configs):
    namespace = argparse.Namespace()
    for key, value in vars(configs).items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    for key, value in vars(args).items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_cnt_attempt(output_dir, id):
    dirs = [d for d in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir,d))]
    cnt = 0
    for d in dirs:
        if d.startswith(id):
            cnt += 1
    return cnt

def copy_file(file1_name, file2_name):
    file1 = open(file1_name,"r")
    file2 = open(file2_name,"w")

    s = file1.read()
    w = file2.write(s)

    file1.close()
    file2.close() 
    

def set_logging_file(output_dir, file_name, mode='a'):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(output_dir, file_name), mode=mode)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.DEBUG)


def init_process_group_and_set_device(backend, world_size, process_id, device_id):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    assert backend == 'nccl'
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    if world_size > 1:
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        setup_for_distributed(process_id == 0)
    return device

def run(process_id, trainer_configs, trainer_mode, trainer_name, device):

    if process_id == 0:
        if trainer_mode == 'train_attmpt':
            set_logging_file(trainer_configs['out_dir'], "stdout_train.txt", mode='w')
        elif trainer_mode == 'train_resume':
            set_logging_file(trainer_configs['out_dir'], "stdout_train.txt")
        elif trainer_mode == 'evaluate_ckpt':
            set_logging_file(trainer_configs['out_dir'], "stdout_eval.txt")
        elif trainer_mode == 'visualize_ckpt':
            set_logging_file(trainer_configs['out_dir'], "stdout_vis.txt")
        elif trainer_mode == 'evaluate_dir':
            set_logging_file(trainer_configs['out_dir'], 'stdout_eval.txt')
        else:
            raise ValueError()
    
    create_trainer = task_entrypoint(trainer_name)
    trainer = create_trainer(configs=trainer_configs,  # configs里的ckpt只要不是空就load ckpt
                            process_id=process_id, 
                            device=device, 
                            num_processes=len(gpu_ids),)
    
    if process_id == 0:
        wandb_configs = trainer_configs['wandb']
        n_parameters = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logging.info(f'number of params:{n_parameters}')
        print(f'number of params:{n_parameters}')
        wandb.init(   
            project=wandb_configs['project'],
            group=wandb_configs['group'],   
            id=wandb_configs['id'], 
            resume=wandb_configs['resume'], 
            name=wandb_configs['name'],
            config=wandb_configs['configs'],
            mode=wandb_configs['mode'],
        )

    if len(gpu_ids) > 1: 
        dist.barrier()

    if 'train' in trainer_mode:
        trainer.train()

    elif trainer_mode == 'evaluate_ckpt':
        trainer.evaluate_ckpt()

    elif trainer_mode == 'visualize_ckpt':
        trainer.visualize_ckpt()

    elif trainer_mode == 'evaluate_dir':
        evaluate_dir = trainer_configs['trainer_ckpts_dir']
        ckpt_dirs = os.listdir(evaluate_dir)
        ckpt_dirs = sorted([a for a in ckpt_dirs if a.startswith('epoch')])
        for ckpt in ckpt_dirs:
            ckpt_epoch = int(ckpt.split('_')[-1])
            trainer_ckpt = os.path.join(evaluate_dir, ckpt, f'{ckpt_epoch:02d}.pth.tar')
            trainer.load_ckpt(trainer_ckpt, strict_load=trainer_configs['strict_load'], resume=trainer_configs['resume'])
            trainer.evaluate_ckpt()
    else:
        raise ValueError()
    
    if process_id == 0:
        wandb.finish()

gpu_ids = list(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
assert len(set(gpu_ids)) == len(gpu_ids)
gpu_ids = list(range(len(gpu_ids)))
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


if __name__=="__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["DGLBACKEND"] = "pytorch"
    logging.getLogger('penman').setLevel(logging.WARNING)    
    logging.getLogger('PIL').setLevel(logging.WARNING)  
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])

    parser.add_argument('--data_dir', type=str, default='/hpc2hdd/home/hxu047/datasets')
    parser.add_argument('--pt_dir', type=str, default='/hpc2hdd/home/hxu047/pt')
    parser.add_argument('--work_dir', type=str, default='/hpc2hdd/home/hxu047/workspace/rvos_encoder')
    
    parser.add_argument('--task', type=str,  required=True,) # rios/rvos/r3os
    parser.add_argument('--group', type=str,  required=True,) # 数据流配置
    parser.add_argument('--config', type=str, required=True,) # 模型配置
    
    parser.add_argument('--mode', type=str)  # train_attmpt train_resume evaluate_ckpt evaluate_dir visualize_ckpt
    parser.add_argument('--trainer_ckpt', type=str, default='') # trainer初始化的ckpt
    parser.add_argument('--trainer_ckpts_dir', type=str, default='') # mode是evaluate_dir, trainer初始化需要是'', 由一个for循环load每个ckpt
    parser.add_argument('--strict_load', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--wandb_mode', type=str, default='online')
    args = parser.parse_args()

    config_file = '.'.join([args.task.upper(), args.group, args.config, args.config])
    config_file_module = importlib.import_module(config_file)
    configs = config_file_module.trainer_configs
    
    configs['out_dir'] = os.path.join('./', args.task.upper(), args.group, args.config)
    configs['mode'] = args.mode
    configs['data']['data_dir'] = args.data_dir
    configs['data']['pt_tokenizer_dir'] = args.pt_dir
    configs['model']['pt_dir'] = args.pt_dir
    configs['model']['work_dir'] = args.work_dir
    configs['seed'] = args.seed
    configs['wandb'] = {
        'project': args.task,
        'group': args.group,
        'name': args.config,
        'id': f'{args.task}_{args.group}_{args.config}',
        'mode': args.wandb_mode,
        'resume': 'must',
        'configs': copy.deepcopy(configs)
    }
    configs['trainer_ckpt'] = args.trainer_ckpt
    configs['trainer_ckpts_dir'] = args.trainer_ckpts_dir
    configs['strict_load'] = args.strict_load
    configs['resume'] = True    

    if args.mode == 'train_attmpt': # 重写
        if os.path.exists(configs['out_dir']):
            answer = input(f'相同的实验存在 {configs["out_dir"]} 重写吗? \n' )
            if answer == 'y':
                pass
            else:
                exit()
        configs['resume'] = False
        configs['wandb']['resume'] = None

    elif args.mode == 'train_resume':
        # /epoch-1/-1.pth.tar
        assert args.trainer_ckpt != ''
        assert '/'.join('/'.split(args.trainer_ckpt)[:-2]) == configs['out_dir']
        

    elif args.mode == 'evaluate_ckpt':
        pass

    elif args.mode == 'visualize_ckpt': 
        pass

    elif args.mode == 'evaluate_dir':
        assert args.trainer_ckpt == '' and os.path.exists(args.trainer_ckpts_dir) 

    else:
        raise ValueError()      
    
    device = init_process_group_and_set_device(backend=args.backend, world_size=WORLD_SIZE, process_id=WORLD_RANK, device_id=gpu_ids[LOCAL_RANK])
    run(process_id=WORLD_RANK, trainer_configs=configs, trainer_mode=args.mode, trainer_name=args.task, device=device)

  