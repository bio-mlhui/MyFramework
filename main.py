import os
import argparse
from argparse import Namespace
import logging
import shutil
import wandb
import torch
import importlib
import copy   

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


def run(process_id, trainer_configs, trainer_mode, trainer_name, gpu_ids):
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
    from trainers import task_entrypoint
    # 所有任务都用相同的rvos trainer
    create_trainer = task_entrypoint('rvos')
    trainer = create_trainer(configs=trainer_configs, 
                            process_id=process_id, 
                            device_id=gpu_ids[process_id], 
                            num_processes=len(gpu_ids))
    
    if 'train' in trainer_mode:
        trainer.train()

    elif trainer_mode == 'evaluate_ckpt':
        trainer.evaluate_ckpt()
        
    elif trainer_mode == 'evaluate_dir':
        evaluate_dir = trainer_configs['trainer_ckpts_dir']
        ckpt_dirs = os.listdir(evaluate_dir)
        ckpt_dirs = sorted([a for a in ckpt_dirs if a.startswith('epoch')])
        for ckpt in ckpt_dirs:
            ckpt_epoch = int(ckpt.split('_')[-1])
            trainer_ckpt = os.path.join(evaluate_dir, ckpt, f'{ckpt_epoch:02d}.pth.tar')
            trainer.load_ckpt(trainer_ckpt)
            trainer.evaluate_ckpt()
            
    elif trainer_mode == 'visualize_ckpt':
        trainer.visualize_ckpt()
    else:
        raise ValueError()
    
    wandb.finish()

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
    parser.add_argument('--data_dir', type=str, default='/hpc2hdd/home/hxu047/datasets')
    parser.add_argument('--pt_dir', type=str, default='/hpc2hdd/home/hxu047/pt')
    parser.add_argument('--work_dir', type=str, default='/hpc2hdd/home/hxu047/workspace/rvos_encoder')
    
    parser.add_argument('--task', type=str,  default='rvos')  # rvos
    parser.add_argument('--group', type=str,  default='a2ds_schedule') # a2ds_schedule / yrvos_schedule / yrvos_v300s1999_schedule
    parser.add_argument('--schedule_model_configs', type=str, required=True,)
    
    parser.add_argument('--mode', type=str) # train_resume / train_attmpt / evaluate_ckpt / visualzie_ckpt
    parser.add_argument('--trainer_ckpt', type=str, default='')
    parser.add_argument('--seed', type=str, default=2023)
    parser.add_argument('--wandb_mode', type=str, default='offline')
    args = parser.parse_args()

    if args.schedule_model_configs == 'model51':
        import yaml
        with open('/home/xhh/workspace/rvos_encoder/RVOS/a2ds_schedule/[model_51]/configs.yaml', 'r') as f:
            configs = yaml.load()
    config_file = '.'.join([args.task.upper(), args.group, args.schedule_model_configs, args.schedule_model_configs])
    config_file_module = importlib.import_module(config_file)
    configs = config_file_module.trainer_configs

    configs['out_dir'] = os.path.join('./', args.task.upper(), args.group, args.schedule_model_configs)
    
    configs['data']['data_dir'] = args.data_dir
    configs['data']['pt_tokenizer_dir'] = args.pt_dir
    configs['model']['pt_dir'] = args.pt_dir
    configs['model']['work_dir'] = args.work_dir
    configs['seed'] = args.seed
    if args.mode == 'train_resume':
        assert args.trainer_ckpt != '' and os.path.exists(args.trainer_ckpt) and os.path.exists(configs['out_dir'])
        configs['wandb'] = {
            'project': args.task,
            'group': args.group,
            'name': args.schedule_model_configs,
            'id': f'{args.task}_{args.group}_{args.schedule_model_configs}_inferbug2',
            'mode': args.wandb_mode,
            'resume': 'must',
            'configs': copy.deepcopy(configs)
        }
        configs['trainer_ckpt'] = args.trainer_ckpt
        
    elif args.mode == 'train_attmpt': # 重写
        if 'stand' not in args.schedule_model_configs:
            # checkpoint文件可能和out_dir不在同一个目录
            if os.path.exists(configs['out_dir']):
                answer = input(f'相同的实验存在 {configs["out_dir"]} 重写吗? \n' )
                if answer == 'y':
                    pass
                else:
                    exit()
            
        configs['wandb'] = {
            'project': args.task,
            'group': args.group,
            'name': args.schedule_model_configs,
            'id': f'{args.task}_{args.group}_{args.schedule_model_configs}_inferbug2',
            'mode': args.wandb_mode,
            'resume': None,
            'configs': copy.deepcopy(configs)
        }
        configs['trainer_ckpt'] = args.trainer_ckpt
        
    elif args.mode == 'evaluate_ckpt':
        if args.trainer_ckpt == '':
            logging.info('你在评估一个没有trainer ckpt指定的 完全初始化的模型')
            print('你在评估一个没有trainer ckpt指定的 完全初始化的模型')
        else:
            assert args.trainer_ckpt != '' and os.path.exists(args.trainer_ckpt) and os.path.exists(configs['out_dir'])
        configs['wandb'] = {
            'project': args.task,
            'group': args.group,
            'name': args.schedule_model_configs,
            'id': f'{args.task}_{args.group}_{args.schedule_model_configs}_inferbug2',
            'mode': args.wandb_mode,
            'resume': None,
            'configs': copy.deepcopy(configs)
        }
        configs['trainer_ckpt'] = args.trainer_ckpt

    elif args.mode == 'evaluate_dir': # attmpt, 重新
        assert args.trainer_ckpt != '' and os.path.exists(args.trainer_ckpt) and os.path.exists(configs['out_dir'])
        configs['wandb'] = {
            'project': args.task,
            'group': args.group,
            'name': args.schedule_model_configs,
            'id': f'{args.task}_{args.group}_{args.schedule_model_configs}_inferbug2',
            'mode': args.wandb_mode,
            'resume': None,
            'configs': copy.deepcopy(configs)
        }   
        configs['trainer_ckpt'] = ''
        configs['trainer_ckpts_dir'] = args.trainer_ckpt
        
    elif args.mode == 'visualize_ckpt':
        assert args.trainer_ckpt != '' and os.path.exists(args.trainer_ckpt)
        configs['wandb'] = None
        configs['trainer_ckpt'] = args.trainer_ckpt
    
    else:
        raise ValueError()      

    gpu_ids = list(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert len(set(gpu_ids)) == len(gpu_ids)
    gpu_ids = list(range(len(gpu_ids)))
    
    print('不管你如何改变代码, 你必须保证每个config.py文件 运行的结果不能变')


    if len(gpu_ids) > 1:
        torch.multiprocessing.spawn(run, nprocs=len(gpu_ids), args=(configs, args.mode, args.task, gpu_ids))
    elif len(gpu_ids) == 1:
        run(process_id=0, trainer_configs=configs, trainer_mode=args.mode, trainer_name=args.task, gpu_ids=gpu_ids)



    
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
    
    