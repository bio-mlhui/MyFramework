import os
import argparse
import logging
import wandb
import importlib
from trainers import task_to_trainer
import detectron2.utils.comm as comm # deepspeed也能用
import logging
import yaml
import torch
from .main import init_process_group_and_set_device, _highlight, _ColorfulFormatter, set_logging_file

def run(rank, configs, world_size):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = "4"
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = "1"
    os.environ["DGLBACKEND"] = "pytorch"
    logging.getLogger('penman').setLevel(logging.WARNING)    
    logging.getLogger('PIL').setLevel(logging.WARNING) 
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)   
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    init_process_group_and_set_device(world_size, process_id=rank, device_id=rank)
    if comm.is_main_process():
        mode = configs['trainer_mode']
        out_dir = configs['out_dir']
        if mode == 'eval':
            num_of_eval_times = len([eval_txt for eval_txt in os.listdir(out_dir) if eval_txt.endswith('eval.txt')])
            set_logging_file(out_dir, f"eval.txt", mode='w')
            path = os.path.join(out_dir, f"config_eval.yaml")
        else:
            num_of_train_times = len([train_txt for train_txt in os.listdir(out_dir) if train_txt.endswith('train.txt')])
            if 'resume' in mode:
                set_logging_file(out_dir, f"train.txt", mode='a')
            else:
                set_logging_file(out_dir, f"train.txt", mode='w')
            path = os.path.join(out_dir, f"config_train.yaml")
            
        logging.debug("Running with full config:\n{}".format(_highlight(yaml.dump(configs, default_flow_style=False), ".yaml")))
        with open(path, "w") as f:
            f.write(yaml.dump(configs, default_flow_style=False))
        logging.debug("Full config saved to {}".format(path))
        wandb.init(   
            project=configs['task'],
            group=configs['group'], 
            name=configs['config'],  
            id=configs['wandb_id'], 
            resume=configs['wandb_resume'],  # resume或者是never
            config=configs,
            mode=configs['wandb_mode'],
        )  
    comm.synchronize()
    # init according to ( initckpt/path, initckpt/load_sampler, initckpt/load_optimizer )
    trainer = task_to_trainer[configs['task']](configs=configs)
    comm.synchronize()
    if configs['trainer_mode'] == 'eval':
        eval_ckpts = configs['eval_ckpts']
        for ckpt in eval_ckpts:
            trainer.load_ckpt(ckpt, load_model=True, load_schedule=True, load_random=False, load_optimize=False)
            trainer.evaluate()

    else:
        if configs['trainer_mode'] == 'train_resume':
            ckpt_dirs = os.listdir(configs['out_dir'])
            # epc1_iter5000/ckpt.pth.tar
            ckpt_dirs = sorted([a for a in ckpt_dirs if a.startswith('epc')], key=lambda x:int(x.split('sap[')[-1][:-1]))
            trainer_ckpt = '/'.join([configs['out_dir'], ckpt_dirs[-1], 'ckpt.pth.tar'])
            trainer.load_ckpt(trainer_ckpt, load_model=True, load_schedule=True, load_random=True, load_optimize=True)
        trainer.train()
    
    if comm.is_main_process():
        wandb.finish()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--trainer_mode', type=str, default='train_attmpt')  # train_attmpt train_resume eval
    parser.add_argument('--eval_path', type=str, default='') #  # 如果是dir， 如果是file
    parser.add_argument('--disable_wandb', action='store_true')  # default: False
    parser.add_argument('--append_wandb_id',type=str, default='')
    parser.add_argument('--resume_wandb', action='store_true') # default: False

    parser.add_argument('--gui_ip', type=str, default="127.0.0.1")
    parser.add_argument('--gui_port', type=int, default=6009)

    args = parser.parse_args()

    task, group, config, config2 = args.config_file.split('/')[-4:]
    assert config == config2[:-3]
    config_file = '.'.join(['output', task, group, config, config])
    configs = importlib.import_module(config_file).trainer_configs
    configs['task'], configs['group'], configs['config'] = task, group, config
    configs['out_dir'] = os.path.join('./', 'output', task, group, config)
    configs['trainer_mode'] = args.trainer_mode
    configs['gui_ip'] = args.gui_ip
    configs['gui_port'] = args.gui_port

    wandb_id = f'{task}_{group}_{config}'
    if args.append_wandb_id != '':
        wandb_id = wandb_id + '_' + args.append_wandb_id
    configs['wandb_id'] = wandb_id
    configs['wandb_mode'] = 'disabled' if args.disable_wandb else os.environ['WANDB_MODE']
    configs['wandb_resume'] = 'must' if args.resume_wandb else 'never' 
    # debug模式下, never也能运行, 直到debug结束; running情况下, 每次不resume的话, wandb_id必须不一样

    if configs['trainer_mode'] == 'eval':
        eval_ckpts = []
        eval_path = args.eval_path # dir/file
        assert eval_path != '', f'evaluate情况下, eval_path: {args.eval_path} 不能是空,'
        
        if os.path.isfile(eval_path):
            eval_ckpts.append(eval_path)

        elif os.path.isdir(eval_path):
            # 按照sap的大小依顺序evaluate每个ckpt
            ckpt_dirs = os.listdir(eval_path) # RVOS/method1/
            ckpt_dirs = [cd for cd in ckpt_dirs if os.path.isdir(os.path.join(eval_path, cd))]
            # epc[1]_iter[5000]_sap[60009]
            ckpt_dirs = sorted([cd for cd in ckpt_dirs if cd.startswith('epc')], key=lambda x:int(x.split('sap[')[-1][:-1]))
            eval_ckpts = [os.path.join(eval_path, cd, f'ckpt.pth.tar') for cd in ckpt_dirs]
            eval_ckpts = [eval_c for eval_c in eval_ckpts if os.path.exists(eval_c)]
        else:
            raise ValueError()
        configs['eval_ckpts'] = eval_ckpts
    else:
        # if (configs['trainer_mode'] == 'train_attmpt') and ('debug' not in configs['config']):
        #     if os.path.exists(os.path.join(configs['out_dir'], 'train.txt')):
        #         answer = input(f'{configs["config"]} 有跑的记录, 要重写整个out_dir嘛\n' )
        #         if answer != 'y':
        #             exit()  
        pass
    
    

    gpu_ids = list(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    assert len(set(gpu_ids)) == len(gpu_ids)
    gpu_ids = list(range(len(gpu_ids)))
    
    if len(gpu_ids) > 1:
        torch.multiprocessing.spawn(run, nprocs=len(gpu_ids), args=(configs, len(gpu_ids)))
    elif len(gpu_ids) == 1:
        run(rank=0, configs=configs, world_size=len(gpu_ids))