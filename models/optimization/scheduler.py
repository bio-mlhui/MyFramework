import torch
from functools import partial
import logging
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver import build_lr_scheduler as build_d2_lr_scheduler
from detectron2.projects.deeplab.lr_scheduler import WarmupPolyLR

def polynomial_decay_lambda(step, initial_learning_rate : float=8e-5, end_learning_rate: float=1.5e-5, decay_steps=25, power=1.0):
    step = min(step, decay_steps)
    return (((initial_learning_rate - end_learning_rate) *
            ((1 - step / decay_steps) ** (power))
            ) + end_learning_rate) / initial_learning_rate

def inverse_sqrt_warmup_lambda(step, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if step < num_warmup_steps:
        return float(step) / float(max(1, num_warmup_steps))
    return max(0.0, (num_warmup_steps / step)**0.5)

def build_scheduler(configs, optimizer):
    name = configs['optim']['scheduler']['name']
    scheduler_configs = configs['optim']['scheduler']

    if name == 'static':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda x : 1,
                                                      last_epoch=-1,
                                                      verbose=False)
        return scheduler

    elif name == 'multistep_lr':
        # 按照iteration进行
        # loader_splits = configs['data']['train_loaders']['splits'] # list[int]
        # max_inputs = configs['data']['train_loaders']['max_inputs'] # int
        # batch_sizes = configs['data']['train_loaders']['batch_size'] # list[int]
        # total_iterations = 0
        # for (last_split, split), bs in zip(zip(loader_splits[:-1], loader_splits[1:]), batch_sizes):
        #     total_iterations += (split - last_split) // bs

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=scheduler_configs['milestones'],
                                                        gamma=scheduler_configs['gamma'],
                                                        verbose=scheduler_configs['verbose'],)
        return scheduler
            
    elif name == "WarmupPolyLR":
        return WarmupPolyLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            power=cfg.SOLVER.POLY_LR_POWER,
            constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
        ), unit
    

    elif name == 'polynomial_split':
        group_names = ['model', 'vbb', 'text']
        poly_lambdas = []
        for gname in group_names:
            g_poly_conf = scheduler_configs[gname]
            poly_lambdas.append(partial(polynomial_decay_lambda, initial_learning_rate=g_poly_conf['initial_learning_rate'],
                                                                        end_learning_rate=g_poly_conf['end_learning_rate'],
                                                                        decay_steps=g_poly_conf['decay_steps'],
                                                                        power=g_poly_conf['power']))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=poly_lambdas,
                                                      last_epoch=-1,)
        return scheduler
    
    elif name == 'polynomial_freezebb':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=partial(polynomial_decay_lambda, 
                                                                        initial_learning_rate=scheduler_configs['initial_learning_rate'],
                                                                        end_learning_rate=scheduler_configs['end_learning_rate'],
                                                                        decay_steps=scheduler_configs['decay_steps'],
                                                                        power=scheduler_configs['power'],
                                                                        ),
                                                      last_epoch=-1,
                                                      verbose=True)
        return scheduler

    elif name == 'polynomial':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 
                                                        total_iters=scheduler_configs['total_iters'],
                                                        power=scheduler_configs['power'],
                                                        last_epoch=-1,
                                                        verbose=True)
        return scheduler

    elif name == 'invert_sqrt':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                      lr_lambda=partial(inverse_sqrt_warmup_lambda,
                                                                         num_warmup_steps=scheduler_configs['num_warmup_steps'],
                                                                         num_training_steps=scheduler_configs['num_training_steps']), last_epoch=-1)
        return scheduler

    else:
        return build_d2_lr_scheduler(configs, optimizer), 'step'



