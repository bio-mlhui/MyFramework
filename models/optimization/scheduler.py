import torch
from functools import partial
import logging
import numpy as np
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

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp  / lr_init

    return helper


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
            
    # elif name == "WarmupPolyLR":
    #     return WarmupPolyLR(
    #         optimizer,
    #         cfg.SOLVER.MAX_ITER,
    #         warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
    #         warmup_iters=cfg.SOLVER.WARMUP_ITERS,
    #         warmup_method=cfg.SOLVER.WARMUP_METHOD,
    #         power=cfg.SOLVER.POLY_LR_POWER,
    #         constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
    #     ), unit
        
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
        raise ValueError()



