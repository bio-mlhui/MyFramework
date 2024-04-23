import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


class ModelAgnostic_Optimizer:
    def __init__(self,
                 optimizer: Optimizer =None,
                 log_lr_group_name_to_idx= None,
                 configs=None,
                 **kwargs
                 ):
        """
        trainer和模型优化之间的缓冲器

        适合Optimizer不对模型进行调整, log_lr在训练的时候不改变的情况

        """
        self.optimizer = optimizer
        self.log_lr_group_name_to_idx = log_lr_group_name_to_idx

        self.set_to_none = configs['set_to_none'] if 'set_to_none' in configs else True
        
    def step(self, 
             closure=None,
             num_iterations=None,
             num_samples=None,
             optimizer_step_dict=None,
             loss_weight=None,
             loss_dict_unscaled=None,
             **kwargs):
        self.optimizer.step(closure=closure)
        
    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(set_to_none=self.set_to_none) # # delete gradient

    def load_state_dict(self, state_dict=None, **kwargs):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self, **kwargs):
        return self.optimizer.state_dict()
    
    @property
    def param_groups(self, **kwargs):
        return self.optimizer.param_groups      


    def get_log_lr_dicts(self,):
        llg = {}
        for log_lr_group_name, log_lr_group_idx in self.log_lr_group_name_to_idx.items():
            if log_lr_group_idx is None:
                llg[f'lr_group_{log_lr_group_name}'] = 0
            else:
                llg[f'lr_group_{log_lr_group_name}'] = self.optimizer.param_groups[log_lr_group_idx]["lr"]
        return llg



class ModelAgnostic_Scheduler:
    def __init__(self, 
                 scheduler:LRScheduler=None,
                 configs=None,
                 **kwargs):
        self.scheduler = scheduler
        
    def step(self,
             num_iterations=None,
             **kwargs):
        self.scheduler.step()
    
    def state_dict(self, 
                   **kwargs):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, 
                        state_dict=None,
                        **kwargs,):
        self.scheduler.load_state_dict(state_dict)
        

  
