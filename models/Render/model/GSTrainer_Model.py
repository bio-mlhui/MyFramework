from models.trainer_model_api import Trainer_Model_API

class Trainer_GSModel:
    def __init__(self, **kwargs) -> None:
        pass

    def train(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def eval(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')
    
    def optimize(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def get_lr_group_dicts(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def optimize_state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')


    def load_state_dict(self, **kwargs):
         raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def load_optimize_state_dict(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')

    def __call__(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')
    
    def sample(self, **kwargs):
        raise ValueError('这是一个virtual method, 需要实现一个新的optimize_setup函数')
