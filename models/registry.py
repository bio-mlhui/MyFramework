_model_entrypoints = {}

def register_model(fn):
    model_name = fn.__name__
    if model_name in _model_entrypoints:
        raise ValueError(f'model name {model_name} has been registered')
    _model_entrypoints[model_name] = fn

    return fn

# def register_model(aux_mapper):
#     def decorator(fn):
#         model_name = fn.__name__
#         if model_name in _model_entrypoints:
#             raise ValueError(f'model name {model_name} has been registered')
#         _model_entrypoints[model_name] = (fn, aux_mapper)
#         return fn
#     return decorator
def model_entrypoint(model_name):
    try:
        return _model_entrypoints[model_name]
    except KeyError as e:
        print(f'Model Name {model_name} not found')

from detectron2.utils.registry import Registry
MODELITY_INPUT_MAPPER_REGISTRY = Registry("MODELITY_INPUT_MAPPER")

"""
detectron的 backbone registry
detectron的 meta registry: 所有module的名字
对于不同任务, 相同模态来说, encoder是一致可复用的, 
    decoder需要定制
对于不同任务, 不同模态来说, encoder不能复用, decoder需要定制
"""