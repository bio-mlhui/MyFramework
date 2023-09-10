_rios_model_entrypoints = {}

def register_rios_model(fn):
    rios_model_name = fn.__name__
    _rios_model_entrypoints[rios_model_name] = fn

    return fn

def rios_model_entrypoint(rios_model_name):
    try:
        return _rios_model_entrypoints[rios_model_name]
    except KeyError as e:
        print(f'rios moel {rios_model_name} not found')