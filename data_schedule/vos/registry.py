_vos_metric_entrypoints = {}

def register_vos_metric(fn):
    vos_metric_name = fn.__name__
    if vos_metric_name in _vos_metric_entrypoints:
        raise ValueError(f'vos_metric name {vos_metric_name} has been registered')
    _vos_metric_entrypoints[vos_metric_name] = fn

    return fn

def vos_metric_entrypoint(vos_metric_name):
    try:
        return _vos_metric_entrypoints[vos_metric_name]
    except KeyError as e:
        print(f'vos_metric Name {vos_metric_name} not found')