_vis_metric_entrypoints = {}

def register_vis_metric(fn):
    vis_metric_name = fn.__name__
    if vis_metric_name in _vis_metric_entrypoints:
        raise ValueError(f'vis_metric name {vis_metric_name} has been registered')
    _vis_metric_entrypoints[vis_metric_name] = fn

    return fn

def vis_metric_entrypoint(vis_metric_name):
    try:
        return _vis_metric_entrypoints[vis_metric_name]
    except KeyError as e:
        print(f'vis_metric Name {vis_metric_name} not found')


# 每个数据集公用的metric
        
