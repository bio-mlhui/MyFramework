_task_entrypoints = {}
def register_task(fn):
    task_name = fn.__name__
    _task_entrypoints[task_name] = fn

    return fn
def task_entrypoint(task_name):
    try:
        return _task_entrypoints[task_name]
    except KeyError as e:
        print(f'RVOS moel {task_name} not found')