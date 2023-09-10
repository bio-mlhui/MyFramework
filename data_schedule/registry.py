
_data_schedule_entrypoints = {}

def register_data_schedule(fn):
    data_schedule_name = fn.__name__
    _data_schedule_entrypoints[data_schedule_name] = fn

    return fn


def data_schedule_entrypoints(data_schedule_name):
    try:
        return _data_schedule_entrypoints[data_schedule_name]
    except KeyError as e:
        print(f'data_schedule {data_schedule_name} not found')
