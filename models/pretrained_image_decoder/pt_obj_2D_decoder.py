_pt_obj_2d_decoder_entrypoints = {}
def register_pt_obj_2d_decoder(fn):
    pt_obj_2d_decoder_name = fn.__name__
    _pt_obj_2d_decoder_entrypoints[pt_obj_2d_decoder_name] = fn

    return fn
def pt_obj_2d_decoder_entrypoint(pt_obj_2d_decoder_name):
    try:
        return _pt_obj_2d_decoder_entrypoints[pt_obj_2d_decoder_name]
    except KeyError as e:
        print(f'RVOS moel {pt_obj_2d_decoder_name} not found')
