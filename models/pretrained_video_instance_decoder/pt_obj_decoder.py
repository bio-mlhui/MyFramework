_pt_3d_obj_decoder_entrypoints = {}
def register_pt_3d_obj_decoder(fn):
    pt_3d_obj_decoder_name = fn.__name__
    _pt_3d_obj_decoder_entrypoints[pt_3d_obj_decoder_name] = fn

    return fn
def pt_3d_obj_decoder_entrypoint(pt_3d_obj_decoder_name):
    try:
        return _pt_3d_obj_decoder_entrypoints[pt_3d_obj_decoder_name]
    except KeyError as e:
        print(f'RVOS moel {pt_3d_obj_decoder_name} not found')
