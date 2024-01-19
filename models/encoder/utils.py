

def find_scale_from_multiscales(multiscale_des, scale_des):
    """
    multiscale_des: list(['1','4'], ['1','8']), ['1','8']
    """
    retrieved_idx = []
    for idx, scale in enumerate(multiscale_des):
        if (scale_des[0] == scale[0]) and (scale_des[1] == scale[1]):
            retrieved_idx.append(idx)
    assert len(retrieved_idx) == 1
    return retrieved_idx[0]

def find_scales_from_multiscales(multiscale_des, scale_deses):
    """
    multiscale_des: list(['1','4'], ['1','8']), ['1','8']
    """
    output = []
    for scale_des in scale_deses:
        output.append(find_scale_from_multiscales(multiscale_des, scale_des))
    return output

