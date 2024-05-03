import os


if os.getenv('CURRENT_TASK') == 'RVOS':
    from . import referential_amr
elif os.getenv('CURRENT_TASK') == 'VIS':
    from . import patch_similarity
    from . import hilbert_curve
elif os.getenv('CURRENT_TASK') == 'Render':
    from . import ray_embeddings
    from . import gaussian_cameras
else:
    raise ValueError()
