import os
import os
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_schedule.render.apis import Scene_Meta
import numpy as np
import logging
from functools import partial
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_camera_fn(scene_path=None,  
                  view_id=None, 
                  only_c2w=True,
                  world_format='opengl',
                  camera_format='opengl',
                  **kwargs):
    """
    6个view_id对应6个c2w
    return proj/c2w;
    """
    if not only_c2w:
        raise NotImplementedError()

    w2c = os.path.join(scene_path, 'zero123_rendered/views_release', scene_id, f'{view_id:03d}.npy')
    w2c = np.load(w2c)  # blender camera
    w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
    w2c = torch.tensor(w2c).float().reshape(4, 4)

    c2w = torch.inverse(w2c)
    # blender world + camera
    # [[Right_x, Up_x, Forward_x, Position_x],
    #  [Right_y, Up_y, Forward_y, Position_y],
    #  [Right_z, Up_z, Forward_z, Position_z],
    #  [0,       0,    0,         1         ]]

    if world_format == 'opengl':
        c2w[1] *= -1  # reverse y
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
    elif world_format == 'colmap' or world_format == 'opencv':
        c2w[1] *= -1  # reverse y
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
        c2w[[1, 2]] *= -1 # reverse y and z
    elif world_format == 'blender':
        pass
    elif world_format == 'unity' or world_format == 'directx':
        c2w[[1, 2]] = c2w[[2, 1]] # switch y and z
    else:
        raise ValueError()

    if camera_format == 'opengl' or camera_format == 'blender':
        pass
    elif camera_format == 'colmap' or camera_format == 'opencv':
        c2w[:, 1:3] *= -1 # reverse up and forward
    else:
        raise ValueError()
    return c2w


def optimize_scene(prompt, 
                   dataset_id,
                   negative_prompt, 
                   views_images_prompt, ):
    Scene_Meta
    scene_id = f'{dataset_id}_0' # optimize_scene_id = dataset_id_0
    view_ids = [haosen['view_id'] for haosen in views_images_prompt]
    metas = [{
        'dataset_id': None,
        'scene_id': scene_id, # optimize scene 的scene_id就是 dataset_id_0
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'views_images_prompt': views_images_prompt,  # list[dict], each of
        # 'view_id': 0,
        # 'view_image': a -> c2w, image = get_view_image_fn(**a)
        'meta_idx': 0
    }]
    MetadataCatalog.get(dataset_id).set(eval_meta_keys={scene_id: view_ids})
    logging.debug(f'{dataset_id} Total metas: [{len(metas)}]')
    return metas

dataset_root = os.path.join(os.getenv('DATASET_PATH'), 'RENDER_dataset')
"""
optimize_recon:
    text
    text_images
    images

    prompt: ''
    negative_prompt: ''
    view_image_prompt: [(view, rendering)]
    images_split: None
"""

def get_view_image_fn(root, 
                      image_path,
                      view):
    image_path = os.path.join(root, image_path)
    image = np.asarray(Image.open(image_path).convert('RGBA')).astype(np.float32) / 255
    image = torch.from_numpy(image) # [512, 512, 4] in [0, 1]
    return None, image, None

dataset_id_to_info = {
    # text
    'Donald_Trump_is_holding_a_puppy_noimagegenerate': {
        'type': 'Text',
        'prompt':  'Donald Trump is holding a puppy',
        'negative_prompt': '',
    },
    'a_tulip_noimagegenerate': {
        'type': 'Text',
        'prompt': 'a tulip',
        'negative_prompt': '',
    },
    'a_photo_of_an_icecream_noimagegenerate': {
        'type': 'Text',
        'prompt': 'icecream',
        'negative_prompt': '',
    },
    # text_images
    'a_person_head_huihui': {
        'type': 'Text_Images',
        'prompt': 'a person head',
        'negative_prompt': 'ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions',

        'views_images_prompt': [{
            'view_id': 0,
            'view_image': {'image_path': 'Text_SingleImage/a_person_head_huihui.jpg', 
                           'view': 'front' # 学习的情况下注册为front_view
                           },
        }],
        'get_view_image_fn': partial(get_view_image_fn, root=dataset_root)
       
    },
    # images
    'a_cup': {
        'type': 'Images',
        'prompt': '',
        'negative_prompt': 'ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions',

        'views_images_prompt': [{
            'view_id': 0,
            'view_image': {'image_path': 'Text_SingleImage/a_person_head_huihui.jpg', 
                          'view': 'front', # 学习的情况下注册为front_view
                          }
        }],
        'get_view_image_fn': partial(get_view_image_fn, root=dataset_root)
    },
}


for datast_id, info in dataset_id_to_info.items(): # 每个text就是一个数据集, 每个数据集
    views_images_prompt = None if info['type'] in ['Text'] else info['views_images_prompt']
    # optimize的情况用DatasetCatalog.get()
    DatasetCatalog.register(datast_id, partial(optimize_scene, 
                                              dataset_id=datast_id,
                                              prompt = info['prompt'],
                                              negative_prompt = info['negative_prompt'],
                                              views_images_prompt = views_images_prompt,))
    MetadataCatalog.get(datast_id).set(mode='all',   
                                       get_view_image_fn=None if info['type'] in ['Text'] else info['get_view_image_fn'],
                                       visualize_meta_idxs=[])
