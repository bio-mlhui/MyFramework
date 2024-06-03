
import os

from PIL import Image
import logging
from data_schedule.render.scene_utils.dataset_readers import sceneLoadTypeCallbacks
from functools import partial
from detectron2.data import DatasetCatalog, MetadataCatalog
from data_schedule.render.apis import Scene_Meta
from data_schedule.render.scene_utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
import random

def get_view_image_fn(camera,
                      **kwargs):
    return None, camera.original_image,None

def DBScene(dataset_id,
            scene_dir,
            prompt,
            negative_prompt,
            resolution_scale=1.0, resolution_config=-1,
            white_background=False,
            llffhold=8,):
        scene_id = f'{dataset_id}_0'
        # return metas
        # mode: train/test/all: 要哪个split
        if os.path.exists(os.path.join(scene_dir, 'sparse')):
            scene_info = sceneLoadTypeCallbacks["Colmap"](scene_dir, 'images', 
                                                          is_eval=True, llffhold=llffhold) # 每隔llffhold选一个test camera
        elif os.path.exists(os.path.join(scene_dir, 'transforms_train.json')):
            logging.info("Found transforms_train.json file, assuming Blender data set!")
            # path, white_background, eval, extension=".png"
            scene_info = sceneLoadTypeCallbacks["Blender"](scene_dir, white_background, is_eval=True)
        else:
            assert False, "Could not recognize scene type!"

        cameras = scene_info.train_cameras + scene_info.test_cameras

        view_cameras = cameraList_from_camInfos(cameras, 
                                                resolution_scale=resolution_scale, resolution_config=resolution_config)
        test_camera_ids = [haosen.uid for haosen in view_cameras][ len(scene_info.train_cameras):]

        MetadataCatalog.get(dataset_id).set(point_cloud=scene_info.point_cloud,  # TODO: 初始的pcd和llfhold有关
                                            cameras_extent=scene_info.nerf_normalization['radius'],
                                            eval_meta_keys={scene_id: test_camera_ids},)        
        
        return [{
            'dataset_id': dataset_id,
            'scene_id': scene_id,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'views_images_prompt': view_cameras, # list[Camera]
            'meta_idx': 0,
        }]


dataset_id_to_info = {
    'deepblend_drjohnson': {
        'scene_dir': 'MultiImage/deepblend/db/drjohnson',
        'prompt':  '',
        'negative_prompt': '',
    },
    'deepblend_playroom': {
        'scene_dir': 'MultiImage/deepblend/db/playroom',
        'prompt':  '',
        'negative_prompt': '',
    }, 
    'deepblend_train': {
        'scene_dir': 'MultiImage/deepblend/tandt/train',
        'prompt':  '',
        'negative_prompt': '',
    },
    'deepblend_truck': {
        'scene_dir': 'MultiImage/deepblend/tandt/truck',
        'prompt':  '',
        'negative_prompt': '',
    },
}

for dataset_id, info in dataset_id_to_info.items():
    Scene_Meta
    # 如果这个scene 需要拿出来test -> 那么一定是作为optimize使用
    DatasetCatalog.register(dataset_id, partial(DBScene,
                                                dataset_id=dataset_id,
                                                scene_dir=os.path.join(os.getenv('DATASET_PATH'), 'RENDER_dataset', info['scene_dir']),
                                                prompt=info['prompt'],
                                                negative_prompt=info['negative_prompt'],)
                                                )
    MetadataCatalog.get(dataset_id).set(mode='all',
                                        gaussian_initialize=None,
                                        get_view_image_fn=get_view_image_fn,
                                        visualize_meta_idxs=[])
        

    


 
        
