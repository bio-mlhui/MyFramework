
"""
3D 模态的一般形式是scene, view, rendering, 一个scene有多个view, 每个view有对应的rendering

3D scene: 一堆(view, rendering), 可以是multi-view 图片相机文件夹; 可以是一个3d 模型文件; 
      抽象成一堆(view, rendering)

注册所有的场景, 组合不同的场景成

任务数据集: 每个meta是一个(view,rendering); 对于基于优化的, 这个dataset的每个meta都属于一个scene; 
           对于基于学习的, 每个meta不一定属于同一个scene
mapper: 


统一的相机接口,


"""

class Scene_Meta:
    """
    generalize_v1版本: video->4D, images->3D, text->3D, text->4D;         如果有其他condition 也可以加进去
        scene_id: 这个场景的scene_id
        scene_text: 这个场景的text
        scene_video_id: 这个场景的video_id
        metalog_name: 通过metalog获得meta_data的名字key

        view_camera: 当前scene的单个视角
        view_cameras: 当前scene的多个视角

        rendering: 这个场景的这个视角的渲染的图像
        meta_idx, visualize
        """

class Scene_Mapper:
    """
    condense: 如果condense=False的话, 每个sample就是一个view; 如果condense=True的话, 每个sample就是(scene_text/scene_video, list[view, rendering])
    get_rendering_fn: 根据(scene_id,  view_camera, scene_text, scene_video_id, ) 获得对应的rendering; 都可以是none, 只要满足一种方式就行
    
    
    mapper: (view,rendering) -> view,rendering
        'scene_dict':
            scene_id
            scene_text
            scene_video
            metalog_name

        'view_dict':
            view_camera

        'rendering_dict':
            rendering: image, 0-1,float
         meta_idx, visualize
    condense_mapper: scene -> scene
        'scene_dict': {
        }
        views_dict: {
        }
        renderings_dict:{
    }
    contrastive mapper: 对于condense数据集
    """

class Scene_Model:
    """
    这个就要看每个model了
    如果是基于优化的方式, 那么expand_view应该是false
    如果是基于学习的方式, 那么expand_view应该是true
    forward:
        text, view, rendering -> loss
        video, view, rendering -> loss
    sample:
        返回优化的repre或者学到的repre
    """

class Scene_Representation:
    """
    repre.render:
        input: 
            view, condition
        ouput:
            rendering
    """

class Representation_Metric:
    """
    4D -> renderings/psnr/ply file
    """

class Scene_Evaluator:
    pass
    """
    """