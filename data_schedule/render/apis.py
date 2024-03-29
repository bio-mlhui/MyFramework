
# learning-based: 在训练的时候

class Scene_Terminology:
    """
    scene_id:
    scene_text: 
    scene_video_id: 
    scene_video

    condense: 如果condense=False的话, 每个sample就是一个view; 如果condense=True的话, 每个sample就是(scene_text/scene_video, list[view, rendering])
    view_camera: 带view的camera对象, 抽象的数据结构
    view_cameras: 当前scene的所有cameras
    condense=True -> view_cameras
    condense=False-> view_camera

    rendering: video/image的渲染的图像
    get_rendering_fn: 根据(scene_id, 
                           view_camera,
                           scene_text,
                           scene_video,
                           scene_video_id,
                           ) 获得对应的rendering; 都可以是none, 只要满足一种方式就行
    """

class Scene_Dataset: 
    """
    scene_id: str
    scene_text: str/None
    scene_video_id: str/None
    
    view_camera: 3DCamera/4DCamera  # 对于condense=False的
    view_cameras: list[3DCamera/4D Camera]  # 对于condense=True的
    """ 


class Scene_Mapper:
    """
    condense_mapper: 每个
        # view_sampler

    mapper: 输入输出一样
        'view_dict': {'viewcamera': data_dict['view_camera'] },
        'targets': {
            'rendering': data_dict['rendering']
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