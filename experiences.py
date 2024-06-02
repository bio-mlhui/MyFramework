# 现存不够的时候，看下是不是输入大小的问题， 输入大小可以大大减低显存

# 阴影是不联通的区域，不太适合point samplede


"""
当前3D模态存在的问题:
1. 3D数据少, nips24
2. 

Point-E
zero-1-to-3
Make-It-3D

基于学习的single image到3D

Zero-1-to-3, Ruoshi Liu
    没有用3D表示, 而是让diffusion直接生成novel-view

    motivation: 当前的2D diffusion已经学到了3D信息,
    view_camera作为diffusion的control, 在合成数据集上专门训练视角转换的能力


LRM, ICLR24, Yicong Hong, 没开源
    用的是nerf-modulation 表示
    image-encoder提出feature map, decoder有一堆nerf-modulation queries, 相机参数和feature map作为decoder的条件

    数据:

    想法: video-4D的话没有足够的数据; 他没有用gs, gs的point的数量是变化的,l
    

DMV3D, ICLR24,
    LRM + diffusion, diffusion的xt是有噪声的multi-view图像, 目的是nerf model


VFusion3D: Junlin Han
    生成Multi-view video, 微调video diffusion model, 然后生成Multi-view dataset, 训练LRM

TripoSR: Dmitry, 开源了
    加速了LRM, 5s -> 0.5s

    
    
"""