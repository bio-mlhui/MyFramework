# 基于 点云diffusion, 改成gs-diffusion,
# blender提取每个model的点云, 渲染图像, loss包含渲染loss, 点云loss
# 可以进行densification, 就是说点的数量可以变
# 
# motivation: 基于splat-image的模型忽略了geometric consistenty,