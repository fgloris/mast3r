# demo.py上展示的参数

### 选择模型(实际上只有一个可以选)

- MASt3R_ViTLarge_BaseDecoder_512_catmlpd

### 对于每张图片生成点云,进行拼接(把多个点云融合成一个点云),对拼接的结果计算3D损失,然后对拼接进行优化,这个过程被称为coarse alignment.而后对拼接结果进行2D重投影回相机平面,计算2D损失,再对拼接进行优化(有点像ba优化),这个过程被称为refinement.
- coarse LR: coarse alignment learning rate
 
    使用梯度下降调整参数优化3D拼接损失时的学习率

- num_iterations (for coarse alignment): 
    
    优化拼接损失的迭代次数,影响输出速度和结果

- fine LR: refinement learning rate

    使用梯度下降调整参数优化2D重投影损失时的学习率
    
- num_iterations (for refinement): 
    
    优化拼接损失的迭代次数,影响输出速度和结果
    
- OptLevel: 优化级别

    - coarse: 优化3D拼接损失
    - refine: 优化3D拼接损失+2D重投影损失
    - refine+depth: 优化3D拼接损失+2D重投影损失+在优化2D重投影损失阶段同时优化深度估计损失
    
### 三维重建:首先模型先生成图像特征信息,然后对两个图像的特征进行匹配,根据特征在不同图像上的位置计算变换关系,从而求解该特征点的3D位置,然后进行三维重建
- Matching Confidence Thr:

    匹配的置信度阈值,低于该阈值的匹配关系将被忽略

- Scenegraph:

    如何进行图像的匹配(将两个图像的特征匹配可以得到一个深度图)
    
    - complete: all possible image pairs

        所有图像两两匹配,可能比较慢但效果好 

    - swin: sliding window

        将所有图像排成一排(或一个圈,可选),然后对所有相邻的n个图像两两匹配,n可调

    - logwin: 和上面一样但支持比较大的n

    - oneref: match one image with all

        仅匹配一张图像和其他所有图像,在你非常信任某一张图像时使用

- min_conf_thr:

    匹配图像时根据 min_conf_thr 判断哪些对应点是有效的,低于该阈值的点将被忽略

- TSDF Threshold: 
    TSDF 是一个用3D体素表示空间体积内的对象的方式,其中每个体素都标有到最近表面的距离.TSDF阈值在进行TSDF深度优化时使用

### 从3维重建图中生成3维重建模型(glb file)

- cam_size:

    生成的glb文件中相机的大小,如果能单独计算glb图像中相机的体积,再和现实中对应,也许可以得到真实体积
    
- as_pointcloud:

    决定3D模型是用点云表示还是mesh表示
    
- Mask sky:

    是否去除天空背景
    
- transparent_cams:

    是否生成相机
    
- Clean-up depthmaps:

    一种后处理:如果一个三维点在某个相机的视图中可见(即它可投影到该相机的图像平面内),并且其深度值小于该视图中对应像素的深度值,那么降低这个三维点的置信度,使重建结果更加可靠
