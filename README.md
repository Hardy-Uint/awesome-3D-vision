# 介绍

> 公众号：[3D视觉工坊](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484684&idx=1&sn=e812540aee03a4fc54e44d5555ccb843&chksm=fbff2e38cc88a72e180f0f6b0f7b906dd616e7d71fffb9205d529f1238e8ef0f0c5554c27dd7&token=691734513&lang=zh_CN#rd)
>
> 主要关注：3D视觉算法、SLAM、vSLAM、计算机视觉、深度学习、自动驾驶、图像处理以及技术干货分享
>
> 运营者和嘉宾介绍：运营者来自国内一线大厂的算法工程师，深研3D视觉、vSLAM、计算机视觉、点云处理、深度学习、自动驾驶、图像处理、三维重建等领域，特邀嘉宾包括国内外知名高校的博士硕士，旷视、商汤、百度、阿里等就职的算法大佬，欢迎一起交流学习

[TOC]

# 硬件

## 相机技术&参数

[事件相机知识点汇总](https://github.com/uzh-rpg/event-based_vision_resources)

## 相机生产厂家汇总

|      公司名称      |             主要业务             |                   岗位                   |        base         |
| :----------------: | :------------------------------: | :--------------------------------------: | :-----------------: |
|      海康威视      |    安防监控、机器视觉解决方案    | 机器视觉算法、三维重建算法、立体视觉算法 |        杭州         |
|        大华        |      视频监控、机器视觉系统      |        机器视觉算法、立体视觉算法        |        杭州         |
|      奥比中光      |      3D相机、3D视觉解决方案      |        3D视觉、三维重建、立体视觉        |   深圳/西安/上海    |
| 北京凌云光技术集团 |    工业视觉、解决方案、3D相机    |                 图像算法                 |        北京         |
|      大恒图像      |  机器视觉系统、3D相机、3D传感器  |          图像算法、机器视觉算法          |        北京         |
|       基恩士       |    机器视觉解决方案、视觉产品    |                                          |      上海/苏州      |
|       康耐视       |     机器视觉解决方案、传感器     |                 机器视觉                 |   上海/苏州/杭州    |
|   埃尔森智能科技   |   机器人3D视觉、3D相机、结构光   |                                          |        郑州         |
|                    |        立体视觉、双目感知        |                                          |        北京         |
|      阿丘科技      |          AI视觉解决方案          |        AI算法、视觉算法、机器视觉        |   北京/苏州/深圳    |
|      图漾科技      |         3D视觉系统、产品         |            视觉算法、图像处理            |        上海         |
|      精锐视觉      |            工业AI视觉            |                 图像算法                 |      深圳/上海      |
|      华夏视科      |        工业视觉、图像检测        |                 图像算法                 |      北京/上海      |
|    Sick（德国）    |         机器视觉解决方案         |              机器视觉工程师              | 北京/上海/深圳/广州 |
|      光鉴科技      |          3D视觉解决方案          |           TOF算法、3D视觉算法            |        上海         |
|      征图新视      |         机器视觉解决方案         |            机器视觉、深度学习            |   深圳/常州/苏州    |
|     中星微电子     |             图像芯片             |          深度学习、视频图像处理          |      北京/上海      |
|      捷尚视觉      |           视频智能分析           |                 图像算法                 |        杭州         |
|    先临三维科技    |       三维扫描仪、三维成像       |          三维视觉算法、点云算法          |        杭州         |
|      华睿科技      |             机器视觉             |              机器视觉工程师              |        杭州         |
|      蓝芯科技      |      视觉系统、3D视觉传感器      |                                          |        杭州         |
|      微视图像      |    机器视觉、工业相机、3D相机    |                                          |        北京         |
|     库柏特科技     |        机器人、3D视觉产品        |                                          |        武汉         |
|      辰视智能      |    结构光、3D系统、双目、多目    |                                          |        深圳         |
|   星上维智能科技   | 结构光、三维机器视觉、三维扫描仪 |                                          |        广州         |
|      创科视觉      |       机器视觉系统、3D相机       |                                          |        深圳         |

# 相机标定

### 综述

1. [线阵相机标定方法综述](http://www.opticsjournal.net/Articles/Abstract?aid=OJe133f6606f9fe076)

### 单相机标定

1. [相机标定误差因素分析](http://www.cnki.com.cn/Article/CJFDTotal-HBYD201201014.htm)
2. [Fully automatic camera calibration method based on circular markers基于圆形标志点的全自动相机标定方法](http://www.cnki.com.cn/Article/CJFDTotal-YQXB200902028.htm)
3. [Accurate camera calibration using iterative refinement of control points](http://xueshu.baidu.com/usercenter/paper/show?paperid=a68a76813662e8a8ee64f377a8516adb&site=xueshu_se)
4. [Accurate Feature Extraction and Control Point Correction for Camera Calibration with a Mono-Plane Target](http://xueshu.baidu.com/usercenter/paper/show?paperid=7a1bfac77a6adb17287b5449a327cd70&site=xueshu_se)
5. [基于主动红外辐射标定板的超广角红外相机标定](http://www.opticsjournal.net/Articles/Abstract?aid=OJ191105000133w3z6B9)
6. [基于相位标靶的相机标定](http://www.opticsjournal.net/Articles/Abstract?aid=OJ1811210000185B8DaG)
7. [基于广义成像模型的Scheimpflug相机标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ1808090000414z6C9F)
8. [多几何约束下的鱼眼相机单像高精度标定](http://www.opticsjournal.net/Articles/Abstract?aid=OJ181115000101pWsZv2)

### 手眼标定

1. [一种新的机器人手眼关系标定方法](http://xueshu.baidu.com/usercenter/paper/show?paperid=ac40e02979ac1aa62cfaf5b3e9365a7b&site=xueshu_se)

### 其它

1. [基于张正友标定法的红外靶标系统](http://www.opticsjournal.net/Articles/Abstract?aid=OJ200119000058dKgMjP)

# 3D全景相机

> 360°环视

### 鱼眼相机标定

[A  Practical  Toolbox  for  Calibrating Omnidirectional Cameras]()

### 多相机拼接

> 主要通过SIFT、SURF、Harris等算法进行特征点对应匹配。

# 3D视觉资源汇总

## 书籍

1. [视觉测量]()[张广军]
2. [机器人视觉测量与控制]()[徐德，谭民，李原]
3. [Machine Vision 2016: Automated Visual Inspection: Theory, Practice and Applications]()

## 资源

[https://github.com/timzhang642/3D-Machine-Learning](https://github.com/timzhang642/3D-Machine-Learning)

https://github.com/sunglok/3dv_tutorial(涉及SLAM、多视图几何代码示例)

# SLAM

## 优秀开源项目汇总

[https://github.com/OpenSLAM/awesome-SLAM-list](https://github.com/OpenSLAM/awesome-SLAM-list)

[https://github.com/tzutalin/awesome-visual-slam](https://github.com/tzutalin/awesome-visual-slam)

https://github.com/kanster/awesome-slam

https://github.com/YoujieXia/Awesome-SLAM

[Recent_SLAM_Research](https://github.com/YiChenCityU/Recent_SLAM_Research)

[https://github.com/youngguncho/awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets)

[https://github.com/marknabil/SFM-Visual-SLAM](https://github.com/marknabil/SFM-Visual-SLAM)

[https://github.com/ckddls1321/SLAM_Resources](https://github.com/ckddls1321/SLAM_Resources)

## 激光SLAM

> 分为前端和后端。其中前端主要完成匹配和位置估计，后端主要完成进一步的优化约束。
>
> 整个SLAM大概可以分为前端和后端，前端相当于VO（视觉里程计），研究帧与帧之间变换关系。首先提取每帧图像特征点，利用相邻帧图像，进行特征点匹配，然后利用RANSAC去除大噪声，然后进行匹配，得到一个pose信息（位置和姿态），同时可以利用IMU（Inertial measurement unit惯性测量单元）提供的姿态信息进行滤波融合。
>
> 后端则主要是对前端出结果进行优化，利用滤波理论（EKF、UKF、PF）、或者优化理论TORO、G2O进行树或者图的优化。最终得到最优的位姿估计。

### 数据预处理

### 点云匹配

### 地图构建



## 视觉SLAM

### Books

- [视觉SLAM十四讲]() 高翔
- [机器人学中的状态估计]()
- [概率机器人]()
- [Simultaneous Localization and Mapping for Mobile Robots: Introduction and Methods](http://www.igi-global.com/book/simultaneous-localization-mapping-mobile-robots/66380) by Juan-Antonio Fernández-Madrigal and José Luis Blanco Claraco, 2012
- [Simultaneous Localization and Mapping: Exactly Sparse Information Filters ](http://www.worldscientific.com/worldscibooks/10.1142/8145/)by Zhan Wang, Shoudong Huang and Gamini Dissanayake, 2011
- [An Invitation to 3-D Vision -- from Images to Geometric Models](http://vision.ucla.edu/MASKS/) by Yi Ma, Stefano Soatto, Jana Kosecka and Shankar S. Sastry, 2005
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004
- [Numerical Optimization](http://home.agh.edu.pl/~pba/pdfdoc/Numerical_Optimization.pdf) by Jorge Nocedal and Stephen J. Wright, 1999

### Courses&&Lectures

- [SLAM Tutorial@ICRA 2016](http://www.dis.uniroma1.it/~labrococo/tutorial_icra_2016/)
- [Geometry and Beyond - Representations, Physics, and Scene Understanding for Robotics](http://rss16-representations.mit.edu/) at Robotics: Science and Systems (2016)
- [Robotics - UPenn](https://www.coursera.org/specializations/robotics) on Coursera by Vijay Kumar (2016)
- [Robot Mapping - UniFreiburg](http://ais.informatik.uni-freiburg.de/teaching/ws15/mapping/) by Gian Diego Tipaldi and Wolfram Burgard (2015-2016)
- [Robot Mapping - UniBonn](http://www.ipb.uni-bonn.de/robot-mapping/) by Cyrill Stachniss (2016)
- [Introduction to Mobile Robotics - UniFreiburg](http://ais.informatik.uni-freiburg.de/teaching/ss16/robotics/) by Wolfram Burgard, Michael Ruhnke and Bastian Steder (2015-2016)
- [Computer Vision II: Multiple View Geometry - TUM](http://vision.in.tum.de/teaching/ss2016/mvg2016) by Daniel Cremers ( Spring 2016)
- [Advanced Robotics - UCBerkeley](http://www.cs.berkeley.edu/~pabbeel/) by Pieter Abbeel (Fall 2015)
- [Mapping, Localization, and Self-Driving Vehicles](https://www.youtube.com/watch?v=x5CZmlaMNCs) at CMU RI seminar by John Leonard (2015)
- [The Problem of Mobile Sensors: Setting future goals and indicators of progress for SLAM](http://ylatif.github.io/movingsensors/) sponsored by Australian Centre for Robotics and Vision (2015)
- [Robotics - UPenn](https://alliance.seas.upenn.edu/~meam620/wiki/index.php?n=Main.HomePage) by Philip Dames and Kostas Daniilidis (2014)
- [Autonomous Navigation for Flying Robots](http://vision.in.tum.de/teaching/ss2014/autonavx) on EdX by Jurgen Sturm and Daniel Cremers (2014)
- [Robust and Efficient Real-time Mapping for Autonomous Robots](https://www.youtube.com/watch?v=_W3Ua1Yg2fk) at CMU RI seminar by Michael Kaess (2014)
- [KinectFusion - Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera](https://www.youtube.com/watch?v=bRgEdqDiOuQ) by David Kim (2012)

### Code

1. [ORB-SLAM](https://github.com/raulmur/ORB_SLAM)
2. [LSD-SLAM](https://github.com/tum-vision/lsd_slam)
3. [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
4. [DVO: Dense Visual Odometry](https://github.com/tum-vision/dvo_slam)
5. [SVO: Semi-Direct Monocular Visual Odometry](https://github.com/uzh-rpg/rpg_svo)
6. [G2O: General Graph Optimization](https://github.com/RainerKuemmerle/g2o)
7. [RGBD-SLAM](https://github.com/felixendres/rgbdslam_v2)

| Project                                                      | Language | License                    |
| ------------------------------------------------------------ | -------- | -------------------------- |
| [COSLAM](http://drone.sjtu.edu.cn/dpzou/project/coslam.php)  | C++      | GNU General Public License |
| [DSO-Direct Sparse Odometry](https://github.com/JakobEngel/dso) | C++      | GPLv3                      |
| [DTSLAM-Deferred Triangulation SLAM](https://github.com/plumonito/dtslam) | C++      | modified BSD               |
| [LSD-SLAM](https://github.com/tum-vision/lsd_slam/)          | C++/ROS  | GNU General Public License |
| [MAPLAB-ROVIOLI](https://github.com/ethz-asl/maplab)         | C++/ROS  | Apachev2.0                 |
| [OKVIS: Open Keyframe-based Visual-Inertial SLAM](https://github.com/ethz-asl/okvis) | C++      | BSD                        |
| [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2)             | C++      | GPLv3                      |
| [REBVO - Realtime Edge Based Visual Odometry for a Monocular Camera](https://github.com/JuanTarrio/rebvo) | C++      | GNU General Public License |
| [SVO semi-direct Visual Odometry](https://github.com/uzh-rpg/rpg_svo) | C++/ROS  | GNU General Public License |

# 计算机视觉

## 资源汇总



## Books

- [Computer Vision: Models, Learning, and Inference](http://www.computervisionmodels.com/) - Simon J. D. Prince 2012
- [Computer Vision: Theory and Application](http://szeliski.org/Book/) - Rick Szeliski 2010
- [Computer Vision: A Modern Approach (2nd edition)](http://www.amazon.com/Computer-Vision-Modern-Approach-2nd/dp/013608592X/ref=dp_ob_title_bk) - David Forsyth and Jean Ponce 2011
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/) - Richard Hartley and Andrew Zisserman 2004
- [Visual Object Recognition synthesis lecture](http://www.morganclaypool.com/doi/abs/10.2200/S00332ED1V01Y201103AIM011) - Kristen Grauman and Bastian Leibe 2011
- [Computer Vision for Visual Effects](http://cvfxbook.com/) - Richard J. Radke, 2012
- [High dynamic range imaging: acquisition, display, and image-based lighting](http://www.amazon.com/High-Dynamic-Range-Imaging-Second/dp/012374914X) - Reinhard, E., Heidrich, W., Debevec, P., Pattanaik, S., Ward, G., Myszkowski, K 2010
- [Numerical Algorithms: Methods for Computer Vision, Machine Learning, and Graphics](https://people.csail.mit.edu/jsolomon/share/book/numerical_book.pdf) - Justin Solomon 2015

## Courses

- [EENG 512 / CSCI 512 - Computer Vision](http://inside.mines.edu/~whoff/courses/EENG512/) - William Hoff (Colorado School of Mines)
- [3D Computer Vision: Past, Present, and Future](https://www.bilibili.com/video/av62437998/)
- [Visual Object and Activity Recognition](https://sites.google.com/site/ucbcs29443/) - Alexei A. Efros and Trevor Darrell (UC Berkeley)
- [Computer Vision](http://courses.cs.washington.edu/courses/cse455/12wi/) - Steve Seitz (University of Washington)
- Visual Recognition [Spring 2016](http://vision.cs.utexas.edu/381V-spring2016/), [Fall 2016](http://vision.cs.utexas.edu/381V-fall2016/) - Kristen Grauman (UT Austin)
- [Language and Vision](http://www.tamaraberg.com/teaching/Spring_15/) - Tamara Berg (UNC Chapel Hill)
- [Convolutional Neural Networks for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/) - Fei-Fei Li and Andrej Karpathy (Stanford University)
- [Computer Vision](http://cs.nyu.edu/~fergus/teaching/vision/index.html) - Rob Fergus (NYU)
- [Computer Vision](https://courses.engr.illinois.edu/cs543/sp2015/) - Derek Hoiem (UIUC)
- [Computer Vision: Foundations and Applications](http://vision.stanford.edu/teaching/cs131_fall1415/index.html) - Kalanit Grill-Spector and Fei-Fei Li (Stanford University)
- [High-Level Vision: Behaviors, Neurons and Computational Models](http://vision.stanford.edu/teaching/cs431_spring1314/) - Fei-Fei Li (Stanford University)
- [Advances in Computer Vision](http://6.869.csail.mit.edu/fa15/) - Antonio Torralba and Bill Freeman (MIT)
- [Computer Vision](http://www.vision.rwth-aachen.de/course/11/) - Bastian Leibe (RWTH Aachen University)
- [Computer Vision 2](http://www.vision.rwth-aachen.de/course/9/) - Bastian Leibe (RWTH Aachen University)
- [Computer Vision](http://klewel.com/conferences/epfl-computer-vision/) Pascal Fua (EPFL):
- [Computer Vision 1](http://cvlab-dresden.de/courses/computer-vision-1/) Carsten Rother (TU Dresden):
- [Computer Vision 2](http://cvlab-dresden.de/courses/CV2/) Carsten Rother (TU Dresden):
- [Multiple View Geometry](https://youtu.be/RDkwklFGMfo?list=PLTBdjV_4f-EJn6udZ34tht9EVIW7lbeo4) Daniel Cremers (TU Munich):

# 深度学习

## Github link

1、[https://github.com/ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)

2、[https://github.com/endymecy/awesome-deeplearning-resources](https://github.com/endymecy/awesome-deeplearning-resources)

# 机器学习

## Github link

1、[https://github.com/josephmisiti/awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)

# 3D点云

## 点云标注工具

### 开源

1. [Semantic.editor](https://github.com/MR-520DAI/semantic-segmentation-editor)

### 商用

> 商用软件很多，阿里、腾讯、百度、京东都有对应业务



## 点云获取

> 传统的点云获取技术包括非接触式测量和接触式测量两种，它们的主要区别在于，在测量过程中测头是否与工件的表面相接触。
>
> 非接触式测量是利用光学原理的方法采集数据，例如结构光法、测距法以及干涉法等。该方法的优点在于测量速度较快、测量精度高，并且能够获得高密度点云数据，但其测量精度易受外界因素干扰，而且测量物
> 体表面的反射光与环境光对测量精度也有一定影响。
>
> 相反，接触式测量是通过将测头上的测量传感器与被测物体的外表面相接触，然后通过移动测头来读取物体表面点的三维坐标值。该方法的优点在于测头的结构相对固定，并且其测量结果不受被测物体表面的材料与表面特性等因素的影响。这种方法的不足在于，由于测头长期与被测物体表面相接触，易产生磨损，并且这种测量方式的测量速度较慢，不适合测量几何结构较复杂的物体。

## 点云应用场景

> 逆向工程、游戏人物重建、文物保护、数字博物馆、医疗辅助、三维城市建模

## 点云种类

> 不同的点云获取技术获取的点云数据类型不同，根据点云数据中点的分布情况可将点云数据划分为以下四种类型

#### 散乱点云

散乱点云是指所有数据点在空间中以散乱状态分布，任意两点之间没有建立拓扑连接关系。一般而言，激光点测量系统获得的点云数据以及坐标测量机在随机扫描状态下获得的点云数据都为散乱点云数据。

#### 扫描线点云

测量设备所获得的三维点云数据是由多条直线或曲线构成，点与点之间有一定的拓扑连接关系。一般而言，这种点云数据类型常见于扫描式点云数据中。

#### 网格化点云

网格化点云是指点云数据中任意一点，均对应于其参数域所对应的一个均匀网格的顶点。当对空间散乱点云进行网格化插值时，所获得的点云数据即为网格化点云数据。

#### 多边形点云

多边形点云是指分布在一组平面内的点云数据，该组平面内的平面两两互相平行，并且一个平面内距离最近的点连接起来可以形成平面多边形。这种点云数据常见于等高线测量、CT 测量等获得的点云数据中。

## 点云去噪&滤波

> 主要包括双边滤波、高斯滤波、条件滤波、直通滤波、随机采样一致滤波、VoxelGrid滤波等
>
> 三角网格去噪算法、

1. [基于K-近邻点云去噪算法的研究与改进](http://www.cnki.com.cn/Article/CJFDTotal-JSJY200904032.htm)
2. [Point cloud denoising based on tensor Tucker decomposition](https://arxiv.org/abs/1902.07602v2)
3. [3D Point Cloud Denoising using Graph Laplacian Regularization of a Low Dimensional Manifold Model](https://arxiv.org/abs/1803.07252v2)

#### 有序点云去噪

> 孤立点排异法、曲线拟合法、弦高差法、全局能量法和滤波法.
>
> 孤立点排异法是通过观察点云数据，然后将与扫描线偏离较大的点剔除掉，从而达到去噪的目的。这类方法简单，可除去比较明显的噪声点，但缺点是只能对点云做初步的去噪处理，并不能滤除与真实点云数据混合在一起的噪声数据点。曲线拟合法是根据给定数据点的首末点，然后通过最小二乘等方法拟合一条曲线，通常为3到4 阶，最后计算中间的点到该曲线的距离，如果该距离大于给定阈值，则该点为噪声点，予以删
> 除，相反，如果该距离小于给定阈值，则该点为正常点，应该保留。弦高差法通过连接给定点集的首末点形成弦，然后求取中间每个点到该弦的距离，如果该距离小于给定阈值，则该点为正常点，予以保留，相反，若大于给定阈值，则该点为噪声点，予以删除。全局能量法通常用于网格式点云去噪，它通过建立整个曲面的能量方程，并求该方程在约束情况下的能量值的最小值。可以看出，这是一个全局最优化问题，因为网格数量比较大，因此会消耗大量的计算机资源与计算时间，而且由于约束方程是建立在整体网格的基础上，所以对于局部形状的去噪效果并不是很好。滤波法也是一种常用的有序点云去噪方法，它通过运用信号处理中的相关方法，使用合适的滤波函数对点云数据进行去噪处理，常用的滤波方法主要包括高斯滤波、均值滤波以及中值滤波法等。

#### 无序点云去噪&空间散乱点云去噪算法

> 目前，针对空间散乱点云数据去噪方法，主要分为两类方法，即基于网格模型的去噪方法和直接对空间点云数据进行去噪的方法。
>
> 其中，基于网格模型的去噪方法需要首先建立点云的三角网格模型，然后计算所有三角面片的纵横比和顶点方向的曲率值，并将该值与相应的阈值进行比较，若小于阈值，则为正常点，予以保留，相反，则为噪声点，予以删除。由于该方法需要对空间点云数据进行三角网格剖分，所以，往往比较复杂，并需要大量计算。

## 点云精简

> 采用三维激光扫描仪获得的点云数据往往**十分密集**，点云数据中点的数量往往高达千万级甚至数亿级，即使对点云数据进行了去噪处理，点云数据中点的数量还是很多，所以往往不会直接使用这些原始点云数据进行曲面重建等工作，因为这会使后续处理过程变得耗时并且消耗过多的计算机资源，而且重构的曲面，其精度也不一定高，甚至出现更大的误差。所以，在进行空间点云曲面重建之前，往往需要对高密度的点云数据进
> 行点云精简操作。点云精简的目的是在保持原始点云的形状特征以及几何特征信息的前提下，尽量删除多余的数据点。
>
> 目前，空间散乱点云数据的精简方法主要分为两大类：**基于三角网格模型的空间点云精简方法**与**直接基于数据点的空间点云精简方法**。
>
> 其中，基于三角网格模型的空间点云精简方法需要先对点云数据进行三角剖分处理，建立其相应的三角网格拓扑结构，然后再对该三角网格进行处理，并将区域内那些形状变化较小的三角形进行合并，最后删除相关的三角网格顶点，从而达到点云数据精简的目的。这种方法需要对点云数据建立其相应的三角网格，该过程比较复杂，且因为需要存储网格数据，故需要消耗大量的计算机系统资源，并且该方法的抗噪能力较弱，对含有噪声的点云数据，构造的三角网格可能会出现变形等情况，因此精简后的点云数据经过曲面重建后的
> 模型与原始点云经过曲面重建后的模型可能大不相同。因此，目前关于直接基于点云数据的精简方法成为点云精简方法的主流。这种方法依据点云数据点之间的空间位置关系来建立点云的拓扑连接关系，并根据建立的拓扑连接关系计算点云数据中每个数据点的几何特征信息，最后根据这些特征信息来对点云数据进行点云精简处理。相比基于三角网格的空间点云精简方法，由于直接基于点云数据点的精简方法无需计算和存储复杂的三角网格结构，使得其精简的效率相对较高。因此，本章只研究直接基于空间点云数据的精简算法。



其中基于空间点云精简方法主要有：空间包围盒法、基于聚类的方法、法向偏差法、曲率精简法、平局点距法以及均匀栅格划分法。

#### Paper

1. 点模型的几何图像简化法
2. 基于相似性的点模型简化算法
3. 基于最小曲面距离的快速点云精简算法
4. 大规模点云选择及精简
5. 一种基于模糊聚类的海量测量数据简化方法
6. 基于均值漂移聚类的点模型简化方法
7. 基于局部曲面拟合的散乱点云简化方法

## 点云关键点

> 常见的三维点云关键点提取算法有一下几种：ISS3D、Harris3D、NARF、SIFT3D，这些算法在PCL库中都有实现，其中NARF算法是用的比较多的

## 点云描述

> 如果要对一个三维点云进行描述，光有点云的位置是不够的，常常需要计算一些额外的参数，比如法线方向、曲率、文理特征等等。如同图像的特征一样，我们需要使用类似的方式来描述三维点云的特征。
>
> 常用的特征描述算法有：法线和曲率计算、特征值分析、PFH、FPFH、SHOT、VFH、CVFH、3D Shape Context、Spin Image等。PFH：点特征直方图描述子，FPFH：跨苏点特征直方图描述子，FPFH是PFH的简化形式。



## 点云线、面拟合

> 针对直线拟合：RANSAC算法、最小二乘法、平面相交法
>
> 针对曲线拟合：拉格朗日插值法、最小二乘法、Bezier曲线拟合法、B样条曲线法（二次、三次B样条曲线拟合）
>
> 针对平面拟合：主成成分分析、最小二乘法、粗差探测法、抗差估计法
>
> 针对曲面拟合：最小二乘法（正交最小二乘、移动最小二乘）、NURBS、 Bezier 

1. 三维激光扫描拟合平面自动提取算法
2. 点云平面拟合新方法
3. 海量散乱点的曲面重建算法研究
4. 一种稳健的点云数据平面拟合方法 
5. 迭代切片算法在点云曲面拟合中的应用
6. [基于最小二乘的点云叶面拟合算法研究](http://www.cqvip.com/QK/97059X/201405/661608237.html)
7. [点云曲面边界线的提取](http://d.wanfangdata.com.cn/Conference/7057652)

## 点云体积计算

> 基于三维点云求取物理模型体积的研究算法大致可分为以下 4 大类。
>
> 1.凸包算法：使用凸包模型近似表示不规则体，再通过把凸包模型切片分割进行累加、或将凸包模型分解为上下两个三角网格面，采用正投影法求取两者的投影体积，其差即所求体积。此方法适用于凸模型，非凸模型误差较大。
>
> 2.模型重建法：在得到点云数据后，使用三角面片构建物理模型的方法求得体积。该算法受点云密度、生成的三角网格数量、点精度影响较大，易产生孔洞。
>
> 3.切片法：将点云沿某一坐标轴方向进行切片处理，再计算切片上下两表面的面积，通过累加切片体积求得总体积。该方法受到切片厚度的影响，切片厚度越小，计算精度越高但会导致计算效率下降。
>
> 4.投影法：先将点云投影进行三角形剖分，再将投影点与其原对应点构建出五面体，通过累加五面体体积求得总体积。该算法同样容易产生孔洞。上述算法，无论是通过三维点云先构建物理模型再求体积、还是基于三维点云通过几何方法直接求体积，当激光雷达采集的三维点云存在密度不均匀、空间物体存在过渡带或过渡线等问题时，重建三维模型的误差较大，体积计算精度不高。

## 点云识别&分类

> 分类：基于点的分类，基于分割的分类，监督分类与非监督分类
>
> 除此之外，还可以基于描述向量/关键点描述进行分类。

1. [3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dvision.princeton.edu/projects/2014/3DShapeNets/paper.pdf)
2. [PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding](https://arxiv.org/abs/1812.02713)
3. [Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/pdf/1908.04616.pdf)
4. [Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models](http://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)[ICCV2017]
5. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)] Colored Point Cloud Registration Revisited. 
6. [[ICRA2017](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds.
7. [[IROS2017](https://ieeexplore.ieee.org/document/8202239)] 3D object classification with point convolution network.
8. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf)] Pointwise Convolutional Neural Networks.
9. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)] SO-Net: Self-Organizing Network for Point Cloud Analysis. 
10. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Uy_PointNetVLAD_Deep_Point_CVPR_2018_paper.pdf)] PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition. 
11. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf)] PointGrid: A Deep Network for 3D Shape Understanding. 
12. [[CVPR2019](https://raoyongming.github.io/files/SFCNN.pdf)] Spherical Fractal Convolutional Neural Networks for Point Cloud Recognition. 
13. [[MM](https://dl.acm.org/citation.cfm?id=3343031.3351009)] MMJN: Multi-Modal Joint Networks for 3D Shape Recognition. 



## 点云匹配&配准&对齐&注册

> 点云配准的概念也可以类比于二维图像中的配准，只不过二维图像配准获取得到的是x，y，alpha，beta等放射变化参数，三维点云配准可以模拟三维点云的移动和对齐，也就是会获得一个旋转矩阵和一个平移向量，通常表达为一个4×3的矩阵，其中3×3是旋转矩阵，1x3是平移向量。严格说来是6个参数，因为旋转矩阵也可以通过罗格里德斯变换转变成1*3的旋转向量。
>
> 常用的点云配准算法有两种：**正太分布变换**和著名的**ICP点云配准**，此外还有许多其它算法，列举如下：
>
> ICP：稳健ICP、point to plane ICP、point to line ICP、MBICP、GICP
>
> NDT 3D、Multil-Layer NDT
>
> FPCS、KFPSC、SAC-IA
>
> Line Segment Matching、ICL

1. [An ICP variant using a point-to-line metric](https://authors.library.caltech.edu/18274/1/Censi2008p84782008_Ieee_International_Conference_On_Robotics_And_Automation_Vols_1-9.pdf)
2. [Generalized-ICP](https://www.researchgate.net/publication/221344436_Generalized-ICP)
3. [Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration](https://www.researchgate.net/publication/228571031_Linear_Least-Squares_Optimization_for_Point-to-Plane_ICP_Surface_Registration)
4. [Metric-Based Iterative Closest Point Scan Matching for Sensor Displacement Estimation](http://webdiis.unizar.es/~jminguez/MbICP_TRO.pdf)
5. [NICP: Dense Normal Based Point Cloud Registration](http://jacoposerafin.com/wp-content/uploads/serafin15iros.pdf)
6. [Efficient Global Point Cloud Alignment using Bayesian Nonparametric Mixtures](http://openaccess.thecvf.com/content_cvpr_2017/papers/Straub_Efficient_Global_Point_CVPR_2017_paper.pdf)[CVPR2017]
7. [3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)[CVPR2017]
8. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] Density Adaptive Point Set Registration.
9. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Vongkulbhisal_Inverse_Composition_Discriminative_CVPR_2018_paper.pdf)] Inverse Composition Discriminative Optimization for Point Cloud Registration.
10. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf)] PPFNet: Global Context Aware Local Features for Robust 3D Point Matching.
11. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf)] Learning and Matching Multi-View Descriptors for Registration of Point Clouds. 
12. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. 
13. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. 
14. [[IROS2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. 
15. [[CVPR2019](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. 
16. [[CVPR2019](https://arxiv.org/abs/1904.03483)] SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. 
17. [[CVPR2019](https://arxiv.org/abs/1811.06879v2)] The Perfect Match: 3D Point Cloud Matching with Smoothed Densities. 
18. [[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. 
19. [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration. 
20. [[ICCV2019](https://arxiv.org/abs/1905.04153v2)] DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration.
21. [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)] Deep Closest Point: Learning Representations for Point Cloud Registration. 
22. [[ICRA2019](https://arxiv.org/abs/1904.09742)] 2D3D-MatchNet: Learning to Match Keypoints across 2D Image and 3D Point Cloud.
23. [[CVPR2019](https://arxiv.org/abs/1811.06879v2)] The Perfect Match: 3D Point Cloud Matching with Smoothed Densities.
24. [[CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration.
25. [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Robust_Variational_Bayesian_Point_Set_Registration_ICCV_2019_paper.pdf)] Robust Variational Bayesian Point Set Registration.
26. [[ICRA2019](https://arpg.colorado.edu/papers/hmrf_icp.pdf)] Robust low-overlap 3-D point cloud registration for outlier rejection. 
27. Learning multiview 3D point cloud registration[[CVPR2020]()]
28. 



## 点云匹配质量评估

1. [[IROS2017](https://ieeexplore.ieee.org/document/8206584)] Analyzing the quality of matched 3D point clouds of objects.

## 点云分割

> 点云的分割也算是一个大Topic了，这里因为多了一维就和二维图像比多了许多问题，点云分割又分为区域提取、线面提取、语义分割与聚类等。同样是分割问题，点云分割涉及面太广，确实是三言两语说不清楚的。只有从字面意思去理解了，遇到具体问题再具体归类。一般说来，点云分割是目标识别的基础。
>
> 分割主要有四种方法：基于边的区域分割、基于面的区域分割、基于聚类的区域分割、混合区域分割方法、深度学习方法
>
> 分割：区域声场、Ransac线面提取、NDT-RANSAC、K-Means（谱聚类）、Normalize Cut、3D Hough Transform(线面提取)、连通分析

1.  [基于局部表面凸性的散乱点云分割算法研究](http://ir.ciomp.ac.cn/handle/181722/57569?mode=full&submit_simple=Show+full+item+record)
2.  [三维散乱点云分割技术综述](http://www.cnki.com.cn/Article/CJFDTotal-ZZGX201005012.htm)
3.  [基于聚类方法的点云分割技术的研究](http://cdmd.cnki.com.cn/Article/CDMD-10213-1015979890.htm)
4.  [SceneEncoder: Scene-Aware Semantic Segmentation of Point Clouds with A Learnable Scene Descriptor](https://arxiv.org/abs/2001.09087v1)
5.  [From Planes to Corners: Multi-Purpose Primitive Detection in Unorganized 3D Point Clouds](https://arxiv.org/abs/2001.07360?context=cs.RO)
6.  [Learning and Memorizing Representative Prototypes for 3D Point Cloud Semantic and Instance Segmentation](http://arxiv.org/abs/2001.01349)
7.  [JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds](https://arxiv.org/abs/1912.09654v1)
8.  [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593v2)
9.  [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413v1)
10.  [SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation,CVPR2017]()
11.  [[ICRA2017](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds.
12.  [[3DV2017](http://segcloud.stanford.edu/segcloud_2017.pdf)] SEGCloud: Semantic Segmentation of 3D Point Clouds. 
13.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Recurrent_Slice_Networks_CVPR_2018_paper.pdf)] Recurrent Slice Networks for 3D Segmentation of Point Clouds. 
14.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)] SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation.
15.  [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf)] Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs.
16.  [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper.pdf)] 3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation.
17.  [[CVPR2019](https://arxiv.org/abs/1904.00699v1)] JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields. 
18.  [[CVPR2019](https://arxiv.org/abs/1903.00709)] PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation. 
19.  [[ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lahoud_3D_Instance_Segmentation_via_Multi-Task_Metric_Learning_ICCV_2019_paper.pdf)] 3D Instance Segmentation via Multi-Task Metric Learning.
20.  [[IROS2019](https://arxiv.org/pdf/1909.01643v1.pdf)] PASS3D: Precise and Accelerated Semantic Segmentation for 3D Point Cloud.

## 点云目标检索

> 这是点云数据处理中一个偏应用层面的问题，简单说来就是Hausdorff距离常被用来进行深度图的目标识别和检索，现在很多三维[人脸识别](https://cloud.tencent.com/product/facerecognition?from=10680)都是用这种技术来做的。

## 点云三维重建

> 我们获取到的点云数据都是一个个孤立的点，如何从一个个孤立的点得到整个曲面呢，这就是三维重建的topic。
>
> 在玩kinectFusion时候，如果我们不懂，会发现曲面渐渐变平缓，这就是重建算法不断迭代的效果。我们采集到的点云是充满噪声和孤立点的，三维重建算法为了重构出曲面，常常要应对这种噪声，获得看上去很舒服的曲面。
>
> 常用的三维重建算法和技术有：
>
> 泊松重建、Delauary triangulatoins(Delauary三角化)
>
> 表面重建，人体重建，建筑物重建，输入重建
>
> 实时重建：重建纸杯或者农作物4D生长台式，人体姿势识别，表情识别

1. [改进的点云数据三维重建算法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ4723ebf4b0b0762)
2. [Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity,CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf)
3. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nan_PolyFit_Polygonal_Surface_ICCV_2017_paper.pdf)] PolyFit: Polygonal Surface Reconstruction from Point Clouds.
4. [[ICCV2017](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ladicky_From_Point_Clouds_ICCV_2017_paper.pdf)] From Point Clouds to Mesh using Regression.
5. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kejie_Li_Efficient_Dense_Point_ECCV_2018_paper.pdf)] Efficient Dense Point Cloud Object Reconstruction using Deformation Vector Fields. 
6. [[ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration.
7. [[AAAI2018](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16530/16302)] Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction. 
8. [[CVPR2019](https://www.researchgate.net/publication/332240602_Robust_Point_Cloud_Based_Reconstruction_of_Large-Scale_Outdoor_Scenes)] Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes. 
9. [[AAAI2019](https://arxiv.org/abs/1811.11731)] CAPNet: Continuous Approximation Projection For 3D Point Cloud Reconstruction Using 2D Supervision. 
10. [[MM](https://dl.acm.org/citation.cfm?id=3350960)] L2G Auto-encoder: Understanding Point Clouds by Local-to-Global Reconstruction with Hierarchical Self-Attention. 
11. [SurfNet: Generating 3D shape surfaces using deep residual networks]()

## 点云其它

1. [[CVPR2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yun_Reflection_Removal_for_CVPR_2018_paper.pdf)] Reflection Removal for Large-Scale 3D Point Clouds. 
2. [[ICML2018](https://arxiv.org/abs/1707.02392)] Learning Representations and Generative Models for 3D Point Clouds.
3. [[3DV](https://arxiv.org/abs/1808.00671)] PCN: Point Completion Network. 
4. [[CVPR2019](https://arxiv.org/abs/1812.02713)] PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding. 
5. [[CVPR2019](http://www.linliang.net/wp-content/uploads/2019/04/CVPR2019_PointClound.pdf)] ClusterNet: Deep Hierarchical Cluster Network with Rigorously Rotation-Invariant Representation for Point Cloud Analysis.
6. [[ICCV2019](https://arxiv.org/pdf/1812.07050.pdf)] LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis. 
7. [[ICRA2019](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_2.html)] Speeding up Iterative Closest Point Using Stochastic Gradient Descent. 

## 点云数据集

1. [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite.
2. [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . 
3. [[ShapeNet](https://www.shapenet.org/)] A collaborative dataset between researchers at Princeton, Stanford and TTIC.
4. [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. 
5. [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. 
6. [[S3DIS](http://buildingparser.stanford.edu/dataset.html#Download)] The Stanford Large-Scale 3D Indoor Spaces Dataset.
7. [[ScanNet](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. 
8. [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. 
9. [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . 
10. [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
11. [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees.
12. [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons.
13. [[Large-Scale Point Cloud Classification Benchmark(ETH)](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. 
14. [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.
15. [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
16. [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. 
17. [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
18. [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
19. [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck.
20. [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
21. [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild.
22. [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition.
23. [[WAD](http://wad.ai/2019/challenge.html)] [[ApolloScape](http://apolloscape.auto/tracking.html)] The datasets are provided by Baidu Inc. 
24. [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.
25. [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] 
26. [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. 
27. [[BLVD](https://github.com/VCCIV/BLVD)] (a) 3D detection, (b) 4D tracking, (c) 5D interactive event recognition and (d) 5D intention prediction. [[ICRA 2019 paper](https://arxiv.org/abs/1903.06405v1)] 
28. [[PedX](https://arxiv.org/abs/1809.03605)] 3D Pose Estimation of Pedestrians, more than 5,000 pairs of high-resolution (12MP) stereo images and LiDAR data along with providing 2D and 3D labels of pedestrians. [[ICRA 2019 paper](https://arxiv.org/abs/1809.03605)] 
29. [[H3D](https://usa.honda-ri.com/H3D)] Full-surround 3D multi-object detection and tracking dataset. [[ICRA 2019 paper](https://arxiv.org/abs/1903.01568)] 
30. [[Matterport3D](https://niessner.github.io/Matterport/)] RGB-D: 10,800 panoramic views from 194,400 RGB-D images. Annotations: surface reconstructions, camera poses, and 2D and 3D semantic segmentations. Keypoint matching, view overlap prediction, normal prediction from color, semantic segmentation, and scene classification. [[3DV 2017 paper](https://arxiv.org/abs/1709.06158)] [[code](https://github.com/niessner/Matterport)] [[blog](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/)]
31. [[SynthCity](https://arxiv.org/abs/1907.04758)] SynthCity is a 367.9M point synthetic full colour Mobile Laser Scanning point cloud. Nine categories. 
32. [[Lyft Level 5](https://level5.lyft.com/dataset/?source=post_page)] Include high quality, human-labelled 3D bounding boxes of traffic agents, an underlying HD spatial semantic map.
33. [[SemanticKITTI](http://semantic-kitti.org/)] Sequential Semantic Segmentation, 28 classes, for autonomous driving. All sequences of KITTI odometry labeled. [[ICCV 2019 paper](https://arxiv.org/abs/1904.01416)] 
34. [[NPM3D](http://npm3d.fr/paris-lille-3d)] The Paris-Lille-3D has been produced by a Mobile Laser System (MLS) in two different cities in France (Paris and Lille).
35. [[The Waymo Open Dataset](https://waymo.com/open/)] The Waymo Open Dataset is comprised of high resolution sensor data collected by Waymo self-driving cars in a wide variety of conditions. 
36. [[A*3D: An Autonomous Driving Dataset in Challeging Environments](https://github.com/I2RDL2/ASTAR-3D)] A*3D: An Autonomous Driving Dataset in Challeging Environments. 
37. [[PointDA-10 Dataset](https://github.com/canqin001/PointDAN)] Domain Adaptation for point clouds.
38. [[Oxford Robotcar](https://robotcar-dataset.robots.ox.ac.uk/)] The dataset captures many different combinations of weather, traffic and pedestrians. 

# 三维重建

资料汇总：https://github.com/openMVG/awesome_3DReconstruction_list

## 单目图像

> 主要分为基于SfM三维重建和基于Deep learning的三维重建方法，sfM方法在下节将会做详细介绍，基于深度学习方式，主要通过RGB图像生成深度图。

### Paper

1. Unsupervised Monocular Depth Estimation with Left-Right Consistency
2. Unsupervised Learning of Depth and Ego-Motion from Video
3. Deep Ordinal Regression Network for Monocular Depth Estimation
4. Depth from Videos in the Wild
5. Attention-based Context Aggregation Network for Monocular Depth Estimation 
6. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network（NIPS2014）
7. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture（ICCV2015)
8. Deeper Depth Prediction with Fully Convolutional Residual Networks
9. Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation(CVPR2017)
10. Single View Stereo Matching

### Project with code

| Project                                                      | Paper                                                        | Framework  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------- |
| [3dr2n2: A unified approach for single and multi-view 3d object Reconstruction](https://github.com/chrischoy/3D-R2N2) | [ECCV 2016](https://arxiv.org/abs/1604.00449)                | Theano     |
| [Learning a predictable and generative vector representation for objects](https://github.com/rohitgirdhar/GenerativePredictableVoxels) | [ECCV 2016](https://arxiv.org/abs/1603.08637)                | Caffe      |
| [Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling](https://github.com/zck119/3dgan-release) | [NIPS 2016](http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf) | Torch 7    |
| [Perspective transformer nets: Learning single-view 3d object reconstruction without 3d supervision](https://github.com/xcyan/nips16_PTN) | [NIPS 2016](https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf) | Torch 7    |
| Deep disentangled representations for volumetric reconstruction | [ECCV 2016](https://arxiv.org/pdf/1610.03777.pdf)            |            |
| [Multi-view 3D Models from Single Images with a Convolutional Network](https://github.com/lmb-freiburg/mv3d) | [ECCV 2016](https://lmb.informatik.uni-freiburg.de/Publications/2016/TDB16a/paper-mv3d.pdf) | Tensorflow |
| [Single Image 3D Interpreter Network](https://github.com/jiajunwu/3dinn) | [ECCV 2016](http://3dinterpreter.csail.mit.edu/papers/3dinn_eccv.pdf) | Torch 7    |
| [Weakly-Supervised Generative Adversarial Networks for 3D Reconstruction](https://github.com/jgwak/McRecon) | [3DV 2017](https://arxiv.org/pdf/1705.10904.pdf)             | Theano     |
| [Hierarchical Surface Prediction for 3D Object Reconstruction](https://github.com/chaene/hsp) | [3DV 2017](https://arxiv.org/pdf/1704.00710.pdf)             | Torch 7    |
| [Octree generating networks: Efficient convolutional architectures for high-resolution 3d outputs](https://github.com/lmb-freiburg/ogn) | [ICCV 2017](https://arxiv.org/pdf/1703.09438.pdf)            | Caffe      |
| [Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency](https://github.com/shubhtuls/drc) | [CVPR 2017](https://arxiv.org/pdf/1704.06254.pdf)            | Torch 7    |
| [SurfNet: Generating 3D shape surfaces using deep residual networks](https://github.com/sinhayan/surfnet) | [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sinha_SurfNet_Generating_3D_CVPR_2017_paper.pdf) | Matlab     |
| [A Point Set Generation Network for 3D Object Reconstruction from a Single Image](https://github.com/fanhqme/PointSetGeneration) | [CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fan_A_Point_Set_CVPR_2017_paper.pdf) | Tensorflow |
| [O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis](https://github.com/Microsoft/O-CNN) | [SIGGRAPH 2017](https://wang-ps.github.io/O-CNN_files/CNN3D.pdf) | Caffe      |
| Rethinking Reprojection: Closing the Loop for Pose-aware Shape Reconstruction from a Single Image | [ICCV 2017](https://jerrypiglet.github.io/pdf/ICCV2017.pdf)  |            |
| Scaling CNNs for High Resolution Volumetric Reconstruction From a Single Image | [ICCV 2017](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Johnston_Scaling_CNNs_for_ICCV_2017_paper.pdf) |            |
| Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55 | [ICCV 2017](https://arxiv.org/pdf/1710.06104.pdf)            |            |
| [Learning a Hierarchical Latent-Variable Model of 3D Shapes](https://github.com/lorenmt/vsl) | [3DV 2018](https://arxiv.org/pdf/1705.05994.pdf)             | Tensorflow |
| [Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction](https://github.com/ericlin79119/3D-point-cloud-generation) | [AAAI 2018](https://chenhsuanlin.bitbucket.io/3D-point-cloud-generation/paper.pdf) | Tensorflow |
| [DeformNet: Free-Form Deformation Network for 3D Shape Reconstruction from a Single Image](https://github.com/jackd/template_ffd) | [ACCV 2018](https://jhonykaesemodel.com/papers/learning_ffd_accv2018-camera_ready.pdf) | Tensorflow |
| [Image2Mesh: A Learning Framework for Single Image 3DReconstruction](https://github.com/jhonykaesemodel/image2mesh) | [ACCV 2018](https://arxiv.org/pdf/1711.10669.pdf)            | Pytorch    |
| [Neural 3D Mesh Renderer](https://github.com/hiroharu-kato/mesh_reconstruction) | [CVPR 2018](https://arxiv.org/pdf/1711.07566.pdf)            | Chainer    |
| [Multi-view Consistency as Supervisory Signal for Learning Shape and Pose Prediction](https://github.com/shubhtuls/mvcSnP) | [CVPR 2018](https://arxiv.org/pdf/1801.03910.pdf)            | Torch 7    |
| [Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers](https://bitbucket.org/visinf/projects-2018-matryoshka) | [CVPR 2018](https://arxiv.org/pdf/1804.10975.pdf)            | Pytorch    |
| [AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation](https://github.com/ThibaultGROUEIX/AtlasNet) | [CVPR 2018](https://arxiv.org/pdf/1802.05384.pdf)            | Pytorch    |
| [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](https://github.com/nywang16/Pixel2Mesh) | [ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf) | Tensorflow |
| [Multiresolution Tree Networks for 3D Point Cloud Processing](https://github.com/matheusgadelha/MRTNet) | [ECCV 2018](https://arxiv.org/pdf/1807.03520.pdf)            | Pytorch    |
| Adaptive O-CNN: A Patch-based Deep Representation of 3D Shapes | [SIGGRAPH Asia 2018](https://wang-ps.github.io/AO-CNN_files/AOCNN.pdf) |            |
| [Learning Implicit Fields for Generative Shape Modeling](https://github.com/czq142857/implicit-decoder) | [CVPR 2019](https://arxiv.org/pdf/1812.02822.pdf)            | Tensorflow |
| [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://github.com/autonomousvision/occupancy_networks) | [CVPR 2019](https://arxiv.org/pdf/1812.03828.pdf)            | Pytorch    |
| [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://github.com/hassony2/shape_sdf) | [CVPR 2019](https://arxiv.org/pdf/1901.05103.pdf)            | Pytorch    |

## 结构光

> 结构光投影三维成像目前是机器人3D 视觉感知的主要方式,结构光成像系统是由若干个投影仪和
> 相机组成, 常用的结构形式有: 单投影仪-单相机、单投影仪-双相机 、单投影仪-多相机、单相机-双投影
> 仪和单相机-多投影仪等典型结构形式.  
>
> 结构光投影三维成像的基本工作原理是：投影仪向目标物体投射特定的结构光照明图案,由相机摄取被目标调制后的图像,再通过图像处理和视觉模型求出目标物体的三维信息. 常用的投影仪主要有下列几种类型:液晶投影(LCD)、数字光调制投影(DLP)[如数字微镜器件 (DMD)]、激光 LED图案直接投影. 根据结构光投影次数划分,结构光投影三维成像可以分成单次投影3D和多次投影3D方法.
>
> 按照扫描方式又可分为：线扫描结构光、面阵结构光

参考链接：[https://zhuanlan.zhihu.com/p/29971801](https://zhuanlan.zhihu.com/p/29971801)

[结构光三维表面成像：综述（一）](https://www.baidu.com/link?url=nj4az5QbljqZTZwS4W1jGzfkYrN8iLk1YwB0RVI1_jR_axeOLDaraHgYFwsNcZ9h&wd=&eqid=e120080700017e0b000000055e5d1f70)

[结构光三维表面成像：综述（二）](https://www.jianshu.com/p/b0e030f3a522)

[结构光三维表面成像：综述（三）](https://www.jianshu.com/p/c8d0afd817cc)

### 综述

[Structured-light 3D surface imaging: a tutorial](https://link.zhihu.com/?target=http%3A//www.rtbasics.com/Downloads/IEEE_structured_light.pdf)

[机器人视觉三维成像技术综述  ](http://www.opticsjournal.net/Articles/Abstract?aid=OJ9aea62c85c7ac99d)

[Real-time structured light profilometry a review  ]()

[A state of the art in structured light patterns for surface profilometry  ]()

[Phase shifting algorithms for fringe projection profilometry: a review  ]()

[Overview of the 3D profilometry of phase shifting fringe projection]()

[Temporal phase unwrapping algorithms for fringe projection profilometry:a comparative review]()

### Lectures&Video

1. [**Build Your Own 3D Scanner: Optical Triangulation for Beginners**](https://link.zhihu.com/?target=http%3A//mesh.brown.edu/byo3d/)
2. [https://github.com/nikolaseu/thesis](https://github.com/nikolaseu/thesis)
3. [**CS6320 3D Computer Vision**, Spring 2015](http://www.sci.utah.edu/~gerig/CS6320-S2015/CS6320_3D_Computer_Vision.html)

### 标定

1. [高效线结构光视觉测量系统标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ590c414efec735a)
2. [一种新的线结构光标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ6ca613096af02c49)
3. [一种结构光三维成像系统的简易标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ8214561beb0d211d)
4. [基于单应性矩阵的线结构光系统简易标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ200109000160x5A8D0)
5. [线结构光标定方法综述](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180307000127TqWtZv)
6. [三线结构光视觉传感器现场标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180908000212OkRnTq)
7. [单摄像机单投影仪结构光三维测量系统标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180808000091UrXu1w)
8. [超大尺度线结构光传感器内外参数同时标定](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180319000030HdJgMj)
9. [单摄像机单投影仪结构光三维测量系统标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180808000039aHdKgM)
10. [三维空间中线结构光与相机快速标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ170329000177sZv2y5)
11. [线结构光传感系统的快速标定方法](http://www.opticsjournal.net/Articles/Abstract?aid=OJ091028000118B8DaGd)



### 单次投影成像

> 单次投影结构光主要采用空间复用编码和频率复用编码形式实现 ,常用的编码形式有:彩色编码 、灰度索引、 几何形状编码和随机斑点. 目前在机器人手眼系统应用中,对于三维测量精度要求不高的场合,如码垛、拆垛、三维抓取等,比较受欢迎的是投射伪随机斑点获得目标三维信息 。

1. One-shot pattern projection for dense and accurate 3D acquisition in structured light

2. A single-shot structured light means by encoding both color and geometrical features  

3. Dynamic 3D surface profilometry using a novel colour pattern encoded with a multiple triangular mode  

4. Review of single-shot 3D shape measurement by phase calculation-based fringe projection techniques

5. Robust pattern decoding in shape-coded structured light

   

### 多次投影成像  

> 多次投影3D方法主要采用时间复用编码方式实现,常用的图案编码形式有:二进制编码、多频相移编码和混合编码法(如格雷码＋相移条纹)等.
>
> 但是格雷码方法仅能在投射空间内进行离散的划分,空间分辨率受到成像器件的限制. 为了提高空间分辨率,需要增加投影条纹幅数,投射条纹宽度更小的格雷码条纹图,但条纹宽度过小会导致格雷码条纹的边缘效应,从而引
> 起解码误差. 
>
> 正弦光栅条纹投影克服了格雷码空间离散划分的缺点,成为使用率最高的结构光类型之一. 众所周知,对于复杂外形,如有空洞、阶梯、遮挡等,采用正弦单频相移法条纹投影时,存在相位解包裹难题.另外为了能够从系列条纹图中求出相位绝对值,需要在条纹中插入特征点,比如一个点、一条线作为参考相位点,但是这些点或线特征标志有可能投影在物体的遮挡或阴影区域,或受到环境光等干扰等,发生丢失,影响测量结果的准确性. 因此,对于复杂轮廓的物体,常采用多频相移技术.   

1. 三维重建的格雷码-相移光编码技术研究
2. Pattern codification strategies in structured light systems  
3. Binary coded linear fringes for three-dimensional shape profiling
4. 3D shape measurement based on complementary Gray-code light
5. Phase shifting algorithms for fringe projection profilometry: a review  
6. Overview of the 3D profilometry of phase shifting fringe projection
7. Temporal phase unwrapping algorithms for fringe projection profilometry:a comparative review
8. A multi-frequency  inverse-phase error compensation method for projectornon linear in3D shape measurement

### 偏折法成像  

> 对于粗糙表面,结构光可以直接投射到物体表面进行视觉成像测量;但对于大反射率光滑表面和镜面物体3D 测量,结构光投影不能直接投射到被测表面,3D测量还需要借助镜面偏折技术 .

1. Principles of shape from specular reflection  
2. Deflectometry: 3D-metrology from nanometer to meter  
3. Three-dimensional shape measurement of a highly reflected specular surface with structured light
   method
4. Three-dimensional shape measurements of specular objects using phase-measuring deflectometry  

> 由于单次投影曝光和测量时间短,抗振动性能好,适合运动物体的3D测量,如机器人实时运动引导,手眼机器人对生产线上连续运动产品进行抓取等操作. 但深度垂直方向上的空间分辨率受到目标视场、镜头倍率和相机像素等因素的影响,大视场情况下不容易提升.
>
> 多次投影方法(如多频条纹方法)具有较高空间分辨率,能有效地解决表面斜率阶跃变化和空洞等难题. 不足之处在于:① 对于连续相移投影方法,3D重构的精度容易受到投影仪、相机的非线性和环境变化的影响;②抗振动性能差,不合适测量连续运动的物体;③在 Eye-in-Hand视觉导引系统中,机械臂不易在连续运动时进行3D成像和引导;④实时性差,不过随着投影仪投射频率和 CCD/CMOS图像传感器采集速度的提高,多次投影方法实时3D
> 成像的性能也在逐步改进.
>
> 偏折法对于复杂面型的测量,通常需要借助多次投影方法,因此具有多次投影方法相同的缺点.另外偏折法对曲率变化大的表面测量有一定的难度,因为条纹偏折后的反射角的变化率是被测表面曲率变化率的２倍,因此对被测物体表面的曲率变化比较敏感,很容易产生遮挡难题.  

### Other Papers

2. 基于面结构光的三维重建阴影补偿算法
3. [Enhanced phase measurement profilometry for industrial 3D inspection automation](https://www.researchgate.net/publication/273481900_Enhanced_phase_measurement_profilometry_for_industrial_3D_inspection_automation)
4. [Profilometry of three-dimensional discontinuous solids by combining two-steps temporal phase unwrapping, co-phased profilometry and phase-shifting interferometry](http://xueshu.baidu.com/usercenter/paper/show?paperid=3056d2277236112a708e4746a73e1e1d&site=xueshu_se)
5. [360-Degree Profilometry of Discontinuous Solids Co-Phasing ２-Projectors and１-Camera  ]()
6. [Coherent digital demodulation of single-camera N-projections for 3D-object shape measurement Co-phased profilometr  ]()
7. [High-speed 3D image acquisition using coded structured light projection](https://www.researchgate.net/publication/224296439_High-speed_3D_image_acquisition_using_coded_structured_light_projection)
8. [Accurate 3D measurement using a Structured Light System](https://www.researchgate.net/publication/222500455_Accurate_3D_measurement_using_a_Structured_Light_System)
9. [Structured light stereoscopic imaging with dynamic pseudo-random patterns  ](https://static.aminer.org/pdf/PDF/000/311/975/a_high_precision_d_object_reconstruction_method_using_a_color.pdf)
10. [Robust one-shot 3D scanning using loopy belief propagation  ](https://www.researchgate.net/publication/224165371_Robust_one-shot_3D_scanning_using_loopy_belief_propagation)
11. [Robust Segmentation and Decoding of a Grid Pattern for Structured Light](https://www.semanticscholar.org/paper/Robust-Segmentation-and-Decoding-of-a-Grid-Pattern-Pag%C3%A8s-Salvi/dcbdd608dcdf03b0d0eba662c68915dcfa90e5a5)
12. [Rapid shape acquisition using color structured light and multi-pass dynamic programming  ](http://ieeexplore.ieee.org/iel5/7966/22019/01024035.pdf?arnumber=1024035)
13. [Improved stripe matching for colour encoded structured light  ]()
14. [Absolute phase mapping for one-shot dense pattern projection  ](https://www.researchgate.net/profile/Joaquim_Salvi/publication/224165341_Absolute_phase_mapping_for_one-shot_dense_pattern_projection/links/56ffaee708ae650a64f805dd.pdf)
15. [3D digital stereophotogrammetry: a practical guide to facial image acquisition  ]()
16. [Method and apparatus for 3D imaging using light pattern having multiple sub-patterns  ]()
17. [High speed laser three-dimensional imager  ]()
18. [Three-dimensional dental imaging method and apparatus having a reflective member  ]()
19. [3D surface profile imaging method and apparatus using single spectral light condition  ]()
20. [Three-dimensional surface profile imaging method and apparatus using single spectral light condition]()
21. [High speed three dimensional imaging method  ]()
22. [A hand-held photometric stereo camera for 3-D modeling  ]()
23. [High-resolution, real-time 3D absolute coordinate measurement based on a phase-shifting method  ]()
24. [A fast three-step phase shifting algorithm  ]()

### Code

1. [https://github.com/jakobwilm/slstudio](https://github.com/jakobwilm/slstudio)
2. [https://github.com/phreax/structured_light](https://github.com/phreax/structured_light)
3. [https://github.com/nikolaseu/neuvision](https://github.com/nikolaseu/neuvision)
4. [https://github.com/pranavkantgaur/3dscan](https://github.com/pranavkantgaur/3dscan)

## 扫描3D成像  

> 扫描3D成像方法可分为扫描测距、主动三角法、色散共焦法等。扫描3D成像的最大优点是测量精度高,其中
> 色散共焦法还有其他方法难以比拟的优点,即非常适合测量透明物体、高反与光滑表面的物体. 但缺点是速度慢、效率低;当用于机械手臂末端时,可实现高精度3D测量,但不适合机械手臂实时3D引导与定位,因此应用场合有限;另外主动三角扫描在测量复杂结构形貌时容易产生遮挡,需要通过合理规划末端路径与姿态来解决.

#### 扫描测距  

> 扫描测距是利用一条准直光束通过一维测距扫描整个目标表面实现3D测量，主要包括：单点飞行时间法、激光散射干涉法、 共焦法。
>
> 单点测距扫描3D方法中,单点飞行时间法适合远距离扫描,测量精度较低,一般在毫米量级. 其他几种单点扫描方法有:单点激光干涉法、共焦法和单点激光主动三角法,测量精度较高,但前者对环境要求高;线扫描精度适中,效率高. 比较适合于机械手臂末端执行3D测量的应是主动激光三角法和色散共焦法.  

##### Paper

1. Active optical range imaging sensor  
2. Active and passive range sensing for robotics  

#### 主动三角法

> 主动三角法是基于三角测量原理,利用准直光束、一条或多条平面光束扫描目标表面完成3D测量的. 光束常采用以下方式获得:激光准直、圆柱或二次曲面柱形棱角扩束,非相干光(如白光、LED 光源)通过小孔、狭缝(光栅)投影或相干光衍射等.  主动三角法可分为三种类型:单点扫描、单线扫描和多线扫描 

##### Paper

1. Review of different 3D scanners and scanning techniques  
2. 3D metrology using a collaborative robot with a laser triangulation sensor  
3. Introductory review on Flying Triangulation a motion-robust optical 3D measurement principle
4. Flying triangulation an optical 3D sensor for the motion-robust acquisition of complex object
5. Hand-Guided 3D Surface Acquisition by Combining Simple Light Sectioning with Real-Time Algorithms  

#### 色彩共焦法  

> 色散共焦似乎可以扫描测量粗糙和光滑的不透明和透明物体,如反射镜面、透明玻璃面等,目前在手机盖板三维检测等领域广受欢迎。色散共焦扫描有三种类型:单点一维绝对测距扫描、多点阵列扫描和连续线扫描。



##### Paper

1. Spectral characteristics of chromatic confocal imaging systems  
2. Spectrally multiplexed chromatic confocal multipoint sensing  
3. Chromatic confocal matrix sensor with actuated pinhole arrays  
4. Multiplex acquisition approach for high speed 3d measurements with a chromatic confocal
   microscope
5. Fast 3D in line-sensor for specular and diffuse surfaces combining the chromatic confocal and triangulation principle
6. Single-shot depth-section imaging through chromatic slit-scan confocal microscopy
7. Three-dimensional surface profile measurement using a beam scanning chromatic confocal microscope
8. 

## 立体视觉3D成像

> 立体视觉字面意思是用一只眼睛或两只眼睛感知三维结构,一般情况下是指从不同的视点获取两
> 幅或多幅图像重构目标物体3D结构或深度信息. 深度感知视觉线索可分为 Monocular cues
> 和 Binocular cues(双目视差). 目前立体视觉3D 可以通过单目视觉、双目视觉、多 (目) 视觉、光场3D
> 成像(电子复眼或阵列相机)实现.  

### 书籍

1. [机器视觉 Robot Vision]()

### 教程

1. [立体视觉书籍推荐&立体匹配十大概念综述---立体匹配算法介绍](https://zhuanlan.zhihu.com/p/20703577)
2. [【关于立体视觉的一切】立体匹配成像算法BM，SGBM，GC，SAD一览](https://zhuanlan.zhihu.com/p/32752535)
3. [StereoVision--立体视觉（1）](https://zhuanlan.zhihu.com/p/30116734)
4. [StereoVision--立体视觉（2）](https://zhuanlan.zhihu.com/p/30333032)
5. [StereoVision--立体视觉（3）](https://zhuanlan.zhihu.com/p/30754263)
6. [StereoVision--立体视觉（4）](https://zhuanlan.zhihu.com/p/31160700)
7. [StereoVision--立体视觉（5）](https://zhuanlan.zhihu.com/p/31500311)

### 综述

1. [Review of Stereo Vision Algorithms: From Software to Hardware  ]()
2. [双目立体视觉的研究现状及进展](http://www.opticsjournal.net/Articles/Abstract?aid=OJ180915000179tZw3z6)

### 单目视觉成像

> 单目视觉深度感知线索通常有:透视、焦距差异 、多视觉成像、覆盖、阴影 、运动视差等.   

1. Depth map extracting based on geometric perspective an applicable２D to３D conversion technology  
2. Focus cues affect perceived depth 
3. 3D image acquisition system based on shape from focus technique
4. Multi-view stereo: a tutorial
5. 3D reconstruction from multiple images part1 principles  
6. Three-dimensional reconstruction of hybrid surfaces using perspective shape from shading  
7. Numerical methods for shape-from-shading a new survey with benchmarks  
8. The neural basis of depth perception from motion parallax
9. Motion parallax in stereo 3D
10. 3D image sensor based on parallax motion  

### 双目视觉

> 在机器视觉里利用两个相机从两个视点对同一个目标场景获取两个视点图像,再计算两个视点图像中同名点的视差获得目标场景的3D深度信息. 典型的双目立体视觉计算过程包含下面四个步骤:图像畸变矫正、立体图像对校正、图像配准和三角法重投影视差图计算.
>
> **双目视觉的难点：**
>
> 1、光照敏感，被动光
>
> 2、双目视觉系统估计视差没那么容易，**立体匹配**是计算机视觉典型的难题，基线宽得到远目标测距准，而基线短得到近目标测距结果好。谈到双目系统的难点，除了立体匹配，还有标定。标定后的系统会出现“漂移”的，所以在线标定是必须具有的。

#### 综述

1. [双目立体视觉匹配技术综述](http://d.wanfangdata.com.cn/Periodical/cqgxyxb201502014)

#### 视差和深度计算

1. Real-time depth computation using stereo imaging  
2. Binocular disparity and the perception of depth  
3. Fast Stereo Disparity Maps Refinement By Fusion of Data-Based And Model-Based Estimations  

#### 立体匹配

> 匹配方法分两种，全局法和局部法，实用的基本是局部法，因为全局法太慢。
>
> （一）基于全局约束的立体匹配算法：在本质上属于优化算法，它是将立体匹配问题转化为寻找全局能量函数的最优化问题，其代表算法主要有图割算法、置信度传播算法和协同优化算法等．全局算法能够获得较低的总误匹配率，但算法复杂度较高,很难满足实时的需求，不利于在实际工程中使用，常见的算法有DP、BP  等。
>
> （二）基于局部约束的立体匹配算法：主要是利用匹配点周围的局部信息进行计算，由于其涉及到的信息量较少，匹配时间较短，因此受到了广泛关注，其代表算法主要有 SAD、SSD、ZSAD、NCC等。

1. [DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch](http://www.researchgate.net/publication/335788330_DeepPruner_Learning_Efficient_Stereo_Matching_via_Differentiable_PatchMatch)
2. [Improved Stereo Matching with Constant Highway Networks and Reflective Confidence Learning](http://www.researchgate.net/publication/320966979_Improved_Stereo_Matching_with_Constant_Highway_Networks_and_Reflective_Confidence_Learning)
3. [PMSC: PatchMatch-Based Superpixel Cut for Accurate Stereo Matching](http://140.98.202.196/document/7744590)
4. [Exact Bias Correction and Covariance Estimation for Stereo Vision](http://openaccess.thecvf.com/content_cvpr_2015/papers/Freundlich_Exact_Bias_Correction_2015_CVPR_paper.pdf)
5. [Efficient minimal-surface regularization of perspective depth maps in variational stereo](https://www.semanticscholar.org/paper/Efficient-minimal-surface-regularization-of-depth-Graber-Balzer/7579d0c7fd9872ff62bdec99335ce25e9fc3bd6a)
6. [Event-Driven Stereo Matching for Real-Time 3D Panoramic Vision](https://www.researchgate.net/publication/279512685_Event-Driven_Stereo_Matching_for_Real-Time_3D_Panoramic_Vision)
7. [Leveraging Stereo Matching with Learning-based Confidence Measures](https://www.researchgate.net/publication/273260542_Leveraging_Stereo_Matching_with_Learning-based_Confidence_Measures)
8. [Graph Cut based Continuous Stereo Matching using Locally Shared Labels](http://openaccess.thecvf.com/content_cvpr_2014/papers/Taniai_Graph_Cut_based_2014_CVPR_paper.pdf)
9. [Cross-Scale Cost Aggregation for Stereo Matching](http://de.arxiv.org/pdf/1403.0316)
10. [Fast Cost-Volume Filtering for Visual Correspondence and Beyond](https://publik.tuwien.ac.at/files/PubDat_202088.pdf)
11. [Constant Time Weighted Median Filtering for Stereo Matching and Beyond](http://openaccess.thecvf.com/content_iccv_2013/papers/Ma_Constant_Time_Weighted_2013_ICCV_paper.pdf)
12. [A non-local cost aggregation method for stereo matching](http://fcv2011.ulsan.ac.kr/files/announcement/592/A%20Non-Local%20Aggregation%20Method%20Stereo%20Matching.pdf)
13. [On building an accurate stereo matching system on graphics hardware](http://nlpr-web.ia.ac.cn/2011papers/gjhy/gh75.pdf)
14. [Efficient large-scale stereo matching](http://www.cs.toronto.edu/~urtasun/publications/geiger_et_al_accv10.pdf)
15. [Accurate, dense, and robust multiview stereopsis](https://www.researchgate.net/publication/221364612_Accurate_Dense_and_Robust_Multi-View_Stereopsis)
16. [A constant-space belief propagation algorithm for stereo matching](http://vision.ai.illinois.edu/publications/yang_cvpr10a.pdf)
17. [Stereo matching with color-weighted correlation, hierarchical belief propagation, and occlusion handling](https://www.computer.org/csdl/journal/tp/2009/03/ttp2009030492/13rRUxBJhnO)
18. [Cost aggregation and occlusion handling with WLS in stereo matching](https://www.researchgate.net/publication/51406942_Cost_Aggregation_and_Occlusion_Handling_With_WLS_in_Stereo_Matching)
19. [Stereo matching: An outlier confidence approach](http://www.cse.cuhk.edu.hk/leojia/all_final_papers/stereo_eccv08.pdf)
20. [ A region based stereo matching algorithm using cooperative optimization](http://vision.middlebury.edu/stereo/eval/papers/CORegion.pdf)
21. [Multi-view stereo for community photo collections](https://www.semanticscholar.org/paper/Multi-View-Stereo-for-Community-Photo-Collections-Goesele-Snavely/b59964ff729bbde324af83743cd3cf424ce69758)
22. [A performance study on different cost aggregation approaches used in real-time stereo matching](https://www.researchgate.net/publication/220659397_A_Performance_Study_on_Different_Cost_Aggregation_Approaches_Used_in_Real-Time_Stereo_Matching)
23. [Evaluation of cost functions for stereo matching]()
24. [Adaptive support-weight approach for correspondence search]()
25. [Segment-based stereo matching using belief propagation and a self-adapting dissimilarity measure]()

### 多目视觉

> 多(目)视觉成像,也称多视点立体成像,用单个或多个相机从多个视点获取同一个目标场景的多幅图像,重构目标场景的三维信息.   

1. Adaptive structure from motion with a contrario model estimation  
2. A comparison and evaluation of multi-view stereo reconstruction algorithms  
3. Multiple view geometry in computer vision  

### 光场成像

> 光场3D成像的原理与传统 CCD和 CMOS相机成像原理在结构原理上有所差异,传统相机成像是光线穿过镜头在后续的成像平面上直接成像,一般是2D图像;光场相机成像是在传感器平面前增加了一个微透镜阵列,将经过主镜头入射的光线再次穿过每个微透镜,由感光阵列接收,从而获得光线的方向与位置信息,使成像结果可在后期处理,达到先拍照,后聚焦的效果.
>
> 光场相机的优点是:单个相机可以进行3D成像,横向和深度方向的空间分辨率可以达到20μm到 mm 量级,景深比普通相机大好几倍,比较适合Eye-in-Hand系统3D测量与引导,但目前精度适中的商业化光场相机价格昂贵.

1. Light field imaging models calibrations reconstructions and applications  
2. Extracting depth information from stereo vision system using a correlation and a feature based methods  
3. 基于微透镜阵列型光场相机的多目标快速测距方法
4. 基于光场相机的四维光场图像水印及质量评价
5. 基于光场相机的深度面光场计算重构
6. 光场相机视觉测量误差分析
7. 一种基于光场图像的聚焦光场相机标定方法
8. 光场相机成像模型及参数标定方法综述
9. 

## SFM

> Structure from Motion（SfM）是一个估计相机参数及三维点位置的问题。一个基本的**SfM pipeline**可以描述为:对每张2维图片检测特征点（feature point），对每对图片中的特征点进行匹配，只保留满足几何约束的匹配，最后执行一个迭代式的、鲁棒的SfM方法来恢复摄像机的内参（intrinsic parameter）和外参(extrinsic parameter)。并由三角化得到三维点坐标，然后使用Bundle Adjustment进行优化。
>
> SFM（Structure From Motion），主要基于多视觉几何原理，用于从运动中实现3D重建，也就是从无时间序列的2D图像中推算三维信息，是计算机视觉学科的重要分支。
>
> 使用同一相机在其内参数不变的条件下,从不同视点获取多幅图像,重构目标场景的三维信息. 该技术常用
> 于跟踪目标场景中大量的控制点,连续恢复场景3D结构信息、相机的姿态和位置.  
>
> SfM方法可以分为增量式（incremental/sequential SfM）,全局式（global SfM），混合式（hybrid SfM）,层次式（hierarchica SfM）。另外有基于语义的SfM(Semantic SfM)和基于Deep learning的SfM。
>
> Incremental SfM
>
> Global SfM
>
> Hierarchical SfM
>
> Multi-Stage SfM
>
> Non Rigid SfM

### 参考

[基于单目视觉的三维重建算法综述](https://zhuanlan.zhihu.com/p/55712813)

#### Turtorial

1. [Open Source Structure-from-Motion](https://blog.kitware.com/open-source-structure-from-motion-at-cvpr-2015/). M. Leotta, S. Agarwal, F. Dellaert, P. Moulon, V. Rabaud. CVPR 2015 Tutorial [(material)](https://github.com/mleotta/cvpr2015-opensfm).
2. Large-scale 3D Reconstruction from Images](https://home.cse.ust.hk/~tshenaa/sub/ACCV2016/ACCV_2016_Tutorial.html). T. Shen, J. Wang, T.Fang, L. Quan. ACCV 2016 Tutorial.

#### Incremental SfM

> 增量式SfM首先使用SIFT特征检测器提取特征点并计算特征点对应的描述子（descriptor），然后使用ANN（approximate nearest neighbor）方法进行匹配，低于某个匹配数阈值的匹配对将会被移除。对于保留下来的匹配对，使用RANSAC和八点法来估计基本矩阵（fundamental matrix），在估计基本矩阵时被判定为外点（outlier）的匹配被看作是错误的匹配而被移除。对于满足以上几何约束的匹配对，将被合并为tracks。然后通过incremental方式的SfM方法来恢复场景结构。首先需要选择一对好的初始匹配对，一对好的初始匹配对应该满足：
>
> （1）足够多的匹配点；
>
> （2）宽基线。之后增量式地增加摄像机，估计摄像机的内外参并由三角化得到三维点坐标，然后使用Bundle Adjustment进行优化。
>
> 增量式SfM从无序图像集合计算三维重建的常用方法，增量式SfM可分为如图 3所示几个阶段：图像特征提取、特征匹配、几何约束、重建初始化、图像注册、三角化、outlier过滤、Bundle adjustment等步骤。
>
> 增量式SfM优势：系统对于特征匹配以及外极几何关系的外点比较鲁棒，重讲场景精度高；标定过程中通过RANSAC不断过滤外点；捆绑调整不断地优化场景结构。
>
> 增量式SfM缺点：对初始图像对选择及摄像机的添加顺序敏感；场景漂移，大场景重建时的累计误差。效率不足，反复的捆绑调整需要大量的计算时间。
>
> 实现增量式SfM框架的包含COLMAP、openMVG、Theia等

1. [Photo Tourism: Exploring Photo Collections in 3D](http://phototour.cs.washington.edu/Photo_Tourism.pdf). N. Snavely, S. M. Seitz, and R. Szeliski. SIGGRAPH 2006.
2. [Towards linear-time incremental structure from motion](http://ccwu.me/vsfm/vsfm.pdf). C. Wu. 3DV 2013.
3. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf). Schöenberger, Frahm. CVPR 2016.

#### Global SfM

> 全局式：估计所有摄像机的旋转矩阵和位置并三角化初始场景点。
>
> 优势：将误差均匀分布在外极几何图上，没有累计误差。不需要考虑初始图像和图像添加顺序的问题。仅执行一次捆绑调整，重建效率高。
>
> 缺点：鲁棒性不足，旋转矩阵求解时L1范数对外点相对鲁棒，而摄像机位置求解时相对平移关系对匹配外点比较敏感。场景完整性，过滤外极几何边，可能丢失部分图像。

1. [Combining two-view constraints for motion estimation](http://www.umiacs.umd.edu/users/venu/cvpr01.pdf) V. M. Govindu. CVPR, 2001.
2. [Lie-algebraic averaging for globally consistent motion estimation](http://www.umiacs.umd.edu/users/venu/cvpr04final.pdf). V. M. Govindu. CVPR, 2004.
3. [Robust rotation and translation estimation in multiview reconstruction](http://imagine.enpc.fr/~monasse/Stereo/Projects/MartinecPajdla07.pdf). D. Martinec and T. Pajdla. CVPR, 2007.
4. [Non-sequential structure from motion](http://www.maths.lth.se/vision/publdb/reports/pdf/enqvist-kahl-etal-wovcnnc-11.pdf). O. Enqvist, F. Kahl, and C. Olsson. ICCV OMNIVIS Workshops 2011.
5. [Global motion estimation from point matches](https://web.math.princeton.edu/~amits/publications/sfm_3dimpvt12.pdf). M. Arie-Nachimson, S. Z. Kovalsky, I. KemelmacherShlizerman, A. Singer, and R. Basri. 3DIMPVT 2012.
6. [Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion](https://hal-enpc.archives-ouvertes.fr/hal-00873504). P. Moulon, P. Monasse and R. Marlet. ICCV 2013.
7. [A Global Linear Method for Camera Pose Registration](http://www.cs.sfu.ca/~pingtan/Papers/iccv13_sfm.pdf). N. Jiang, Z. Cui, P. Tan. ICCV 2013.
8. [Global Structure-from-Motion by Similarity Averaging](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf). Z. Cui, P. Tan. ICCV 2015.
9. [Linear Global Translation Estimation from Feature Tracks](http://arxiv.org/abs/1503.01832) Z. Cui, N. Jiang, C. Tang, P. Tan, BMVC 2015.

#### 混合式

> 混合式SfM[5]在一定程度上综合了incremental SfM和global SfM各自的优点。HSfM的整个pipeline可以概括为全局估计摄像机旋转矩阵，增量估计摄像机位置，三角化初始场景点。
>
> 用全局的方式提出一种基于社区的旋转误差平均法，该方法既考虑了对极几何的精度又考虑了成对几何的精度。基于已经估计的相机的绝对旋转姿态，用一种增量的方式估计相机光心位置。对每个添加的相机，其旋转和内参保持不变，同时使用改进的BA细化光心和场景结构。
>
> 层次式SfM同样借鉴incremental SfM和global SfM各自优势，但是基于分段式的incremental SfM和全局式SfM，没有像混合式SfM分成两个阶段进行。
>
> SfM中我们用来做重建的点是由特征匹配提供的，所以SfM获得特征点的方式决定了它不可能直接生成密集点云。而MVS则几乎对照片中的每个像素点都进行匹配，几乎重建每一个像素点的三维坐标，这样得到的点的密集程度可以较接近图像为我们展示出的清晰度。

#### Hierarchical SfM

1. [Structure-and-Motion Pipeline on a Hierarchical Cluster Tree](http://www.diegm.uniud.it/fusiello/papers/3dim09.pdf). A. M.Farenzena, A.Fusiello, R. Gherardi. Workshop on 3-D Digital Imaging and Modeling, 2009.
2. [Randomized Structure from Motion Based on Atomic 3D Models from Camera Triplets](https://www.researchgate.net/publication/224579249_Randomized_structure_from_motion_based_on_atomic_3D_models_from_camera_triplets). M. Havlena, A. Torii, J. Knopp, and T. Pajdla. CVPR 2009.
3. [Efficient Structure from Motion by Graph Optimization](https://dspace.cvut.cz/bitstream/handle/10467/62206/Havlena_stat.pdf?sequence=1&isAllowed=y). M. Havlena, A. Torii, and T. Pajdla. ECCV 2010.
4. [Hierarchical structure-and-motion recovery from uncalibrated images](http://www.diegm.uniud.it/fusiello/papers/cviu15.pdf). Toldo, R., Gherardi, R., Farenzena, M. and Fusiello, A.. CVIU 2015.

#### Multi-Stage SfM

1. [Parallel Structure from Motion from Local Increment to Global Averaging](https://arxiv.org/abs/1702.08601). S. Zhu, T. Shen, L. Zhou, R. Zhang, J. Wang, T. Fang, L. Quan. arXiv 2017.
2. [Multistage SFM : Revisiting Incremental Structure from Motion](https://researchweb.iiit.ac.in/~rajvi.shah/projects/multistagesfm/). R. Shah, A. Deshpande, P. J. Narayanan. 3DV 2014. -> [Multistage SFM: A Coarse-to-Fine Approach for 3D Reconstruction](http://arxiv.org/abs/1512.06235), arXiv 2016.
3. [HSfM: Hybrid Structure-from-Motion](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf). H. Cui, X. Gao, S. Shen and Z. Hu, ICCV 2017.

#### Non Rigid SfM

1. [Robust Structure from Motion in the Presence of Outliers and Missing Data](http://arxiv.org/abs/1609.02638). G. Wang, J. S. Zelek, J. Wu, R. Bajcsy. 2016.

### Project&code

| Project                                                 | Language | License                                              |
| ------------------------------------------------------- | -------- | ---------------------------------------------------- |
| [Bundler](https://github.com/snavely/bundler_sfm)       | C++      | GNU General Public License - contamination           |
| [Colmap](https://github.com/colmap/colmap)              | C++      | BSD 3-clause license - Permissive                    |
| [TeleSculptor](https://github.com/Kitware/TeleSculptor) | C++      | BSD 3-Clause license - Permissive                    |
| [MicMac](https://github.com/micmacIGN)                  | C++      | CeCILL-B                                             |
| [MVE](https://github.com/simonfuhrmann/mve)             | C++      | BSD 3-Clause license + parts under the GPL 3 license |
| [OpenMVG](https://github.com/openMVG/openMVG)           | C++      | MPL2 - Permissive                                    |
| [OpenSfM](https://github.com/mapillary/OpenSfM/)        | Python   | Simplified BSD license - Permissive                  |
| [TheiaSfM](https://github.com/sweeneychris/TheiaSfM)    | C++      | New BSD license - Permissive                         |

## TOF

> 飞行时间 (TOF) 相机每个像素利用光飞行的时间差来获取物体的深度。TOF成像可用于大视野、远距离、低精度、低成本的3D图像采集. 其特点是:检测速度快、视野范围较大、工作距离远、价格便宜,但精度低,易受环境
> 光的干扰 。

#### 分类

##### 直接TOF

> D-TOF通常用于单点测距系统, 为了实现面积范围3D成像,通常需要采用扫描技术 。 



##### 间接TOF

> 间接 TOF(I-TOF),时间往返行程是从光强度的时间选通测量中间接外推获得 ，I-TOF不需要精确的
> 计时,而是采用时间选通光子计数器或电荷积分器,它们可以在像素级实现.  



#### 教程

1. [ToF技术是什么？和结构光技术又有何区别?](https://zhuanlan.zhihu.com/p/51218791)
2. [3D相机--TOF相机](https://zhuanlan.zhihu.com/p/85519428)

#### Paper

1. https://arxiv.org/pdf/1511.07212.pdf)

## Multi-view Stereo

> 多视角立体视觉（Multiple View Stereo，MVS）是对立体视觉的推广，能够在多个视角（从外向里）观察和获取景物的图像，并以此完成匹配和深度估计。某种意义上讲，SLAM/SFM其实和MVS是类似的，只是前者是摄像头运动，后者是多个摄像头视角（**可以是单相机的多个视角图像，也可以是多相机的多视角图像**）。也可以说，前者可以在环境里面“穿行”，而后者更像在环境外“旁观”。
>
> 多视角立体视觉的pipelines如下：
>
> 1. 收集图像；
> 2. 针对每个图像计算相机参数；
> 3. 从图像集和相应的摄像机参数重建场景的3D几何图形；
> 4. 可选择地重建场景的形状和纹理颜色。

参考链接：[多视角立体视觉MVS简介](https://zhuanlan.zhihu.com/p/73748124)

### paper

1. [Learning Inverse Depth Regression for Multi-View Stereo with Correlation Cost Volume](https://arxiv.org/pdf/1912.11746v1.pdf)
2. [Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching](http://arxiv.org/abs/1912.06378v2)
3. [Point-Based Multi-View Stereo Network](http://arxiv.org/abs/1908.04422v1)
4. [Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference](http://arxiv.org/abs/1902.10556v1)
5. [NRMVS: Non-Rigid Multi-View Stereo](http://arxiv.org/abs/1901.03910v1)
6. [Multi-View Stereo 3D Edge Reconstruction](http://arxiv.org/abs/1801.05606v1)
7. [Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference](https://arxiv.org/abs/1902.10556)

#### 综述

1. [Multi-view stereo: A tutorial](https://www.mendeley.com/catalogue/multiview-stereo-tutorial/)
2. [State of the Art 3D Reconstruction Techniques](https://docs.google.com/file/d/0B851Hlh7xL0KNGx3X09VcEYzSjg/preview) N. Snavely, Y. Furukawa, CVPR 2014 tutorial slides. [Introduction](http://www.cse.wustl.edu/~furukawa/papers/cvpr2014_tutorial_intro.pdf) [MVS with priors](http://www.cse.wustl.edu/~furukawa/papers/cvpr2014_tutorial_mvs_prior.pdf) - [Large scale MVS](http://www.cse.wustl.edu/~furukawa/papers/cvpr2014_tutorial_large_scale_mvs.pdf)

#### Point cloud computation（点云计算）

1. [Accurate, Dense, and Robust Multiview Stereopsis](http://www.cs.wustl.edu/~furukawa/papers/cvpr07a.pdf). Y. Furukawa, J. Ponce. CVPR 2007. [PAMI 2010](http://www.cs.wustl.edu/~furukawa/papers/pami08a.pdf)
2. [State of the art in high density image matching](https://www.researchgate.net/publication/263465866_State_of_the_art_in_high_density_image_matching﻿). F. Remondino, M.G. Spera, E. Nocerino, F. Menna, F. Nex . The Photogrammetric Record 29(146), 2014.
3. [Progressive prioritized multi-view stereo](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Locher_Progressive_Prioritized_Multi-View_CVPR_2016_paper.pdf). A. Locher, M. Perdoch and L. Van Gool. CVPR 2016.
4. [Pixelwise View Selection for Unstructured Multi-View Stereo](https://demuc.de/papers/schoenberger2016mvs.pdf). J. L. Schönberger, E. Zheng, M. Pollefeys, J.-M. Frahm. ECCV 2016.
5. [TAPA-MVS: Textureless-Aware PAtchMatch Multi-View Stereo](https://arxiv.org/pdf/1903.10929.pdf). A. Romanoni, M. Matteucci. ICCV 2019

#### Surface computation & refinements（曲面计算与优化）

1. [Efficient Multi-View Reconstruction of Large-Scale Scenes using Interest Points, Delaunay Triangulation and Graph Cuts](http://www.di.ens.fr/sierra/pdfs/07iccv_a.pdf). P. Labatut, J-P. Pons, R. Keriven. ICCV 2007
2. [Multi-View Stereo via Graph Cuts on the Dual of an Adaptive Tetrahedral Mesh](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/SinhaICCV07.pdf). S. N. Sinha, P. Mordohai and M. Pollefeys. ICCV 2007.
3. [Towards high-resolution large-scale multi-view stereo](https://www.researchgate.net/publication/221364700_Towards_high-resolution_large-scale_multi-view_stereo).  H.-H. Vu, P. Labatut, J.-P. Pons, R. Keriven. CVPR 2009.
4. [Refinement of Surface Mesh for Accurate Multi-View Reconstruction](http://cmp.felk.cvut.cz/ftp/articles/tylecek/Tylecek-IJVR2010.pdf). R. Tylecek and R. Sara. IJVR 2010.
5. [High Accuracy and Visibility-Consistent Dense Multiview Stereo](https://hal.archives-ouvertes.fr/hal-00712178/).  H.-H. Vu, P. Labatut, J.-P. Pons, R. Keriven. Pami 2012.
6. [Exploiting Visibility Information in Surface Reconstruction to Preserve Weakly Supported Surfaces](https://www.researchgate.net/publication/275064596_Exploiting_Visibility_Information_in_Surface_Reconstruction_to_Preserve_Weakly_Supported_Surfaces) M. Jancosek et al. 2014.
7. [A New Variational Framework for Multiview Surface Reconstruction](http://urbanrobotics.net/pdf/A_New_Variational_Framework_for_Multiview_Surface_Reconstruction_86940719.pdf). B. Semerjian. ECCV 2014.
8. [Photometric Bundle Adjustment for Dense Multi-View 3D Modeling](https://www.inf.ethz.ch/personal/pomarc/pubs/DelaunoyCVPR14.pdf). A. Delaunoy, M. Pollefeys. CVPR2014.
9. [Global, Dense Multiscale Reconstruction for a Billion Points](https://lmb.informatik.uni-freiburg.de/people/ummenhof/multiscalefusion/). B. Ummenhofer, T. Brox. ICCV 2015.
10. [Efficient Multi-view Surface Refinement with Adaptive Resolution Control](http://slibc.student.ust.hk/pdf/arc.pdf). S. Li, S. Yu Siu, T. Fang, L. Quan. ECCV 2016.
11. [Multi-View Inverse Rendering under Arbitrary Illumination and Albedo](http://www.ok.ctrl.titech.ac.jp/~torii/project/mvir/), K. Kim, A. Torii, M. Okutomi, ECCV2016.
12. [Shading-aware Multi-view Stereo](http://www.gcc.tu-darmstadt.de/media/gcc/papers/Langguth-2016-SMV.pdf), F. Langguth and K. Sunkavalli and S. Hadap and M. Goesele, ECCV 2016.
13. [Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity](https://arxiv.org/abs/1705.00949), C. Mostegel, R. Prettenthaler, F. Fraundorfer and H. Bischof. CVPR 2017.
14. [Multi-View Stereo with Single-View Semantic Mesh Refinement](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Romanoni_Multi-View_Stereo_with_ICCV_2017_paper.pdf), A. Romanoni, M. Ciccone, F. Visin, M. Matteucci. ICCVW 2017

#### Machine Learning based MVS

1. [Matchnet: Unifying feature and metric learning for patch-based matching](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf), X. Han, Thomas Leung, Y. Jia, R. Sukthankar, A. C. Berg. CVPR 2015.
2. [Stereo matching by training a convolutional neural network to compare image patches](https://github.com/jzbontar/mc-cnn), J., Zbontar,  and Y. LeCun. JMLR 2016.
3. [Efficient deep learning for stereo matching](https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf), W. Luo, A. G. Schwing, R. Urtasun. CVPR 2016.
4. [Learning a multi-view stereo machine](https://arxiv.org/abs/1708.05375), A. Kar, C. Häne, J. Malik. NIPS 2017.
5. [Learned multi-patch similarity](https://arxiv.org/abs/1703.08836), W. Hartmann, S. Galliani, M. Havlena, L. V. Gool, K. Schindler.I CCV 2017.
6. [Surfacenet: An end-to-end 3d neural network for multiview stereopsis](https://github.com/mjiUST/SurfaceNet), Ji, M., Gall, J., Zheng, H., Liu, Y., Fang, L. ICCV2017.
7. [DeepMVS: Learning Multi-View Stereopsis](https://github.com/phuang17/DeepMVS), Huang, P. and Matzen, K. and Kopf, J. and Ahuja, N. and Huang, J. CVPR 2018.
8. [RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials](https://avg.is.tuebingen.mpg.de/publications/paschalidou2018cvpr), D. Paschalidou and A. O. Ulusoy and C. Schmitt and L. Gool and A. Geiger. CVPR 2018.
9. [MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/abs/1804.02505), Y. Yao, Z. Luo, S. Li, T. Fang, L. Quan. ECCV 2018.
10. [Learning Unsupervised Multi-View Stereopsis via Robust Photometric Consistency](https://tejaskhot.github.io/unsup_mvs/), T. Khot, S. Agrawal, S. Tulsiani, C. Mertz, S. Lucey, M. Hebert. 2019.
11. [DPSNET: END-TO-END DEEP PLANE SWEEP STEREO](https://openreview.net/pdf?id=ryeYHi0ctQ), Sunghoon Im, Hae-Gon Jeon, Stephen Lin, In So Kweon. 2019.
12. [Point-based Multi-view Stereo Network](http://hansf.me/projects/PMVSNet/), Rui Chen, Songfang Han, Jing Xu, Hao Su. ICCV 2019.

#### Multiple View Mesh Texturing（多视图网格纹理）

1. [Seamless image-based texture atlases using multi-band blending](http://imagine.enpc.fr/publications/papers/ICPR08a.pdf). C. Allène,  J-P. Pons and R. Keriven. ICPR 2008.
2. [Let There Be Color! - Large-Scale Texturing of 3D Reconstructions](http://www.gcc.tu-darmstadt.de/home/proj/texrecon/). M. Waechter, N. Moehrle, M. Goesele. ECCV 2014

#### Texture Mapping(纹理贴图)

1. [3D Textured Model Encryption via 3D Lu Chaotic Mapping](https://arxiv.org/abs/1709.08364v1)

### Courses

- [Image Manipulation and Computational Photography](http://inst.eecs.berkeley.edu/~cs194-26/fa14/) - Alexei A. Efros (UC Berkeley)
- [Computational Photography](http://graphics.cs.cmu.edu/courses/15-463/2012_fall/463.html) - Alexei A. Efros (CMU)
- [Computational Photography](https://courses.engr.illinois.edu/cs498dh3/) - Derek Hoiem (UIUC)
- [Computational Photography](http://cs.brown.edu/courses/csci1290/) - James Hays (Brown University)
- [Digital & Computational Photography](http://stellar.mit.edu/S/course/6/sp12/6.815/) - Fredo Durand (MIT)
- [Computational Camera and Photography](http://ocw.mit.edu/courses/media-arts-and-sciences/mas-531-computational-camera-and-photography-fall-2009/) - Ramesh Raskar (MIT Media Lab)
- [Computational Photography](https://www.udacity.com/course/computational-photography--ud955) - Irfan Essa (Georgia Tech)
- [Courses in Graphics](http://graphics.stanford.edu/courses/) - Stanford University
- [Computational Photography](http://cs.nyu.edu/~fergus/teaching/comp_photo/index.html) - Rob Fergus (NYU)
- [Introduction to Visual Computing](http://www.cs.toronto.edu/~kyros/courses/320/) - Kyros Kutulakos (University of Toronto)
- [Computational Photography](http://www.cs.toronto.edu/~kyros/courses/2530/) - Kyros Kutulakos (University of Toronto)
- [Computer Vision for Visual Effects](https://www.ecse.rpi.edu/~rjradke/cvfxcourse.html) - Rich Radke (Rensselaer Polytechnic Institute)
- [Introduction to Image Processing](https://www.ecse.rpi.edu/~rjradke/improccourse.html) - Rich Radke (Rensselaer Polytechnic Institute)

### Software

- [MATLAB Functions for Multiple View Geometry](http://www.robots.ox.ac.uk/~vgg/hzbook/code/)
- [Peter Kovesi's Matlab Functions for Computer Vision and Image Analysis](http://staffhome.ecm.uwa.edu.au/~00011811/Research/MatlabFns/index.html)
- [OpenGV ](http://laurentkneip.github.io/opengv/)- geometric computer vision algorithms
- [MinimalSolvers](http://cmp.felk.cvut.cz/mini/) - Minimal problems solver
- [Multi-View Environment](http://www.gcc.tu-darmstadt.de/home/proj/mve/)
- [Visual SFM](http://ccwu.me/vsfm/)
- [Bundler SFM](http://www.cs.cornell.edu/~snavely/bundler/)
- [openMVG: open Multiple View Geometry](http://imagine.enpc.fr/~moulonp/openMVG/) - Multiple View Geometry; Structure from Motion library & softwares
- [Patch-based Multi-view Stereo V2](http://www.di.ens.fr/pmvs/)
- [Clustering Views for Multi-view Stereo](http://www.di.ens.fr/cmvs/)
- [Floating Scale Surface Reconstruction](http://www.gris.informatik.tu-darmstadt.de/projects/floating-scale-surface-recon/)
- [Large-Scale Texturing of 3D Reconstructions](http://www.gcc.tu-darmstadt.de/home/proj/texrecon/)
- [Multi-View Stereo Reconstruction](http://vision.middlebury.edu/mview/)

### Project&code

| Project                                                      | Language             | License                                                      |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------------------------ |
| [Colmap](https://github.com/colmap/colmap)                   | C++ CUDA             | BSD 3-clause license - Permissive (Can use CGAL -> GNU General Public License - contamination) |
| [GPUIma + fusibile](https://github.com/kysucix)              | C++ CUDA             | GNU General Public License - contamination                   |
| [HPMVS](https://github.com/alexlocher/hpmvs)                 | C++                  | GNU General Public License - contamination                   |
| [MICMAC](http://logiciels.ign.fr/?Micmac)                    | C++                  | CeCILL-B                                                     |
| [MVE](https://github.com/simonfuhrmann/mve)                  | C++                  | BSD 3-Clause license + parts under the GPL 3 license         |
| [OpenMVS](https://github.com/cdcseacave/openMVS/)            | C++  (CUDA optional) | AGPL3                                                        |
| [PMVS](https://github.com/pmoulon/CMVS-PMVS)                 | C++ CUDA             | GNU General Public License - contamination                   |
| [SMVS Shading-aware Multi-view Stereo](https://github.com/flanggut/smvs) | C++                  | BSD-3-Clause license                                         |

## 深度相机三维重建

> 主要是基于Kinect这类深度相机进行三维重建，包括KinectFusion、Kintinuous，ElasticFusion、InfiniTAM，BundleFusion

## 基于线条/面的三维重建

1. Surface Reconstruction from 3D Line Segments

## Planar Reconstruction

> 参考：https://github.com/BigTeacher-777/Awesome-Planar-Reconstruction

### Papers

-  **[PlaneRCNN]** PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image [[CVPR2019(Oral)](https://arxiv.org/pdf/1812.04072.pdf)][[Pytorch](https://github.com/NVlabs/planercnn)]
-  **[PlanarReconstruction]** Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding [[CVPR2019](https://arxiv.org/pdf/1902.09777.pdf)][[Pytorch](https://github.com/svip-lab/PlanarReconstruction)]
-  **[Planerecover]** Recovering 3D Planes from a Single Image via Convolutional Neural Networks [[ECCV2018](https://faculty.ist.psu.edu/zzhou/paper/ECCV18-plane.pdf)][[Tensorflow](https://github.com/fuy34/planerecover)]
-  **[PlaneNet]** PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image [[CVPR2018](https://arxiv.org/pdf/1804.06278.pdf)][[Tensorflow](https://github.com/art-programmer/PlaneNet)]

### Datasets

-  ScanNet Dataset (PlaneNet) [[Train](https://drive.google.com/file/d/1NyDrgI02ao18WmXyepgVkWGqtM3YS3_4/view)][[Test](https://drive.google.com/file/d/1kfd-kreGQQLSRNF66t447R9WgDqsTh-3/view)]
-  ScanNet Dataset (PlaneRCNN)[[Link](https://www.dropbox.com/s/u2wl4ji700u4shq/ScanNet_planes.zip?dl=0)]
-  NYU Depth Dataset [[Link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]
-  SYNTHIA Dataset [[Link](https://psu.app.box.com/s/6ds04a85xqf3ud3uljjxnedmux169ebf)]

## 3D人脸重建

1、[Nonlinear 3D Face Morphable Model](https://arxiv.org/abs/1804.03786)

2、[On Learning 3D Face Morphable Model from In-the-wild Images](https://arxiv.org/abs/1808.09560)

3、[Cascaded Regressor based 3D Face Reconstruction from a Single Arbitrary View Image](https://arxiv.org/abs/1509.06161v1)

4、[JointFace Alignment and 3D Face Reconstruction](http://xueshu.baidu.com/usercenter/paper/show?paperid=4dcdab9e3941e82563f82009a2ef3125&site=xueshu_se)

5、[Photo-Realistic Facial Details Synthesis From Single Image](https://arxiv.org/pdf/1903.10873.pdf)

6、[FML: Face Model Learning from Videos](https://arxiv.org/pdf/1812.07603.pdf)

7、[Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric](https://arxiv.org/abs/1703.07834)

8、[Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://arxiv.org/pdf/1803.07835.pdf)

9、[Joint 3D Face Reconstruction and Dense Face Alignment from A Single Image with 2D-Assisted Self-Supervised Learning](https://arxiv.org/abs/1903.09359)

10、[Face Alignment Across Large Poses: A 3D Solution]()

## 纹理/材料分析与合成

1. Texture Synthesis Using Convolutional Neural Networks (2015)[[Paper\]](https://arxiv.org/pdf/1505.07376.pdf)
2. Two-Shot SVBRDF Capture for Stationary Materials (SIGGRAPH 2015) [[Paper\]](https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/)
3. Reflectance Modeling by Neural Texture Synthesis (2016) [[Paper\]](https://mediatech.aalto.fi/publications/graphics/NeuralSVBRDF/)
4. Modeling Surface Appearance from a Single Photograph using Self-augmented Convolutional Neural Networks (2017)[[Paper\]](http://msraig.info/~sanet/sanet.htm)
5. High-Resolution Multi-Scale Neural Texture Synthesis (2017) [[Paper\]](https://wxs.ca/research/multiscale-neural-synthesis/)
6. Reflectance and Natural Illumination from Single Material Specular Objects Using Deep Learning (2017) [[Paper\]](https://homes.cs.washington.edu/~krematas/Publications/reflectance-natural-illumination.pdf)
7. Joint Material and Illumination Estimation from Photo Sets in the Wild (2017) [[Paper\]](https://arxiv.org/pdf/1710.08313.pdf)
8. TextureGAN: Controlling Deep Image Synthesis with Texture Patches (2018 CVPR) [[Paper\]](https://arxiv.org/pdf/1706.02823.pdf)
9. Gaussian Material Synthesis (2018 SIGGRAPH) [[Paper\]](https://users.cg.tuwien.ac.at/zsolnai/gfx/gaussian-material-synthesis/)
10. Non-stationary Texture Synthesis by Adversarial Expansion (2018 SIGGRAPH) [[Paper\]](http://vcc.szu.edu.cn/research/2018/TexSyn)
11. Synthesized Texture Quality Assessment via Multi-scale Spatial and Statistical Texture Attributes of Image and Gradient Magnitude Coefficients (2018 CVPR) [[Paper\]](https://arxiv.org/pdf/1804.08020.pdf)
12. LIME: Live Intrinsic Material Estimation (2018 CVPR) [[Paper\]](https://gvv.mpi-inf.mpg.de/projects/LIME/)
13. Learning Material-Aware Local Descriptors for 3D Shapes (2018) [[Paper\]](http://www.vovakim.com/papers/18_3DV_ShapeMatFeat.pdf)

## **场景合成/重建**

1. Make It Home: Automatic Optimization of Furniture Arrangement (2011, SIGGRAPH) [[Paper\]](http://people.sutd.edu.sg/~saikit/projects/furniture/index.html)
2. Interactive Furniture Layout Using Interior Design Guidelines (2011) [[Paper\]](http://graphics.stanford.edu/~pmerrell/furnitureLayout.htm)
3. Synthesizing Open Worlds with Constraints using Locally Annealed Reversible Jump MCMC (2012) [[Paper\]](http://graphics.stanford.edu/~lfyg/owl.pdf)
4. Example-based Synthesis of 3D Object Arrangements (2012 SIGGRAPH Asia) [[Paper\]](http://graphics.stanford.edu/projects/scenesynth/)
5. Sketch2Scene: Sketch-based Co-retrieval and Co-placement of 3D Models (2013) [[Paper\]](http://sweb.cityu.edu.hk/hongbofu/projects/sketch2scene_sig13/#.WWWge__ysb0)
6. Action-Driven 3D Indoor Scene Evolution (2016) [[Paper\]](https://www.cs.sfu.ca/~haoz/pubs/ma_siga16_action.pdf)
7. The Clutterpalette: An Interactive Tool for Detailing Indoor Scenes (2015) [[Paper\]](https://www.cs.umb.edu/~craigyu/papers/clutterpalette.pdf)
8. Relationship Templates for Creating Scene Variations (2016) [[Paper\]](http://geometry.cs.ucl.ac.uk/projects/2016/relationship-templates/)
9. IM2CAD (2017) [[Paper\]](http://homes.cs.washington.edu/~izadinia/im2cad.html)
10. Predicting Complete 3D Models of Indoor Scenes (2017) [[Paper\]](https://arxiv.org/pdf/1504.02437.pdf)
11. Complete 3D Scene Parsing from Single RGBD Image (2017) [[Paper\]](https://arxiv.org/pdf/1710.09490.pdf)
12. Adaptive Synthesis of Indoor Scenes via Activity-Associated Object Relation Graphs (2017 SIGGRAPH Asia) [[Paper\]](http://arts.buaa.edu.cn/projects/sa17/)
13. Automated Interior Design Using a Genetic Algorithm (2017) [[Paper\]](https://publik.tuwien.ac.at/files/publik_262718.pdf)
14. SceneSuggest: Context-driven 3D Scene Design (2017) [[Paper\]](https://arxiv.org/pdf/1703.00061.pdf)
15. A fully end-to-end deep learning approach for real-time simultaneous 3D reconstruction and material recognition (2017)[[Paper\]](https://arxiv.org/pdf/1703.04699v1.pdf)
16. Human-centric Indoor Scene Synthesis Using Stochastic Grammar (2018, CVPR)[[Paper\]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis.pdf) [[Supplementary\]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis_supplementary.pdf) [[Code\]](https://github.com/SiyuanQi/human-centric-scene-synthesis)
17. FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans (2018) [[Paper\]](https://arxiv.org/pdf/1804.00090.pdf) [[Code\]](http://art-programmer.github.io/floornet.html)
18. ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans (2018) [[Paper\]](https://arxiv.org/pdf/1712.10215.pdf)
19. Configurable 3D Scene Synthesis and 2D Image Rendering with Per-Pixel Ground Truth using Stochastic Grammars (2018) [[Paper\]](https://arxiv.org/pdf/1704.00112.pdf)
20. Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image (ECCV 2018) [[Paper\]](http://siyuanhuang.com/holistic_parsing/main.html)
21. Automatic 3D Indoor Scene Modeling from Single Panorama (2018 CVPR) [[Paper\]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Automatic_3D_Indoor_CVPR_2018_paper.pdf)
22. Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding (2019 CVPR) [[Paper\]](https://arxiv.org/pdf/1902.09777.pdf) [[Code\]](https://github.com/svip-lab/PlanarReconstruction)
23. 3D Scene Reconstruction with Multi-layer Depth and Epipolar Transformers (ICCV 2019) [[Paper\]](https://research.dshin.org/iccv19/multi-layer-depth/)

# 姿态估计

> 主要基于RGB、RGB-D以及点云数据，估计物体和相机/基准坐标系的关系。
>
> 主要有整体方式、霍夫投票方式、Keypoint-based方式和Dense Correspondence方式

## 实现方式不同

### 整体方式

> 整体方法直接估计给定图像中物体的三维位置和方向。经典的基于模板的方法构造刚性模板并扫描图像以计算最佳匹配姿态。这种手工制作的模板对集群场景不太可靠。最近，人们提出了一些基于深度神经网络的方法来直接回归相机或物体的6D姿态。然而，旋转空间的非线性使得数据驱动的DNN难以学习和推广。

1. Discriminative mixture-of-templates for viewpoint classification  
2. Gradient response maps for realtime detection of textureless objects.  
3. Comparing images using the hausdorff distance  
4. Densefusion: 6d object pose estimation by iterative dense fusion  
5. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes.  
6. Viewpoints and keypoints  
7. Implicit 3d orientation learning for 6d object detection from rgb images.
8. Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views.  

### Keypoint-based方式

> 目前基于关键点的方法首先检测图像中物体的二维关键点，然后利用PnP算法估计6D姿态

1. Surf: Speeded up robust features.  
2. Object recognition from local scaleinvariant features  
3. 3d object modeling and recognition using local affine-invariant image descriptors and multi-view spatial constraints.  
4. Stacked hourglass networks for human pose estimation  
5. Making deep heatmaps robust to partial occlusions for 3d object pose estimation.   
6. Bb8: A scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth 
7. Real-time seamless single shot 6d object pose prediction.  
8. Discovery of latent 3d keypoints via end-toend geometric reasoning.  
9. Pvnet: Pixel-wise voting network for 6dof pose estimation.  

### Dense Correspondence and 霍夫投票方式

1. independent object class detection using 3d feature maps.  
2. Depthencoded hough voting for joint object detection and shape recovery.   
3. aware object detection and pose estimation.  
4. Learning 6d object pose estimation using 3d object coordinates.  
5. Global hypothesis generation for 6d object pose estimation.  
6. Deep learning of local rgb-d patches for 3d object detection and 6d pose estimation.   
7. Cdpn: Coordinates-based disentangled pose network for real-time rgb-based 6-dof object
   pose estimation.  
8. Pix2pose: Pixel-wise coordinate regression of objects for 6d pose estimation.  
9. Normalized object coordinate space for categorylevel 6d object pose and size estimation.  
10. Recovering 6d object pose and predicting next-bestview in the crowd.
11. 

## 数据格式不同

### 基于点云方式

1. PointFusion
2. Frustum PointNets
3. VoteNet

### 基于RGB方式



### 基于RGB-D方式



# 无序抓取



# 视觉伺服

### 综述

1. [Survey on Visual Servoing for Manipulation](http://xueshu.baidu.com/usercenter/paper/show?paperid=ec62933c2dc0edd1c24ba39d2e28d9f4&site=xueshu_se)
2. [A review on vision-based control of robot manipulators  ]()

### 基于图像（2D）

1. [Kinematics-based incremental visual servo for robotic capture of non-cooperative target](https://www.baidu.com/link?url=bPpedLCJ7n1GR2u8fDlVnJZ0HZRL15k0OlRXzPjPYzpPzELq79znoqkbnO7CG9CDTh2fmC-C-wbgc1xjXfrjDacDBkiztLvViFj1FGnbfVIlu_ur8gAKyBPywPHgnxfxasqSKF5NapFdvaMSZ7Z4A6YKY2aqAY1DQf2LdXbyXPaIsbOov2PiBj7Twb73KrAIkCfcpu2ZvODrzVhG0RGM9K&wd=&eqid=f29332170003bec5000000055e59ff3f)
2. [Position and attitude control of Eye-In-Hand System by visual servoing using Binocular Visual Space](http://xueshu.baidu.com/usercenter/paper/show?paperid=1788c32af8c3c191062de46f599fcf55&site=xueshu_se)
3. [Progressive 3D Reconstruction of Unknown Objects using one Eye-in-Hand Camera]()
4. 

### 基于位置（3D）

1. [A hybrid positioning method for eye-in-hand industrial robot by using 3D reconstruction and IBVS](http://xueshu.baidu.com/usercenter/paper/show?paperid=233c5ff9ae58ec24cc2d9572af577412&site=xueshu_se)

### 基于图像和位置（2.5D）

1.  [Moment-Based 2.5-D Visual Servoing for Textureless Planar Part Grasping](https://ieeexplore.ieee.org/document/8584462)

# 深度图补全&修复

### 深度图补全

1、[HMS-Net: Hierarchical Multi-scale Sparsity-invariant Network for Sparse Depth Completion](https://arxiv.org/abs/1808.08685)

2、[Sparse and noisy LiDAR completion with RGB guidance and uncertainty](https://arxiv.org/abs/1902.05356)

3、[3D LiDAR and Stereo Fusion using Stereo Matching Network with Conditional Cost Volume Normalization](https://arxiv.org/pdf/1904.02917.pdf)

4、[Deep RGB-D Canonical Correlation Analysis For Sparse Depth Completion](https://arxiv.org/pdf/1906.08967.pdf)

5、[Confidence Propagation through CNNs for Guided Sparse Depth Regression](https://arxiv.org/abs/1811.01791)

6、[Learning Guided Convolutional Network for Depth Completion](https://arxiv.org/pdf/1908.01238.pdf)

7、[DFineNet: Ego-Motion Estimation and Depth Refinement from Sparse, Noisy Depth Input with RGB Guidance](http://arxiv.org/abs/1903.06397)

8、[PLIN: A Network for Pseudo-LiDAR Point Cloud Interpolation](https://arxiv.org/abs/1909.07137)

9、[Depth Completion from Sparse LiDAR Data with Depth-Normal Constraints](https://arxiv.org/pdf/1910.06727v1.pdf)

### 深度图修复

#### 基于Kinect的深度图修复

> 主要有高斯滤波、双边滤波、插值、多尺度各向异性扩散、连通域分析

#### paper

1. 基于Kinect的深度图像修复技术研究
2. 深度图像自适应邻域修复方法

# 软件&系统开发

## Github链接

1. https://github.com/DovAmir/awesome-design-patterns

