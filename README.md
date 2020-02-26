# awesome-3D-vision

3D computer vision incuding SLAM，VSALM，Deep Learning，Structured light，Stereo，Three-dimensional reconstruction，Computer vision，Machine Learning and so on

# 介绍

> 公众号：[3D视觉工坊](https://mp.weixin.qq.com/s?__biz=MzU1MjY4MTA1MQ==&mid=2247484684&idx=1&sn=e812540aee03a4fc54e44d5555ccb843&chksm=fbff2e38cc88a72e180f0f6b0f7b906dd616e7d71fffb9205d529f1238e8ef0f0c5554c27dd7&token=691734513&lang=zh_CN#rd)
>
> 主要关注：3D视觉算法、SLAM、vSLAM、计算机视觉、深度学习、自动驾驶、图像处理以及技术干货分享
>
> 运营者和嘉宾介绍：运营者来自国内一线大厂的算法工程师，深研3D视觉、深度学习、图像处理、自动驾驶、vSLAM等领域，特邀嘉宾包括国内外知名高校的博士硕士，旷视、商汤、百度、阿里等就职的算法大佬，欢迎一起交流学习

# 3D视觉实战代码汇总

1. https://github.com/sunglok/3dv_tutorial(涉及SLAM、多视图几何代码示例)

# SLAM

## Books

- [视觉SLAM十四讲]() 高翔
- [机器人学中的状态估计]()
- [概率机器人]()
- [Simultaneous Localization and Mapping for Mobile Robots: Introduction and Methods](http://www.igi-global.com/book/simultaneous-localization-mapping-mobile-robots/66380) by Juan-Antonio Fernández-Madrigal and José Luis Blanco Claraco, 2012
- [Simultaneous Localization and Mapping: Exactly Sparse Information Filters ](http://www.worldscientific.com/worldscibooks/10.1142/8145/)by Zhan Wang, Shoudong Huang and Gamini Dissanayake, 2011
- [An Invitation to 3-D Vision -- from Images to Geometric Models](http://vision.ucla.edu/MASKS/) by Yi Ma, Stefano Soatto, Jana Kosecka and Shankar S. Sastry, 2005
- [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004
- [Numerical Optimization](http://home.agh.edu.pl/~pba/pdfdoc/Numerical_Optimization.pdf) by Jorge Nocedal and Stephen J. Wright, 1999

## Courses&&Lectures

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

## Code

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

## 点云识别

1、[3D ShapeNets: A Deep Representation for Volumetric Shapes](http://3dvision.princeton.edu/projects/2014/3DShapeNets/paper.pdf)

2、[PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding](https://arxiv.org/abs/1812.02713)

3、[Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data](https://arxiv.org/pdf/1908.04616.pdf)

## 点云匹配

1、[An ICP variant using a point-to-line metric](https://authors.library.caltech.edu/18274/1/Censi2008p84782008_Ieee_International_Conference_On_Robotics_And_Automation_Vols_1-9.pdf)

2、[Generalized-ICP](https://www.researchgate.net/publication/221344436_Generalized-ICP)

3、[Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration](https://www.researchgate.net/publication/228571031_Linear_Least-Squares_Optimization_for_Point-to-Plane_ICP_Surface_Registration)

4、[Metric-Based Iterative Closest Point Scan Matching for Sensor Displacement Estimation](http://webdiis.unizar.es/~jminguez/MbICP_TRO.pdf)

5、[NICP: Dense Normal Based Point Cloud Registration](http://jacoposerafin.com/wp-content/uploads/serafin15iros.pdf)

## 点云分割

1、[SceneEncoder: Scene-Aware Semantic Segmentation of Point Clouds with A Learnable Scene Descriptor](https://arxiv.org/abs/2001.09087v1)

2、[From Planes to Corners: Multi-Purpose Primitive Detection in Unorganized 3D Point Clouds](https://arxiv.org/abs/2001.07360?context=cs.RO)

3、[Learning and Memorizing Representative Prototypes for 3D Point Cloud Semantic and Instance Segmentation](http://arxiv.org/abs/2001.01349)

4、[JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds](https://arxiv.org/abs/1912.09654v1)

5、[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593v2)

6、[PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413v1)

# 三维重建

## 结构光

> 先扫描结构光
>
> 面阵结构光

参考链接：[https://zhuanlan.zhihu.com/p/29971801](https://zhuanlan.zhihu.com/p/29971801)

### Code

1. [https://github.com/jakobwilm/slstudio](https://github.com/jakobwilm/slstudio)
2. [https://github.com/phreax/structured_light](https://github.com/phreax/structured_light)
3. [https://github.com/nikolaseu/neuvision](https://github.com/nikolaseu/neuvision)
4. [https://github.com/pranavkantgaur/3dscan](https://github.com/pranavkantgaur/3dscan)

### Lectures

1. [**Build Your Own 3D Scanner: Optical Triangulation for Beginners**](https://link.zhihu.com/?target=http%3A//mesh.brown.edu/byo3d/)
2. [https://github.com/nikolaseu/thesis](https://github.com/nikolaseu/thesis)

### Video

1. [**CS6320 3D Computer Vision**, Spring 2015](http://www.sci.utah.edu/~gerig/CS6320-S2015/CS6320_3D_Computer_Vision.html)

### Papers

1. [Structured-light 3D surface imaging: a tutorial](https://link.zhihu.com/?target=http%3A//www.rtbasics.com/Downloads/IEEE_structured_light.pdf)
2. [High-speed 3D image acquisition using coded structured light projection](https://www.researchgate.net/publication/224296439_High-speed_3D_image_acquisition_using_coded_structured_light_projection)
3. [Accurate 3D measurement using a Structured Light System](https://www.researchgate.net/publication/222500455_Accurate_3D_measurement_using_a_Structured_Light_System)
4. [Structured light stereoscopic imaging with dynamic pseudo-random patterns  ](https://static.aminer.org/pdf/PDF/000/311/975/a_high_precision_d_object_reconstruction_method_using_a_color.pdf)
5. [Robust one-shot 3D scanning using loopy belief propagation  ](https://www.researchgate.net/publication/224165371_Robust_one-shot_3D_scanning_using_loopy_belief_propagation)
6. [Robust Segmentation and Decoding of a Grid Pattern for Structured Light](https://www.semanticscholar.org/paper/Robust-Segmentation-and-Decoding-of-a-Grid-Pattern-Pag%C3%A8s-Salvi/dcbdd608dcdf03b0d0eba662c68915dcfa90e5a5)
7. [Rapid shape acquisition using color structured light and multi-pass dynamic programming  ](http://ieeexplore.ieee.org/iel5/7966/22019/01024035.pdf?arnumber=1024035)
8. [Improved stripe matching for colour encoded structured light  ]()
9. [Absolute phase mapping for one-shot dense pattern projection  ](https://www.researchgate.net/profile/Joaquim_Salvi/publication/224165341_Absolute_phase_mapping_for_one-shot_dense_pattern_projection/links/56ffaee708ae650a64f805dd.pdf)
10. [3D digital stereophotogrammetry: a practical guide to facial image acquisition  ]()
11. [Method and apparatus for 3D imaging using light pattern having multiple sub-patterns  ]()
12. [High speed laser three-dimensional imager  ]()
13. [Three-dimensional dental imaging method and apparatus having a reflective member  ]()
14. [3D surface profile imaging method and apparatus using single spectral light condition  ]()
15. [Three-dimensional surface profile imaging method and apparatus using single spectral light condition]()
16. [High speed three dimensional imaging method  ]()
17. [A hand-held photometric stereo camera for 3-D modeling  ]()
18. [High-resolution, real-time 3D absolute coordinate measurement based on a phase-shifting method  ]()
19. [A fast three-step phase shifting algorithm  ]()

## 立体视觉

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

### 立体匹配

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



## SFM

> Incremental SfM
>
> Global SfM
>
> Hierarchical SfM
>
> Multi-Stage SfM
>
> Non Rigid SfM

#### Turtorial

1. [Open Source Structure-from-Motion](https://blog.kitware.com/open-source-structure-from-motion-at-cvpr-2015/). M. Leotta, S. Agarwal, F. Dellaert, P. Moulon, V. Rabaud. CVPR 2015 Tutorial [(material)](https://github.com/mleotta/cvpr2015-opensfm).
2. Large-scale 3D Reconstruction from Images](https://home.cse.ust.hk/~tshenaa/sub/ACCV2016/ACCV_2016_Tutorial.html). T. Shen, J. Wang, T.Fang, L. Quan. ACCV 2016 Tutorial.

#### Incremental SfM

1. [Photo Tourism: Exploring Photo Collections in 3D](http://phototour.cs.washington.edu/Photo_Tourism.pdf). N. Snavely, S. M. Seitz, and R. Szeliski. SIGGRAPH 2006.
2. [Towards linear-time incremental structure from motion](http://ccwu.me/vsfm/vsfm.pdf). C. Wu. 3DV 2013.
3. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf). Schöenberger, Frahm. CVPR 2016.

#### Global SfM

1. [Combining two-view constraints for motion estimation](http://www.umiacs.umd.edu/users/venu/cvpr01.pdf) V. M. Govindu. CVPR, 2001.
2. [Lie-algebraic averaging for globally consistent motion estimation](http://www.umiacs.umd.edu/users/venu/cvpr04final.pdf). V. M. Govindu. CVPR, 2004.
3. [Robust rotation and translation estimation in multiview reconstruction](http://imagine.enpc.fr/~monasse/Stereo/Projects/MartinecPajdla07.pdf). D. Martinec and T. Pajdla. CVPR, 2007.
4. [Non-sequential structure from motion](http://www.maths.lth.se/vision/publdb/reports/pdf/enqvist-kahl-etal-wovcnnc-11.pdf). O. Enqvist, F. Kahl, and C. Olsson. ICCV OMNIVIS Workshops 2011.
5. [Global motion estimation from point matches](https://web.math.princeton.edu/~amits/publications/sfm_3dimpvt12.pdf). M. Arie-Nachimson, S. Z. Kovalsky, I. KemelmacherShlizerman, A. Singer, and R. Basri. 3DIMPVT 2012.
6. [Global Fusion of Relative Motions for Robust, Accurate and Scalable Structure from Motion](https://hal-enpc.archives-ouvertes.fr/hal-00873504). P. Moulon, P. Monasse and R. Marlet. ICCV 2013.
7. [A Global Linear Method for Camera Pose Registration](http://www.cs.sfu.ca/~pingtan/Papers/iccv13_sfm.pdf). N. Jiang, Z. Cui, P. Tan. ICCV 2013.
8. [Global Structure-from-Motion by Similarity Averaging](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf). Z. Cui, P. Tan. ICCV 2015.
9. [Linear Global Translation Estimation from Feature Tracks](http://arxiv.org/abs/1503.01832) Z. Cui, N. Jiang, C. Tang, P. Tan, BMVC 2015.

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

#### 教程

1. [ToF技术是什么？和结构光技术又有何区别?](https://zhuanlan.zhihu.com/p/51218791)
2. [3D相机--TOF相机](https://zhuanlan.zhihu.com/p/85519428)

#### Paper

1. https://arxiv.org/pdf/1511.07212.pdf)

## Multi-view Stereo

> 多视角立体视觉（Multiple View Stereo，MVS）是对立体视觉的推广，能够在多个视角（从外向里）观察和获取景物的图像，并以此完成匹配和深度估计。某种意义上讲，SLAM/SFM其实和MVS是类似的，只是前者是摄像头运动，后者是多个摄像头视角。也可以说，前者可以在环境里面“穿行”，而后者更像在环境外“旁观”。
>
> 多视角立体视觉的pipelines如下：
>
> 1. 收集图像；
> 2. 针对每个图像计算相机参数；
> 3. 从图像集和相应的摄像机参数重建场景的3D几何图形；
> 4. 可选择地重建场景的形状和纹理颜色。

参考链接：[多视角立体视觉MVS简介](https://zhuanlan.zhihu.com/p/73748124)

### paper

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

10、[Face Alignment Across Large Poses: A 3D Solution](

# 深度图补全

1、[HMS-Net: Hierarchical Multi-scale Sparsity-invariant Network for Sparse Depth Completion](https://arxiv.org/abs/1808.08685)

2、[Sparse and noisy LiDAR completion with RGB guidance and uncertainty](https://arxiv.org/abs/1902.05356)

3、[3D LiDAR and Stereo Fusion using Stereo Matching Network with Conditional Cost Volume Normalization](https://arxiv.org/pdf/1904.02917.pdf)

4、[Deep RGB-D Canonical Correlation Analysis For Sparse Depth Completion](https://arxiv.org/pdf/1906.08967.pdf)

5、[Confidence Propagation through CNNs for Guided Sparse Depth Regression](https://arxiv.org/abs/1811.01791)

6、[Learning Guided Convolutional Network for Depth Completion](https://arxiv.org/pdf/1908.01238.pdf)

7、[DFineNet: Ego-Motion Estimation and Depth Refinement from Sparse, Noisy Depth Input with RGB Guidance](http://arxiv.org/abs/1903.06397)

8、[PLIN: A Network for Pseudo-LiDAR Point Cloud Interpolation](https://arxiv.org/abs/1909.07137)

9、[Depth Completion from Sparse LiDAR Data with Depth-Normal Constraints](https://arxiv.org/pdf/1910.06727v1.pdf)
