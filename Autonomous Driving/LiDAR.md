<font face="微软雅黑" size=5>
Lidar Point clound processing for Autonomous Driving
</font>

Thanks are given to：

&emsp;&emsp;@takeitallsource (Manfred Diaz, https://github.com/takeitallsource/awesome-autonomous-vehicles),https://zhuanlan.zhihu.com/p/27686577

&emsp;&emsp;@beedotkiran (B Ravi Kiran, https://github.com/beedotkiran/Lidar_For_AD_references)

&emsp;&emsp;@daviddao (David Dao, https://github.com/daviddao/deep-autonomous-driving-papers)

&emsp;&emsp;@timzhang642 (Yuxuan (Tim) Zhang, https://github.com/timzhang642/3D-Machine-Learning, indoor Scene Understanding)

&emsp;&emsp;@hoya012 (Lee hoseong, https://github.com/hoya012/deep_learning_object_detection#2014 )

&emsp;&emsp;@Ewenwan（WAN Youwen, https://github.com/Ewenwan/MVision）

&emsp;&emsp;@ETH Robotics and Perception Group (http://rpg.ifi.uzh.ch/), （https://github.com/uzh-rpg/event-based_vision_resources)

&emsp;&emsp;@JudasDie (ZP ZHANG, https://github.com/JudasDie/deeplearning.ai)

&emsp;&emsp;@Handong1587 (https://github.com/handong1587/handong1587.github.io)

&emsp;&emsp;@zziz (https://github.com/zziz/pwc)

&emsp;&emsp;@unrealcv (https://github.com/unrealcv/synthetic-computer-vision)

&emsp;&emsp;@Zeyu, https://github.com/nan0755

&emsp;&emsp;@amusi, https://github.com/amusi

&emsp;&emsp;GITHUB TOPIC: LiDAR, https://github.com/topics/lidar


## Section 1: Algorithms

### Lidar Datasets and Simulators

   &emsp;&emsp;*Udacity based simulator 
    [__[Project](http://wangyangevan.weebly.com/lidar-simulation.html)__]
    [__[Github](https://github.com/EvanWY/USelfDrivingSimulator)__]

   &emsp;&emsp;*Udacity Driving Dataset 
    [__[Github](https://github.com/udacity/self-driving-car/tree/master/datasets)__]

   &emsp;&emsp;*Microsoft AirSim 
    [__[Doc](https://microsoft.github.io/AirSim/docs/use_precompiled/)__]
    [__[pdf](https://arxiv.org/pdf/1705.05065.pdf?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)__]
    [__[Github](https://github.com/Microsoft/AirSim)__]

   &emsp;&emsp;*MVIG, Shanghai Jiao Tong University and SCSC Lab, Xiamen University, DBNet 
    [__[pdf](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1645.pdf)__]
    [__[Github](https://github.com/driving-behavior/DBNet)__]
    [__[website](http://www.dbehavior.net/)__]

   &emsp;&emsp;*OXFORD ROBOTCAR DATASET [__[website](https://robotcar-dataset.robots.ox.ac.uk/)__]
    [__[SDK](https://github.com/ori-drs/robotcar-dataset-sdk)__]

   &emsp;&emsp;*Ford Campus Vision and Lidar Data Set (FCVL) 
    [[website](http://robots.engin.umich.edu/SoftwareData/Ford)]

   &emsp;&emsp;*University of Michigan North Campus Long-Term Vision and LIDAR Dataset (NCLT) 
    [[website](http://robots.engin.umich.edu/SoftwareData/NCLT)]

   &emsp;&emsp;*SEMANTIC3D.NET: A new large-scale point cloud classification benchmark 
    [[website](http://www.semantic3d.net/)]

   &emsp;&emsp;*KAIST URBAN DATA SET 
    [[webiste](http://irap.kaist.ac.kr/dataset/)]

   &emsp;&emsp;*The ApolloScape Dataset for Autonomous Driving 
    [[pdf](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Huang_The_ApolloScape_Dataset_CVPR_2018_paper.pdf)]
    [[website](http://apolloscape.auto/)]

   &emsp;&emsp;*DeepGTAV 
    [[Github](https://github.com/aitorzip/DeepGTAV)]

   &emsp;&emsp;*MINES ParisTech, PSL Research University, Centre for Robotics, Paris-Lille-3D
    [[pdf](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w40/Roynard_Paris-Lille-3D_A_Point_CVPR_2018_paper.pdf)]
    [[website](http://npm3d.fr/paris-lille-3d)]



### Clustering/Segmentation (road/ground extraction, plane extraction)

   &emsp;&emsp;[2016] Fast semantic segmentation of 3d point clounds with strongly varying density [[pdf](https://www.ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/timo-jan-isprs2016.pdf)]

   &emsp;&emsp;[2017] Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications [[Github](https://github.com/VincentCheungM/Run_based_segmentation)]
   
   &emsp;&emsp;[2017] Time-series LIDAR Data Superimposition for Autonomous Driving [[pdf](http://lab.cntl.kyutech.ac.jp/~nishida/paper/2016/ThBT3.3.pdf)]
   
   &emsp;&emsp;[2017] An Improved RANSAC for 3D Point Cloud Plane Segmentation Based on Normal Distribution Transformation Cells [[pdf]()]
   
   &emsp;&emsp;[2017] A Fast Ground Segmentation Method for 3D Point Cloud [[pdf](http://jips-k.org/file/down?pn=463)]
   
   &emsp;&emsp;[2017] Ground Estimation and Point Cloud Segmentation using SpatioTemporal Conditional Random Field [[pdf](https://hal.inria.fr/hal-01579095/document)]
   
   &emsp;&emsp;[2017] Efficient Online Segmentation for Sparse 3D Laser Scans [[pdf](http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16pfg.pdf)] [[git](https://github.com/PRBonn/depth_clustering)]
   
