<font face="微软雅黑" size=5>
LiDAR Point cloud processing for Autonomous Driving
</font>

Thanks are given to:

&emsp;&emsp;@takeitallsource (Manfred Diaz, https://github.com/takeitallsource/awesome-autonomous-vehicles),https://zhuanlan.zhihu.com/p/27686577

&emsp;&emsp;@beedotkiran (B Ravi Kiran, https://github.com/beedotkiran/Lidar_For_AD_references)

&emsp;&emsp;@daviddao (David Dao, https://github.com/daviddao/deep-autonomous-driving-papers)

&emsp;&emsp;@timzhang642 (Yuxuan (Tim) Zhang, https://github.com/timzhang642/3D-Machine-Learning, indoor Scene Understanding)

&emsp;&emsp;@hoya012 (Lee hoseong, https://github.com/hoya012/deep_learning_object_detection#2014 )

&emsp;&emsp;@Ewenwan(WAN Youwen, https://github.com/Ewenwan/MVision)

&emsp;&emsp;@ETH Robotics and Perception Group (http://rpg.ifi.uzh.ch/), (https://github.com/uzh-rpg/event-based_vision_resources)

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
   
&emsp;&emsp;[2017] Efficient Online Segmentation for Sparse 3D Laser Scans [[pdf](http://www.ipb.uni-bonn.de/pdfs/bogoslavskyi16pfg.pdf)] [[Github](https://github.com/PRBonn/depth_clustering)]
   
&emsp;&emsp;[2018] Real-Time Road Segmentation Using LiDAR Data Processing on an FPGA [[pdf](https://arxiv.org/pdf/1711.02757.pdf)]
   
&emsp;&emsp;[2018] CNN for Very Fast Ground Segmentation in Velodyne LiDAR Data [[pdf](https://arxiv.org/pdf/1709.02128.pdf)] [[Github](https://github.com/nan0755/cnn_pcl_seg)]

&emsp;&emsp;[2018] LiSeg: Lightweight Road-object Semantic Segmentation In 3D LiDAR Scans For Autonomous Driving [[pdf](https://ieeexplore.ieee.org/document/8500701)]
   
&emsp;&emsp;[2018] PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation [[pdf](https://arxiv.org/abs/1807.00652)][[Github](https://github.com/MVIG-SJTU/pointSIFT)]
   
&emsp;&emsp;[2018] Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation ][pdf](https://arxiv.org/pdf/1804.09915.pdf)]
   
&emsp;&emsp;[2018] Semantic Segmentation of 3D LiDAR Data in Dynamic Scene Using Semi-supervised Learning [[pdf](https://arxiv.org/pdf/1809.00426.pdf)]
   
&emsp;&emsp;[2018] RoadNet-v2: A 10 ms Road Segmentation Using Spatial Sequence Layer [[pdf](https://arxiv.org/abs/1808.04450)]
   
&emsp;&emsp;[2018] PointSeg: Real-Time Semantic Segmentation Based on 3D LiDAR Point Cloud [[pdf](https://arxiv.org/abs/1807.06288])[[Github](https://github.com/ywangeq/PointSeg)]
   
   
### Registration and Localization

&emsp;&emsp;[2011] Automatic Merging of Lidar Point-Clouds Using Data from Low-Cost GPS/IMU Systems [[pdf](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1081&context=ece_facpub)]

&emsp;&emsp;[2016] Point Clouds Registration with Probabilistic Data Association [[pdf](https://github.com/ethz-asl/robust_point_cloud_registration)][[Github](https://github.com/ethz-asl/robust_point_cloud_registration)]

&emsp;&emsp;[2017] Robust LIDAR Localization using Multiresolution Gaussian Mixture Maps for Autonomous Driving [[pdf](https://pdfs.semanticscholar.org/7292/1fc6b181cf75790664e482963d982ec9ac48.pdf)]

&emsp;&emsp;[2017] 3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Elbaz_3D_Point_Cloud_CVPR_2017_paper.pdf)] [[Github](https://github.com/gilbaz/LORAX)]

&emsp;&emsp;[2018] Integrating Deep Semantic Segmentation Into 3-D Point Cloud Registration [[pdf](http://eprints.lincoln.ac.uk/32390/1/integrating-semantic-knowledge.pdf)]

&emsp;&emsp;[2018] Ground-Edge-Based LIDAR Localization Without a Reflectivity Calibration for Autonomous Driving [[pdf](Ground-Edge-Based LIDAR Localization Without a Reflectivity Calibration for Autonomous Driving)]

&emsp;&emsp;[2018] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] [[Github](https://github.com/yewzijian/3DFeatNet)]


### Feature Extraction

&emsp;&emsp;[2007] A Fast RANSAC–Based Registration Algorithm for Accurate Localization in Unknown Environments using LIDAR Measurements [[pdf](http://vision.ucla.edu/papers/fontanelliRS07.pdf)]

&emsp;&emsp;[2008] Fast Feature Detection and Stochastic Parameter Estimation of Road Shape using Multiple LIDAR [[pdf](https://www.ri.cmu.edu/pub_files/2008/9/peterson_kevin_2008_1.pdf)]

&emsp;&emsp;[2010] A Fast and Accurate Plane Detection Algorithm for Large Noisy Point Clouds Using Filtered Normals and Voxel Growing [[pdf](https://hal-mines-paristech.archives-ouvertes.fr/hal-01097361/document)]

&emsp;&emsp;[2012] Online detection of planes in 2D lidar [[pdf](https://pdfs.semanticscholar.org/6857/b602dd702664c20febd41dc984451fd97bb3.pdf)]

&emsp;&emsp;[2013] Finding Planes in LiDAR Point Clouds for Real-Time Registration [[pdf](http://ilab.usc.edu/publications/doc/Grant_etal13iros.pdf)]

&emsp;&emsp;[2014] Hierarchical Plane Extraction (HPE): An Efficient Method For Extraction Of Planes From Large Pointcloud Datasets [[pdf](https://pdfs.semanticscholar.org/8217/61a207088e6015de845cc3f9e556e1c94be1.pdf)]


### Object detection and Tracking

&emsp;&emsp;[2010] Learning a Real-Time 3D Point Cloud Obstacle Discriminator via Bootstrapping [[pdf](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.385.6290)]
    
&emsp;&emsp;[2011] 3D Object Detection from Roadside Data Using Laser Scanners [[pdf](http://www-video.eecs.berkeley.edu/papers/JYT/spie-paper.pdf)]

&emsp;&emsp;[2012] Moving Object Detection with Laser Scanners [[pdf](https://www.ri.cmu.edu/pub_files/2013/1/rob21430.pdf)]

&emsp;&emsp;[2016] 3D Lidar-based static and moving obstacle detection in driving environments: An approach based on voxels and multi-region ground planes [[pdf](http://patternrecognition.cn/perception/negative2016a.pdf)] [[Github](https://github.com/alirezaasvadi/ObstacleDetection)]
    
&emsp;&emsp;[2016] Terrain-Adaptive Obstacle Detection [[pdf](https://ieeexplore.ieee.org/document/7759531)]
    
&emsp;&emsp;[2016] Motion-based Detection and Tracking in 3D LiDAR Scans [[pdf](http://ais.informatik.uni-freiburg.de/publications/papers/dewan16icra.pdf)]
    
&emsp;&emsp;[2016] Deep Tracking on the Move: Learning to Track the World from a Moving Vehicle using Recurrent Neural Networks. [[pdf](https://arxiv.org/abs/1602.00991)]
    
&emsp;&emsp;[2016] Deep Tracking: Seeing Beyond Seeing Using Recurrent Neural Networks. [[pdf](https://arxiv.org/abs/1602.00991)] [[Github](https://github.com/pondruska/DeepTracking)] [[project](https://robotcar-dataset.robots.ox.ac.uk/)]
    
&emsp;&emsp;[2017] Lidar-histogram for fast road and obstacle detection [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)]
    
&emsp;&emsp;[2017] 3D-LIDAR Multi Object Tracking for Autonomous Driving [[pdf](https://repository.tudelft.nl/islandora/object/uuid%3Af536b829-42ae-41d5-968d-13bbaa4ec736)]
    
&emsp;&emsp;[2017] DepthCN: Vehicle Detection Using 3D-LIDAR and ConvNet [[pdf](http://home.isr.uc.pt/~cpremebida/files_cp/DepthCN_preprint.pdf)] [[Github](https://github.com/alirezaasvadi/Multimodal)]
    
&emsp;&emsp;[2018] End-to-end Learning of Multi-sensor 3D Tracking by Detection [[pdf](https://arxiv.org/pdf/1806.11534.pdf)]
    
&emsp;&emsp;[2018] Focal Loss in 3D Object Detection [[pdf](https://arxiv.org/abs/1809.06065)][[Github](https://github.com/pyun-ram/FL3D/tree/master/VoxelNet)]
    
&emsp;&emsp;[2018] PIXOR: Real-time 3D Object Detection from Point Clouds [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf)] [[Github](https://github.com/ankita-kalra/PIXOR)]
    
&emsp;&emsp;[2018] LMNet: Real-time Multiclass Object Detection on CPU using 3D LiDAR [[pdf](https://arxiv.org/abs/1805.04902)] [[Github](https://github.com/CPFL/Autoware/tree/feature/cnn_lidar_detection)]
    
&emsp;&emsp;[2018] BirdNet: a 3D Object Detection Framework from LiDAR information [[pdf](https://arxiv.org/abs/1805.01195)]
    
&emsp;&emsp;[2018] RT3D: Real-Time 3-D Vehicle Detection in LiDAR Point Cloud for Autonomous Driving [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8403277&tag=1)]
    
&emsp;&emsp;[2018] SBNet: Sparse Blocks Network for Fast Inference [[pdf](https://arxiv.org/abs/1801.02108)][[Github](https://github.com/uber/sbnet)]
    
&emsp;&emsp;[2018] Joint 3D Proposal Generation and Object Detection from View Aggregation [[pdf](https://arxiv.org/abs/1712.02294)] [[Github](https://github.com/kujason/avod)]
    
&emsp;&emsp;[2018] PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation [[pdf](https://arxiv.org/abs/1711.10871)] [[report](http://cs230.stanford.edu/files_winter_2018/projects/6939556.pdf)] [[Github](https://github.com/malavikabindhi/CS230-PointFusion)]
    
&emsp;&emsp;[2018] Deep Tracking in the Wild: End-to-End Tracking Using Recurrent Neural Networks. [[pdf](https://journals.sagepub.com/doi/10.1177/0278364917710543)] [[Github](https://github.com/pondruska/DeepTracking)] [[project](https://robotcar-dataset.robots.ox.ac.uk/)]
    
&emsp;&emsp;[2018] (ContFuse) Deep Continuous Fusion for Multi-Sensor 3D Object Detection [[pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)]
    
&emsp;&emsp;[2018] Leveraging Heteroscedastic Aleatoric Uncertainties for Robust Real-Time LiDAR 3D Object Detection [[pdf](https://arxiv.org/abs/1809.05590)]
    
&emsp;&emsp;[2018] (AVOD) Joint 3D Proposal Generation and Object Detection from View Aggregation [[pdf](https://arxiv.org/abs/1712.02294v4)] [[Github](https://github.com/kujason/avod)][[AVOD_SSD](https://github.com/melfm/avod-ssd)]
    
&emsp;&emsp;[2018] Road-Segmentation-Based Curb Detection Method for Self-Driving via a 3D-LiDAR Sensor [[pdf](https://ieeexplore.ieee.org/document/8291612/)]
    
&emsp;&emsp;[2018] YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud [[pdf](https://arxiv.org/pdf/1808.02350.pdf)]
    
&emsp;&emsp;[2018] SINet: A Scale-insensitive Convolutional Neural Network for Fast Vehicle Detection [[pdf](https://arxiv.org/abs/1804.00433)] [[Github](https://github.com/xw-hu/SINet)]

### Classification/Supervised Learning

&emsp;&emsp;[2015] Improving LiDAR Point Cloud Classification using Intensities and Multiple Echoes [[pdf](https://hal.archives-ouvertes.fr/hal-01182604/document)]

&emsp;&emsp;[2017] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation [[pdf](http://stanford.edu/~rqi/pointnet/)] [[Github](https://github.com/charlesq34/pointnet)]

&emsp;&emsp;[2017] Pointnet++: Deep hierarchical feature learning on point sets in a metric space [[pdf](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf)][[Github](https://github.com/charlesq34/pointnet2)]

&emsp;&emsp;[2017] SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud [[pdf](https://arxiv.org/pdf/1710.07368.pdf)] [[Github](https://github.com/BichenWuUCB/SqueezeSeg)]

&emsp;&emsp;[2018] SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud [[pdf](https://arxiv.org/pdf/1809.08495.pdf)]

&emsp;&emsp;[2017] Fast LIDAR-based Road Detection Using Fully Convolutional Neural Networks [[pdf](https://arxiv.org/abs/1703.03613)]

&emsp;&emsp;[2018] ChipNet: Real-Time LiDAR Processing for Drivable Region Segmentation on an FPGA [[pdf](https://arxiv.org/pdf/1808.03506.pdf)] [[Github](https://github.com/YechengLyu/ChipNet)]


### Maps / Grids / HD Maps / Occupancy grids/ Prior Maps

&emsp;&emsp;[2014] Fast 3-D Urban Object Detection on Streaming Point Clouds [[pdf](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/workshops/w15/Paper%202.pdf)]

&emsp;&emsp;[2015] Detection and Tracking of Moving Objects Using 2.5D Motion Grids [[pdf](http://a-asvadi.ir/wp-content/uploads/itsc15.pdf)]

&emsp;&emsp;[2016] 3D Lidar-based Static and Moving Obstacle Detection in Driving Environments: an approach based on voxels and multi-region ground planes [[pdf](http://patternrecognition.cn/perception/negative2016a.pdf)]

&emsp;&emsp;[2016] Spatio–Temporal Hilbert Maps for Continuous Occupancy Representation in Dynamic Environments [[pdf](https://papers.nips.cc/paper/6541-spatio-temporal-hilbert-maps-for-continuous-occupancy-representation-in-dynamic-environments.pdf)]

&emsp;&emsp;[2017] Dynamic Occupancy Grid Prediction for Urban Autonomous Driving: A Deep Learning Approach with Fully Automatic Labeling [[pdf](https://arxiv.org/pdf/1705.08781.pdf)]

&emsp;&emsp;[2017] LIDAR-Data Accumulation Strategy To Generate High Definition Maps For Autonomous Vehicles [[pdf](https://ieeexplore.ieee.org/abstract/document/8170357)]

&emsp;&emsp;[2018] Mobile Laser Scanned Point-Clouds for Road Object Detection and Extraction: A Review [[pdf](https://www.mdpi.com/2072-4292/10/10/1531)]

&emsp;&emsp;[2018] HDNET: Exploiting HD Maps for 3D Object Detection [[pdf](http://proceedings.mlr.press/v87/yang18b/yang18b.pdf)]

&emsp;&emsp;[2018] Hierarchical Probabilistic Fusion Framework for Matching and Merging of 3-D Occupancy Maps [[pdf](https://ieeexplore.ieee.org/abstract/document/8451911)]

&emsp;&emsp;[2018] Recurrent-OctoMap: Learning State-Based Map Refinement for Long-Term Semantic Mapping With 3-D-Lidar Data [[pdf](https://arxiv.org/pdf/1807.00925.pdf)]

### End-To-End Learning

&emsp;&emsp;[2017] VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[pdf](https://arxiv.org/abs/1711.06396)] [[TensorFlow 1](https://github.com/qianguih/voxelnet)] [[TensorFlow 2](https://github.com/jeasinema/VoxelNet-tensorflow)] [[PyTorch](https://github.com/skyhehe123/VoxelNet-pytorch)][[Keras](https://github.com/baudm/VoxelNet-Keras)][[chainer](https://github.com/yukitsuji/voxelnet_chainer)]

&emsp;&emsp;[2018] (DBNet) LiDAR-Video Driving Dataset: Learning Driving Policies Effectively [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_LiDAR-Video_Driving_Dataset_CVPR_2018_paper.pdf)] [[website](http://www.dbehavior.net/)]

&emsp;&emsp;[2018] Monocular Fisheye Camera Depth Estimation Using Semi-supervised Sparse Velodyne Data [[pdf](https://arxiv.org/pdf/1803.06192.pdf)]

&emsp;&emsp;[2018] Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net [[pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)]


## Section 2: Teams and Projects


&emsp;&emsp;1. Oxford Robotics Institute, Oxford RobotCar,https://robotcar-dataset.robots.ox.ac.uk/
	
&emsp;&emsp;2. Center for Automotive Research at Stanford, Standford Artificial Intelligence Laboratory (SAIL), http://ai.stanford.edu/
     SAIL-Toyota Center for AI Research, https://aicenter.stanford.edu/

&emsp;&emsp;3. Berkeley DeepDrive (BDD), https://deepdrive.berkeley.edu/

&emsp;&emsp;4. Princeton Vision & Robotics Autonomous Vehicle Engineering (PAVE), https://pave.princeton.edu/; http://deepdriving.cs.princeton.edu/
	
&emsp;&emsp;5. University of Maryland Autonomous Vehicle Laboratory, http://www.avl.umd.edu/

&emsp;&emsp;6. University of Waterloo WAVE Laboratory (Waterloo Autonomous Vehicles Lab), http://wavelab.uwaterloo.ca/

&emsp;&emsp;7. General Motors-Carnegie Mellon Autonomous Driving Collaborative Research Lab, http://varma.ece.cmu.edu/GM-CMU-AD-CRL/

&emsp;&emsp;8. Toyota-CSAIL Joint Research Center at MIT, https://toyota.csail.mit.edu/

&emsp;&emsp;9. The Perceptual Robotics Laboratory (PeRL) at the University of Michigan, Next Generation Vehicle (NGV),http://robots.engin.umich.edu/Projects/NGV

&emsp;&emsp;10. Univerity of Toronto and Uber ATG, Urtasun, http://www.cs.toronto.edu/~urtasun/, http://www.cs.toronto.edu/~urtasun/group/group.html, https://www.uber.com/info/atg/

&emsp;&emsp;11. Tsinghua University, 3D Image Lab, http://3dimage.ee.tsinghua.edu.cn/

&emsp;&emsp;12. Karlsruher Institut für Technologie Institut für Mess- und Regelungstechnik (MRT), http://www.mrt.kit.edu/index.php

&emsp;&emsp;13. University of Tübingen, Max Planck Research Group for Autonomous Vision,  https://avg.is.tuebingen.mpg.de/

&emsp;&emsp;14. Baidu Inc., Apollo,  https://github.com/ApolloAuto/apollo (https://github.com/Ewenwan/apollo); 

&emsp;&emsp;15. General Motors, Cruise,  https://getcruise.com/

&emsp;&emsp;16. COMMA.AI, https://comma.ai/; https://github.com/commaai

&emsp;&emsp;17. Auro Robotics (Ridecell), http://auro.ai/

&emsp;&emsp;18. Deepdrive,  https://deepdrive.io/； https://github.com/deepdrive

&emsp;&emsp;19. SJTU Machine Vision and Intelligence Group,  http://mvig.sjtu.edu.cn/

&emsp;&emsp;20. Tier IV, Inc., CPFL, https://autoware.ai/; https://github.com/CPFL/Autoware

&emsp;&emsp;21. Intelligent Systems Lab (LSI) from Universidad Carlos III de Madrid, http://portal.uc3m.es/portal/page/portal/dpto_ing_sistemas_automatica/investigacion/IntelligentSystemsLab

## Section 3: LiDAR companies and their products


&emsp;&emsp;1. BLK360: https://lasers.leica-geosystems.com/blk360 (BLK 360, 3D Reconstruction)
    
&emsp;&emsp;2. Velodyne: https://www.velodynelidar.com/ (HDL-64, HDL-32, VLS-128, PUCK, and Ultra PUCK)
   
&emsp;&emsp;3. ouster: https://www.ouster.io/ (OS-1)
    
&emsp;&emsp;4. blackmore: https://blackmoreinc.com/auto (Auto)
    
&emsp;&emsp;5. quanergy: https://quanergy.com/(S1, S3, M8)
    
&emsp;&emsp;6. AEye: https://www.aeye.ai/ (iDAR)
    
&emsp;&emsp;7. INNOVIZ: https://innoviz.tech/ (InnovizOne, InnovizPro)
    
&emsp;&emsp;8. Renesas & Dibotics: http://augmentedlidar.com/
    
&emsp;&emsp;9. CEPTON: http://www.cepton.com/ (HR80, SORA, VISTA)
    
&emsp;&emsp;10. Oryx Vision: http://oryxvision.com/
    
&emsp;&emsp;11. Innovusion: https://www.innovusion.com/
    
&emsp;&emsp;12. Luminar: https://www.luminartech.com/

&emsp;&emsp;13. LeddarTech: https://leddartech.com/

&emsp;&emsp;14. Leishen Intelligent System(镭神智能): http://www.leishen-lidar.com/ (C16,  C32)
    
&emsp;&emsp;15. HESAI(禾赛科技): http://www.hesaitech.com (Hesai Pandora 40)
    
&emsp;&emsp;16. SureStar(北科天绘): http://www.isurestar.com/ (R-FANs 16,  R-FANs 32, C-FANs 128)
    
&emsp;&emsp;17. RoboSense(速腾聚创): http://www.robosense.cn/ (RS-Lidar-16, RS-Lidar-32, RS-Fusion-P3)
    
&emsp;&emsp;18. Princeton Lightwave: https://www.princetonlightwave.com/ (GeigerCruizer)(Princeton Lightwave is now Argo AI)
    
&emsp;&emsp;19. Sick: https://www.sick.com/cn/zh/detection-and-ranging-solutions/3d-lidar-/c/g282752 (LMS511, MRSx000, LD-MRS)
    
&emsp;&emsp;20. Hokuyo: https://www.hokuyo-aut.jp/

&emsp;&emsp;21. Ibeo Automative: https://www.ibeo-as.com/aboutibeo/lidar/
