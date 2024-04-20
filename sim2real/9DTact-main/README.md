# 9DTact

# Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [3D Shape Reconstruction](#reconstruction)
   1. [Camera Calibration](#camera_calibration)
   2. [Sensor Calibration](#sensor_calibration)
   3. [Shape Reconstruction](#shape_reconstruction)
4. [6D Force Estimation](#estimation)
   1. [BOTA Driver](#bota)
   2. [Data Collection](#collection)
   3. [Data Processing](#processing)
   4. [Model Training](#training)
   5. [Force Estimation](#inference)
5. [Run in ROS](#ros)
   1. [Shape Reconstruction in ROS](#shape_ros)
   2. [Force Estimation in ROS](#force_ros)
   3. [Simultaneous Shape Reconstruction and Force Estimation (SSAF) in ROS](#shape_force)

## Overview <a name="overview"></a>
**This repository provides open-source files of the paper:**

![](source/pipelie.png)

<b>9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation</b> <br>
[Changyi Lin](https://linchangyi1.github.io/),
[Han Zhang](https://doublehan07.github.io/),
Jikai Xu, Lei Wu, and
[Huazhe Xu](http://hxu.rocks/) <br>
RAL, 2023 <br>
[Website](https://linchangyi1.github.io/9DTact/) /
[Arxiv Paper](https://arxiv.org/abs/2308.14277) /
[Video Tutorial](https://www.youtube.com/watch?v=-oRtW398JDY)

```
@article{lin20239dtact,
  title={9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation},
  author={Lin, Changyi and Zhang, Han and Xu, Jikai and Wu, Lei and Xu, Huazhe},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```



## Installation <a name="installation"></a>
#### Create a conda environment:
```bash
conda create -n 9dtact python=3.8
```
#### Install pytorch (choose the version that is compatible with your computer):
```bash
conda activate 9dtact
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
#### In this repository, install the other requirements:
```bash
pip install -e .
```


## 3D Shape Reconstruction <a name="reconstruction"></a>
For all the terminals used in this section, they are located in the **shape_reconstruction** directory and based on the **9dtact** conda environment:
```bash
cd shape_reconstruction
conda activate 9dtact
```

### 1. Camera Calibration <a name="camera_calibration"></a>
Before you start, 3d print the [calibration board](9DTact_Design/fabrication/calibration_board.STL).<br>
Run:
```bash
python3 _1_Camera_Calibration.py
```
Just follow the printed instructions.

### 2. Sensor Calibration <a name="sensor_calibration"></a>
Firstly, prepare a ball with a radius of 4.0 mm.
(The radius of the ball depends on the thickness of the sensor surface.
4.0 mm is just a recommendation.)<br>
Then, run:
```bash
python3 _2_Sensor_Calibration.py
```
Just follow the printed instructions.

### 3. Shape Reconstruction <a name="shape_reconstruction"></a>
```bash
python3 _3_Shape_Reconstruction.py
```

## 6D Force Estimation <a name="estimation"></a>
### 1. BOTA Driver <a name="bota"></a>
**If you want to collect force data with a BOTA MiniONE Pro sensor, you need to:**<br>
Create a directory named 'bota_ws' as the ROS workspace, and install the [bota driver package](https://gitlab.com/botasys/bota_driver).

### 2. Data Collection <a name="collection"></a>
#### At the first terminal, open the BOTA sensor:
```bash
cd ~/xxx/bota_ws # Modify 'xxx' to enter the workspace directory
source devel/setup.bash
roslaunch rokubimini_serial rokubimini_serial.launch
```
#### At the second terminal, run:
```bash
source ~/xxx/bota_ws/devel/setup.bash
cd data_collection
conda activate 9dtact
python3 collect_data.py
```
#### At the third terminal, open the 9DTact sensor:
```bash
cd shape-force-ros
conda activate 9dtact
python3 _1_Sensor_ros.py
```

### 3. Data Processing <a name="processing"></a>
#### Open a terminal, normalize the wrench:
```bash
cd data_collection
conda activate 9dtact
python3 wrench_normalization.py  # remember to modify the object_num
```
#### At the same terminal, split the data by running:
```bash
python3 split_train_test.py
```
and also:
```bash
python3 split_train_test(objects).py
```

### 4. Model Training <a name="training"></a>
If using the defaulted parameters, just run:
```bash
cd force_estimation
python3 train.py
```
You may also choose to use [Weights and Bias (wandb)](https://docs.wandb.ai/quickstart) by setting use_wandb as True,
which helps to track the training performance.

### 5. Force Estimation <a name="inference"></a>
You need to specify a model saved in the 'saved_models' directory as an estimator,
by modifying the 'weights' parameters in the [force_config.yaml](force_estimation/force_config.yaml).<br>
After that, run:
```bash
cd force_estimation
python3 _1_Force_Estimation.py
```


## Run in ROS <a name="ros"></a>
### 1. Shape Reconstruction in ROS <a name="shape_ros"></a>
#### At the first terminal, open the 9DTact sensor:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _1_Sensor_ros.py
```
#### At the second terminal, run:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _2_Shape_Reconstruction_ros.py
```

### 2. Force Estimation in ROS <a name="force_ros"></a>
#### At the first terminal, open the 9DTact sensor:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _1_Sensor_ros.py
```
#### At the second terminal, run:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _3_Force_Estimation_ros.py
```
#### (Optional for visualization) At the third terminal, open the visualization window:
```bash
cd force_estimation
conda activate 9dtact
python3 force_visualizer.py
```
### 3. Simultaneous Shape Reconstruction and Force Estimation (SSAF) in ROS <a name="shape_force"></a>
#### At the first terminal, open the force estimator:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _3_Force_Estimation_ros.py
```
#### At the second terminal, run:
```bash
cd shape-force_ros
conda activate 9dtact
python3 _4_Shape_Force_ros.py
```

