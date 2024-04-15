## Dependencies
Some dependencies can be installed by

```sh
pip install -r ./requirements.txt
```
### [Isaac Gym](https://developer.nvidia.com/isaac-gym)

Our framework is implemented on Isaac Gym simulator, the version we used is Preview Release 4. You may encounter errors in installing packages, most solutions can be found in the official docs.

### [Pointnet2](https://github.com/daerduoCarey/where2act/tree/main/code)

Install pointnet++ manually.

```sh
cd {the dir for packages}
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
pip install -r requirements.txt
pip install -e .
```

Finally, run the following to install other packages.

```sh
# make sure you are at the repository root directory
pip install -r requirements.txt
```

## Task test
### Lift object
```sh
python3 train.py --task ur5pickup --test
```
### Pick and Place
```sh
python3 train.py --task ur5pickandplace --test
```
### Cabinet draw
```sh
python3 train.py --task ur5cabinet --test
```
### Open cabinet door
```sh
python3 train.py --task ur5cabinet_door --test
```
## Training

Tensorboard logdir :/run

### 1. RL 
Modify this file :   cfg/task/ur5xxxx.py  <br>
  numEnvs : 512+                                         
  obs_type: ["oracle","contact_force"] <br>

```sh
python3 train.py --task ur5xxxx --algo sac --headless
```
### 2. VTA
Modify this file :cfg/task/ur5xxxx.py 
  numEnvs : 16+
  obs_type: ["oracle","contact_force","pointcloud","tactile"]
```sh
python3 train.py --task ur5xxxx --algo vta --headless
```
For testing:
```sh
python3 train.py --task ur5xxxx --algo vta --test --headless
```
### 3. VTP
```sh
python3 train.py --task ur5xxxx --algo vtp --headless
```
For testing:
```sh
python3 train.py --task ur5xxxx --algo vtp --test --headless
```
