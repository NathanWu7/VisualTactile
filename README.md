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
For env  modify this file :   cfg/task/ur5xxxx.yaml  <br>
  numEnvs : 512+        <br>                                   
  obs_type: ["oracle"] <br>
  
For algo modify this file:    cfg/train/sac/sac_ur5xxxx.yaml   <br>
  load_iter: when rl model saved   <br>
```sh
python3 train.py --task ur5xxxx --algo sac --headless
```
### 2. VTA
For env modify this file :  cfg/task/ur5xxxx.yaml  <br>
  numEnvs : 16+  <br>
  obs_type: ["oracle","pointcloud","tactile"]  <br>
  
For algo modify this file:   cfg/train/vta/vta_ur5xxx.yaml  <br>   
  rl_algo: "sac"     --> Choose algorithm <br>
  rl_iter: 10000     -->  When RL model saved (iter)  <br>
  max_iterations: 10000  <br>
  
```sh
python3 train.py --task ur5xxxx --algo vta --headless
```

### 3. VTP
For algo modify this file:   cfg/train/vtp/vtp_ur5xxx.yaml    
  sample_batch_size: 32 + (sample from replay_size * numEnvs)<br>
  replay_size: 300 + (total data: replay_size * numEnvs, each step update -> numEnvs)<br>
  lr: 0.001   -->  learning rate   <br>
Other config files are the same as VTA  <br>
```sh
python3 train.py --task ur5xxxx --algo vtp --headless
```
For testing:
```sh
python3 train.py --task ur5xxxx --algo vtp --test --headless
```
