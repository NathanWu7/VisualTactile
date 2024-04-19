# Experiment Log

| Task | date |
| --- | --- |
| ur5lift | 4/17 |
| ur5pickandplace | 4/18 |
|     |     |
|     |     |

##

---

## ur5pickandplace

1. RL
  

```yaml
cfg/task/ur5pickandplace.yaml
    numenvs: 1024
    obs_type: [["oracle"] ]


python3 train.py --task ur5pickandplace--algo sac --headless
python3 train.py --task ur5pickandplace --algo vta --headless
```

2. VTA
  

```yaml
cfg/task/ur5pickandplace.yaml
    numenvs: 32
    obs_type: ["oracle","pointcloud","tactile"]

cfg/train/vta/vta_ur5pickandplace.yaml
    rl_iter:6000

python3 train.py --task ur5pickandplace --algo vta --headless
    
```

````yaml
```yaml
cfg/task/ur5pickandplace.yaml
    numenvs: 32
    obs_type: ["oracle","pointcloud","tactile"]
cfg/train/vtp/vtp_ur5pickandplace_door.yaml
    replay_size: 1000
    sample_batch_size: 256

python3 train.py --task ur5pickandplace --algo vtp --headless


python3 train.py --task ur5pickandplace --algo vtp --test --headless 
    success_rate:  0.0   in 160 cases.
```
````

## ur5lift

1. RL
  

```yml
cfg/task/ur5lift.yaml
    numenvs: 1024
    obs_type: [["oracle"] ]

python3 train.py --task ur5lift --algo sac --headless

cfg/train/sac/sac_ur5lift.yaml
    load_iter: 4000

python3 train.py --task ur5lift--algo sac --headless --test
    success_rate:  0.9862   in 1086 cases.
```

2. VTA
  

```yaml
cfg/task/ur5lift.yaml
    numenvs: 32
    obs_type: ["oracle","pointcloud","tactile"]

python3 train.py --task ur5lift --algo vta --headless
```

3. VTP
  

```yaml
cfg/task/ur5lift.yaml
    numenvs: 32
    obs_type: ["oracle","pointcloud","tactile"]
cfg/train/vtp/vtp_ur5lift_door.yaml
    replay_size: 1000
    sample_batch_size: 256

python3 train.py --task ur5lift --algo vtp --headless


python3 train.py --task ur5lift --algo vtp --test --headless 
    success_rate:  0.9231   in 1001 cases.
```