# Experiment Log

| Task | date |
| --- | --- |
| ur5lift | 4/18 |

## ur5lift 

1. RL
  

```yml
cfg/task/ur5lift.yaml
    numenvs: 1024
    obs_type: [["oracle"] ]

python3 train.py --task ur5lift --algo sac --headless
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
cfg/train/vtp/vtp_ur5cabinet_door.yaml
    replay_size: 1000
    sample_batch_size: 256

python3 train.py --task ur5lift --algo vtp --headless


python3 train.py --task ur5lift --algo vtp --test --headless 
success_rate:  0.9231   in 1001 cases.
```