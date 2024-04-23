## RL parameters
Edit tasks/xxxx.py <br>
Function name: compute_reach_reward() <br>
The reward must lower than 0 <br>

```sh
# left finger to cube (distance)
d_lf = torch.norm(states["cube_pos"] - states["eef_lf_pos"], dim=-1)

# right finger to cube (distance)
d_rf = torch.norm(states["cube_pos"] - states["eef_rf_pos"], dim=-1)

# left finger to right finger (distance)
d_ff = torch.norm(states["eef_lf_pos"] - states["eef_rf_pos"], dim=-1)

# left finger to right finger (force)
force = states["force"].squeeze(1)

# object to goal (distance)
d_g = torch.norm(states["cube_to_goal"], dim=-1)

# cabinet dof pos
d_cabinet = states["cabinet_dof_pos"].squeeze(1)
```

### 1. lift

The goal is to lift a object to 0.1 m.
```sh
    cubeA_height = states["cube_pos"][:, 2] - 0.86
    cubeA_lifted = cubeA_height > 0.01
    cubeA_reached = cubeA_height > 0.1
    success_buf = cubeA_reached
    force[force > 200] = 200

    rew_buf = - 0.3 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) + cubeA_lifted * cubeA_height * 2\
                + cubeA_reached * 100 \
                + force * 0.0005
```

### 2. pickandplace
The goal is to lift a object to 0.2m, then move it to goal.
```sh
    cubeA_height = states["cube_pos"][:, 2] - 0.86
    cubeA_height[cubeA_height>0.2] = 0.2 
    cubeA_lifted = cubeA_height > 0.01
    cubeA_picked = cubeA_height >= 0.2
    cubeA_reached = d_g < 0.03
    success_buf = cubeA_reached
    force[force > 200] = 200

    rew_buf = - 0.4 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) + cubeA_lifted * cubeA_height * 5\
                + cubeA_picked * (1-torch.tanh(d_g * 2)) * 2 \
                + force * 0.0005 \
                + cubeA_reached * 300
```

### 3. cabinet
The goal is to open the cabinet drawer for 0.1m.
```sh
    force[force > 200] = 200
    ungrasp = force == 0
    goal = d_cabinet > 0.1

    rew_buf = - 0.4 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) * ungrasp \
                + force * 0.0005 \
                + d_cabinet \
                + goal * 100
```

### 4. cabinet_door
The goal is to open the cabinet door for 0.3 rad.
```sh
    force[force > 200] = 200
    touch = force > 0
    goal = d_cabinet > 0.3
    close = d_cabinet < 0.01

    rew_buf =   - 0.6 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) * close \
                + touch * 0.1 \
                + torch.tanh(3 * d_cabinet) * 0.5 \
                + goal * 600
```
