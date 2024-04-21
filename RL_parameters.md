## RL parameters
Edit tasks/ur5xxxx.py

### 1. ur5lift
'''
    cubeA_height = states["cube_pos"][:, 2] - 0.86
    cubeA_lifted = cubeA_height > 0.01
    cubeA_reached = cubeA_height > 0.1
    #cubeA_unreached = cubeA_height < 0.1
    #cubeA_droped = cubeA_height < -0.01
    success_buf = cubeA_reached
    force[force > 200] = 200

    rew_buf = - 0.3 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) + cubeA_lifted * cubeA_height * 2\
                + cubeA_reached * 100 \
                + force * 0.0005
'''

### 2. ur5pickandplace
'''
    cubeA_height = states["cube_pos"][:, 2] - 0.86
    cubeA_height[cubeA_height>0.2] = 0.2 
    cubeA_lifted = cubeA_height > 0.01
    cubeA_picked = cubeA_height >= 0.2
    cubeA_reached = d_g < 0.03
    #cubeA_unreached = cubeA_height < 0.1
    #cubeA_droped = cubeA_height < -0.01
    success_buf = cubeA_reached
    force[force > 200] = 200

    rew_buf = - 0.4 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) + cubeA_lifted * cubeA_height * 5\
                + cubeA_picked * (1-torch.tanh(d_g * 2)) * 2 \
                + force * 0.0005 \
                + cubeA_reached * 300
'''

### 3. ur5cabinet
'''
    force[force > 200] = 200
    ungrasp = force == 0
    goal = d_cabinet > 0.1

    rew_buf = - 0.4 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) * ungrasp \
                + force * 0.0005 \
                + d_cabinet \
                + goal * 100
'''

### 4. ur5cabinet_door
'''
    force[force > 200] = 200
    touch = force > 0
    goal = d_cabinet > 0.3
    close = d_cabinet < 0.01

    rew_buf =   - 0.6 - torch.tanh(5.0 * ( d_lf + d_rf - d_ff / 2)) * close \
                + touch * 0.1 \
                + torch.tanh(3 * d_cabinet) * 0.5 \
                + goal * 600
'''
