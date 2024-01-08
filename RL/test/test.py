from datetime import datetime
import numpy as np
import torch



class TEST:
    def __init__(self,
                 vec_env,
                 device='cpu'
                 ):
        self.vec_env = vec_env
        self.device = device

    def run(self,num_learning_iterations=0, log_interval=1):
        current_obs = self.vec_env.reset()
        counter = torch.zeros((self.vec_env.num_envs), device = self.device)
        while True:
            with torch.no_grad():
                # Compute the action
                actions = torch.zeros((self.vec_env.num_envs, 7), device = self.device)
                
                #(x,y,z,rx,ry,rz,gripper)

                # print("lf_x",current_obs[:,14])
                # print("lf_y",current_obs[:,15])
                # print("lf_z",current_obs[:,16])

                # print("rf_z",current_obs[:,19])

                # print("f_x",current_obs[:,7])
                # print("f_y",current_obs[:,8])

                # print("cube_x",current_obs[:,23])
                # print("cube_y",current_obs[:,24])
                # print("cube_z",current_obs[:,25])
                
                # PID policy
                actions[:,2] = (current_obs[:,25] -0.01 - (current_obs[:,16] + current_obs[:,19]) / 2) * 2
                actions[:,0] = (current_obs[:,23] - (current_obs[:,14] + current_obs[:,17]) / 2) * 2
                actions[:,1] = (current_obs[:,24] - (current_obs[:,15] + current_obs[:,18]) / 2) * 2
                if torch.any(actions[:, 2] > -0.04):
                    # 对于满足条件的元素，将对应位置的 actions[:, 6] 设置为 0.01
                    actions[actions[:, 2] > -0.04, 6] = 0.1
                    counter[actions[:, 2] > -0.04] += 1

                #gripper_dis = (current_obs[:,14] - current_obs[:,17]) ** 2 + (current_obs[:,15] - current_obs[:,18]) ** 2
                #print(gripper_dis)gripper_dis  < 0.004 0.06**2

                if torch.any(counter  > 100 ):
                    actions[counter  > 100, 2] = 0.05
                    

                # Step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                counter[dones==1] = 0
                #print(next_obs[:,])

                #obs = ["q_dof_pos_state", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat"]
                
                current_obs.copy_(next_obs)
  