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
                #print("q",current_obs[:,0:7])
                #print(current_obs)
                # print("lf_x",current_obs[:,14])
                # print("lf_y",current_obs[:,15])
                # print("lf_z",current_obs[:,16])

                # print("rf_z",current_obs[:,19])

                # print("f_x",current_obs[:,7])
                # print("f_y",current_obs[:,8])

                # print("1_x",current_obs[:,24])

                # print("2_x",current_obs[:,25])
                # print("3_x",current_obs[:,26])
                # print("4_x",current_obs[:,27])
                
                # PID policy
                actions[:,2] = (current_obs[:,23] - (current_obs[:,16] + current_obs[:,19]) / 2) * 2
                actions[:,0] = (current_obs[:,21] - (current_obs[:,14] + current_obs[:,17]) / 2) * 2
                actions[:,1] = (current_obs[:,22] - (current_obs[:,15] + current_obs[:,18]) / 2) * 2

                counter += 1
                if torch.any(counter  > 200 ):
                    actions[counter  > 200, 6] = 0.05  

                if torch.any(counter  > 300 ):
                    actions[counter  > 300, 0] = -0.05  


                # Step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(actions)
                next_states = self.vec_env.get_state()
                #print(next_states.shape)
                print(rews.mean())
                counter[dones==1] = 0
                #print(next_obs[:,])

                #obs = ["q_dof_pos_state", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat"]
                
                current_obs.copy_(next_obs)
  