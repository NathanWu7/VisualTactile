from RL.pc_vtafford.pcmodule import Network
from RL.pc_vtafford.rlmodule import Student
from RL.sac import MLPActorCritic

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np 

import copy
import open3d as o3d


class vtafford:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 log_dir='run',
                 device='cpu'
                 ):
        self.is_testing = True
        self.pc_debug = True
        self.pointCloudVisualizerInitialized = False

        self.vec_env = vec_env
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.device = device
        self.cfg_train = cfg_train
        self.num_transitions_per_env = 4

        self.pointclouds_shape = self.cfg_train["PCDownSampleNum"]
        self.tactile_shape = self.cfg_train["TDownSampleNum"] * 2
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]

        self.rl_model_path = self.cfg_train["rl_model_path"]

        self.latent_shape = self.cfg_train["latent_shape"]
        self.prop_shape = self.cfg_train["proprioception_shape"]
 
        self.TAN_path = self.cfg_train["adapter_model_path"]
        self.Student_model_path = self.cfg_train["student_model_path"]
        self.input_shape = self.latent_shape + self.prop_shape 
        self.origin_shape =  self.cfg_train["origin_shape"]

        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        ac_kwargs = dict(hidden_sizes=[self.model_cfg["hidden_nodes"]]* self.model_cfg["hidden_layer"])

        self.learning_rate = 0.0001

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.actor_critic =  MLPActorCritic(self.origin_shape, vec_env.action_space, **ac_kwargs).to(self.device)

        self.actor_critic.to(self.device)
        self.actor_critic.load_state_dict(torch.load(self.rl_model_path))
        self.actor_critic.eval()
        
        #self.encoded_obs = torch.zeros((self.vec_env.num_envs, self.input_shape), dtype=torch.float, device=self.device)

        self.TAN = Network(4, 16).to(device)

        self.optimizer = optim.Adam([
	        {'params': self.TAN.parameters(), 'lr': self.learning_rate,}
        	])
        self.criterion = nn.BCELoss()
        
        #debug
        if self.pc_debug:

            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.pcd = o3d.geometry.PointCloud()

    def run(self,num_learning_iterations=0,log_interval=1):
        current_obs = self.vec_env.reset()

        current_pcs = self.vec_env.get_pointcloud()

        actions = torch.zeros((self.vec_env.num_envs, 7), device = self.device)
        current_dones = torch.zeros((self.vec_env.num_envs), device = self.device)
        pointclouds = torch.zeros((self.vec_env.num_envs, (self.pointclouds_shape + self.tactile_shape), 4), device = self.device)
        
        update_step = 0
        iter = 0
        all_indices = set(torch.arange(pointclouds.size(0)).numpy())
        

        if self.is_testing:
            self.TAN.load_state_dict(torch.load(self.TAN_path))
            self.TAN.eval()
            while True:
                with torch.no_grad():

                    actions = self.actor_critic.act(current_obs)   
                    pointclouds[:,:,0:3] = current_pcs[:,:,0:3]
                    tactiles = current_pcs[:,self.pointclouds_shape:,0:3]
                    is_zero = torch.all(tactiles == 0, dim=-1)
                    num_zero_points = torch.sum(is_zero, dim=-1)
                    zero_indices = torch.nonzero(num_zero_points == 128)[:, 0]
                    touch_indices = torch.tensor(list( all_indices - set(zero_indices.cpu().numpy())))

                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_pointcloud = self.vec_env.get_pointcloud()  


                    if len(touch_indices) > 0:
                        pointclouds[:,:,3] = 0
                        tactile_part = pointclouds[:,self.pointclouds_shape:,:]
                        is_nonzero = (tactile_part[:,:,:3]!=0).any(dim=2)
                        pointclouds[:,self.pointclouds_shape:,3][is_nonzero] = 1
                        #shuffled = pointclouds[:, torch.randperm(pointclouds.size(1)), :]
                        pcs = pointclouds[:, -self.pointclouds_shape:, :]
                        labels = pcs[:,:,3].clone()
                        pcs[:,:,3] = 1 #relabel
             
                        output = self.TAN(pcs)  
  
                    else:
                        pcs = pointclouds[:, :self.pointclouds_shape, :]
                        pcs[:,:,3] = 1 
                        output = self.TAN(pcs)
                    pcs[:,:,3] = output.detach()

                    if self.pc_debug:
                        test = pcs[1, :, :3].cpu().numpy()
                        #print(test.shape)
                        color = pcs[1, :, 3].unsqueeze(1).cpu().numpy()
                        color = (color - min(color)) / (max(color)-min(color))
                        colors_blue = o3d.utility.Vector3dVector( color * [[1,0,0]])

                        self.pcd.points = o3d.utility.Vector3dVector(list(test))
                        self.pcd.colors = o3d.utility.Vector3dVector(list(colors_blue))

                        if self.pointCloudVisualizerInitialized == False :
                            self.pointCloudVisualizer.add_geometry(self.pcd)
                            self.pointCloudVisualizerInitialized = True
                        else :
                            self.pointCloudVisualizer.update(self.pcd)  
                
                    # Step the vec_environment
                    #next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs = next_obs
                    current_pcs = next_pointcloud


        while True:

            actions = self.actor_critic.act(current_obs)   
            
            pointclouds[:,:,0:3] = current_pcs[:,:,0:3]
            tactiles = current_pcs[:,self.pointclouds_shape:,0:3]
            is_zero = torch.all(tactiles == 0, dim=-1)
            num_zero_points = torch.sum(is_zero, dim=-1)
            zero_indices = torch.nonzero(num_zero_points == 128)[:, 0]
            
            touch_indices = torch.tensor(list( all_indices - set(zero_indices.cpu().numpy())))

            next_obs, rews, dones, infos = self.vec_env.step(actions)

            next_pointcloud = self.vec_env.get_pointcloud()  
            #print(torch.mean(next_pointcloud,axis=1))
            
            if len(touch_indices) > 0:
                pointclouds[:,:,3] = 0
                tactile_part = pointclouds[:,self.pointclouds_shape:,:]
                is_nonzero = (tactile_part[:,:,:3]!=0).any(dim=2)
                pointclouds[:,self.pointclouds_shape:,3][is_nonzero] = 1

                #shuffled = pointclouds[:, torch.randperm(pointclouds.size(1)), :]

                pcs = pointclouds[:, -self.pointclouds_shape:, :]
                labels = pcs[:,:,3].clone()
                pcs[:,:,3] = 1 
                         
                update_step += 1            
                output = self.TAN(pcs)  
                # print("output:", output)
                #print("label:", label)
                loss = self.criterion(output[touch_indices,:],labels[touch_indices,:])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()       
                self.writer.add_scalar('Loss/pc', loss,update_step)      
            else:
                pcs = pointclouds[:, :self.pointclouds_shape, :]
                pcs[:,:,3] = 1 
                output = self.TAN(pcs)
                #print(output[1])
            pcs[:,:,3] = output.detach()

            if self.pc_debug:
                test = pcs[1, :, :3].cpu().numpy()
                #print(test.shape)
                color = pcs[1, :, 3].unsqueeze(1).cpu().numpy()
                color = (color - min(color)) / (max(color)-min(color))
                colors_blue = o3d.utility.Vector3dVector( color * [[1,0,0]])

                self.pcd.points = o3d.utility.Vector3dVector(list(test))
                self.pcd.colors = o3d.utility.Vector3dVector(list(colors_blue))

                if self.pointCloudVisualizerInitialized == False :
                    self.pointCloudVisualizer.add_geometry(self.pcd)
                    self.pointCloudVisualizerInitialized = True
                else :
                    self.pointCloudVisualizer.update(self.pcd)  
                #self.writer.add_scalar('Loss/Adapter_action', action_loss,update_step)   
                #self.writer.add_scalar('Loss/Adapter_env', env_loss,update_step)       
                #else:

                #print(next_obs[:,])
            if update_step % 100 == 0:
                torch.save(self.TAN.state_dict(), self.TAN_path)
                print("Save at:", update_step)

            current_obs = next_obs
            current_pcs = next_pointcloud
            current_dones = dones.clone()
            

