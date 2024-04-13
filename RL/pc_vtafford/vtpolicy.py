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
import time
import open3d as o3d


class vtpolicy:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 log_dir='run',
                 device='cpu'
                 ):
        self.is_testing = True
        self.pc_debug = False
        self.pointCloudVisualizerInitialized = False

        self.vec_env = vec_env
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.device = device
        self.cfg_train = cfg_train
        self.num_transitions_per_env = 4

        self.pointclouds_shape = self.cfg_train["PCDownSampleNum"]
        self.tactile_shape = 64 * 2
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]

        self.rl_model_path = self.cfg_train["rl_model_path"]

        self.latent_shape = self.cfg_train["latent_shape"]
        self.prop_shape = self.cfg_train["proprioception_shape"]
 
        self.TAN_path = self.cfg_train["TAN_model_path"]
        self.Student_model_path = self.cfg_train["student_model_path"]
        self.input_shape = self.latent_shape + self.prop_shape 
        self.origin_shape =  self.cfg_train["origin_shape"]

        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.student_cfg = self.cfg_train["student"]
        ac_kwargs = dict(hidden_sizes=[self.model_cfg["hidden_nodes"]]* self.model_cfg["hidden_layer"])

        self.learning_rate = 0.001
        self.dagger_iter = 11

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.actor_critic =  MLPActorCritic(self.origin_shape, vec_env.action_space, **ac_kwargs).to(self.device)
        self.student_actor = Student(self.input_shape, self.latent_shape, self.action_space.shape, self.vec_env.num_envs, self.device, self.student_cfg)

        self.actor_critic.to(self.device)
        self.student_actor.to(self.device)
        self.actor_critic.load_state_dict(torch.load(self.rl_model_path))
        self.actor_critic.eval()
        
        #self.encoded_obs = torch.zeros((self.vec_env.num_envs, self.input_shape), dtype=torch.float, device=self.device)

        self.TAN = Network(4, 16).to(device)
        self.TAN.load_state_dict(torch.load(self.TAN_path))
        self.TAN.eval()

        self.optimizer = optim.Adam([
	        {'params': self.student_actor.parameters(), 'lr': self.learning_rate,}
        	])
        self.criterion = nn.MSELoss()
        
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

        update_step = 1
        iter = 0
        all_indices = set(torch.arange(pointclouds.size(0)).numpy())
        counter = torch.zeros((self.vec_env.num_envs), device = self.device)
        action_labels = torch.zeros((self.vec_env.num_envs, 7), device = self.device)
        

        if self.is_testing:

            self.student_actor.load_state_dict(torch.load(self.Student_model_path))
            self.student_actor.eval()
            while True:
                with torch.no_grad():
            
                    pointclouds[:,:,0:3] = current_pcs[:,:,0:3]
                    tactiles = current_pcs[:,self.pointclouds_shape:,0:3]
                    is_zero = torch.all(tactiles == 0, dim=-1)
                    num_zero_points = torch.sum(is_zero, dim=-1)
                    zero_indices = torch.nonzero(num_zero_points == 128)[:, 0]
                    
                    touch_indices = torch.tensor(list( all_indices - set(zero_indices.cpu().numpy())))
        
                    if len(touch_indices) > 0:

                        pointclouds[:,:,3] = 0
                        tactile_part = pointclouds[:,self.pointclouds_shape:,:]
                        is_nonzero = (tactile_part[:,:,:3]!=0).any(dim=2)
                        pointclouds[:,self.pointclouds_shape:,3][is_nonzero] = 1

                        shuffled = pointclouds[:, torch.randperm(pointclouds.size(1)), :]

                        pcs = shuffled[:, :self.pointclouds_shape, :]
                        pcs[:,:,3] = 1 
                                
                        update_step += 1            
                        output = self.TAN(pcs)  

                    else:
                        pcs = pointclouds[:, :self.pointclouds_shape, :]
                        pcs[:,:,3] = 1 
                        output = self.TAN(pcs)

                    pcs[:,:,3] = output.detach()
                    mu, sigma, pi = self.student_actor.act(pcs,current_obs[:,:self.prop_shape])  
                    action_pre = self.student_actor.mdn_sample(mu, sigma, pi)
                    #action_pre = self.actor_critic.act_inference(current_obs)  
                    #action_pre = self.student_actor(pcs,current_obs[:,:self.prop_shape])   

                    next_obs, rews, dones, infos = self.vec_env.step(action_pre)

                    next_pointcloud = self.vec_env.get_pointcloud()  

                    if self.pc_debug:
                        test = pcs[0, :, :3].cpu().numpy()
                        #print(test.shape)
                        color = output[0].unsqueeze(1).detach().cpu().numpy()
                        color = (color - min(color)) / (max(color)-min(color))
                        colors_blue = o3d.utility.Vector3dVector( color * [[1,0,0]])
                        #print(color * [[0,0,1]])
                        self.pcd.points = o3d.utility.Vector3dVector(list(test))
                        self.pcd.colors = o3d.utility.Vector3dVector(list(colors_blue))

                        if self.pointCloudVisualizerInitialized == False :
                            self.pointCloudVisualizer.add_geometry(self.pcd)
                            self.pointCloudVisualizerInitialized = True
                        else :
                            self.pointCloudVisualizer.update(self.pcd)
               
                    # Step the vec_environment
                    current_obs = next_obs
                    current_pcs = next_pointcloud

       
        while True:
            beta = iter / (self.dagger_iter - 1)
            with torch.no_grad():
                action_labels = self.actor_critic.act(current_obs)   
            
                pointclouds[:,:,0:3] = current_pcs[:,:,0:3]
                tactiles = current_pcs[:,self.pointclouds_shape:,0:3]
                is_zero = torch.all(tactiles == 0, dim=-1)
                num_zero_points = torch.sum(is_zero, dim=-1)
                zero_indices = torch.nonzero(num_zero_points == 128)[:, 0]
                
                touch_indices = torch.tensor(list( all_indices - set(zero_indices.cpu().numpy())))
    
                if len(touch_indices) > 0:

                    pointclouds[:,:,3] = 0
                    tactile_part = pointclouds[:,self.pointclouds_shape:,:]
                    is_nonzero = (tactile_part[:,:,:3]!=0).any(dim=2)
                    pointclouds[:,self.pointclouds_shape:,3][is_nonzero] = 1

                    shuffled = pointclouds[:, torch.randperm(pointclouds.size(1)), :]

                    pcs = shuffled[:, :self.pointclouds_shape, :]
                    pcs[:,:,3] = 1 
                            
                            
                    output = self.TAN(pcs)  

                else:
                    pcs = pointclouds[:, :self.pointclouds_shape, :]
                    pcs[:,:,3] = 1 
                    output = self.TAN(pcs)
                
                pcs[:,:,3] = output.detach()

                # print("current_obs: ",current_obs)

                mu, sigma, pi = self.student_actor.act(pcs,current_obs[:,:self.prop_shape])    #[:,:self.prop_shape]
                action_pre = self.student_actor.mdn_sample(mu, sigma, pi)
                if random.random() < beta:
                    action_mix = action_labels
                else:
                    action_mix = action_pre

                #loss = self.student_actor.mdn_loss(mu, sigma, pi, action_labels)
                self.student_actor.add_transitions(pcs,current_obs[:,:self.prop_shape],action_labels)
             # 0.02s

            if self.student_actor.fullfill:
                data_pcs_batch,data_obs_batch, labels_batch = self.student_actor.batch_sampler()
                mu_batch, sigma_batch, pi_batch = self.student_actor.act(data_pcs_batch,data_obs_batch)
                loss = self.student_actor.mdn_loss(mu_batch, sigma_batch, pi_batch, labels_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()       
                update_step += 1
                self.writer.add_scalar('Loss/Imitation', loss,update_step)      

            next_obs, rews, dones, infos = self.vec_env.step(action_mix)

            next_pointcloud = self.vec_env.get_pointcloud()
            #counter[dones==1] = 0  

            if self.pc_debug:
                test = pcs[1, :, :3].cpu().numpy()
                #print(test.shape)
                color = output[1].unsqueeze(1).detach().cpu().numpy()
                color = (color - min(color)) / (max(color)-min(color))
                colors_blue = o3d.utility.Vector3dVector( color * [[1,0,0]])
                #print(color * [[0,0,1]])
                self.pcd.points = o3d.utility.Vector3dVector(list(test))
                self.pcd.colors = o3d.utility.Vector3dVector(list(colors_blue))

                if self.pointCloudVisualizerInitialized == False :
                    self.pointCloudVisualizer.add_geometry(self.pcd)
                    self.pointCloudVisualizerInitialized = True
                else :
                    self.pointCloudVisualizer.update(self.pcd)  

            if update_step % 100 == 0:
                torch.save(self.student_actor.state_dict(), self.Student_model_path)
                print("Save at:", update_step, " Iter:",iter)
                iter = iter + 1 if iter < 10 else 0
            
            current_obs = next_obs
            current_pcs = next_pointcloud
            current_dones = dones.clone()
