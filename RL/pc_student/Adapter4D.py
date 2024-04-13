from RL.pc_student.Adamodule import Network
from RL.pc_student.module import Student
from RL.pc_student.module import ActorCritic

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
import random

import copy
import open3d as o3d


class PCSampling:
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

        self.pointclouds_shape = 2048
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]

        self.rl_model_path = self.cfg_train["rl_model_path"]

        self.latent_shape = self.cfg_train["latent_shape"]
        self.prop_shape = self.cfg_train["proprioception_shape"]
 
        self.Adapternet_path = self.cfg_train["adapter_model_path"]
        self.Student_model_path = self.cfg_train["student_model_path"]
        self.input_shape = self.latent_shape + self.prop_shape 
        self.origin_shape =  self.cfg_train["origin_shape"]

        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]

        self.learning_rate = 0.0001
        self.dagger_iter = 11

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.actor_critic = ActorCritic(self.origin_shape, self.state_space.shape, self.action_space.shape,
                                               self.init_noise_std, self.model_cfg, asymmetric=False)
        self.student_actor = Student(self.input_shape,self.action_space.shape,self.model_cfg)

        self.actor_critic.to(self.device)
        self.student_actor.to(self.device)
        self.actor_critic.load_state_dict(torch.load(self.rl_model_path))
        self.actor_critic.eval()
        
        #self.encoded_obs = torch.zeros((self.vec_env.num_envs, self.input_shape), dtype=torch.float, device=self.device)

        self.Adapternet = Network(4, 16).to(device)

        self.optimizer = optim.Adam([
	        {'params': self.Adapternet.parameters(), 'lr': self.learning_rate,}, 
	        {'params': self.student_actor.parameters(), 'lr': self.learning_rate},
        	])
        self.criterion = nn.MSELoss()
        
        #debug
        if self.pc_debug:

            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()

    # def generate_inputs(self,pointcloud_batch):
    #     points = pointcloud_batch[:,:,0:3]
    #     features = pointcloud_batch[:,:,3:]
    #     coords, feats = ME.utils.sparse_collate([point for point in points], [action for action in features])
    #     input = ME.SparseTensor(feats, coordinates=coords, device=self.device)
    #     return input

    def encode_pcs(self, current_obs,pc_features):
        return torch.cat((current_obs[:,0:self.prop_shape],pc_features),dim=1)

    def run(self,num_learning_iterations=0,log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        current_pcs = current_states[:,current_obs.size(1):].view(self.vec_env.num_envs,self.pointclouds_shape,3)

        actions = torch.zeros((self.vec_env.num_envs, 7), device = self.device)
        current_dones = torch.zeros((self.vec_env.num_envs), device = self.device)
        pointclouds = torch.zeros((self.vec_env.num_envs, self.pointclouds_shape, 4), device = self.device)

        update_step = 0
        iter = 0

        if self.is_testing:
            self.Adapternet.load_state_dict(torch.load(self.Adapternet_path))
            self.student_actor.load_state_dict(torch.load(self.Student_model_path))
            self.Adapternet.eval()
            self.student_actor.eval()
            while True:
                with torch.no_grad():

                    pointclouds[:,:,0:3] = current_pcs
                    pointclouds[:,0:1920,3] = 0
                    pointclouds[:,1920:,3] = 1

                    output = self.Adapternet(pointclouds)  
                    encoded_obs = self.encode_pcs(current_obs,output)
                    # Compute the action
                    action_pre = self.student_actor(encoded_obs)      

                    next_obs, rews, dones, infos = self.vec_env.step(action_pre)
                    next_states = self.vec_env.get_state()

                    pcs = next_states[:,next_obs.size(1):].view(self.vec_env.num_envs,self.pointclouds_shape,3)

                    if self.pc_debug:
                        test = pcs[0, :, :3].cpu().numpy()


                        self.o3d_pc.points = o3d.utility.Vector3dVector(test)

                        if self.pointCloudVisualizerInitialized == False :
                            self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                            self.pointCloudVisualizerInitialized = True
                        else :
                            self.pointCloudVisualizer.update(self.o3d_pc)
               
                    # Step the vec_environment
                    #next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs = next_obs
                    current_pcs = pcs


        while True:
            beta = iter / 10

            action_labels = self.actor_critic.act_inference(current_obs)   

            pointclouds[:,:,0:3] = current_pcs
            pointclouds[:,0:896,3] = 0
            pointclouds[:,896:,3] = 1


            output = self.Adapternet(pointclouds)  
            encoded_obs = self.encode_pcs(current_obs,output)
            # Compute the action
            action_pre = self.student_actor(encoded_obs)      

            if random.random() < beta:
                action_mix = action_labels
            else:
                action_mix = action_pre

            next_obs, rews, dones, infos = self.vec_env.step(action_mix)

            next_states = self.vec_env.get_state()  

            pcs = next_states[:,current_obs.size(1):].view(self.vec_env.num_envs,self.pointclouds_shape,3)              


            update_step += 1


            action_loss = self.criterion(action_labels,action_pre)
            #env_loss = self.criterion(labels_batch[:,self.prop_shape:], output_batch.features)
            loss = action_loss
            #print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()       
            self.writer.add_scalar('Loss/Imitation', loss,update_step)        
            #self.writer.add_scalar('Loss/Adapter_action', action_loss,update_step)   
            #self.writer.add_scalar('Loss/Adapter_env', env_loss,update_step)       
            #else:

            #print(next_obs[:,])
            if update_step % 100 == 0:
                torch.save(self.Adapternet.state_dict(), self.Adapternet_path)
                torch.save(self.student_actor.state_dict(), self.Student_model_path)
                print("Save at:", update_step)
                iter = iter + 1 if iter < 10 else 0


            #print(dones)
            #obs = ["q_dof_pos_state", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat"]
            #print("Iter: ", i)
            current_obs = next_obs
            current_pcs = pcs
            current_dones = dones.clone()

