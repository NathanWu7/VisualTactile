from RL.pc_sampling.Adamodule import AdapterNetwork
from RL.pc_sampling.PCstorage import RolloutPointClouds
from RL.pc_sampling.module import EnvEncoder

from torch.utils.tensorboard import SummaryWriter

import MinkowskiEngine as ME
import torch
import torch.optim as optim
import torch.nn as nn

import copy

class PCSampling:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 log_dir='run',
                 device='cpu'
                 ):
        self.vec_env = vec_env
        self.device = device
        self.cfg_train = cfg_train
        self.num_transitions_per_env = 4
        self.action_space = vec_env.action_space
        self.pointclouds_shape = 1024
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.env_encoder_cfg = self.cfg_train["env_encoder"]
        self.env_shape = self.cfg_train["env_shape"]
        self.latent_shape = self.cfg_train["latent_shape"]
        self.prop_shape = self.cfg_train["proprioception_shape"]
        self.encoder_path = self.cfg_train["encoder_model_path"]
        self.total_shape = self.latent_shape + self.prop_shape 

        self.mini_batch_size = self.cfg_train["mini_batch_size"]
        self.learning_rate = 0.001

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.encoded_obs = torch.zeros((self.vec_env.num_envs, self.total_shape), dtype=torch.float, device=self.device)

        self.Adapternet = AdapterNetwork(in_feat=self.action_space.shape[0], out_feat=32).to(device)

        self.env_encoder = EnvEncoder(self.env_shape,self.latent_shape, self.env_encoder_cfg)
        self.env_encoder.to(self.device)
        self.env_encoder.load_state_dict(torch.load(self.encoder_path))
        self.env_encoder.eval()

        self.storage = RolloutPointClouds(self.vec_env.num_envs, self.num_transitions_per_env, self.pointclouds_shape, self.action_space.shape[0], device=device) 

        self.optimizer = optim.Adam(self.Adapternet.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def generate_inputs(self,pointcloud_batch):
        points = pointcloud_batch[:,:,0:4]
        features = pointcloud_batch[:,:,4:]
        coords, feats = ME.utils.sparse_collate([point for point in points], [action for action in features])
        input = ME.SparseTensor(feats, coordinates=coords, device=self.device)
        return input

    def encode_obs(self, current_obs):
        self.encoded_obs[:,:self.prop_shape] = current_obs[:,0:self.prop_shape]
        #print(current_obs[:,self.prop_shape:].size())
        encoded_env = self.env_encoder(current_obs[:,self.prop_shape:])
        self.encoded_obs[:,self.prop_shape:] = encoded_env

    def run(self,num_learning_iterations=0,log_interval=1):
        current_obs = self.vec_env.reset()
        counter = torch.zeros((self.vec_env.num_envs), device = self.device)
        actions = torch.zeros((self.vec_env.num_envs, 7), device = self.device)
        update_step = 0
        
        while True:

            # Compute the action
            
            #(x,y,z,rx,ry,rz,gripper)
            #print("q",current_obs[:,0:7])

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

            next_states = self.vec_env.get_state()
            #print(next_obs[:,self.prop_shape:].size())
            #self.encode_obs(next_obs)
            pcs = next_states[:,next_obs.size(1):].view(self.vec_env.num_envs,1024,3)
            
            labels = self.env_encoder(next_obs[:,self.prop_shape:])

            full = self.storage.add_transitions(pcs,actions,dones,labels)
            if full:
                pointcloud_batch, labels_batch = self.storage.mini_batch_generator(self.mini_batch_size)
                #pointcloud_batch, labels = self.storage.mini_batch_generator(self.mini_batch_size)
                if pointcloud_batch is not None:
                    update_step += 1
                    input = self.generate_inputs(pointcloud_batch)
                    output = self.Adapternet(input)
                    #print(output.features)   #result
                    loss = self.criterion(labels_batch, output.features)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()       
                    self.writer.add_scalar('Loss/Adapter', loss,update_step)            

            counter[dones==1] = 0
            #print(next_obs[:,])

            #obs = ["q_dof_pos_state", "eef_pos", "eef_quat", "eef_lf_pos", "eef_rf_pos","goal_pos", "cube_pos", "cube_quat"]
            
            current_obs.copy_(next_obs)
        self.writer.close()