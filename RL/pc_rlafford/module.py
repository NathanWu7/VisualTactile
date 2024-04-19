import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg):
        super(ActorCritic, self).__init__()
        if model_cfg is None:
            latent_size = 16
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            latent_size = model_cfg['pc_latent_size']
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])
        self.pointnet = PointNet(latent_size) 

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []

        critic_layers.append(nn.Linear(obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.pointnet)
        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, prio, points, mpo):
        latent = self.pointnet(points)
        observations = torch.cat((prio, latent, mpo),dim=1)
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self,  prio, points, mpo):
        latent = self.pointnet(points)
        observations = torch.cat((prio, latent, mpo),dim=1)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, prio, points, mpo, actions):
        latent = self.pointnet(points)
        observations = torch.cat((prio, latent, mpo),dim=1)
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()


        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class PointNet(nn.Module):
    def __init__(self, latent_size):
        super(PointNet,self).__init__()
        self.conv1 = nn.Conv1d(4, 64, 1) # (envs,n,64)
        self.conv2 = nn.Conv1d(64, 128, 1) # (envs,n,128)
        self.conv3 = nn.Conv1d(128, 256, 1) # (envs,n,256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.linear1 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        #self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, latent_size)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(x))) # (envs,n,64)
        x = F.relu(self.bn2(self.conv2(x))) # (envs,n,128)
        x = F.relu(self.bn3(self.conv3(x))) # (envs,n,256)

        x = F.adaptive_max_pool1d(x, 1).squeeze(2) # (envs, 256)

        x = F.relu(self.bn4(self.linear1(x))) # (envs,256)

        #x = self.dp1(x)
        x = self.linear2(x) # (envs, output_channels)       
        #print(x)
        return x 
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
