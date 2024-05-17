import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import MultivariateNormal

import random


class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])


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
        if self.asymmetric:
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

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

    def act(self, observations, states):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class Student(nn.Module):
    def __init__(self, obs_shape,  prop_shape, pointclouds_shape, latent_size, actions_shape, num_envs, device, model_cfg):
        super(Student, self).__init__()
        self.hidden_size = model_cfg['pi_hid_sizes'][-1]
        self.num_gaussians = model_cfg['num_gaussians']
        self.actions_space = actions_shape[0]
        self.replay_size = model_cfg['replay_size']
        self.total_size = self.replay_size * num_envs
        self.batch_size = model_cfg['sample_batch_size']
        self.device = device
        self.fullfill = False
        self.pc_shape = pointclouds_shape
        self.step = 0

        if model_cfg is None:
            actor_hidden_dim = [256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.pointnet = PointNet(latent_size) 
        self.input_shape = obs_shape
        self.prio_shape = prop_shape

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.input_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], self.hidden_size))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        print("student_actor:", self.actor)
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

        self.current_obs = torch.zeros(self.replay_size, num_envs, self.prio_shape, device=device)
        self.pcs = torch.zeros(self.replay_size, num_envs, self.pc_shape, 6, device=device)
        self.labels = torch.zeros(self.replay_size, num_envs, self.actions_space, device=device)
        self.z_pi = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_gaussians),
            nn.Softmax(dim=1)
        )
        self.z_mu = nn.Linear(self.hidden_size, self.num_gaussians * self.actions_space)
        self.z_sigma = nn.Linear(self.hidden_size, self.num_gaussians * self.actions_space)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def mdn_loss(self, mu, sigma, pi, target):
        mix = D.Categorical(pi)    
        comp = D.Independent(D.Normal(mu,sigma), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return -gmm.log_prob(target).mean()

    def mdn_sample(self, mu, sigma, pi):
        mix=D.Categorical(pi)
        comp=D.Independent(D.Normal(mu,sigma), 1)
        gmm=D.MixtureSameFamily(mix,comp)
        return gmm.sample()

    def act(self,points,prio):
        latent = self.pointnet(points)
        input = torch.cat((latent, prio),dim=1)
        #input = prio

        output = self.actor(input)

        pi = self.z_pi(output)
        mu = self.z_mu(output)
        mu = mu.view(-1, self.num_gaussians, self.actions_space)
        sigma = torch.exp(self.z_sigma(output))  #positive
        sigma = sigma.view(-1, self.num_gaussians, self.actions_space)

        return mu, sigma, pi
    
    def add_transitions(self,pcs,current_obs,labels):
        if self.step >= self.replay_size:
            self.step = (self.step + 1) % self.replay_size
            self.fullfill = True
            # raise AssertionError("Rollout buffer overflow")
        #print(self.step)

        self.pcs[self.step].copy_(pcs)
        self.current_obs[self.step].copy_(current_obs)
        self.labels[self.step].copy_(labels)

        self.step += 1


    def batch_sampler(self):
        random_indices = torch.randint(low = 1, high = self.total_size -1, size=(self.batch_size,)).to(self.device)
        data_pcs = self.pcs.reshape(-1,self.pc_shape,6)
        data_obs = self.current_obs.reshape(-1,self.prio_shape)
        labels = self.labels.reshape(-1,self.actions_space)
        return data_pcs[random_indices], data_obs[random_indices], labels[random_indices]
    
    def forward(self):
        raise NotImplementedError

class PointNet(nn.Module):
    def __init__(self, latent_size):
        super(PointNet,self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1) # (envs,n,64)
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

if __name__ == '__main__':
    device = "cuda:0"
    latent_size = 8
    actor = Student(obs_shape=31, latent_size=8, actions_shape=7, num_envs=4, device=device, model_cfg=None)
    predictions = torch.ones(6,7).uniform_(0,1).to(device) 
    labels = torch.ones(6,7).uniform_(0,1).to(device) 
    for i in range(3000):
        actor.add_transitions(predictions,labels)
        predictions = torch.ones(6,7).uniform_(0,1).to(device) 
        labels = torch.ones(6,7).uniform_(0,1).to(device) 
        if actor.fullfill:
            data_batch,labels_batch = actor.batch_sampler()
            #print(labels_batch.shape)
    #prio = 