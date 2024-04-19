from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from RL.pc_rlafford.storage import RolloutStorage
from RL.pc_rlafford.module import ActorCritic
from RL.pc_vtafford.pcmodule import Network

import copy

class rlafford:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space  #TODO DEBUG

        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.device = device
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env=learn_cfg["nsteps"]
        self.learning_rate=learn_cfg["optim_stepsize"]
        self.TAN_lr = learn_cfg["tan_lr"]

        self.pc_latent_shape = self.model_cfg['pc_latent_size']
        self.pointclouds_shape = self.cfg_train["PCDownSampleNum"]
        self.tactile_shape = self.cfg_train["TDownSampleNum"] * 2
        self.prop_shape = self.cfg_train["proprioception_shape"]
        self.mpo_shape = 4 #points 3 affordence map 1
        self.total_input_shape = self.pc_latent_shape + self.prop_shape + self.mpo_shape

        #self.iter = cfg_train["load_iter"]
        #print(self.total_shape)
        # PPO components
        self.vec_env = vec_env
        self.task_name = vec_env.task_name
        self.actor_critic = ActorCritic(self.total_input_shape, self.action_space.shape,
                                               self.init_noise_std, self.model_cfg) #TODO debug observation
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, self.num_transitions_per_env, self.prop_shape,
                                      self.pointclouds_shape, self.mpo_shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)


        self.TAN = Network(4, 16).to(device)
        #self.TAN.load_state_dict(torch.load(os.path.join(self.model_dir,'TAN_model.pt')))
        #self.TAN.eval()

        self.TAN_optimizer = optim.Adam([
	        {'params': self.TAN.parameters(), 'lr': self.learning_rate,}
        	])
        self.TAN_criterion = nn.BCELoss()
        # PPO parameters
        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.num_transitions_per_env = self.num_transitions_per_env
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = False
        self.current_learning_iteration = 0
        self.model_dir = os.path.join(log_dir,self.task_name) 
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def get_labeled_pcs(current_pcs, current_tactiles):
        is_nonzero = (current_tactiles != 0).any(dim=2)
        
        current_tactiles[:,:,3][is_nonzero] = 1
        pcs_labeled = torch.cat(current_pcs, current_tactiles, dim=1)
        return pcs_labeled
    
    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()

        current_pcs_all = self.vec_env.get_pointcloud()

        current_pcs = torch.zeros((self.vec_env.num_envs, self.pointclouds_shape, 4), device = self.device) 
        current_pcs[:,:,0:3] = current_pcs_all[:,:self.pointclouds_shape,:]

        current_tactiles = torch.zeros((self.vec_env.num_envs, self.tactile_shape, 4), device = self.device) 
        current_tactiles[:,:,0:3] = current_pcs_all[:,self.pointclouds_shape:,:]  #(num_envs, tactile_shape, 3)

        current_prios = current_obs[:,:self.prop_shape]
        
        all_indices = set(torch.arange(self.vec_env.num_envs).numpy())

        current_mpos = torch.zeros((self.vec_env.num_envs, 4), device = self.device) 

        if self.is_testing:
            pass
            # self.test(os.path.join(self.model_dir,'sac_model_{}.pt'.format(self.iter)))
            # while True:
            #     with torch.no_grad():
            #         if self.apply_reset:
            #             current_obs = self.vec_env.reset()
            #         # Compute the action
            #         actions = self.actor_critic.act_inference(current_obs)
            #         next_obs, rews, dones, infos = self.vec_env.step(actions)
            #         current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()

                        current_pcs_all = self.vec_env.get_pointcloud()

                        current_pcs[:,:,0:3] = current_pcs_all[:,:self.pointclouds_shape,:]
                        current_tactiles[:,:,0:3] = current_pcs_all[:,self.pointclouds_shape:,:]

                        current_prios = current_obs[:,:self.prop_shape]

                    current_pcs[:,:,0:3] = current_pcs_all[:,:self.pointclouds_shape,:]
                    current_tactiles[:,:,0:3] = current_pcs_all[:,self.pointclouds_shape:,:]
                    current_prios = current_obs[:,:self.prop_shape]

                    is_zero = torch.all(current_tactiles == 0, dim=-1)
                    num_zero_points = torch.sum(is_zero, dim=-1)
                    zero_indices = torch.nonzero(num_zero_points == 128)[:, 0]
                    touch_indices = torch.tensor(list( all_indices - set(zero_indices.cpu().numpy())))

                    if len(touch_indices) > 0:
                        pcs_labeled = self.get_labeled_pcs(current_pcs, current_tactiles)

                        #shuffled = pointclouds[:, torch.randperm(pointclouds.size(1)), :]
                        current_pcs[touch_indices,:,:] = pcs_labeled[touch_indices, -self.pointclouds_shape:, :]
                        labels = current_pcs[:,:,3].clone()

                        current_pcs[:,:,3] = 1 
                        output = self.TAN(current_pcs)  
                        # print("output:", output)
                        #print("label:", label)
                        loss = self.TAN_criterion(output[touch_indices,:],labels[touch_indices,:])
                        #print(loss)

                        self.TAN_optimizer.zero_grad()
                        loss.backward()
                        self.TAN_optimizer.step()       
                        self.writer.add_scalar('Loss/pc', loss,it)      
                    else:
                        current_pcs[:,:,3] = 1 
                        output = self.TAN(current_pcs)

                    current_pcs[:,:,3] = output.detach()

                    top_values, top_indices = torch.topk(output.detach(), 1, dim=1)
                    current_mpos = torch.gather(current_pcs, 1, top_indices.unsqueeze(-1).expand(-1, -1, current_pcs.size(-1))).squeeze(1)

                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_prios, current_pcs, current_mpos)
                    # Step the vec_environment
                    next_obs, rews, dones,successes, infos = self.vec_env.step(actions)
                    rews += (current_mpos[:,3]-1) * 0.2

                    next_pcs = self.vec_env.get_pointcloud()

                    # pcs = next_states[:,next_obs.size(1):]#.view(self.vec_env.num_envs,1024,3)
                    # print(pcs.size())
                    # Record the transition
                    self.storage.add_transitions(current_prios, current_pcs, current_mpos, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    current_pcs_all.copy_(next_pcs)
                    current_obs.copy_(next_obs)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_prios, current_pcs, current_mpos)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.model_dir,'rla_model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.model_dir, 'rla_model_{}.pt'.format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
 
                #obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                prios_batch = self.storage.prios.view(-1, self.storage.prio_shape)[indices]
                pcs_batch = self.storage.pcs.view(-1, self.pointclouds_shape, 4)[indices]
                mpos_batch = self.storage.mpos.view(-1, 4)[indices]

                # else:
                #     states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                #print(states_batch.size())
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(prios_batch,
                                                                                                                       pcs_batch,
                                                                                                                       mpos_batch,
                                                                                                                       actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(
                        sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss
