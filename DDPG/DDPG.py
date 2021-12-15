#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project ：RL_with_pytorch_gym
@File    ：DDPG.py
@Author  ：Lee W
@Date    ：2021/12/14 下午9:57
"""

import numpy as np
import torch
import torch.nn as nn

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.Tensor(action_bound)
        # layer
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.tanh(self.output(a))
        # 对action进行缩放，实际上 a in [-1, 1]
        scaled_a = a * self.action_bound
        return scaled_a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val


class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=1000, gamma=0.95, lr_a=0.001,
                 lr_c=0.002, batch_size=32):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        # Define Replay Buffer
        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0
        # Define actor net
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        # Define critic net
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # define optim
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # define loss function
        self.mse_loss = nn.MSELoss()

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices,:]

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):
        # soft update and hard update
        if self.replacement["name"] == "soft":
            tau = self.replacement["tau"]
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for a_l in a_layers:
                a_l[1].weight.data.mul_((1-tau))
                a_l[1].weight.data.add_(tau * self.actor.state_dict()[a_l[0]+".weight"])
                a_l[1].bias.data.mul_((1-tau))
                a_l[1].bias.data.add_(tau * tau * self.critic.state_dict()[a_l[0]+'.bias'])
            for c_l in c_layers:
                c_l[1].weight.data.mul_((1-tau))
                c_l[1].weight.data.add_(tau * self.critic.state_dict()[c_l[0]+'.weight'])
                c_l[1].bias.data.mul_((1-tau))
                c_l[1].bias.data.add_(tau * self.critic.state_dict()[c_l[0]+'.bias'])
        else:
            if self.t_replace_counter % self.replacement["rep_iter"] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for a_l in a_layers:
                    a_l[1].weight.data = self.actor.state_dict()[a_l[0] + ".weight"]
                    a_l[1].bias.data = self.actor.state_dict()[a_l[0]+".bias"]
                for c_l in c_layers:
                    c_l[1].weight.data = self.critic.state_dict()[c_l[0]+".weight"]
                    c_l[1].bias.data = self.critic.state_dict()[c_l[0]+".bias"]
            self.t_replace_counter += 1

        # sample from replay buffer
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim: self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim+1])
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])

        # train actor
        a = self.actor(bs)
        q = self.critic(bs, ba)
        a_loss = -torch.mean(q)
        self.actor_optim.zero_grad()
        a_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # train critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

    def store_transition(self, s, a, r, s_):
        transition =np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1
