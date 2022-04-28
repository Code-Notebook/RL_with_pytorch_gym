#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project ：RL_with_pytorch_gym 
@File    ：DQN.py
@Author  ：Lee W
@Date    ：2021/12/19 上午12:03 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

from util.image import draw_train_reward

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(N_STATES, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(32, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __int__(self, args=None):
        self.eval_net, self.target_net = Net(), Net()  # DQN需要使用两个神经网络
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.model_name = args.algorithm
        self.replay_buffer = list(np.zeros((MEMORY_CAPACITY, 4)))  # 初始化记忆库用,numpy生成一个(容量,4)大小的全0矩阵
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)  # torch 的优化器
        self.loss_func = nn.MSELoss()  # 误差公式

    def choose_action(self, state, epsilon):
        pass

    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass

    def save_model(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass

if __name__ == "__main__":
    pass

