#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project ：RL_with_pytorch_gym
@File    ：DDPG.py
@Author  ：Lee W
@Date    ：2021/12/14 下午9:57
"""

import gym
import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self, state):
        pass


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

    def forward(self, s, a):
        pass


class DDPG(object):
    def __init__(self):
        pass

    def sample(self):
        pass

    def choose_action(self, s):
        pass

    def learn(self):
        pass

    def store_transition(self, s, a, r, s_):
        pass
