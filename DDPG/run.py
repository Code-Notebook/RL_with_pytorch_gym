#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project ：RL_with_pytorch_gym 
@File    ：run.py
@Author  ：Lee W
@Date    ：2021/12/14 下午11:27 
"""

import gym
import time
import numpy as np
from DDPG import DDPG

if __name__ == "__main__":
    # hyper parameters
    VAR = 3
    MAX_EPISODES = 1000
    MAX_EP_STEP = 200
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name="soft", tau=0.01),
        dict(name="hard", rep_ter=600)
    ][0]
    ENV_NAME = "Pendulum-v1"
    RENDER = True

    # train
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    ddpg = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        replacement=REPLACEMENT,
        memory_capacity=MEMORY_CAPACITY
    )
    start_train_time = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEP):
            if RENDER:
                env.render()

            # add exploration noise
            a = ddpg.choose_action(s)
            # a = np.clip(np.random.normal(a, VAR), -2, 2)  # 在动作选择上添加随机噪声

            s_, r, done, _ = env.step(a)
            ddpg.store_transition(s, a, r/10, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= 0.9995  # 减弱action 的随机性
                ddpg.learn()
            s = s_
            ep_reward += r
            if j == MAX_EP_STEP - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                if ep_reward > -300:
                    RENDER = True
                    break
    print("Running time: ", time.time() - start_train_time)
