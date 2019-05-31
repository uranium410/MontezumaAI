


import gym
import numpy as numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as finished

RESULT_DISPLAY_TIMES = 10

EPISODE_NUM = 50

env = gym.make('MontezumaRevenge-v0')

total_reward = 0

# Add representative state for first partition

for episode in range(EPISODE_NUM):

    #reset episode
    env.reset()
    total_reward = 0

    while True:
        # Determine the current partition

        # Update the set of visited partitions

        # Update the best candidate according to the
        # distance measure defined by Equation 6

        # Add a new rep. state every T_add seps



        # DQN()

        # EELearning()
        action = 5
        next_obserbation, reward, done, info = env.step(action)
        #env.render()

        if done:
            break

    # Update all partitions' visit counts based on v
    if episode % RESULT_DISPLAY_TIMES == 0 or episode == EPISODE_NUM-1:
        print('Episode[',episode,']  total_reward:', total_reward)

print('training has finished.')