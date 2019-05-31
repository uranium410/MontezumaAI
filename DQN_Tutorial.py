import gym

import sys

import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

args = sys.argv

env = gym.make('MontezumaRevenge-v0')

EPISODEMAX = 20
FRAMEMAX = 10000

for eisode in range(EPISODEMAX):

    observation = env.reset()

    for frame in range(FRAMEMAX):

        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        env.render()

        if done:
            break

