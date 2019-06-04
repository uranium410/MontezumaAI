
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T

import random
import math
from collections import namedtuple

RESULT_DISPLAY_TIMES = 10
EPISODE_NUM = 1
ACTION_NUM = 18

env = gym.make('MontezumaRevenge-v0')

total_reward = 0


#DQN Network
#reference: http://torch.classcat.com/2018/05/15/pytorch-tutorial-intermediate-reinforcement-q-learning/
class DQN(nn.Module):
 
    def __init__(self):
        super(DQN, self).__init__()
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        #self.head = nn.Linear(448, 18)
        self.head = nn.Linear(32*23*23,18)
 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


#Epsilon Greedy Method
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
 
def Epsilon_Greedy(action, episode):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episode / EPS_DECAY)
    if sample > eps_threshold:
        return action
    else:
        return random.randrange(ACTION_NUM)

Transition = namedtuple('Transition',
                         ('state', 'action', 'next_state', 'reward'))

#Replay memory
class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def  push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(sel.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#Update DQN
def Update_Network():
    return

# Add representative state for first partition

network = DQN()

for episode in range(EPISODE_NUM):

    #reset episode
    next_obserbation = env.reset()
    total_reward = 0

    while True:
        # Determine the current partition

        # Update the set of visited partitions

        # Update the best candidate according to the
        # distance measure defined by Equation 6

        # Add a new rep. state every T_add seps

        # DQN()

        # EELearning()
        reshapeArray = np.zeros([210,25,3])
        inputGraph = np.concatenate((reshapeArray,next_obserbation),axis = 1)
        inputGraph = np.concatenate((inputGraph, reshapeArray),axis = 1)

        #conversion Inputdata for network
        nnInput = torch.FloatTensor([inputGraph]).transpose(1,3).transpose(2,3)

        Qtable = network.forward(nnInput)

        #select action from Qtable
        action = Epsilon_Greedy(np.argmax(Qtable.detach().numpy()),episode)

        #print(network.forward(tensorObs))

        # tentative
        action = env.action_space.sample()
        next_obserbation, reward, done, info = env.step(action)
        env.render()

        if done:
            break

    # Update all partitions' visit counts based on v
    if episode % RESULT_DISPLAY_TIMES == 0 or episode == EPISODE_NUM-1:
        print('Episode[',episode,']  total_reward:', total_reward)

print('training has finished.')