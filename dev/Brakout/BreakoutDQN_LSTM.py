# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pickle

RENDER = True

EPISODES_NUM = 500

BATCH_SIZE = 32
GAMMA = 0.999


#epsilon_greedy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

env = gym.make('Breakout-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Type:",device)
print("rendering:", RENDER)
print("Batch size:", BATCH_SIZE)
print("GAMMA:", GAMMA)


# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN algorithm

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.linear = nn.Linear(linear_input_size, 128)

        self.lstm = nn.LSTM(input_size=128,hidden_size=32,batch_first=True)
        self.c = torch.zeros(32)

        #self.head = nn.Linear(linear_input_size, outputs)
        self.head = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        x, self.c = self.lstm(x.view(1,1,128))
        x = x.view(x.size(0), -1)
        return self.head(x)


# Input extraction

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(next_obserbation):
        nnInput = torch.FloatTensor([inputGraph])

        #conversion Inputdata for network
        nnInput = nnInput.transpose(1,3).to(device)#.transpose(2,3)
        a,b,c = torch.chunk(nnInput, 3, dim = 1)
        nnInput = (a + b + c)/3

        return nnInput


inputGraph = env.reset()
#plt.figure()
#plt.imshow(get_screen(inputGraph).cpu().squeeze(0).permute(1, 2, 0).numpy(),
#           interpolation='none')
#plt.title('Example extracted screen')
#plt.show()


# Training
init_screen = get_screen(inputGraph)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #print("policy_net")
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #print("random")
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_rewards = []


def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_forplot = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_forplot.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_forplot) >= 100:
        means = rewards_forplot.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

#Main loop

continue_epsode = True

num_episodes = EPISODES_NUM
while continue_epsode:
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        inputGraph = env.reset()
        last_screen = get_screen(inputGraph)
        current_screen = last_screen
        state = current_screen

        totalReward = 0
        inputGraph, reward, done, beforeLives = env.step(0)
        

        for t in count():
            # Select and perform an action
            action = select_action(state)
            #print(action.item())
            inputGraph, reward, done, lives = env.step(action.item())
            if not beforeLives == lives:
                reward -= 1
            beforeLives = lives


            reward = torch.tensor([reward], device=device)
            if reward > 0:
                totalReward += reward.item()

            if RENDER:
                env.render()

            # Observe new stat
            last_screen = current_screen
            current_screen = get_screen(inputGraph)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_rewards.append(totalReward)
                plot_rewards()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    while True:
        print("continue?[y/n]")
        ans = input()
        if ans == "y":
            print("input episodes num to add:")
            num_episodes = int(input())
            break
        elif ans == "n":
            continue_epsode = False
            while True:
                print("save graph dat?[y/n]")
                ans = input()
                if ans == "y":
                    print("Input FileName")
                    fn = input()
                    with open(fn, 'wb') as f:
                        pickle.dump(episode_rewards,f)
                        break
                elif ans == "n":
                    break
            break
        else:
            print("please answer y/n")

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()