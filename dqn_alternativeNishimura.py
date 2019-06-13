

import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import deque
from collections import namedtuple

import cv2
from skimage.color import rgb2gray
from skimage.transform import resize


RESULT_DISPLAY_TIMES = 1
EPISODE_NUM = 300
MAX_MEMORY = 20000
BATCH_SIZE = 32
DISCOUNT_RATE = 0.99
TARGET_UPDATE = 10

resize_x = 84
resize_y = 84

file_history = open("history.txt", "w")
file_history.close()
file_history = open("graph.txt", "w")
file_history.close()

# GPU
device = torch.device('cuda:0')
device = torch.device('cpu')


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay memory 
class ReplayMemory(object):
 
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, *args):
        # Save transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
 
    def __len__(self):
        return len(self.memory)



#Agent
#DQN_reference: http://torch.classcat.com/2018/05/15/pytorch-tutorial-intermediate-reinforcement-q-learning/
# http://torch.classcat.com/2018/05/15/pytorch-tutorial-intermediate-reinforcement-q-learning/


class Agent(nn.Module):
 
    def __init__(self):
        super(Agent, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(7 * 7 * 64, 512)
        self.head2 = nn.Linear(512, 18)
                
    def forward(self, x):
        # x = Input(shape=(resize_y, resize_x, 1))
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        model = self.head2(x)
        return model


# Create network
q_net = Agent().to(device)
t_net = Agent().to(device)
t_net.load_state_dict(q_net.state_dict())
t_net.eval()

# Replay memory
memory = ReplayMemory(MAX_MEMORY)


def getAction(state, epsilon, env):

    # print('\n')
    # print(q_net(state))
    # print(q_net(state).max(1)[1].view(1, 1))

    # epsilon-greedy
    if np.random.rand() <= epsilon:
        act = [env.action_space.sample()]
        return torch.LongTensor([act]).to(device)

    # state = state[np.newaxis,:,:,:]
    return q_net(state).max(1)[1].view(1, 1).to(device)

def PrintInfo(episode, reward, epsilon):
    msg = "[episode:" + str(episode) + " } {reward:" + str(reward) + "} {epsilon:" + str(epsilon) + "}]"
    print(msg)


def Preprocess(image, _resize_x=resize_x, _resize_y=resize_y):
    image = resize(image, (_resize_x, _resize_y))
    return image


optimizer = optim.RMSprop(q_net.parameters())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
 
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).view(-1, 1).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
 
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = q_net(state_batch).gather(1, action_batch)
 
    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = t_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DISCOUNT_RATE) + reward_batch
 
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
 
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():
             
    # epsilon = 1.0
    eps_start = 1.0
    eps_end = 0.01
    eps_endt = 100000 # 1000000 #100000 good
    # eps_decay = 200 * EPISODE_NUM
    learn_start = 100000 # 50000
    steps_done = 0
    # min_epsilon = 0.1
    # reduction_epsilon = (1. - min_epsilon) / EPISODE_NUM

    # 18
    # 9
    # 18
    # 4
    # 6

    env_name = 'MontezumaRevenge-v0'
    # env_name = 'MsPacman-v0'
    # env_name = 'Centipede-v0'
    # env_name = 'Breakout-v0'
    # env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)
   

    continue_epsode = True

    num_episodes = EPISODE_NUM
    
    while continue_epsode:
        for episode in range(num_episodes):

            #reset episode
            env.reset()
            last_screen = env.render(mode='rgb_array')
            current_screen = env.render(mode='rgb_array')
            current_observation = current_screen - last_screen
            current_observation = Preprocess(current_observation, resize_x, resize_y)
            done = False
            total_reward = 0

            while not done:
 
                #Input
                nnInput = torch.FloatTensor([current_observation])
                nnInput = nnInput.transpose(1,3)
                nnInput = nnInput.transpose(2,3)
                nnInput = nnInput.to(device)

                # update epsilon
                # epsilon -= reduction_epsilon
                # epsilon = min_epsilon + (1. - min_epsilon) * (EPISODE_NUM - episode) / EPISODE_NUM
                # epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
                epsilon = eps_end + max(0, (eps_start - eps_end) * (eps_endt - max(0, steps_done - learn_start)) / eps_endt)

                # Choose an action
                action = getAction(nnInput, epsilon, env)
                # print(action)

                # Observe new state
                _, reward, done, info = env.step(action)

                last_screen = current_screen
                current_screen = env.render(mode='rgb_array')

                next_observation = current_screen - last_screen
                # Get learning faster
                next_observation = Preprocess(next_observation, resize_x, resize_y)

                nnInput2 = torch.FloatTensor([next_observation])
                nnInput2 = nnInput.transpose(1,3)
                nnInput2 = nnInput.transpose(2,3)
                nnInput2 = nnInput.to(device)

                # reward = torch.tensor([reward], device=device)
                total_reward += reward
                        
                # Store the transition in memory
                reward = torch.FloatTensor([reward]).to(device)
                memory.push(nnInput, action, nnInput2, reward)
                                   
                # Update state
                current_observation = next_observation

                # Perform one step of the optimization (on the target network)
                # if(steps_done % SKIPPING_FRAME == 0):
                optimize_model()
              
                # Visualize
                env.render()

                # print(steps_done)

                # print(nnInput.is_cuda)
                     
                # print(nnInput.size())
                # action = agent.forward(nnInput)
                # print(agent.forward(tensorObs))

                steps_done += 1

            # Update the target network
            if episode % TARGET_UPDATE == 0:
                t_net.load_state_dict(q_net.state_dict())

            # Update all partitions' visit counts based on v
            if episode % RESULT_DISPLAY_TIMES == 0 or episode == EPISODE_NUM-1:
                print('Episode[',episode,']  total_reward:', total_reward, '  epsilon:', epsilon, ' steps_done:', steps_done)

            data = str(total_reward) + '\n'

            file_history = open("graph.txt", "a")
            file_history.write(data)
            file_history.close()

            if(episode + 1 == num_episodes):
                while True:
                    print("continue?[y/n]")
                    ans = input()
                    if ans == "y":
                        print("input episodes num to add:")
                        num_episodes = int(input())
                        break
                    elif ans == "n":
                        continue_epsode = False
                        break
                    else:
                        print("please answer y/n")



    print('training has finished.')

    # Show graph
    f = open("graph.txt", "r")
    x = []
    y = []
    for i, line in enumerate(f):
        x.append(i)
        y.append(float(line))
    plt.plot(x,y)
    # Save graph
    plt.savefig("dqn.png")
    plt.show()
    
    f.close()


    env.close()
    
if __name__ == '__main__':
    main()