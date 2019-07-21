import numpy as np
import random
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from .utils import MlpPolicyUtils as mlpu
from .utils import MathUtils as mathu
from .utils import AdvantageEstimatorsUtils as aeu
from .utils import NetworkUtils as netu
import sys
from torch.nn.utils import clip_grad_value_

class NatureDQN(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super(NatureDQN, self).__init__()
        self.netutil = neu()

        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        height = self.netutil.conv2d_output_size(8, 4, h)
        width = self.netutil.conv2d_output_size(8, 4, w)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        height = self.netutil.conv2d_output_size(4, 2, height)
        width = self.netutil.conv2d_output_size(4, 2, width)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        height = self.netutil.conv2d_output_size(3, 1, height)
        width = self.netutil.conv2d_output_size(3, 1, width)
        

        self.fc1 = nn.Linear(height*width*64, 512)
        self.fc2 = nn.Linear(512, out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class FullyConnectedDQN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullyConnectedDQN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 512)
        self.fc5 = nn.Linear(512, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


transitions = namedtuple('transitions', 
    ('state', 'action', 'next_state', 'reward'))

class ExperienceReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.experience_memory = {}
        self.spot = 0

    def store(self, *args):
        if len(self.experience_memory) < self.buffer_size:
            self.experience_memory[self.spot] = None 
        self.experience_memory[self.spot] = [*args]
        self.spot = (self.spot+1) % self.buffer_size

    def sample(self, bs):
        return random.sample(list(self.experience_memory.values()), bs)

    def __len__(self):
        return len(self.experience_memory)


class DQNtraining:
    def __init__(self, env, network, buffer_size=10000, epsilon=0.9, gamma=.99, bs=128, optimizer=optim.RMSprop, 
        anneal_epsilon=True, epsilon_end=0.05, num_anneal_steps=200, loss=F.smooth_l1_loss):

        self.env = env

        self.policy_net = network
        self.target_net = network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.experience_buffer = ExperienceReplayBuffer(buffer_size)
        self.eps = epsilon
        self.optimizer=optimizer(self.policy_net.parameters())

        self.anneal = anneal_epsilon
        if anneal_epsilon:
            self.eps_vals = np.linspace(self.eps, epsilon_end, num_anneal_steps)

        self.loss = loss
        self.bs = bs
        self.gamma = gamma

    def action_choice(self, state, i=None):
        if self.anneal:
            i = i if i < len(self.eps_vals) else len(self.eps_vals)-1
            do_random = np.random.binomial(1, self.eps_vals[i])
        else:
            do_random = np.random.binomial(1, self.eps)
        if do_random:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(self.policy_net(state.float()))
        return action

    def update(self):
        if len(self.experience_buffer) < self.bs:
            return 

        ts_ = self.experience_buffer.sample(self.bs)

        states = [ts_[i][0] for i in range(len(ts_))]
        actions = [ts_[i][1] for i in range(len(ts_))]
        next_states = [ts_[i][2] for i in range(len(ts_)) if ts_[i][2] is not None]
        rewards = [ts_[i][3] for i in range(len(ts_))]
        
        mask_fn = lambda s: s is not None
        mask = [mask_fn(next_states[i]) for i in range(len(next_states))]
        
        qs = torch.tensor([torch.max(self.policy_net(state.float())) for state in states])
        
        next_state_vs = torch.zeros(self.bs)
        next_state_vs[mask] = torch.tensor([torch.max(self.target_net(next_state.float())) for next_state in next_states])

        expected_qs = (next_state_vs * self.gamma) + torch.tensor(rewards)

        self.optimizer.zero_grad()
        loss = self.loss(qs, expected_qs)
        loss.requires_grad = True
        loss.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def train_loop_(self, num_epochs, verbose=True, n=5):
        running_reward = 0
        eplen, eprew = [], []
        for i in range(num_epochs):
            obs, ep_reward = self.env.reset(), 0
            newobs = np.copy(obs)
            obs = torch.tensor(obs)
            done = False
            s = 0
            while not done:
                action = self.action_choice(obs, i)
                newobs, reward, done, _ = self.env.step(int(action))
                newobs = torch.tensor(newobs)
                self.experience_buffer.store(obs, torch.tensor([action]), newobs, torch.tensor([reward]))
                obs = newobs
                ep_reward += reward
                running_reward += (1-self.gamma) * ep_reward  + (self.gamma) * running_reward
                s += 1
                self.update()
                if done:
                    eplen.append(s)
                    eprew.append(ep_reward)
                    if verbose:
                        print('\rEpoch {} of {}'.format(i, num_epochs), '\t Episode reward: {}'.format(ep_reward), end="")
                        sys.stdout.flush()
            if i % n == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.env.close()
        print('\n')
        return eprew, eplen