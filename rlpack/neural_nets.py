import numpy as np
import random
from collections import namedtuple
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from rlpack.utils import MlpPolicyUtils as mlpu
from rlpack.utils import MathUtils as mathu
from rlpack.utils import AdvantageEstimatorsUtils as aeu
from rlpack.utils import NetworkUtils as netu
import sys
from torch.nn.utils import clip_grad_value_
import gym

class ActorCritic(nn.Module):
    def __init__(self, in_size, out_size, continuous=False):
        super(ActorCritic, self).__init__()
        self.save_log_probs = []
        self.save_states = []
        self.save_rewards = []
        self.save_values = []
        self.save_actions = []
        self.continuous = continuous

        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        if self.continuous:
            self.mu = nn.Linear(64, out_size)
            self.sigma = nn.Linear(64, out_size)
        else:
            self.layer3 = nn.Linear(64, out_size)
        self.val = nn.Linear(64, 1)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.val(x)
        if self.continuous:
            mean = torch.tanh(self.mu(x))
            #mean = self.mu(x)
            var = F.softplus(self.sigma(x))
            return mean, var, value
        else:
            action = self.layer3(x)
            return F.softmax(action, dim=-1), value

    def eval(self, state, action):
        if not self.continuous:
            action_ps, values = self.forward(state)
            action_dist = torch.distributions.Categorical(action_ps)
            action_logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
        elif self.continuous:
            mu, sig, values = self.forward(state)
            action_dist = torch.distributions.Normal(mu, sig)
            action_logprobs = action_dist.log_prob(action)
            entropy = action_dist.entropy()
        return action_logprobs, torch.squeeze(values), entropy

class PolicyNet(nn.Module):
    def __init__(self, in_size, out_size, continuous=False, is_val_func=False):
        super(PolicyNet, self).__init__()
        self.is_val = is_val_func
        self.save_log_probs = []
        self.save_rewards = []
        self.save_values = []
        self.continuous = continuous

        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        if self.continuous:
            self.mu = nn.Linear(64, out_size)
            self.sigma = nn.Linear(64, out_size)
        else:
            self.layer3 = nn.Linear(64, out_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        if self.is_val:
            return self.layer3(x)
        elif self.continuous:
            mean = torch.tanh(self.mu(x))
            var = F.softplus(self.sigma(x))
            return mean, var
        else:
            return F.softmax(self.layer3(x), dim=-1)



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

