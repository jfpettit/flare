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
    def __init__(self, in_size, out_size):
        super(ActorCritic, self).__init__()
        self.save_log_probs = []
        self.save_states = []
        self.save_rewards = []
        self.save_values = []
        self.save_actions = []

        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_size)
        self.val = nn.Linear(64, 1)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        value = self.val(x)
        action = self.layer3(x)
        return F.softmax(action, dim=-1), value

    def evaluate(self, state, action):
        action_ps, values = self.forward(state)
        action_dist = torch.distributions.Categorical(action_ps)
        action_logprobs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action_logprobs, torch.squeeze(values), entropy

class ContinuousPolicy(nn.Module):
    def __init__(self, in_size, out_size):
        super(ContinuousActorCritic, self).__init__()
        self.layer1 = nn.Linear(in_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.mu_out = nn.Linear(32, out_size)
        self.sig_sq_out = nn.Linear(32, out_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mu = self.mu_out(x)
        sig_sq = self.sig_sq_out(x)
        return mu, sig_sq, value


class ContinuousPolicyNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(ContinuousPolicyNet, self).__init__()
        self.layer1 = nn.Linear(in_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.mu_out = nn.Linear(32, out_size)
        self.sig_sq_out = nn.Linear(32, out_size)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh((self.layer2(x)))
        mu = self.mu_out(x)
        sig_sq = self.sig_sq_out(x)
        return mu, sig_sq

    def evaluate(self, states, actions):
        action_mu, action_sig = self.forward(states)
        action_dist = torch.distributions.normal.Normal(action_mu, action_sig)
        action_logprobs = action_dist.log_prob(actions).sum(1)
        self.entropy = action_dist.entropy()
        return action_logprobs, self.entropy

class PolicyNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_size)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return F.softmax(self.layer3(x), dim=-1)

    def evaluate(self, state, action):
        action_ps = self.forward(state)
        action_dist = torch.distributions.Categorical(action_ps)
        action_logprobs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action_logprobs, entropy

class ValueNet(nn.Module):
    def __init__(self, in_size):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(in_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

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

    def evaluate(self, states, actions):
        action_values = self.forward(states.float())
        return action_values.gather(1, actions)
