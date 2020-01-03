import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from flare.utils import NetworkUtils as netu
import gym
from scipy.signal import lfilter

class MLP(nn.Module):
    def __init__(self, layer_szs, activations=torch.tanh, out_act=None, out_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations
        self.out_act = out_act
        self.out_squeeze = out_squeeze

        for i, l in enumerate(layer_szs[1:]):
            self.layers.append(nn.Linear(layer_szs[i], l))
        
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activations(l(x))
        
        if self.out_act is None:
            x = self.layers[-1](x)
        else:
            x = self.out_act(self.layers[-1](x))

        return x.squeeze() if self.out_squeeze else x

class CategoricalPolicy(nn.Module):
    def __init__(self, state_features, hidden_sizes, activation, out_activation, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.mlp = MLP([state_features]+list(hidden_sizes)+[action_dim], activations=activation)

    def forward(self, x, a=None):
        logits = self.mlp(x)

        policy = torch.distributions.Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi

class GaussianPolicy(nn.Module):
    def __init__(self, state_features, hidden_sizes, activation, out_activation, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mlp = MLP([state_features]+list(hidden_sizes)+[action_dim], activations=activation, out_act=out_activation)
        self.logstd = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        mu = self.mlp(x)
        std = torch.exp(self.logstd)
        policy = torch.distributions.Normal(mu, std)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

class FireActorCritic(nn.Module):
    def __init__(self, state_features, action_space, hidden_sizes=(32, 32), activation=torch.tanh, out_activation=None, policy=None):
        super(FireActorCritic, self).__init__()

        if policy is None and isinstance(action_space, gym.spaces.Box):
            self.policy = GaussianPolicy(state_features, hidden_sizes, activation, out_activation, action_space.shape[0])
        elif policy is None and isinstance(action_space, gym.spaces.Discrete):
            self.policy = CategoricalPolicy(state_features, hidden_sizes, activation, out_activation, action_space.n)
        else:
            self.policy = policy(state_features, hidden_sizes, activation, out_activation, action_space)

        self.value_f = MLP([state_features]+list(hidden_sizes)+[1], activations=activation, out_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x)
        value = self.value_f(x)

        return pi, logp, logp_pi, value

class ActorCritic(nn.Module):
    def __init__(self, in_size, out_size):
        super(ActorCritic, self).__init__()
        self.layer1 = nn.Linear(in_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, out_size)
        self.val = nn.Linear(32, 1)


    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        value = self.val(x)
        action = self.layer3(x)
        return action, value

    def evaluate_acts(self, states):
        return self.forward(states)


    def buffer_init(self, epoch_interaction_size, gamma=0.99, lam=0.95):
        self.size = epoch_interaction_size
        self.save_log_probs = []
        self.save_states = []
        self.save_rewards = np.zeros(self.size, dtype=np.float32)
        self.save_values = np.zeros(self.size, dtype=np.float32)
        self.old_log_probs = np.zeros(self.size, dtype=np.float32)
        self.save_value_tensors = []
        self.save_actions = []
        
        self.gamma = gamma
        self.lam = lam

        self.adv_record = np.zeros(self.size, dtype=np.float32)
        self.ret_record = np.zeros(self.size, dtype=np.float32)

        self.ptr, self.strt = 0, 0

    def store(self, state, action, reward, value, logprob):
        assert self.ptr < self.size
        self.save_states.append(state)
        self.save_log_probs.append(logprob)
        self.save_rewards[self.ptr] = reward
        self.save_values[self.ptr] = value
        self.save_actions.append(action)
        self.save_value_tensors.append(value.clone().detach().requires_grad_(True))

        self.ptr += 1

    def discount_cumulative_sum(self, x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def end_traj(self, last_val=0):
        traj_slice = slice(self.strt, self.ptr)

        rews = np.append(self.save_rewards[traj_slice], last_val)
        vals = np.append(self.save_values[traj_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_record[traj_slice] = self.discount_cumulative_sum(deltas, self.gamma * self.lam)
        self.ret_record[traj_slice] = self.discount_cumulative_sum(rews, self.gamma*self.lam)[:-1]
        #self.adv_record[traj_slice] = self.discount_cumulative_sum(rews[:-1] - vals[:-1], self.gamma*self.lam)

        self.strt = self.ptr

    def gather(self):
        assert self.ptr == self.size, 'Buffer must be full before you gather.'

        self.ptr, self.strt = 0, 0

        self.adv_record = (self.adv_record - self.adv_record.mean()) / (self.adv_record.std() + 1e-8)

        return [self.save_states, self.save_actions, torch.tensor(self.adv_record), torch.tensor(self.ret_record), self.save_log_probs, self.save_value_tensors]

    def clear_mem(self):
        self.save_log_probs = []
        self.save_states = []
        self.save_actions = []
        self.save_value_tensors = []

        self.save_rewards = np.zeros(self.size, dtype=np.float32)
        self.save_values = np.zeros(self.size, dtype=np.float32)

        self.adv_record = np.zeros(self.size, dtype=np.float32)
        self.ret_record = np.zeros(self.size, dtype=np.float32)

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
