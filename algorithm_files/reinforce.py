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


class SimplePolicyNet(nn.Module):
    def __init__(self, in_size, out_size, is_val_func=False):
        super(SimplePolicyNet, self).__init__()
        self.is_val = is_val_func
        self.save_log_probs = []
        self.save_rewards = []
        self.save_values = []
        
        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        if self.is_val:
            return x
        return F.softmax(x, dim=-1)

class REINFORCE:
    def __init__(self, gamma, env, model, optimizer=None):
        self.gamma = gamma
        self.env = env
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters())
    
    def end_episode_(self):
        return_ = 0
        policy_loss = []
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / returns.std()
        for log_prob, return_ in zip(self.model.save_log_probs, returns):
            policy_loss.append(-log_prob * return_)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]

    def action_choice(self, state):
        state = np.asarray(state)
        print(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probabilities = self.model(state)
        m_ = torch.distributions.Categorical(action_probabilities)
        choice = m_.sample()
        self.model.save_log_probs.append(m_.log_prob(choice))
        return choice.item()
    
    def train_loop_(self, render, epochs, verbose=True):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        for i in range(epochs):
            state, episode_reward = self.env.reset(), 0
            for s in range(1, 10000):
                action = self.action_choice(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                self.model.save_rewards.append(reward)
                episode_reward += reward
                if done:
                    self.ep_length.append(s)
                    self.ep_reward.append(episode_reward)
                    break
                
            running_reward += 0.05 * episode_reward  + (1-0.05) * running_reward
            print('\r Episode {} of {}'.format(i, epochs), '\t Episode reward:', episode_reward, end="")
            sys.stdout.flush()
            self.end_episode_()
            self.env.close()
        return self.ep_reward, self.ep_length








    




        


    