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

class SimpleActorCritic(nn.Module):
    def __init__(self, in_size, out_size):
        super(SimpleActorCritic, self).__init__()
        self.save_log_probs = []
        self.save_old_log_probs = []
        self.save_rewards = []
        self.save_values = []

        self.fc1 = nn.Linear(in_size, 20)
        self.act = nn.Linear(20, out_size)
        self.val = nn.Linear(20, 1)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action = self.act(x)
        value = self.val(x)
        return F.softmax(action, dim=-1), value


class REINFORCE:
    def __init__(self, env, model, adv_fn=None, gamma=.99, lam=.95, lr=3e-4, steps_per_epoch=1000, optimizer='adam', standardize_rewards=True):
        self.env = env
        self.model=model
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.standardize_rewards = standardize_rewards

        if optimizer is not 'adam':
            self.optimizer=optimizer(self.model.parameters())
        else:
            self.optimizer = optim.Adam(self.model.parameters())
        
        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(self.gamma, self.lam)
            self.adv_fn = self.aeu_.basic_adv_estimator
        

    def action_choice(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        action_probabilities, state_value = self.model(state)
        #print(action_probabilities)
        m_ = torch.distributions.Categorical(action_probabilities)
        choice = m_.sample()
        self.model.save_log_probs.append(m_.log_prob(choice))
        self.model.save_values.append(state_value)
        return choice.item()    

    def update_(self):
        return_ = 0
        log_probs = self.model.save_log_probs
        values = self.model.save_values
        rewards = self.model.save_rewards
        policy_loss = []
        value_loss = []
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.Tensor(returns)
        if self.standardize_rewards:
            if len(returns) > 1:
                if not torch.equal(returns, torch.full(returns.shape, returns[0])):
                    returns = (returns - returns.mean()) / returns.std()
            
        ls = torch.as_tensor(np.array([range(0, len(returns)-1)]), dtype=torch.float)
        for index, obj in enumerate(zip(log_probs, values, returns)):
            log_prob, val, return_ = obj
            adv=self.adv_fn(return_, val, values[index-1], index)
            policy_loss.append(-log_prob * adv)
            value_loss.append((return_ - val)**2)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean() + torch.stack(value_loss).mean()
        loss.backward()
        self.optimizer.step()
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]

        
    def train_loop_(self, render, epochs, verbose=True):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        for i in range(epochs):
            state, episode_reward = self.env.reset(), 0
            for s in range(1, self.steps_per_epoch):
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
                
            running_reward += (1-self.gamma) * episode_reward  + (self.gamma) * running_reward
            self.update_()
            self.env.close()
            print('\rEpisode {} of {}'.format(i, epochs), '\t Episode reward: ', episode_reward, end='')
            sys.stdout.flush()
        return self.ep_reward, self.ep_length