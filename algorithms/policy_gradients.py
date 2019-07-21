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

class SimplePolicyTraining:
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
    
    def train_loop_(self, render, epochs):
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
            self.end_episode_()
            self.env.close()
        return self.ep_reward, self.ep_length


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


class VanillaPolicyGradient:
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

        
    def train_loop_(self, render, epochs):
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
        return self.ep_reward, self.ep_length

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
    def __init__(self, env, buffer_size, epsilon, network, gamma=.99, bs=32, optimizer=optim.RMSprop, 
        anneal_epsilon=False, epsilon_end=None, num_anneal_steps=None, loss=F.smooth_l1_loss):

        self.env = env

        self.policy_net = network
        self.target_net = network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.experience_buffer = ExperienceReplayBuffer(buffer_size)
        self.eps = epsilon
        self.optimizer=optimizer(self.policy_net.parameters())

        if anneal_epsilon:
            self.eps_vals = np.linspace(self.eps, epsilon_end, num_anneal_steps)

        self.loss = loss
        self.bs = bs
        self.gamma = gamma

    def action_choice(self, state, i):
        i = i if i < len(self.eps_vals) else len(self.eps_vals)-1
        do_random = np.random.binomial(1, self.eps_vals[i])
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
        next_states = [ts_[i][2] for i in range(len(ts_)) if ts_[i][2] is not None]
        rewards = [ts_[i][3] for i in range(len(ts_))]
        actions = [ts_[i][1] for i in range(len(ts_))]
        mask_fn = lambda s: s is not None
        mask = [mask_fn(next_states[i]) for i in range(len(next_states))]
        
        qs = torch.tensor([torch.max(self.policy_net(state.float())) for state in states])
        
        next_state_vs = torch.zeros(self.bs)
        next_state_vs[mask] = torch.tensor([self.target_net(next_states[i].float()).max() for i in range(len(next_states))])

        expected_qs = (next_state_vs * self.gamma) + torch.tensor(rewards)

        self.optimizer.zero_grad()
        loss = self.loss(qs, expected_qs.unsqueeze(1).view(self.bs))
        loss.requires_grad = True
        loss.backward()
        clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def train_loop_(self, num_epochs, verbose=True, n=10):
        target_update = num_epochs//n
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
                        print('\rEpoch {} of {}'.format(i, num_epochs), '\tEp len: {}; Ep rew: {}\t'.format(s, ep_reward), end="")
                        sys.stdout.flush()
            if i % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.env.close()
        print('\n')
        return eprew, eplen








    




        


    