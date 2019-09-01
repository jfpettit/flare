import numpy as np
import gym
import roboschool
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
from torch.nn.utils import clip_grad_value_, clip_grad_norm_

use_gpu = True if torch.cuda.is_available() else False

class REINFORCE:
    def __init__(self, env, model, gamma=0.99, optimizer=optim.Adam, steps_per_epoch=1000):
        self.gamma = gamma
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False
        self.env = env

        self.model = model
        self.entropies = []
        self.steps_per_epoch = steps_per_epoch
        if use_gpu:
            self.model.cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=1e-3)
    
    def action_choice(self, state):
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        if not self.continuous:
            action_probabilities = self.model(state)
            m_ = torch.distributions.Categorical(action_probabilities)
            action = m_.sample()
            lp = m_.log_prob(action)

        elif self.continuous:
            mu, sig_sq = self.model(state)
            sig_sq = F.relu(sig_sq)
            eps = torch.randn(mu.size())
            #eps = torch.distributions.Normal(mu, sig_sq.sqrt()).sample()
            action = (mu + sig_sq.sqrt()*eps).data
            self.normal = torch.distributions.Normal(mu, sig_sq.sqrt())
            #action = self.normal.sample(self.env.action_space.shape)
            #action = torch.clamp(action, float(self.env.action_space.low), float(self.env.action_space.high))
            lp = self.normal.log_prob(action)
            self.entropies.append(self.normal.entropy())

        self.model.save_log_probs.append(lp)
        return action if self.continuous else action.item() 

    def update_(self):
        return_ = 0
        policy_loss = []
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std()

        logps = torch.stack(self.model.save_log_probs)

        if use_gpu: 
            logps = logps.cuda()
            returns = returns.cuda()
        self.optimizer.zero_grad()
        policy_loss = ((-logps.detach() * returns) - self.normal.entropy()*1e-1) if self.continuous else (-logps * returns)
        policy_loss.mean().backward()
        #clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.entropies[:]
    
    def learn(self, epochs, render=False, verbose=True, solved_threshold=None):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        for i in range(epochs):
            state, episode_reward = self.env.reset(), 0
            for s in range(self.steps_per_epoch):
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
            if solved_threshold and len(self.ep_reward) > 100:
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length
            if verbose:
                print('\r Episode {} of {}'.format(i+1, epochs), '\t Episode reward:', episode_reward, end="")
            sys.stdout.flush()
            self.update_()
            self.env.close()
        print('\n')
        return self.ep_reward, self.ep_length

    def exploit(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 

class A2C:
    def __init__(self, env, model, adv_fn=None, gamma=.99, lam=.95, steps_per_epoch=1000, optimizer=optim.Adam, standardize_rewards=True,
        policy_train_iters=80, val_loss=nn.MSELoss(), verbose=True):
        self.env = env

        self.model=model
        if use_gpu:
            self.model.cuda()
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.standardize_rewards = standardize_rewards

        self.optimizer = optimizer(self.model.parameters())
        
        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(self.gamma, self.lam)
            self.adv_fn = self.aeu_.basic_adv_estimator

        self.policy_train_iters = policy_train_iters
        self.val_loss = val_loss
        self.verbose = verbose
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False
        

    def action_choice(self, state):
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        action_probabilities, state_value = self.model(state)
        m_ = torch.distributions.Categorical(action_probabilities)
        action = m_.sample()
        lp = m_.log_prob(action)

        self.model.save_log_probs.append(lp)
        self.model.save_values.append(state_value)
        self.model.save_actions.append(action)
        self.model.save_states.append(state)
        return action.item()    

    def update_(self):
        return_ = 0
        rewards = self.model.save_rewards
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.tensor(returns)
        if self.standardize_rewards:
            returns = (returns - returns.mean()) / returns.std()
        
        logprobs_ = torch.stack(self.model.save_log_probs)
        vals_ = torch.stack(self.model.save_values).squeeze().detach()

        if use_gpu:
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            vals_ = vals_.cuda()

        pol_loss, val_loss = [], []
        for i in range(len(returns)):
            adv = returns[i] - vals_[i]
            pol_loss.append(-logprobs_[i] * adv)
            val_loss.append(0.5 * self.val_loss(returns[i], vals_[i]))

        self.optimizer.zero_grad()
        loss = torch.stack(pol_loss).sum() + torch.stack(val_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]
        del self.model.save_actions[:]
        del self.model.save_states[:]
        
    def learn(self, epochs, render=False, solved_threshold=None):
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
            if i == 0:
                self.log_probs_old = self.model.save_log_probs
            self.update_()
            self.env.close()
            if solved_threshold and len(self.ep_reward) > 100:
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length
            if self.verbose:
                print('\rEpisode {} of {}'.format(i+1, epochs), '\t Episode reward: ', episode_reward, end='')
                sys.stdout.flush()
        print('\n')
        return self.ep_reward, self.ep_length

    def exploit(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities, value = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 

class PPO(A2C):
    def __init__(self, env, network, epsilon=0.2, adv_fn=None, gamma=.99, lam=.95, 
        steps_per_epoch=1000, optimizer=optim.Adam, standardize_rewards=True, lr=3e-4, target_kl=0.03,
        policy_train_iters=80, verbose=True, kl_max=None):
        self.env = env

        self.model = network
        if use_gpu:
            self.model.cuda()
        self.epsilon = epsilon
        self.verbose = verbose

        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(gamma, lam)
            self.adv_fn = self.aeu_.basic_adv_estimator

        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.standardize = standardize_rewards
        self.steps_per_epoch = steps_per_epoch
        self.target_kl = target_kl
        self.kl_max = kl_max

        self.policy_train_iters = policy_train_iters
        self.val_loss = nn.MSELoss()

    def update_(self):
        return_ = 0
        rewards = self.model.save_rewards
        policy_loss = []
        value_loss = []
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.tensor(returns)
        
        if self.standardize:
            if len(returns) > 1:
                if not torch.equal(returns, torch.full(returns.shape, returns[0])):
                    returns = (returns - returns.mean()) / returns.std()
        
        states_ = torch.stack(self.model.save_states).detach()
        actions_ = torch.stack(self.model.save_actions).detach()
        logprobs_ = torch.stack(self.model.save_log_probs).detach()

        if use_gpu:
            states_ = states_.cuda()
            actions_ = actions_.cuda()
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()

        
        for step in range(self.policy_train_iters):
            probs, values, entropy = self.model.eval(states_, actions_)
            pol_ratio = torch.exp(probs - logprobs_.detach())
            approx_kl = (logprobs_ - probs).mean()
            if approx_kl > 1.5 * self.target_kl:
                if self.verbose: print('\r Early stopping due to {} hitting max KL'.format(approx_kl), '\n', end="")
                sys.stdout.flush()
                break
            adv = returns - values.detach()
            g_ = torch.clamp(pol_ratio, 1-self.epsilon, 1+self.epsilon) * adv
            self.optimizer.zero_grad()
            loss_fn = -torch.min(pol_ratio*adv, g_).mean() + (0.5 * self.val_loss(returns, values))
            loss_fn.backward()
            self.optimizer.step()

        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]
        del self.model.save_actions[:]
        del self.model.save_states[:]