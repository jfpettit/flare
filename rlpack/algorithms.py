import numpy as np
import gym
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

use_gpu = True if torch.cuda.is_available() else False

class REINFORCE:
    def __init__(self, env, model, gamma=0.99, optimizer=optim.Adam):
        self.gamma = gamma
        self.env = env
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        self.model = model
        if use_gpu:
            self.model.cuda()
        self.optimizer = optimizer(self.model.parameters())
    
    def update_(self):
        return_ = 0
        policy_loss = []
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)

        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / returns.std()

        logps = torch.stack(self.model.save_log_probs)
        if use_gpu: 
            logps = logps.cuda()
            returns = returns.cuda()
        self.optimizer.zero_grad()
        policy_loss = (-logps * returns) - (self.normal.entropy()*1e-4) if self.continuous else (-logps * returns)
        policy_loss.mean().backward()
        self.optimizer.step()
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]

    def action_choice(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        if not self.continuous:
            action_probabilities = self.model(state)
            m_ = torch.distributions.Categorical(action_probabilities)
            action = m_.sample()
            lp = m_.log_prob(action)

        elif self.continuous:
            mu, sig = self.model(state)
            self.normal = torch.distributions.Normal(mu, sig)
            action = self.normal.sample(self.env.action_space.shape)
            action = torch.clamp(action, float(self.env.action_space.low), float(self.env.action_space.high))
            lp = self.normal.log_prob(action)

        self.model.save_log_probs.append(lp)
        return action if self.continuous else action.item() 

    def exploit(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 
    
    def train_loop_(self, epochs, render=False, verbose=True, solved_threshold=None):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        for i in range(epochs):
            state, episode_reward = self.env.reset(), 0
            for s in range(0, 10000):
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

class A2C:
    def __init__(self, env, model, adv_fn=None, gamma=.99, lam=.95, steps_per_epoch=1000, optimizer='adam', standardize_rewards=True,
        policy_train_iters=80, val_loss=nn.MSELoss(), verbose=True):
        self.env = env
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        self.model=model
        if use_gpu:
            self.model.cuda()
        self.gamma = gamma
        self.lam = lam
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
            #self.adv_fn = self.aeu_.basic_adv_estimator
            self.adv_fn = self.aeu_.gae_lambda

        self.policy_train_iters = policy_train_iters
        self.val_loss = val_loss
        self.verbose = verbose
        

    def action_choice(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        if not self.continuous:
            action_probabilities, state_value = self.model(state)
            m_ = torch.distributions.Categorical(action_probabilities)
            action = m_.sample()
            lp = m_.log_prob(action)

        elif self.continuous:
            mu, sig, state_value = self.model(state)
            self.normal = torch.distributions.Normal(mu, torch.sqrt(sig))
            action = self.normal.sample(self.env.action_space.shape)
            action = torch.clamp(action, float(self.env.action_space.low), float(self.env.action_space.high))
            lp = self.normal.log_prob(action)

        self.model.save_log_probs.append(lp)
        self.model.save_values.append(state_value)
        self.model.save_actions.append(action)
        self.model.save_states.append(state)
        return action if self.continuous else action.item()    

    def update_(self):
        return_ = 0
        rewards = self.model.save_rewards
        returns = []
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.Tensor(returns)
        if self.standardize_rewards:
            if len(returns) > 1:
                if not torch.equal(returns, torch.full(returns.shape, returns[0])):
                    returns = (returns - returns.mean()) / returns.std()
        
        states_ = torch.stack(self.model.save_states)
        actions_ = torch.stack(self.model.save_actions)
        logprobs_ = torch.stack(self.model.save_log_probs)
        vals_ = torch.stack(self.model.save_values).squeeze()

        if use_gpu:
            states_ = states_.cuda()
            actions_ = actions_.cuda()
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            vals_ = vals_.cuda()

        #adv = returns - vals_
        adv = self.adv_fn(returns, vals_)
        pol_loss = -(logprobs_ * adv).mean()
        v_loss = 0.5 * self.val_loss(returns, vals_)
        if self.continuous:
            loss_fn = pol_loss + v_loss - (1e-2 * self.normal.entropy())
        else:
            loss_fn = pol_loss + v_loss
        self.optimizer.zero_grad()
        loss_fn.backward(retain_graph=True)
        self.optimizer.step()

        #for step in range(self.policy_train_iters):
        #    probs, values, entropy = self.model.eval(states_, actions_)
        #    v_loss = 0.5 * self.val_loss(returns, values)
            #adv = returns - values
            #loss_fn = -(logprobs_ * adv).mean() + (0.5 * self.val_loss(returns, values))
        #    self.optimizer.zero_grad()
        #    v_loss.backward(retain_graph=True)
        #    self.optimizer.step()
            
        
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]
        del self.model.save_actions[:]
        del self.model.save_states[:]


        
    def train_loop_(self, epochs, render=False, solved_threshold=None):
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

class PPO(A2C):
    def __init__(self, env, network, epsilon=0.2, adv_fn=None, gamma=.99, lam=.95, 
        steps_per_epoch=1000, optimizer=optim.Adam, standardize_rewards=True, lr=3e-4, target_kl=0.03,
        policy_train_iters=80, verbose=True, kl_max=None):
        self.env = env
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        self.model = network
        if use_gpu:
            self.model.cuda()
        self.epsilon = epsilon
        self.verbose = verbose

        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(gamma, lam)
            #self.adv_fn = self.aeu_.basic_adv_estimator
            #self.adv_fn = self.aeu_.td_residual
            self.adv_fn = self.aeu_.gae_lambda

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
        returns = torch.Tensor(returns)
        if self.standardize:
            if len(returns) > 1:
                if not torch.equal(returns, torch.full(returns.shape, returns[0])):
                    returns = (returns - returns.mean()) / returns.std()
        
        states_ = torch.stack(self.model.save_states)
        actions_ = torch.stack(self.model.save_actions)
        logprobs_ = torch.stack(self.model.save_log_probs)

        if use_gpu:
            states_ = states_.cuda()
            actions_ = actions_.cuda()
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()

        
        for step in range(self.policy_train_iters):
            probs, values, entropy = self.model.eval(states_, actions_)
            pol_ratio = torch.exp(probs - logprobs_.detach())
            approx_kl = (probs - logprobs_).mean()
            if approx_kl > 2.0 * self.target_kl:
                if self.verbose: print('\r Early stopping due to {} hitting max KL'.format(approx_kl), '\n', end="")
                sys.stdout.flush()
                break
            #adv = returns - values.detach()
            adv = self.adv_fn(returns, values)
            g_ = torch.clamp(pol_ratio, 1-self.epsilon, 1+self.epsilon) * adv
            if self.continuous:
                loss_fn = -torch.min(pol_ratio*adv, g_) + (0.5 * self.val_loss(returns, values)) + (1e-4 * self.normal.entropy())
            else:
                loss_fn = -torch.min(pol_ratio*adv, g_) + (0.5 * self.val_loss(returns, values))
            self.optimizer.zero_grad()
            loss_fn.mean().backward(retain_graph=True)
            self.optimizer.step()
        
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]
        del self.model.save_actions[:]
        del self.model.save_states[:]


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
    def __init__(self, env, network, buffer_size=10000, epsilon=1., gamma=.99, bs=128, optimizer=optim.RMSprop, 
        anneal_epsilon=True, epsilon_end=0.05, num_anneal_steps=200, loss=F.smooth_l1_loss):

        self.env = env

        self.policy_net = network
        self.target_net = network
        if use_gpu:
            self.policy_net.cuda()
            self.target_net.cuda()
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
            if use_gpu: state = state.cuda()
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

        if use_gpu:
            states = states.cuda()
            next_states = next_states.cuda()
        
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












