# import needed packages
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
from rlpack.utils import Buffer
import sys
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from rlpack.neural_nets import PolicyNet, ValueNet
from gym.spaces import Box, Discrete
import math

# figure whether a GPU is available. In the future, this will be changed to use torch.tensor().to(device) syntax to maximize GPU usage
use_gpu = True if torch.cuda.is_available() else False

# REINFORCE policy gradient. Currently only discrete actions are seriously supported. Continuous is under development. 
class REINFORCE:
    def __init__(self, env, model, gamma=0.99, lam=0.95, optimizer=optim.Adam, steps_per_epoch=1000):
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
        # take in the state, convert it to a torch tensor with dtype=float
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        if not self.continuous:
            # get action probabilities from the network
            action_probabilities = self.model(state)
            # parameterize a categorical distribution over actions with the action probabilities from the network
            m_ = torch.distributions.Categorical(action_probabilities)
            # sample an action from the categorical distribution
            action = m_.sample()
            # compute the log probability of that action
            lp = m_.log_prob(action)

        elif self.continuous:
            mu, sig_sq = self.model(state)
            sig_sq = F.relu(sig_sq)
            eps = torch.randn(mu.size())
            action = (mu + sig_sq.sqrt()*eps).data
            self.normal = torch.distributions.Normal(mu, sig_sq.sqrt())
            lp = self.normal.log_prob(action)
            self.entropies.append(self.normal.entropy())

        # save the log probability of the selected action and return the action picked
        self.model.save_log_probs.append(lp)
        return action if self.continuous else action.item() 

    def update_(self):
        return_ = 0
        policy_loss = []
        returns = []
        # compute returns for the last collected batch
        for reward in self.model.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)

        # normalize the returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # convert log probs to stacked torch tensor
        logps = torch.stack(self.model.save_log_probs)

        if use_gpu: 
            logps = logps.cuda()
            returns = returns.cuda()

        # zero the optimizer gradient
        self.optimizer.zero_grad()
        # compute the REINFORCE policy loss. Just negative log probability times returns
        policy_loss = ((-logps.detach() * returns) - self.normal.entropy()*1e-1) if self.continuous else (-logps * returns)
        # take the mean of the loss and backpropagate
        policy_loss.mean().backward()
        # step the optimizer
        self.optimizer.step()
        # delete saved batch
        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.entropies[:]
    
    def learn(self, epochs, render=False, verbose=True, solved_threshold=None):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        # train for the number of epochs input. currently, one epoch is one episode in the environment
        for i in range(epochs):
            # reset environment and episode reward
            state, episode_reward = self.env.reset(), 0
            # limit the amount of interaction allowed in one epoch
            for s in range(self.steps_per_epoch):
                # pick an action
                action = self.action_choice(state)
                # step the environment 
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                # save the collected reward
                self.model.save_rewards.append(reward)
                episode_reward += reward
                if done:
                    # if the episode is over, save how long it was and reward earned. break for next episode
                    self.ep_length.append(s)
                    self.ep_reward.append(episode_reward)
                    break
                
            if solved_threshold and len(self.ep_reward) > 100:
                # if the reward over last 100 episodes is greater than the threshold, break training
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length
            if verbose:
                print('\r Episode {} of {}'.format(i+1, epochs), '\t Episode reward:', episode_reward, end="")
            sys.stdout.flush()
            # update every epoch
            self.update_()
            self.env.close()
        print('\n')
        return self.ep_reward, self.ep_length

    def exploit(self, state):
        # used to evaluate the agent after training
        # convert state to torch tensor dtype=float
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        # deterministically choose the most likely action instead of paramaterizing a categorical distribution and sampling from it
        action_probabilities = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 

class A2C:
    def __init__(self, env, pol_model, val_model, adv_fn=None, gamma=.99, lam=.97, steps_per_epoch=1000, optimizer=optim.Adam, 
        standardize_rewards=True, value_train_iters=80, val_loss=nn.MSELoss(), verbose=True):
        self.env = env
        self.policy_net = pol_model
        self.value_net = val_model
        if use_gpu:
            self.policy_net.cuda()
            self.valu_net.cuda()

        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.standardize_rewards = standardize_rewards

        self.policy_optimizer = optimizer(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optimizer(self.value_net.parameters(), lr=1e-3)
        
        # advantage estimator setup
        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(self.gamma, self.lam)
            self.adv_fn = self.aeu_.basic_adv_estimator

        self.value_train_iters = value_train_iters
        self.val_loss = val_loss
        self.verbose = verbose
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        if isinstance(env.action_space, Box):
            action_space = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_space = 1
        self.buffer = Buffer(env.observation_space.shape[0], action_space, steps_per_epoch, gamma=gamma, lam=lam)
        

    def calc_log_probs(self, action_mu, action_logstd, action):
        lp = -0.5 * (((action-action_mu)/(torch.exp(action_logstd)+1e-8))**2 + 2*action_logstd + torch.log(torch.tensor(2*math.pi)))
        return lp

    def action_choice(self, state):
        # convert state to torch tensor, dtype=float
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        # same as with REINFORCE, sample action from categorical distribution paramaterizes by the network's output of action probabilities
        if isinstance(self.env.action_space, Box):
            action_mu, action_logstd = self.policy_net(state)
            state_value = self.value_net(state)
            action = action_mu + torch.randn(self.env.action_space.shape) * torch.exp(action_logstd)
            lp = -0.5 * (((action-action_mu)/(torch.exp(action_logstd)+1e-8))**2 + 2*action_logstd + torch.log(torch.tensor(2*math.pi)))
        elif isinstance(self.env.action_space, Discrete):
            action_probabilities = self.policy_net(state)
            state_value = self.value_net(state)
            m_ = torch.distributions.Categorical(action_probabilities)
            action = m_.sample()
            lp = m_.log_prob(action)
        
        self.approx_entropy = -lp.mean()
        return action, lp.mean(), state_value    

    def update_(self):
        # update function is more or less the same as REINFORCE. I'll highlight the important differences. 

        states, actions, advs, returns, logprobs_ = self.buffer.gather()

        if use_gpu:
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            advs = advs.cuda()

        #action_mu, action_logstd = self.policy_net(states.detach())
        #logprobs_ = self.calc_log_probs(action_mu.detach(), action_logstd.detach(), actions.detach())
        logprobs_, entropy = self.policy_net.evaluate(states.detach(), actions.detach())

        self.policy_optimizer.zero_grad()
        #logprobs_ = logprobs_.detach().requires_grad_()
        pol_loss = -(logprobs_ * advs).mean()
        pol_loss.backward()
        self.policy_optimizer.step()

        # estimate advantage and policy and value loss for each sample in the batch
        for _ in range(self.value_train_iters):
            vals = self.value_net(states)
            val_loss = (self.val_loss(torch.squeeze(returns), torch.squeeze(vals)))

            self.value_optimizer.zero_grad()
            # compute A2C loss, it is sum(policy loss) + sum(value loss). Policy loss is negative log probabilities times advantage and 
            # value loss is mean squared error loss
            loss = val_loss
            loss.backward()
            self.value_optimizer.step()

        return pol_loss.detach(), val_loss.detach()
        
    def learn(self, epochs, solved_threshold=None, render_epochs=None, render_frames=1000):
        # functionality and stuff happening in the learn function is the same as REINFORCE
        self.ep_length = []
        self.ep_reward = []
        state, episode_reward, episode_len = self.env.reset(), 0, 0
        for i in range(epochs):
            epochrew = []
            epochlen = []
            for s in range(self.steps_per_epoch):

                action, logprob, value = self.action_choice(state)
                state, reward, done, _ = self.env.step(action.detach().numpy())

                self.buffer.push(state, action, reward, value.detach().numpy(), logprob)
                episode_reward += reward
                episode_len += 1

                if done or s == self.steps_per_epoch-1:
                    state = torch.from_numpy(state).float()
                    last_value = reward if done else self.value_net(state).detach().numpy()
                    self.buffer.end_trajectory(last_value)
                    if done:
                        self.ep_length.append(episode_len)
                        self.ep_reward.append(episode_reward)
                        epochlen.append(episode_len)
                        epochrew.append(episode_reward)
                    
                    state, reward, done, episode_reward, episode_len = self.env.reset(), 0, False, 0, 0
            
            pol_loss, val_loss = self.update_()
            self.buffer.reset()
            
            if solved_threshold and len(self.ep_reward) > 100:
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length
            
            if self.verbose:
                print(
                    '-------------------------------\n'
                    'Epoch {} of {}\n'.format(i+1, epochs), 
                    'EpRewardMean: {}\n'.format(np.mean(epochrew)),
                    'EpRewardStdDev: {}\n'.format(np.std(epochrew)), 
                    'EpRewardMax: {}\n'.format(np.max(epochrew)), 
                    'EpRewardMin: {}\n'.format(np.min(epochrew)), 
                    'EpLenMean: {}\n'.format(np.mean(epochlen)),
                    'PolicyEntropy: {}\n'.format(self.approx_entropy),
                    'PolicyLoss: {}\n'.format(pol_loss),
                    'ValueLoss: {}\n'.format(val_loss),
                    '\n', end='')
            if render_epochs is not None and i in render_epochs:
                state = self.env.reset()
                for i in range(render_frames):
                    action, logprob, value = self.action_choice(state)
                    state, reward, done, _ = self.env.step(action.detach().numpy())
                    self.env.render()
                    if done:
                        state = self.env.reset()
                self.env.close()
        print('\n')
        return self.ep_reward, self.ep_length

    def exploit(self, state):
        # this is also the same as REINFORCE
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities, value = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 

class PPO(A2C):
    # PPO subclasses A2C and the only function it needs set up is the update() function. Everything else stays the same.
    # OpenAI PPO blog post: https://openai.com/blog/openai-baselines-ppo/
    def __init__(self, env, policy_network, value_network, epsilon=0.2, adv_fn=None, gamma=.99, lam=.95, 
        steps_per_epoch=1000, optimizer=optim.Adam, lr=3e-4, target_kl=0.03,
        policy_train_iters=80, value_train_iters=80, verbose=True, entropy_coeff=0):
        self.env = env

        self.policy_net = policy_network
        self.value_net = value_network

        if use_gpu:
            self.policy_net.cuda()
            self.value_net.cuda()

        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff

        self.verbose = verbose

        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(gamma, lam)
            self.adv_fn = self.aeu_.basic_adv_estimator

        self.policy_optimizer = optimizer(self.policy_net.parameters())
        self.value_optimizer = optimizer(self.value_net.parameters())

        
        self.steps_per_epoch = steps_per_epoch
        self.target_kl = target_kl

        self.policy_train_iters = policy_train_iters
        self.value_train_iters = value_train_iters
        self.val_loss = nn.MSELoss()

        if isinstance(env.action_space, Box):
            action_space = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_space = 1
        self.buffer = Buffer(env.observation_space.shape[0], action_space, steps_per_epoch, gamma=gamma, lam=lam)

    def update_(self):
        # the main difference between the PPO update function and A2C update function is in the loss function update rule
        states, actions, advs, returns, logprobs_ = self.buffer.gather()

        if use_gpu:
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            advs = advs.cuda()

        logprobs_ = logprobs_.detach()
        states = states.detach()
        actions = actions.detach()

        # train on the collected batch of data self.policy_train_iters times
        for step in range(self.policy_train_iters):
            # get action log probabilities, values, and action distribution entropy for the states and actions in the batch
            # The models and their functions are defined in neural_nets.py
            probs, entropy = self.policy_net.evaluate(states, actions)
            #print('probs {} \n entropy {}'.format(probs, entropy))
            # compute the policy ratio between the new and old policies
            pol_ratio = torch.exp(probs - logprobs_) * advs
            #print('policy ratio {}'.format(pol_ratio))
            # calculate the approximate KL divergence
            approx_kl = 0.5 * ((logprobs_ - probs)**2).mean()
            
            g_ = torch.clamp(pol_ratio, 1-self.epsilon, 1+self.epsilon) * advs
            pol_loss = -(torch.min(pol_ratio, g_) - (entropy*self.entropy_coeff)).mean()

            values = self.value_net(states)
            val_loss = (self.val_loss(torch.squeeze(values), torch.squeeze(returns)))
            # if the policy update is too big, skip this update
            #if approx_kl > 1.5 * self.target_kl:
                #if self.verbose: print('\r Warning: {} hit max KL of {} on policy update {}'.format(approx_kl, self.target_kl, step),
                #'\n', end="")
                #sys.stdout.flush()
                #break
            # estimate the advantage
            # clamp the PPO-clip update to 1-epsilon, 1+epsilon
            # zero optimizer
            self.policy_optimizer.zero_grad()
            # calculate PPO loss function. min(policy ratio * advantage, clip(policy ratio, 1-epsilon, 1+epsilon)*advantage)
            # backpropagate loss, step optimizer
            pol_loss.backward()
            #nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            #for step in range(self.value_train_iters):
            self.value_optimizer.zero_grad()
            val_loss.backward()
            self.value_optimizer.step()

        return pol_loss.detach(), val_loss.detach()
