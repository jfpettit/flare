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
import rlpack.neural_nets as nets
import sys
from torch.nn.utils import clip_grad_value_, clip_grad_norm_

# figure whether a GPU is available. In the future, this will be changed to use torch.tensor().to(device) syntax to maximize GPU usage
use_gpu = True if torch.cuda.is_available() else False

# REINFORCE policy gradient. Currently only discrete actions are seriously supported. Continuous is under development. 
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
        returns = (returns - returns.mean()) / returns.std()

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



class PPO(A2C):
    # PPO subclasses A2C and the only function it needs set up is the update() function. Everything else stays the same.
    # OpenAI PPO blog post: https://openai.com/blog/openai-baselines-ppo/
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
        # the main difference between the PPO update function and A2C update function is in the loss function update rule
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

        # train on the collected batch of data self.policy_train_iters times
        for step in range(self.policy_train_iters):
            # get action log probabilities, values, and action distribution entropy for the states and actions in the batch
            # The models and their functions are defined in neural_nets.py
            probs, values, entropy = self.model.evaluate(states_, actions_)
            # compute the policy ratio between the new and old policies
            pol_ratio = torch.exp(probs - logprobs_.detach())
            # calculate the approximate KL divergence
            approx_kl = (logprobs_ - probs).mean()
            # if the policy update is too big, skip this update
            if approx_kl > 1.5 * self.target_kl:
                if self.verbose: print('\r Early stopping due to {} hitting max KL'.format(approx_kl), '\n', end="")
                sys.stdout.flush()
                break
            # estimate the advantage
            adv = returns - values.detach()
            # clamp the PPO-clip update to 1-epsilon, 1+epsilon
            g_ = torch.clamp(pol_ratio, 1-self.epsilon, 1+self.epsilon) * adv
            # zero optimizer
            self.optimizer.zero_grad()
            # calculate PPO loss function. min(policy ratio * advantage, clip(policy ratio, 1-epsilon, 1+epsilon)*advantage)
            loss_fn = -torch.min(pol_ratio*adv, g_).mean() + (0.5 * self.val_loss(returns, values))
            # backpropagate loss, step optimizer
            loss_fn.backward()
            self.optimizer.step()

        del self.model.save_rewards[:]
        del self.model.save_log_probs[:]
        del self.model.save_values[:]
        del self.model.save_actions[:]
        del self.model.save_states[:]