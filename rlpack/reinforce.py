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
from rlpack import utils
import sys
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from rlpack.neural_nets import PolicyNet, ValueNet
from gym.spaces import Box, Discrete
import math

use_gpu = True if torch.cuda.is_available() else False

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
            act_dims = list(self.env.action_space.shape)[-1]
            mu = self.model(state)
            log_std = -0.5 * torch.ones(act_dims)
            std = torch.exp(log_std)
            action = mu + torch.randn(mu.size()) * std
            lp = utils.gaussian_likelihood(action, mu, log_std)

        # save the log probability of the selected action and return the action picked
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
        policy_loss = ((-logps.detach() * returns)) if self.continuous else (-logps * returns)
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
