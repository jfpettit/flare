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
from rlpack.a2c import A2C

# figure whether a GPU is available. In the future, this will be changed to use torch.tensor().to(device) syntax to maximize GPU usage
use_gpu = True if torch.cuda.is_available() else False

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
