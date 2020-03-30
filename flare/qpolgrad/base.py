# pylint: disable=import-error
# pylint: disable=no-member
import numpy as np
import time
import torch
import flare.neural_nets as nets
from flare import utils
import abc
from termcolor import cprint
from gym.spaces import Box
import torch.nn as nn
from flare.logging import EpochLogger
import pickle as pkl
import scipy

class ReplayBuffer(Buffer):
    """
    A replay buffer for off-policy RL agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs1_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def get_batch(self, n_sample=32):
        sample_inds = np.random.randint(0, self.size, size=n_sample)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.act_buf[idxs],
            rews=self.rew_buf[idxs],
            done=self.done_buf[idxs])

class BaseQPolicyGradient:
    def __init__(self, env, actorcritic=nets.FireQActorCritic, gamma=.99, polyak_avg=.995)