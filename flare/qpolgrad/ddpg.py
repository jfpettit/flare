import flare.neural_nets as nets
import numpy as np
import torch
import gym
import torch.nn.functional as F
from termcolor import cprint
from flare.utils import ReplayBuffer

class DDPG:
    """
    Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
    """
    def __init__(self, env, hidden_sizes=(400, 300), actorcritic=nets.FireQActorCritic, steps_per_epoch=5000, replay_size=int(1e6), gamma=0.99, polyakavg=0.995, pol_lr=1e-3, q_lr=1e-3, bs=100, starting_steps=10000, noise=0.1, )