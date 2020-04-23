# pylint: disable=no-member
# pylint: disable=not-callable
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from flare.kindling.utils import NetworkUtils as netu
import gym
from scipy.signal import lfilter
from typing import Optional, Iterable, List, Dict, Callable, Union, Tuple


class MLP(nn.Module):
    def __init__(
        self,
        layer_szs: Union[List, Tuple],
        activations: Optional[Callable] = torch.tanh,
        out_act: Optional[bool] = None,
        out_squeeze: Optional[bool] = False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations
        self.out_act = out_act
        self.out_squeeze = out_squeeze

        for i, l in enumerate(layer_szs[1:]):
            self.layers.append(nn.Linear(layer_szs[i], l))

    def forward(self, x: torch.Tensor):
        for l in self.layers[:-1]:
            x = self.activations(l(x))

        if self.out_act is None:
            x = self.layers[-1](x)
        else:
            x = self.out_act(self.layers[-1](x))

        return x.squeeze() if self.out_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        state_features: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
        action_dim: int,
    ):
        super(CategoricalPolicy, self).__init__()
        self.mlp = MLP(
            [state_features] + list(hidden_sizes) + [action_dim], activations=activation
        )

    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None):
        logits = self.mlp(x)

        policy = torch.distributions.Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()

        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_features: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
        action_dim: int,
    ):
        super(GaussianPolicy, self).__init__()

        self.mlp = MLP(
            [state_features] + list(hidden_sizes) + [action_dim],
            activations=activation,
            out_act=out_activation,
        )
        self.logstd = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None):
        mu = self.mlp(x)
        std = torch.exp(self.logstd)
        policy = torch.distributions.Normal(mu, std)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi


class FireActorCritic(nn.Module):
    def __init__(
        self,
        state_features: int,
        action_space: int,
        hidden_sizes: Optional[Union[Tuple, List]] = (32, 32),
        activation: Optional[Callable] = torch.tanh,
        out_activation: Optional[Callable] = None,
        policy: Optional[nn.Module] = None,
    ):
        super(FireActorCritic, self).__init__()

        if policy is None and isinstance(action_space, gym.spaces.Box):
            self.policy = GaussianPolicy(
                state_features,
                hidden_sizes,
                activation,
                out_activation,
                action_space.shape[0],
            )
        elif policy is None and isinstance(action_space, gym.spaces.Discrete):
            self.policy = CategoricalPolicy(
                state_features, hidden_sizes, activation, out_activation, action_space.n
            )
        else:
            self.policy = policy(
                state_features, hidden_sizes, activation, out_activation, action_space
            )

        self.value_f = MLP(
            [state_features] + list(hidden_sizes) + [1],
            activations=activation,
            out_squeeze=True,
        )

    def forward(self, x: torch.Tensor, a: Optional[torch.Tensor] = None):
        pi, logp, logp_pi = self.policy(x, a)
        value = self.value_f(x)

        return pi, logp, logp_pi, value


class MLPQActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        policy_layer_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.policy = MLP(policy_layer_sizes, activation, torch.tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.policy(obs)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.qfunc = MLP([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.qfunc(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class FireDDPGActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=torch.relu,
    ):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = MLPQActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.qfunc = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.policy(obs).numpy()


class FireTD3ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=torch.relu,
    ):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = MLPQActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.qfunc1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qfunc2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.policy(obs).numpy()


"""
From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = MLP([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class FireSACActorCritic(nn.Module):
    """
    From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=torch.relu,
    ):
        super().__init__()

        obs_dim = observation_space
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.qfunc1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qfunc2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.policy(obs, deterministic, False)
            return a.numpy()


class FireQActorCritic(nn.Module):
    def __init__(
        self,
        state_features: int,
        action_space: int,
        hidden_sizes: Optional[Union[Tuple, List]] = (256, 128),
        activation: Optional[Callable] = torch.relu,
        out_activation: Optional[Callable] = nn.Identity,
    ):
        super(FireQActorCritic, self).__init__()

        action_dim = action_space.shape[0]
        action_lim = action_space.high[0]

        self.policy = MLP(
            [state_features] + list(hidden_sizes) + [action_dim],
            activations=activation,
            out_act=out_activation,
        )
        self.qfunc = MLP(
            [state_features] + list(hidden_sizes) + [action_dim],
            activations=activation,
            out_squeeze=True,
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        act = self.policy(x)
        q = self.qfunc(torch.cat(x, a, dim=1))
        q_act = self.qfunc(torch.cat(x, act, dim=1))

        return act, q, q_act


class NatureDQN(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super(NatureDQN, self).__init__()
        self.netutil = netu()

        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        height = self.netutil.conv2d_output_size(8, 4, h)
        width = self.netutil.conv2d_output_size(8, 4, w)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        height = self.netutil.conv2d_output_size(4, 2, height)
        width = self.netutil.conv2d_output_size(4, 2, width)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        height = self.netutil.conv2d_output_size(3, 1, height)
        width = self.netutil.conv2d_output_size(3, 1, width)

        self.fc1 = nn.Linear(height * width * 64, 512)
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

    def evaluate(self, states, actions):
        action_values = self.forward(states.float())
        return action_values.gather(1, actions)
