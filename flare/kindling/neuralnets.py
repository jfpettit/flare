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
from flare.kindling.utils import conv2d_output_shape, conv2d_output_size


class MLP(nn.Module):
    r"""
    A class for building a simple MLP network.

    Args:
        layer_sizes (list or tuple): Layer sizes for the network.
            Example::

                sizes = (4, 64, 64, 2)
                mlp = MLP(sizes)
        activations (Function): Activation function for MLP net.
        out_act (Function): Output activation function
        out_squeeze (bool): Whether to squeeze the output of the network.
    """

    def __init__(
        self,
        layer_sizes: Union[List, Tuple],
        activations: Optional[Callable] = torch.tanh,
        out_act: Optional[bool] = None,
        out_squeeze: Optional[bool] = False,
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = activations
        self.out_act = out_act
        self.out_squeeze = out_squeeze

        for i, l in enumerate(layer_sizes[1:]):
            self.layers.append(nn.Linear(layer_sizes[i], l))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = self.activations(l(x))

        if self.out_act is None:
            x = self.layers[-1](x)
        else:
            x = self.out_act(self.layers[-1](x))

        return torch.squeeze(x, -1) if self.out_squeeze else x

class CNN(nn.Module):
    """
    Create a PyTorch CNN module.
    :param kernel_size: Convolutional kernel size
    :param stride: convolutional kernel stride
    :param outpu_size: size of network output
    :param input_channels: number of channels in the input
    :param output_activation: if any, activation to apply to the output layer
    :param input_height: size of one side of input (currently assumes square input)
    :param channels: List of channel sizes for each convolutional layer
    :param linear_layer_sizes: list of (if any) sizes of linear layers to add after convolutional layers
    :param activation: activation function
    :param dropout_layers: if any, layers to apply dropout to
    :param dropout_p: probability of dropout to use
    :param out_squeeze: whether to squeeze the output
    """
    def __init__(self, 
                input_channels: int,
                input_height: int,
                output_size: int,
                kernel_size: int = 3,
                stride: int = 1,
                channels: list = [64, 64],
                linear_layer_sizes: list = [512],
                activation: Callable = torch.relu,
                output_activation: Callable = None,
                dropout_layers: list = None,
                dropout_p: float = None,
                out_squeeze: bool = False):
        
        super(CNN, self).__init__()

        conv_sizes = [input_channels] + channels
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.out_squeeze = out_squeeze

        self.dropout_p = dropout_p
        self.dropout_layers = dropout_layers

        self.hw=input_height
        for i, l in enumerate(conv_sizes[1:]):
            self.hw = conv2d_output_size(kernel_size=kernel_size, stride=stride, sidesize=self.hw)
            self.layers.append(nn.Conv2d(conv_sizes[i], l, kernel_size=kernel_size, stride=stride))

        self.hw = (self.hw, self.hw)
        conv_out_size = 1
        for num in self.hw:
            conv_out_size *= num
        conv_out_size *= conv_sizes[-1]

        linear_sizes = [conv_out_size] + linear_layer_sizes + [output_size]
        self.layers.append(nn.Flatten())
        for i, l in enumerate(linear_sizes[1:]):
            self.layers.append(nn.Linear(linear_sizes[i], l))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            print(l)

            if self.dropout_layers is not None and l in self.dropout_layers:
                x = F.dropout(x, p=self.dropout_p)

        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))

        return x.squeeze() if self.out_squeeze else x

class Actor(nn.Module):
    
    def action_distribution(self, states):
        raise NotImplementedError

    def logprob_from_distribution(self, policy, action):
        raise NotImplementedError

    def forward(self, x, a = None):
        policy = self.action_distribution(x)
        logp_a = None
        if a is not None:
            logp_a = self.logprob_from_distribution(policy, a)
        return policy, logp_a

class CategoricalPolicy(Actor):
    r"""
    A class for a Categorical Policy network. Used in discrete action space environments. The policy is an :func:`~MLP`.

    Args:
        state_features (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the network.
        out_activation (Function): Output activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
        useconv: bool = False,
        channels: int = 3,
        height: int = 64,
    ):
        super().__init__()
        if not useconv:
            self.net = MLP(
                [state_features] + list(hidden_sizes) + [action_dim], activations=activation
            )

        elif useconv:
            self.net = CNN(
                channels, height, action_dim 
            )

    def action_distribution(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

    def logprob_from_distribution(self, policy, actions):
        return policy.log_prob(actions)


class GaussianPolicy(Actor):
    r"""
    A class for a Gaussian Policy network. Used in continuous action space environments. The policy is an :func:`~MLP`.

    Args:
       state_features (int): Dimensionality of the state space.
       action_dim (int): Dimensionality of the action space.
       hidden_sizes (list or tuple): Hidden layer sizes.
       activation (Function): Activation function for the network.
       out_activation (Function): Output activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[List, Tuple],
        activation: Callable,
        out_activation: Callable,
        useconv: bool = False,
        channels: int = 3,
        height: int = 64,
    ):
        super().__init__()

        if not useconv:
            self.net = MLP(
                [state_features] + list(hidden_sizes) + [action_dim],
                activations=activation,
                out_act=out_activation,
            )

        elif useconv:
            self.net = CNN(
                channels, height, action_dim 
            ) 
        self.logstd = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def action_distribution(self, states):
        mus = self.net(states)
        std = torch.exp(self.logstd)
        return torch.distributions.Normal(mus, std)

    def logprob_from_distribution(self, policy, actions):
        return policy.log_prob(actions).sum(axis=-1)


class FireActorCritic(nn.Module):
    r"""
    An Actor Critic class for Policy Gradient algorithms.

    Has built-in capability to work with continuous (gym.spaces.Box) and discrete (gym.spaces.Discrete) action spaces. The policy and value function are both :func:`~MLP`. If working with a different action space, the user can pass in a custom policy class for that action space as an argument.

    Args:
       state_features (int): Dimensionality of the state space.
       action_space (gym.spaces.Space): Action space of the environment.
       hidden_sizes (list or tuple): Hidden layer sizes.
       activation (Function): Activation function for the network.
       out_activation (Function): Output activation function for the network.
       policy (nn.Module): Custom policy class for an environment where the action space is not gym.spaces.Box or gym.spaces.Discrete 

    """

    def __init__(
        self,
        state_features: int,
        action_space: int,
        hidden_sizes: Optional[Union[Tuple, List]] = (32, 32),
        activation: Optional[Callable] = torch.tanh,
        out_activation: Optional[Callable] = None,
        policy: Optional[nn.Module] = None,
        useconv: Optional[bool] = False,
        channels: Optional[int] = 3,
        height: Optional[int] = 64
    ):
        super(FireActorCritic, self).__init__()
    
        obs_dim = state_features 

        if isinstance(action_space, gym.spaces.Discrete):
            act_dim = action_space.n
            self.policy = CategoricalPolicy(
                obs_dim, 
                act_dim, 
                hidden_sizes, 
                activation,
                out_activation,
                useconv=useconv,
                channels=channels,
                height=height
                )
        elif isinstance(action_space, gym.spaces.Box):
            act_dim = action_space.shape[0]
            self.policy = GaussianPolicy(
                obs_dim, 
                act_dim, 
                hidden_sizes, 
                activation, 
                out_activation,
                useconv=useconv,
                channels=channels,
                height=height
                )
        else:
            self.policy = policy(
                obs_dim,
                action_space,
                hidden_sizes,
                activation,
                out_activation,
                useconv=useconv,
                channels=channels,
                height=height
            )

        self.value_f = MLP(
            [state_features] + list(hidden_sizes) + [1],
            activations=activation,
            out_squeeze=True,
        )

    def step(self, x):
        with torch.no_grad():
            policy = self.policy.action_distribution(x)
            action = policy.sample()
            logp_action = self.policy.logprob_from_distribution(policy, action)
            value = self.value_f(x)
        return action.numpy(), logp_action.numpy(), value.numpy()

    def act(self, x):
        return self.step(x)[0]


class MLPQActor(nn.Module):
    r"""
    An actor for Q policy gradient algorithms. 
    
    The policy is an :func:`~MLP`. This differs from the :func:`~FireActorCritic` class because the output from the policy network is scaled to action space limits on the forward pass.

    Args:
       state_features (int): Dimensionality of the state space.
       action_dim (int): Dimensionality of the action space.
       hidden_sizes (list or tuple): Hidden layer sizes.
       activation (Function): Activation function for the network.
       action_limit (float or int): Limits of the action space.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[list, tuple],
        activation: Callable,
        action_limit: Union[float, int],
    ):
        super(MLPQActor, self).__init__()
        policy_layer_sizes = [state_features] + list(hidden_sizes) + [action_dim]
        self.policy = MLP(policy_layer_sizes, activation, torch.tanh)
        self.action_limit = action_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output from the policy network scaled to the limits of the env action space."""
        return self.action_limit * self.policy(x)


class MLPQFunction(nn.Module):
    r"""
    A Q function network for Q policy gradient methods. 

    The Q function is an :func:`~MLP`. It always takes in a (state, action) pair and returns a Q-value estimate for that pair.

    Args:
        state_features (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[tuple, list],
        activation: Callable,
    ):
        super().__init__()
        self.qfunc = MLP(
            [state_features + action_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Return Q-value estimate for state, action pair (x, a).
        
        Args:
            x (torch.Tensor): Environment state.
            a (torch.Tensor): Action taken by the policy.
        """
        q = self.qfunc(torch.cat([x, a], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class FireDDPGActorCritic(nn.Module):
    r"""
    An Actor Critic for the DDPG algorithm. 

    The policy is an :func:`~MLPQActor` and the q-value function is an :func:`~MLPQFunction`. 

    Args:
        state_features (int): Dimensionality of the state space.
        action_space (gym.spaces.Box): Environment action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the network.
    """

    def __init__(
        self,
        state_features: int,
        action_space: gym.spaces.Box,
        hidden_sizes: Optional[Union[tuple, list]] = (256, 256),
        activation: Optional[Callable] = torch.relu,
    ):
        super().__init__()

        obs_dim = state_features
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = MLPQActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.qfunc = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get an action from the policy.

        Args:
            x (torch.Tensor): Observations from the environment.
        """
        with torch.no_grad():
            return self.policy(x).numpy()


class FireTD3ActorCritic(nn.Module):
    r"""
    Actor Critic for the TD3 algorithm.

    The policy is an :func:`~MLPQActor` and the q-function is an :func:`~MLPQFunction`.

    Args:
        state_features (int): Dimensionality of the state space.
        action_space (gym.spaces.Box): Environment action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the network. 
    """

    def __init__(
        self,
        state_features: int,
        action_space: gym.spaces.Box,
        hidden_sizes: Optional[Union[list, tuple]] = (256, 256),
        activation: Optional[Callable] = torch.relu,
    ):
        super(FireTD3ActorCritic, self).__init__()

        obs_dim = state_features
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = MLPQActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.qfunc1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qfunc2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get an action from the policy.

        Args:
            x (torch.Tensor): Observations from the environment.
        """
        with torch.no_grad():
            return self.policy(x).numpy()


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    """
    GaussianMLP Actor for SAC. From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    
    Policy network is an :func:`~MLP` with heads for mean and log standard deviation of the action distribution.

    Args:
        state_features (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the network.
        action_limit (float or int): Limit of the action space.
    """

    def __init__(
        self,
        state_features: int,
        action_dim: int,
        hidden_sizes: Union[list, tuple],
        activation: Callable,
        action_limit: Union[float, int],
    ):
        super().__init__()
        self.net = MLP([state_features] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.act_limit = action_limit

    def forward(
        self, x: torch.Tensor, deterministic: bool = False, with_logprob: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an action and action log prob from the policy.
        
        Args:
            x (torch.Tensor): state from the environment.
            deterministic (bool): whether to act deterministically or not.
            with_logprob (bool): whether to return with action log probability or not.
        """
        net_out = self.net(x)
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
    An SAC Actor Critic class. From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    
    The policy is a :func:`~SquashedGaussianMLPActor` and the q-functions are both :func:`~MLPQFunctions`.

    Args:
        state_features (int): Dimensionality of state space.
        action_space (gym.spaces.Box): Environment action space.
        hidden_sizes (list or tuple): Hidden layer sizes.
        activation (Function): Activation function for the networks.
    """

    def __init__(
        self,
        state_features: int,
        action_space: gym.spaces.Box,
        hidden_sizes: Optional[Union[tuple, list]] = (256, 256),
        activation: Optional[Callable] = torch.relu,
    ):
        super().__init__()

        obs_dim = state_features
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.policy = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.qfunc1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qfunc2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, x: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        r"""
        Get action from policy.

        Args:
            x (torch.Tensor): State from the environment.
            deterministic (bool): Whether to act deterministically.
        """
        with torch.no_grad():
            a, _ = self.policy(x, deterministic, False)
            return a.numpy()


class FireQActorCritic(nn.Module):
    r"""
    Generic Q Actor Critic class.

    Policy is an :func:`~MLP`. Q function is a :func:`~MLP` as well.

    Args:
        state_features (int): Dimensionality of state space.
        action_space (gym.spaces.Box): Environment action space.
        hidden_sizes (tuple or list): Hidden layer sizes.
        activation (Function): Activation function for the networks.
        out_activation (Function): Output activation for the networks.
    """

    def __init__(
        self,
        state_features: int,
        action_space: gym.spaces.Box,
        hidden_sizes: Optional[Union[Tuple, List]] = (256, 128),
        activation: Optional[Callable] = torch.relu,
        out_activation: Optional[Callable] = nn.Identity,
    ):
        super(FireQActorCritic, self).__init__()

        action_dim = action_space.shape[0]

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

    def forward(
        self, x: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Get action, q value estimates for action taken, and q value estimates for previous actions.

        Args:
            x (torch.Tensor): State from the environment.
            a (torch.Tensor): Action taken in the environment.
        """
        act = self.policy(x)
        q = self.qfunc(torch.cat(x, a, dim=1))
        q_act = self.qfunc(torch.cat(x, act, dim=1))

        return act, q, q_act
