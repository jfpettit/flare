import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import gym
from gym import wrappers
import math
import scipy

class Buffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class MathUtils:
    def __init__(self):
        super(MathUtils, self).__init__()
        self.EPS = 1e-8

    def reduce_sum(self, x, axis=None):
        x = np.asarray(x)
        if axis is None:
            return x.sum()
        return x.sum(axis=axis)

    def reduce_mean(self, x, axis=None):
        x = np.asarray(x)
        if axis is None:
            return x.mean()
        return x.mean(axis=axis )

    def one_hot_encoder(self, vec, depth):
        vec = np.asarray(vec)
        encoding = np.copy(vec)
        vec_imax = vec[i].max()
        encoding[encoding < vec_imax] = 0
        encoding[encoding == vec_imax] = 1
        return encoding[:, :depth]

    def conjugate_gradient(self, Ax, b, iters=10, tol=1e-4):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotold = torch.dot(r, r)
        i = 0
        while i < iters or tol > 1e-4:
            z = Ax(p)
            alpha = rdotold/(np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            rdotnew = torch.dot(r,r)
            p = r + (rdotnew/rdotold) * p
            tol = abs(rdotold - rdotnew)/rdotold
            rdotold=rdotnew
            i += 1
        return x

def gaussian_likelihood(x, mu, log_std):
    vals = -.5 * (((x - mu)/torch.exp(log_std)+1e-8))**2 + 2 * log_std + torch.log(torch.tensor(2*math.pi))
    return vals.sum()

class MlpPolicyUtils:
    def __init__(self, mlp, action_space):
        self.mlp = mlp
        self.action_space = action_space
        self._math = MathUtils()

    def categorical_mlp(self, observations, actions):
        act_dim = self.action_space.n

        logits = mlp(observations)
        logprob_all = F.log_softmax(logits)
        policy = torch.squeeze(torch.distributions.multinomial.Multinomial(logits=logits), dim=1)
        logprobs = self._math.reduce_sum(self._math.one_hot_encoder(actions, depth=act_dim) * logprob_all, axis=1)
        logprobs_policy = self._math.reduce_sum(self._math.one_hot_encoder(policy.numpy(), depth=act_dim) * logprob_all, axis=1)
        return policy. logprobs, logprobs_policy

    def gaussian_mlp(self, observations, actions):
        act_dim = actions.shape.as_list()[-1]
        mu = self.mlp(observations)
        mu_ = mu.numpy()
        log_std = -.5 * np.ones(act_dim, dtype=np.float32)
        std = np.exp(log_std)
        policy = mu_ * np.random.normal(size=mu_) * std
        logprobs = self._math.gaussian_likelihood(actions, mu_, log_std)
        logprobs_policy = self._math.gaussian_likelihood(policy, mu_, log_std)
        return policy, logprobs, logprobs_policy

    def actor_critic_mlp(self, observations, actions):
        if isinstance(self.action_space, Box):
            policy = self.gaussian_mlp
        elif isinstance(self.action_space, Discrete):
            policy = self.categorical_mlp

        policy, logprobs, logprobs_policy = policy(observations, actions)
        values = torch.squeeze(self.mlp(observations), axis=1)
        return policy, logprobs, logprobs_policy, values

class AdvantageEstimatorsUtils:
    def __init__(self, gamma, lam):
        super(AdvantageEstimatorsUtils, self).__init__()
        self.gamma = gamma
        self.lam = lam
        
    def gae_lambda(self, return_, values):
        deltas = self.td_residual(return_, values)
        ls = torch.arange(1, values.size()[0]+1).float()
        adv = torch.pow((self.gamma*self.lam), ls) * deltas
        return adv

    def td_residual(self, return_, values):
        residuals = []
        residuals.append(return_[0] + values[1] - values[0])
        for i in range(1, len(values)):
            residuals.append(return_[i-1] + (self.gamma * values[i]) - values[i-1])
        return torch.stack(residuals).float()

    def basic_adv_estimator(self, return_, values):
        return return_ - values

class NetworkUtils:
    def __init__(self):
        super(NetworkUtils, self).__init__()

    def conv2d_output_size(self, kernel_size, stride, sidesize):
        return (sidesize - (kernel_size - 1) - 1) // stride + 1

    def squared_error_loss(self, target, actual):
        return (actual - target)**2

import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif

    This code from this floydhub blog post: https://blog.floydhub.com/spinning-up-with-deep-reinforcement-learning/
    """
    #patch = plt.imshow(frames[0])
    fig = plt.figure()
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    #anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim = animation.ArtistAnimation(fig, frames, interval=50)
    if filename:
        anim.save(filename, writer='imagemagick')


import gym


class NormalizedActions(gym.ActionWrapper):
    '''
    Normalize actions for continuous policy

    From here: https://github.com/JamesChuanggg/pytorch-REINFORCE/blob/master/normalized_actions.py
    '''

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return actions

    





            
