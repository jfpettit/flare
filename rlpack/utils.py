import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

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

    def gaussian_likelihood(self, x, mu, log_std):
        vals = -.5 * (((x - mu)/np.exp(log_std)+self.EPS))**2 + 2 * log_std + np.log(2*np.pi)
        return self.reduce_sum(vals, axis=1)

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
        anim.save(filename)


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

    





            
