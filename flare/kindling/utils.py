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
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str,
    color: int,
    bold: Optional[bool] = False,
    highlight: Optional[bool] = False,
):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def calc_logstd_anneal(n_anneal_cycles, anneal_start, anneal_end, epochs):
    if n_anneal_cycles > 0:
        logstds = np.linspace(anneal_start, anneal_end, num=epochs // n_anneal_cycles)
        for _ in range(n_anneal_cycles):
            logstds = np.hstack((logstds, logstds))
    else:
        logstds = np.linspace(anneal_start, anneal_end, num=epochs)

    return logstds


def gaussian_likelihood(x, mu, log_std):
    vals = (
        -0.5 * (((x - mu) / torch.exp(log_std) + 1e-8)) ** 2
        + 2 * log_std
        + torch.log(torch.tensor(2 * math.pi))
    )
    return vals.sum()


class NetworkUtils:
    def __init__(self):
        super(NetworkUtils, self).__init__()

    def conv2d_output_size(self, kernel_size, stride, sidesize):
        return (sidesize - (kernel_size - 1) - 1) // stride + 1

    def squared_error_loss(self, target, actual):
        return (actual - target) ** 2


def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif

    This code from this floydhub blog post: https://blog.floydhub.com/spinning-up-with-deep-reinforcement-learning/
    """
    # patch = plt.imshow(frames[0])
    fig = plt.figure()
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    # anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim = animation.ArtistAnimation(fig, frames, interval=50)
    if filename:
        anim.save(filename, writer="imagemagick")


class NormalizedActions(gym.ActionWrapper):
    """
    Normalize actions for continuous policy

    From here: https://github.com/JamesChuanggg/pytorch-REINFORCE/blob/master/normalized_actions.py
    """

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= self.action_space.high - self.action_space.low
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= self.action_space.high - self.action_space.low
        action = action * 2 - 1
        return action
