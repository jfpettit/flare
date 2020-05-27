import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import gym
from flare.kindling.neuralnets import MLP, GaussianPolicy, CategoricalPolicy, Actor
from flare.kindling.utils import _discount_cumsum
from flare.kindling import PGBuffer
from flare.polgrad import BasePolicyGradient
# pylint: disable=import-error
# pylint: disable=no-member
import time
import flare.kindling as fk
from flare.kindling import utils
from gym.spaces import Box
from flare.kindling import EpochLogger
from flare.kindling import TensorBoardWriter
import pickle as pkl
from typing import Optional, Any, Union, Callable, Tuple, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
from flare.polgrad import BasePolicyGradient


class REINFORCE(BasePolicyGradient):
    r"""
    REINFORCE Policy Gradient Class, written using PyTorch + PyTorch Lightning

    Args:
        env (function): environment to train in
        hidden_sizes (list or tuple): hidden layer sizes for MLP actor
        steps_per_epoch (int): Number of environment interactions to collect per epoch
        minibatch_size (int or None): size of minibatches of interactions to train on
        gamma (float): gamma discount factor for reward discounting
        lam (float): Used in advantage estimation, not used here. REINFORCE does not learn a value function so can't calculate advantage.
    """

    def __init__(
        self,
        env: Callable,
        actor = fk.FireActorCritic,
        hidden_sizes: Optional[Union[Tuple, List]] = (64, 64),
        steps_per_epoch: Optional[int] = 4000,
        minibatch_size: Optional[Union[None, int]] = None,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.97,
        seed = 0,
        hparams = None
    ):
        super().__init__(
            env,
            ac=actor,
            hidden_sizes=hidden_sizes,
            steps_per_epoch=steps_per_epoch,
            minibatch_size=minibatch_size,
            gamma=gamma,
            lam=lam,
            seed=seed,
            hparams=hparams
        )

    def configure_optimizers(self) -> tuple:
        r"""
        Set up optimizers for agent.
        """
        return torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
    
    def inner_loop(self) -> None:
        r"""
        Run agent-env interaction loop. 

        Stores agent environment interaction tuples to the buffer. Logs reward mean/std/min/max to tracker dict. Collects data at loop end.

        Slightly modified from :func:`~LitBasePolicyGradient`. REINFORCE does not learn a value function, so the portions which get value estimates had to be removed.
        """
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0
        rewlst = []
        lenlst = []

        for i in range(self.steps_per_epoch):
            action, logp, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

            next_state, reward, done, _ = self.env.step(action)

            self.buffer.store(
                state,
                action,
                reward,
                0,
                logp
            )

            state = next_state
            episode_length += 1
            episode_reward += reward

            
            timeup = episode_length == 1000
            over = done or timeup
            epoch_ended = i == self.steps_per_epoch - 1
            if over or epoch_ended:
                if timeup or epoch_ended:
                    last_val = reward
                else:
                    last_val = 0
                self.buffer.finish_path(last_val)

                if over:
                    rewlst.append(episode_reward)
                    lenlst.append(episode_length)
                state, episode_reward, episode_length = self.env.reset(), 0, 0

        trackit = {
            "MeanEpReturn": np.mean(rewlst),
            "StdEpReturn": np.std(rewlst),
            "MaxEpReturn": np.max(rewlst),
            "MinEpReturn": np.min(rewlst),
            "MeanEpLength": np.mean(lenlst)
        }
        self.tracker_dict.update(trackit)

        self.data = self.buffer.get()

    def calc_pol_loss(self, logps: torch.Tensor, rets: torch.Tensor) -> torch.Tensor:
        r"""
        Reinforce Policy gradient loss. -(action_log_probabilities * returns)

        Args:
            logps (PyTorch Tensor): Action log probabilities.
            returns (PyTorch Tensor): Returns from the environment.
        """
        return -(logps * rets).mean()

    def training_step(self, batch: Tuple, batch_idx: int) -> dict:
        r"""
        Calculate policy loss over input batch.

        Also compute and log policy entropy and KL divergence.

        Args:
            batch (Tuple of PyTorch tensors): Batch to train on.
            batch_idx: batch index.
        """
        states, acts, _, rets, logps_old = batch

        policy, logps = self.ac.policy(states, acts)
        pol_loss = self.calc_pol_loss(logps, rets)

        ent = policy.entropy().mean() 
        kl = (logps_old - logps).mean()
        log = {"PolicyLoss": pol_loss, "Entropy": ent, "KL": kl}
        self.tracker_dict.update(log)

        return {"loss": pol_loss, "log": log, "progress_bar": log}


def learn(
    env_name,
    epochs: Optional[int] = 100,
    minibatch_size: Optional[int] = None,
    steps_per_epoch: Optional[int] = 4000,
    hidden_sizes: Optional[Union[Tuple, List]] = (64, 32),
    gamma: Optional[float] = 0.99,
    lam: Optional[float] = 0.97,
    hparams = None,
    seed = 0
):
    from flare.polgrad.base import runner 
    minibatch_size = 4000 if minibatch_size is None else minibatch_size
    runner(
        env_name, 
        REINFORCE, 
        epochs=epochs, 
        minibatch_size=minibatch_size, 
        hidden_sizes=(64, 32),
        gamma=gamma,
        lam=lam,
        hparams=hparams,
        seed = seed
        )
