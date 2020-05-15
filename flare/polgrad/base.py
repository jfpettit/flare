import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import gym
import time
import flare.kindling as fk
from flare.kindling import utils
from typing import Optional, Any, Union, Callable, Tuple, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from flare.kindling.datasets import PolicyGradientRLDataset
import sys
import abc
from argparse import Namespace

class BasePolicyGradient(pl.LightningModule):
    r"""
    Base Policy Gradient Class, written using PyTorch + PyTorch Lightning

    Args:
        env (function): environment to train in
        ac (nn.Module): actor-critic class to use
        hidden_sizes (list or tuple): hidden layer sizes for MLP actor
        steps_per_epoch (int): Number of environment interactions to collect per epoch
        minibatch_size (int or None): size of minibatches of interactions to train on
        gamma (float): gamma discount factor for reward discounting
        lam (float): Used in advantage estimation, not used here. REINFORCE does not learn a value function so can't calculate advantage.
        pol_lr (float): policy optimizer learning rate
        val_lr (float): value function optimizer learning rate
        train_iters (int): number of training steps on each batch of data
        seed (int): random seed
        hparams (argparse.Namespace): any hyperparameters to log to experiment logger
    """

    def __init__(
        self,
        env: Callable,
        ac: nn.Module,
        hidden_sizes: Optional[Union[Tuple, List]] = (64, 64),
        steps_per_epoch: Optional[int] = 4000,
        minibatch_size: Optional[Union[None, int]] = None,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.97,
        pol_lr: Optional[float] = 3e-4,
        val_lr: Optional[float] = 1e-3,
        train_iters = 80,
        seed = 0,
        hparams: Namespace = None
    ):
        super().__init__()

        if hparams is None:
            pass
        else:
            self.hparams = hparams


        seed += 10000
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env()
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        self.ac = ac(
            self.env.observation_space.shape[0],
            self.env.action_space,
            hidden_sizes
        )

        self.buffer = fk.PGBuffer(
            self.env.observation_space.shape[0],
            self.env.action_space.shape,
            steps_per_epoch,
            gamma=gamma,
            lam=lam
        )

        self.steps_per_epoch = steps_per_epoch
        self.pol_lr = pol_lr
        self.val_lr = val_lr
        self.train_iters = train_iters

        self.tracker_dict = {}

        self.inner_loop()

        self.minibatch_size = steps_per_epoch // 10
        if minibatch_size is not None:
            self.minibatch_size = minibatch_size

    def forward(self, x: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        r"""
        Forward pass for the agent.
        """
        return self.ac(x, a) 

    def configure_optimizers(self) -> tuple:
        r"""
        Set up optimizers for agent.
        """
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=self.pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=self.val_lr) 
        return self.policy_optimizer, self.value_optimizer

    def inner_loop(self) -> None:
        r"""
        Run agent-env interaction loop. 

        Stores agent environment interaction tuples to the buffer. Logs reward mean/std/min/max to tracker dict. Collects data at loop end.

        """
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0
        rewlst = []
        lenlst = []

        for i in range(self.steps_per_epoch):
            action, logp, value = self.ac.step(torch.as_tensor(state, dtype=torch.float32))

            next_state, reward, done, _ = self.env.step(action)

            self.buffer.store(
                state,
                action,
                reward,
                value,
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
                    last_val = self.ac.value_f(torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
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

    @abc.abstractmethod
    def calc_pol_loss(self, *args) -> torch.Tensor:
        r"""
        Loss for policy gradient agent.
        """
        pass

    def calc_val_loss(self, values: torch.Tensor, rets: torch.Tensor) -> torch.Tensor:
        return ((values - rets)**2).mean()

    @abc.abstractmethod
    def training_step(self, batch: Tuple, batch_idx: int, optimizer_idx: int) -> dict:
        r"""
        Policy gradient agent training step.

        Args:
            batch (Tuple of PyTorch tensors): Batch to train on.
            batch_idx: batch index.
            optimizer_idx: Index of optimizer to use
        """
        pass

    def training_step_end(
        self,
        step_dict
    ):
        step_dict['log'] = self.add_to_log_dict(step_dict['log'])
        return step_dict

    def add_to_log_dict(self, log_dict):
        add_to_dict = {
            "MeanEpReturn": self.tracker_dict["MeanEpReturn"],
            "MaxEpReturn": self.tracker_dict["MaxEpReturn"],
            "MinEpReturn": self.tracker_dict["MinEpReturn"],
            "MeanEpLength": self.tracker_dict["MeanEpLength"]}
        log_dict.update(add_to_dict)
        return log_dict


    def train_dataloader(self) -> DataLoader:
        r"""
        Define a PyTorch dataset with the data from the last :func:`~inner_loop` run and return a dataloader.
        """
        dataset = PolicyGradientRLDataset(self.data)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, sampler=None)
        return dataloader

    def printdict(self, out_file: Optional[str] = sys.stdout) -> None:
        r"""
        Print the contents of the epoch tracking dict to stdout or to a file.
        """
        self.print("\n", file=out_file)
        for k, v in self.tracker_dict.items():
            self.print(f"{k}: {v}", file=out_file)
        self.print("\n", file=out_file)
    
    def on_epoch_end(self) -> None:
        r"""
        Print tracker_dict, reset tracker_dict, and generate new data with inner loop.
        """
        self.printdict()
        self.tracker_dict = {}
        self.inner_loop()


def runner(
    env_name: str, 
    algo: BasePolicyGradient,
    ac: nn.Module = fk.FireActorCritic,
    epochs: Optional[int] = 100, 
    minibatch_size: Optional[Union[int, None]] = None, 
    steps_per_epoch: Optional[int] = 4000,
    hidden_sizes: Optional[Union[Tuple, List]] = (64, 64),
    gamma: Optional[float] = 0.99,
    lam: Optional[float] = 0.97,
    hparams: Optional[Namespace] = None,
    seed: Optional[int] = 0
    ):

    env = lambda: gym.make(env_name)
    
    agent = algo(
        env,
        ac,
        hidden_sizes=hidden_sizes,
        seed=seed,
        steps_per_epoch=steps_per_epoch, 
        minibatch_size=minibatch_size,
        gamma=gamma,
        lam=lam,
        hparams=hparams
        )

    trainer = pl.Trainer(
        reload_dataloaders_every_epoch=True,
        early_stop_callback=False,
        max_epochs=epochs
    )

    trainer.fit(agent)