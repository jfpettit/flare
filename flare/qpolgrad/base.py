# pylint: disable=import-error
# pylint: disable=no-member
from copy import deepcopy
import numpy as np
import time
import torch
import abc
from termcolor import cprint
from gym.spaces import Box
import torch.nn as nn
import pickle as pkl
import scipy
import flare.kindling as fk
from flare.kindling import ReplayBuffer
from typing import Optional, Union, Callable, Tuple, List
import pytorch_lightning as pl
from flare.kindling.datasets import QPolicyGradientRLDataset
from argparse import Namespace
import sys
import gym
import pybullet_envs


class BaseQPolicyGradient(pl.LightningModule):
    def __init__(
        self,
        env_fn: Callable,
        actorcritic: Callable,
        epochs: int,
        seed: Optional[int] = 0,
        steps_per_epoch: Optional[int] = 4000,
        horizon: Optional[int] = 1000,
        replay_size: Optional[int] = int(1e6),
        gamma: Optional[float] = 0.99,
        polyak: Optional[float] = 0.95,
        pol_lr: Optional[float] = 1e-3,
        q_lr: Optional[float] = 1e-3,
        hidden_sizes: Optional[Union[tuple, list]] = (256, 128),
        bs: Optional[int] = 100,
        warmup_steps: Optional[int] = 10000,
        update_after: Optional[int] = 1000,
        update_every: Optional[int] = 50,
        act_noise: Optional[float] = 0.1,
        num_test_episodes: Optional[int] = 10,
        buffer: Optional[float] = ReplayBuffer,
        hparams = None
    ):
        super().__init__()

        if hparams is None:
            pass
        else:
            self.hparams = hparams

        self.env, self.test_env = env_fn(), env_fn()

        self.ac = actorcritic(
            self.env.observation_space.shape[0],
            self.env.action_space,
            hidden_sizes=hidden_sizes,
        )
        self.buffer = buffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            replay_size,
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.steps_per_epoch = steps_per_epoch
        self.tracker_dict = {}
        self.horizon = horizon
        self.num_test_episodes = num_test_episodes
        self.t = 0
        self.start = 0
        self.warmup_steps = warmup_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pol_lr = pol_lr
        self.q_lr = q_lr
        self.hidden_sizes = hidden_sizes
        self.bs = bs
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]
        self.steps = self.steps_per_epoch * epochs 


        self.ac_targ = deepcopy(self.ac)


        for param in self.ac_targ.parameters():
            param.requires_grad = False


        self.saver = fk.Saver(out_dir='tmp')

    def get_name(self):
        return self.__class__.__name__

    def on_train_start(self):
       self.inner_loop(self.steps)

    def forward(self, x, a):
        return self.ac(x, a)

    @abc.abstractmethod
    def configure_optimizers(self):
        """Function to initialize optimizers"""
        return

    @abc.abstractmethod
    def calc_pol_loss(self, data):
        """Function to compute policy loss"""
        return

    @abc.abstractmethod
    def calc_qfunc_loss(self, data):
        """Function to compute q-function loss"""
        return

    @abc.abstractmethod
    def training_step(self, data, timer=None):
        """Update rule for algorithm"""
        return

    def training_step_end(
        self,
        step_dict: dict
    ) -> dict:
        r"""
        Method for end of training step. Makes sure that episode reward and length info get added to logger.

        Args:
            step_dict (dict): dictioanry from last training step.
        
        Returns:
            step_dict (dict): dictionary from last training step with episode return and length info from last epoch added to log.
        """
        step_dict['log'] = self.add_to_log_dict(step_dict['log'])
        return step_dict

    def add_to_log_dict(self, log_dict) -> dict:
        r"""
        Adds episode return and length info to logger dictionary.

        Args:
            log_dict (dict): Dictionary to log to.
        
        Returns:
            log_dict (dict): Modified log_dict to include episode return and length info.
        """
        log_dict.update(self.tracker_dict)
        return log_dict

    def train_dataloader(self):
        dataset = QPolicyGradientRLDataset(self.buffer.sample_batch(self.bs))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.bs)
        return dataloader

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self, num_test_episodes, max_ep_len):
        test_return = []
        test_length = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            test_return.append(ep_ret)
            test_length.append(ep_len)
            trackit = dict(MeanTestEpReturn=np.mean(test_return), MeanTestEpLength=np.mean(test_length))
        return trackit

    def update(self):
        dataloader = self.train_dataloader()
        for i, batch in enumerate(dataloader):
            out1 = self.training_step(batch, i, 0)
            out2 = self.training_step(batch, i, 1)
            self.tracker_dict.update(out1)
            self.tracker_dict.update(out2)

    def inner_loop(self, steps):
        max_ep_len = self.horizon
        state, episode_return, episode_length = self.env.reset(), 0, 0
        rewlst = []
        lenlst = []
        for i in range(self.start, steps):
            # Main loop: collect experience in env and update/log each epoch

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if self.t > self.warmup_steps:
                action = self.get_action(state, self.act_noise)
            else:
                action = self.env.action_space.sample()

            # Step the env
            next_state, reward, done, _ = self.env.step(action)
            episode_return += reward
            episode_length += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if episode_length == max_ep_len else done

            # Store experience to replay buffer
            self.buffer.store(state, action, reward, next_state, done)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            state = next_state

            # End of trajectory handling
            if done or (episode_length == max_ep_len):
                rewlst.append(episode_return)
                lenlst.append(episode_length)
                state, episode_return, episode_length = self.env.reset(), 0, 0

            self.t += 1
            if self.t > self.update_after and self.t % self.update_every == 0:
                trackit = {
                    "MeanEpReturn": np.mean(rewlst),
                    "StdEpReturn": np.std(rewlst),
                    "MaxEpReturn": np.max(rewlst),
                    "MinEpReturn": np.min(rewlst),
                    "MeanEpLength": np.mean(lenlst),
                }
                self.tracker_dict.update(trackit)
                self.update()
            # End of epoch handling
            if (self.t + 1) % self.steps_per_epoch == 0:

                # Test the performance of the deterministic version of the agent.
                testtrack = self.test_agent(
                    num_test_episodes=self.num_test_episodes, max_ep_len=max_ep_len
                )
                self.tracker_dict.update(testtrack)
                self.printdict()
        
        self.start = i

        
    def printdict(self, out_file: Optional[str] = sys.stdout) -> None:
        r"""
        Print the contents of the epoch tracking dict to stdout or to a file.

        Args:
            out_file (sys.stdout or string): File for output. If writing to a file, opening it for writing should be handled in :func:`on_epoch_end`.
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
        self.saver.store(**self.tracker_dict)
        self.tracker_dict = {}
        self.inner_loop(self.steps)

    def on_train_end(self):
        self.saver.save()

def runner(
    env_name: str, 
    algo: BaseQPolicyGradient,
    ac: nn.Module,
    epochs: Optional[int] = 10000, 
    steps_per_epoch: Optional[int] = 4000,
    bs: Optional[Union[int, None]] = 50, 
    hidden_sizes: Optional[Union[Tuple, List]] = (256, 256),
    gamma: Optional[float] = 0.99,
    hparams: Optional[Namespace] = None,
    seed: Optional[int] = 0
    ):
    r"""
    Runner function to train algorithms in env.

    Args:
        algo (BasePolicyGradient subclass): The policy gradient algorithm to run. Included are A2C, PPO, and REINFORCE.
        ac (nn.Module): Actor-Critic network following same API as :func:`~FireActorCritic`.
        epochs (int): Number of epochs to train for.
        steps_per_epoch (int): Number of agent - environment interaction steps to train on each epoch.
        minibatch_size (int): Size of minibatches to sample from the batch collected over the epoch and train on. Default is None. When set to None, trains on minibatches one tenth the size of the full batch.
        hidden_sizes (tuple or list): Hidden layer sizes for MLP Policy and MLP Critic.
        gamma (float): Discount factor for return discounting and GAE-Lambda.
        hparams (Namespace): Hyperparameters to log. Defaults to None.
        seed (int): Random seeding for environment, PyTorch, and NumPy.
    """

    env = lambda: gym.make(env_name)
    
    agent = algo(
        env,
        ac,
        epochs=epochs,
        hidden_sizes=hidden_sizes,
        seed=seed,
        steps_per_epoch=steps_per_epoch, 
        bs=bs,
        gamma=gamma,
        hparams=hparams
        )

    trainer = pl.Trainer(
        reload_dataloaders_every_epoch=True,
        early_stop_callback=False,
        max_epochs=epochs
    )

    trainer.fit(agent)
