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
from typing import Optional, Union, Callable


class BaseQPolicyGradient:
    def __init__(
        self,
        env_fn: Callable,
        actorcritic: Callable,
        seed: Optional[int] = 0,
        steps_per_epoch: Optional[int] = 4000,
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
        buffer: Optional[float] = ReplayBuffer,
        save_freq: Optional[int] = 1,
        state_preproc: Optional[Callable] = None,
        state_sze: Optional[Union[int, tuple]] = None,
        logger_dir: Optional[str] = None,
        tensorboard: Optional[bool] = True,
        save_states: Optional[bool] = False,
        save_screen: Optional[bool] = False,
    ):

        self.env, self.test_env = env_fn(), env_fn()

        if state_preproc is None:
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
            self.state_preproc = lambda x: x

        elif state_preproc is not None:
            assert (
                state_sze is not None
            ), "If using some state preprocessing, must specify state size after preprocessing."
            self.ac = actorcritic(
                state_sze, self.env.action_space, hidden_sizes=hidden_sizes
            )
            self.buffer = buffer(state_sze, self.env.action_space.shape, replay_size,)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pol_lr = pol_lr
        self.q_lr = q_lr
        self.hidden_sizes = hidden_sizes
        self.bs = bs
        self.warmup_steps = warmup_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.save_freq = save_freq

        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        self.ac_targ = deepcopy(self.ac)

        for param in self.ac_targ.parameters():
            param.requires_grad = False

        self.tensorboard = tensorboard
        if self.tensorboard:
            if logger_dir is None:
                name = self.get_name()
                logger_dir = f"flare_runs/run_at_time_{int(time.time())}_{name}_on_{self.env.unwrapped.spec.id}"
                self.tb_logger = fk.TensorBoardWriter(fpath=logger_dir)
            else:
                self.tb_logger = fk.TensorBoardWriter(fpath=logger_dir)

            self.saver = fk.Saver(out_dir=self.tb_logger.full_logdir)

            self.logger = fk.EpochLogger(output_dir=self.tb_logger.full_logdir)

        elif not self.tensorboard:
            self.logger = fk.EpochLogger(logger_dir)
            self.saver = fk.Saver(out_dir=self.logger.output_dir)

        self.logger.setup_pytorch_saver(self.ac)

        self.setup_optimizers(pol_lr=pol_lr, q_lr=q_lr)

    def get_name(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def setup_optimizers(self, pol_lr, q_lr):
        """Function to initialize optimizers"""
        return

    @abc.abstractmethod
    def calc_policy_loss(self, data):
        """Function to compute policy loss"""
        return

    @abc.abstractmethod
    def calc_qfunc_loss(self, data):
        """Function to compute q-function loss"""
        return

    @abc.abstractmethod
    def update(self, data, timer=None):
        """Update rule for algorithm"""
        return

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self, num_test_episodes, max_ep_len):
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpReturn=ep_ret, TestEpLength=ep_len)

    def learn(self, epochs, horizon=1000, num_test_episodes=10, **kwargs):
        max_ep_len = horizon
        total_steps = self.steps_per_epoch * epochs
        last_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if t > self.warmup_steps:
                a = self.get_action(o, self.act_noise)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            self.buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                self.logger.store(EpReturn=ep_ret, EpLength=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.buffer.sample_batch(self.bs)
                    self.update(data=batch, timer=j)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == epochs):
                    self.logger.save_state({"env": self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(
                    num_test_episodes=num_test_episodes, max_ep_len=max_ep_len
                )

                ep_dict = self.logger.epoch_dict_copy
                if self.tensorboard:
                    self.tb_logger.add_vals(ep_dict, step=epoch)
                # Log info about epoch
                new_time = time.time()
                self.logger.log_tabular("Iteration", epoch)
                self.logger.log_tabular("EpReturn", with_min_and_max=True)
                self.logger.log_tabular("TestEpReturn", with_min_and_max=True)
                self.logger.log_tabular("EpLength", average_only=True)
                self.logger.log_tabular("TestEpLength", average_only=True)
                self.logger.log_tabular("TotalEnvInteracts", t)
                self.logger_tabular_to_dump()
                self.logger.log_tabular("IterationTime", new_time - last_time)
                last_time = new_time
                self.logger.dump_tabular()

    @abc.abstractmethod
    def logger_tabular_to_dump(self):
        """Function to log tabular logger outputs for the particular algorithm. Should not call logger.dump_tabular(). This will be called for you."""
        return
