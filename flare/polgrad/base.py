# pylint: disable=import-error
# pylint: disable=no-member
import numpy as np
import time
import torch
import flare.kindling as fk
from flare.kindling import utils
from flare.kindling import PGBuffer
import abc
from termcolor import cprint
import gym
from gym.spaces import Box
import torch.nn as nn
from flare.kindling import EpochLogger
from flare.kindling import TensorBoardWriter
from flare.kindling.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from flare.kindling.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
import pickle as pkl
from typing import Optional, Any, Union, Callable


class BasePolicyGradient:
    def __init__(
        self,
        env_fn: callable,
        actorcritic: Optional[nn.Module] = fk.FireActorCritic,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.97,
        steps_per_epoch: Optional[int] = 4000,
        hid_sizes: Optional[tuple] = (32, 32),
        seed: Optional[int] = 0,
        state_preproc: Optional[Callable] = None,
        state_sze: Optional[Union[int, tuple]] = None,
        logger_dir: Optional[str] = None,
        tensorboard: Optional[bool] = True,
        save_states: Optional[bool] = False,
        save_screen: Optional[bool] = False,
    ):
        setup_pytorch_for_mpi()

        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env_fn()
        self.state_preproc = state_preproc

        steps_per_epoch = int(steps_per_epoch / num_procs())

        if state_preproc is None:
            self.ac = actorcritic(
                self.env.observation_space.shape[0],
                self.env.action_space,
                hidden_sizes=hid_sizes,
            )
            self.buffer = PGBuffer(
                self.env.observation_space.shape,
                self.env.action_space.shape,
                steps_per_epoch,
                gamma,
                lam,
            )
            self.state_preproc = lambda x: x

        elif state_preproc is not None:
            assert (
                state_sze is not None
            ), "If using some state preprocessing, must specify state size after preprocessing."
            self.ac = actorcritic(
                state_sze, self.env.action_space, hidden_sizes=hid_sizes
            )
            self.buffer = PGBuffer(
                state_sze, self.env.action_space.shape, steps_per_epoch, gamma, lam
            )

        sync_params(self.ac)

        self.steps_per_epoch = steps_per_epoch

        self.save_states = save_states
        self.save_screen = save_screen

        self.tensorboard = tensorboard
        if self.tensorboard:
            if logger_dir is None:
                name = self.get_name()
                logger_dir = f"flare_runs/run_at_time_{int(time.time())}_{name}_on_{self.env.unwrapped.spec.id}"
                self.tb_logger = TensorBoardWriter(fpath=logger_dir)
            else:
                self.tb_logger = TensorBoardWriter(fpath=logger_dir)
    
        self.saver = fk.saver.Saver(out_dir=self.tb_logger.full_logdir)

        self.logger = EpochLogger(output_dir=self.tb_logger.full_logdir)
        self.logger.setup_pytorch_saver(self.ac)

    @abc.abstractmethod
    def get_name(self):
        """Return name of subclass"""
        pass

    @abc.abstractmethod
    def update(self):
        """Update rule for policy gradient algo."""
        return

    def learn(
        self, epochs, render=False, horizon=1000, logstd_anneal=None, n_anneal_cycles=0,
    ):
        if render and "Bullet" in self.env.unwrapped.spec.id and proc_id() == 0:
            self.env.render()

        if logstd_anneal is not None:
            assert isinstance(
                self.env.action_space, Box
            ), "Log standard deviation only used in environments with continuous action spaces. Your current environment uses a discrete action space."
            logstds = utils.calc_logstd_anneal(
                n_anneal_cycles, logstd_anneal[0], logstd_anneal[1], epochs
            )

        last_time = time.time()
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0

        for i in range(epochs):
            self.ep_length = []
            self.ep_reward = []

            if logstd_anneal is not None:
                self.ac.logstds = nn.Parameter(
                    logstds[i] * torch.ones(self.env.action_space.shape[0])
                )

            self.ac.eval()
            for _ in range(self.steps_per_epoch):
                if self.save_states:
                    self.saver.store(state_saver=state)
                if self.save_screen:
                    screen = self.env.render(mode="rgb_array")
                    self.saver.store(screen_saver=screen)

                state = self.state_preproc(state)

                action, _, logp, value = self.ac(torch.Tensor(state.reshape(1, -1)))
                self.logger.store(Values=np.array(value.detach().numpy()))

                if (
                    render
                    and "Bullet" not in self.env.unwrapped.spec.id
                    and proc_id() == 0
                ):
                    self.env.render()

                self.buffer.store(
                    state,
                    action.detach().numpy(),
                    reward,
                    value.item(),
                    logp.detach().numpy(),
                )
                state, reward, done, _ = self.env.step(action.detach().numpy()[0])
                episode_reward += reward
                episode_length += 1

                over = done or (episode_length == horizon)
                if over or (_ == self.steps_per_epoch - 1):
                    if self.state_preproc is not None:
                        state = self.state_preproc(state)

                    last_val = (
                        reward
                        if done
                        else self.ac.value_f(torch.Tensor(state.reshape(1, -1))).item()
                    )
                    self.buffer.finish_path(last_val)

                    if over:
                        self.logger.store(
                            EpReturn=episode_reward, EpLength=episode_length
                        )

                    state = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False
                    reward = 0

            self.saver.save()
            self.update()

            ep_dict = self.logger.epoch_dict_copy
            if self.tensorboard:
                self.tb_logger.add_vals(ep_dict, step=i)

            self.logger.log_tabular("Iteration", i)
            self.logger.log_tabular("EpReturn", with_min_and_max=True)
            self.logger.log_tabular("EpLength", average_only=True)
            self.logger.log_tabular("Values", with_min_and_max=True)
            self.logger.log_tabular("TotalEnvInteracts", (i + 1) * self.steps_per_epoch)
            self.logger.log_tabular("PolicyLoss", average_only=True)
            self.logger.log_tabular("ValueLoss", average_only=True)
            self.logger.log_tabular("DeltaPolLoss", average_only=True)
            self.logger.log_tabular("DeltaValLoss", average_only=True)
            self.logger.log_tabular("Entropy", average_only=True)
            self.logger.log_tabular("KL", average_only=True)
            self.logger.log_tabular("IterationTime", time.time() - last_time)
            last_time = time.time()

            if logstd_anneal is not None:
                self.logger.log_tabular("CurrentLogStd", logstds[i])

            self.logger.log_tabular("Env", self.env.unwrapped.spec.id)
            self.logger.dump_tabular()

        return self.ep_reward, self.ep_length
