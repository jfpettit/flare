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
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset




class REINFORCE(BasePolicyGradient):
    def __init__(
        self,
        env,
        hidden_sizes=(64, 32),
        actorcritic=Actor,
        gamma=0.99,
        lam=0.97,
        steps_per_epoch=4000,
        pol_lr=3e-4,
        val_lr=1e-3,
        seed=0,
        state_preproc=None,
        state_sze=None,
        logger_dir=None,
        tensorboard=True,
        save_screen=False,
        save_states=False,
    ):
        super().__init__(
            env,
            actorcritic=actorcritic,
            gamma=gamma,
            lam=lam,
            steps_per_epoch=steps_per_epoch,
            hidden_sizes=hidden_sizes,
            seed=seed,
            state_sze=state_sze,
            state_preproc=state_preproc,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_screen=save_screen,
            save_states=save_states,
        )

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)

    def calc_pol_loss(self, logps, returns):
        return (-logps * returns).mean()
        
    def update(self):
        states, actions, advs, returns, logps_old = [torch.as_tensor(x, dtype=torch.float32) for x in self.buffer.get()]
        pol_loss_old = self.calc_pol_loss(logps_old, returns)

        self.policy_optimizer.zero_grad()    
        _, logps, _ = self.ac(states, actions) 
        kl = mpi_avg((logps_old - logps).mean().item())
        policy_loss = self.calc_pol_loss(logps, returns)
        policy_loss.backward()
        mpi_avg_grads(self.ac)
        self.policy_optimizer.step()

        policy_entropy = (-logps).mean().detach().numpy()

        self.logger.store(PolicyLoss=policy_loss, Entropy=policy_entropy, DeltaPolLoss=(policy_loss - pol_loss_old).item(), KL=kl)

    def learn(
        self, epochs, render=False, horizon=1000, logstd_anneal=None, n_anneal_cycles=0,
    ):
        """
        Training loop for policy gradient algorithm.

        Args:
            epochs: Number of epochs to train for in the environment.
            render: Whether to render the agent during training
            horizon: Maximum allowed episode length
            logstd_anneal: None or two values. Anneals log standard deviation of action distribution from the first value to the second if it is not None.
                Example::
                    logstd_anneal = np.array([-1.6, -0.7])
                    agent.learn(100, logstd_anneal=logstd_anneal)
            n_anneal_cycles: Integer greater than or equal to zero. If logstd_anneal is specified, this variable allows the algorithm to cycle through the anneal schedule n times.
                Example::
                    agent.learn(100, logstd_anneal=np.array([-1.6, -0.7]), n_anneal_cycles=2)
        """

        v = 0 # no value function is learned 

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

            for j in range(self.steps_per_epoch):
                if self.save_states:
                    self.saver.store(state_saver=state)
                if self.save_screen:
                    screen = self.env.render(mode="rgb_array")
                    self.saver.store(screen_saver=screen)

                state = self.state_preproc(state)

                action, _, logp = self.ac(torch.Tensor(state.reshape(1, -1)))
                self.logger.store(Values=v)

                next_state, reward, done, _ = self.env.step(action.detach().numpy()[0])

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
                    v,
                    logp.detach().numpy(),
                )

                state = next_state
                episode_reward += reward
                episode_length += 1

                over = done or (episode_length == horizon)
                if over or (j == self.steps_per_epoch - 1):
                    if self.state_preproc is not None:
                        state = self.state_preproc(state)

                    last_val = (
                        v
                    )
                    self.buffer.finish_path(last_val)

                    if over:
                        self.logger.store(
                            EpReturn=episode_reward, EpLength=episode_length
                        )

                    state = self.env.reset()
                    self.ep_reward.append(episode_reward)
                    self.ep_length.append(episode_length)
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
            self.logger.log_tabular("TotalEnvInteracts", (i + 1) * self.steps_per_epoch)
            self.logger.log_tabular("PolicyLoss", average_only=True)
            self.logger.log_tabular("DeltaPolLoss", average_only=True)
            self.logger.log_tabular("Entropy", average_only=True)
            self.logger.log_tabular("KL", average_only=True)
            self.logger.log_tabular("IterationTime", time.time() - last_time)
            last_time = time.time()

            if logstd_anneal is not None:
                self.logger.log_tabular("CurrentLogStd", logstds[i])

            self.logger.log_tabular("Env", self.env.unwrapped.spec.id)
            self.logger.dump_tabular()

        return self.ep_reward, self.ep_length


if __name__ == '__main__':
    ncpu = 2
    epochs = 50
    steps_per_epoch = 4000

    mpi_fork(ncpu)
    env = lambda: gym.make("CartPole-v0")

    reinforce = REINFORCE(env) 
    r, l = reinforce.learn(epochs)

