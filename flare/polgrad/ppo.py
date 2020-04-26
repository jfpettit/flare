from flare.polgrad import BasePolicyGradient
import flare.kindling as fk
import numpy as np
import torch
import gym
import torch.nn.functional as F
from termcolor import cprint
from flare.kindling.mpi_tools import (
    mpi_fork,
    mpi_avg,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from flare.kindling.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from typing import Callable, Optional, Union
from pathlib import Path


class PPO(BasePolicyGradient):
    r"""
    Implementation of the Proximal Policy Optimization (PPO) reinforcement learning algorithm.

    Args:
        env_fn: lambda function making the desired gym environment.
            Example::

                import gym
                env_fn = lambda: gym.make("CartPole-v1")
                agent = PPO(env_fn)
        hidden_sizes: Tuple of integers representing hidden layer sizes for the MLP policy.
        actorcritic: Class for policy and value networks.
        gamma: Discount factor for GAE-lambda estimation.
        lam: Lambda for GAE-lambda estimation.
        steps_per_epoch: Number of state, action, reward, done tuples to train on per epoch.
        epsilon: Clipping ratio for policy loss term in PPO.
        maxkl: Maximum allowed KL-divergence between new and old policies during update. Breaks update loop if the KL is estimated to be greater than this.
        train_steps: Number of training steps to do over the collected data.
        pol_lr: Learning rate for the policy optimizer.
        val_lr: Learning rate for the value optimizer.
        seed: random seeding for NumPy and PyTorch.
        state_preproc: An optional state preprocessing function. Any desired manipulations to the state before it is passed to the agent can be performed here. The state_preproc function must take in and return a NumPy array.
            Example::

                def state_square(state):
                    state = state**2
                    return state
                agent = PPO(env_fn, state_preproc=state_square, state_sze=shape_of_state_after_preprocessing)
        state_sze: If a state preprocessing function is included, the size of the state after preprocessing must be passed in as well.
        logger_dir: Directory to log results to.
        tensorboard: Whether or not to use tensorboard logging.
        save_screen: Whether to save rendered screen images to a pickled file. Saves within logger_dir.
        save_states: Whether to save environment states to a pickled file. Saves within logger_dir.
    """
    def __init__(
        self,
        env_fn: Callable,
        hidden_sizes: Optional[tuple] = (64, 32),
        actorcritic: Optional[torch.nn.Module] = fk.FireActorCritic,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.97,
        steps_per_epoch: Optional[int] = 4000,
        epsilon: Optional[float] = 0.2,
        maxkl: Optional[float] = 0.01,
        train_steps: Optional[int] = 80,
        pol_lr: Optional[float] = 3e-4,
        val_lr: Optional[float] = 1e-3,
        seed: Optional[int] = 0,
        state_preproc: Optional[Callable] = None,
        state_sze: Optional[int] = None,
        logger_dir: Optional[Union[str, Path]]=None,
        tensorboard: Optional[bool] = True,
        save_screen: Optional[bool] = False,
        save_states: Optional[bool] = False,
    ):
        super().__init__(
            env_fn,
            actorcritic=actorcritic,
            gamma=gamma,
            lam=lam,
            steps_per_epoch=steps_per_epoch,
            hidden_sizes=hidden_sizes,
            seed=seed,
            state_preproc=state_preproc,
            state_sze=state_sze,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_screen=save_screen,
            save_states=save_states,
        )

        self.eps = epsilon
        self.maxkl = maxkl
        self.train_steps = train_steps

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=val_lr)

    def get_name(self):
        """Function to return class name. Used in logging to directory"""
        return self.__class__.__name__

    def update(self):
        """
        Update rule for PPO
        """
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [
            torch.Tensor(x) for x in self.buffer.get()
        ]
        _, logp, _ = self.ac.policy(states, acts)
        pol_ratio = (logp - logprobs_old).exp()
        min_adv = torch.clamp(pol_ratio, 1 - self.eps, 1 + self.eps) * advs
        pol_loss_old = -(torch.min(pol_ratio * advs, min_adv)).mean()
        approx_ent = (-logp).mean()

        for i in range(self.train_steps):
            self.policy_optimizer.zero_grad()
            _, logp, _ = self.ac.policy(states, acts)
            pol_ratio = (logp - logprobs_old).exp()
            min_adv = torch.clamp(pol_ratio, 1 - self.eps, 1 + self.eps) * advs
            pol_loss = -(torch.min(pol_ratio * advs, min_adv)).mean()

            _, logp, _ = self.ac.policy(states, acts)
            kl = mpi_avg((logprobs_old - logp).mean().item())
            if kl > 1.5 * self.maxkl:
                self.logger.log(
                    f"Early stopping at step {i} due to reaching max kl.", "yellow"
                )
                break
            pol_loss.backward()
            mpi_avg_grads(self.ac.policy)
            self.policy_optimizer.step()

        vals = self.ac.value_f(states)
        val_loss_old = F.mse_loss(vals, rets)

        for _ in range(self.train_steps):
            self.value_optimizer.zero_grad()
            vals = self.ac.value_f(states)
            val_loss = F.mse_loss(vals, rets)
            val_loss.backward()
            mpi_avg_grads(self.ac.value_f)
            self.value_optimizer.step()

        self.logger.store(
            PolicyLoss=pol_loss_old.detach().numpy(),
            ValueLoss=val_loss_old.detach().numpy(),
            KL=kl,
            Entropy=approx_ent.detach().numpy(),
            DeltaPolLoss=(pol_loss - pol_loss_old).detach().numpy(),
            DeltaValLoss=(val_loss - val_loss_old).detach().numpy(),
        )
        return (
            pol_loss_old.detach().numpy(),
            val_loss_old.detach().numpy(),
            approx_ent.detach().numpy(),
            kl,
        )
