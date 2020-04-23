import numpy as np
import torch
import gym
import torch.nn.functional as F
from termcolor import cprint
from flare.qpolgrad import BaseQPolicyGradient
import flare.kindling as fk
from flare.kindling import ReplayBuffer
from typing import Optional, Union, Callable


class DDPG(BaseQPolicyGradient):
    """
    Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
    """

    def __init__(
        self,
        env_fn: Callable,
        actorcritic: Callable = fk.FireDDPGActorCritic,
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

        super().__init__(
            env_fn,
            actorcritic,
            seed=seed,
            steps_per_epoch=steps_per_epoch,
            replay_size=replay_size,
            gamma=gamma,
            polyak=polyak,
            pol_lr=pol_lr,
            q_lr=q_lr,
            hidden_sizes=hidden_sizes,
            bs=bs,
            warmup_steps=warmup_steps,
            update_after=update_after,
            update_every=update_every,
            act_noise=act_noise,
            save_freq=save_freq,
            buffer=buffer,
            state_preproc=state_preproc,
            state_sze=state_sze,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_states=save_states,
            save_screen=save_screen,
        )

    def setup_optimizers(self, pol_lr, q_lr):
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.q_optimizer = torch.optim.Adam(self.ac.qfunc.parameters(), lr=q_lr)

    def calc_policy_loss(self, data):
        o = data["obs"]
        q_pi = self.ac.qfunc(o, self.ac.policy(o))
        return -q_pi.mean()

    def calc_qfunc_loss(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q = self.ac.qfunc(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.qfunc(o2, self.ac_targ.policy(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QValues=q.detach().numpy())

        return loss_q, loss_info

    def update(self, data, timer=None):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.calc_qfunc_loss(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.qfunc.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.policy_optimizer.zero_grad()
        loss_pi = self.calc_policy_loss(data)
        loss_pi.backward()
        self.policy_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.qfunc.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(QLoss=loss_q.item(), PolicyLoss=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def logger_tabular_to_dump(self):
        self.logger.log_tabular("QValues", with_min_and_max=True)
        self.logger.log_tabular("PolicyLoss", average_only=True)
        self.logger.log_tabular("QLoss", average_only=True)
