import numpy as np
import torch
import gym
import torch.nn.functional as F
from termcolor import cprint
from flare.qpolgrad.base import BaseQPolicyGradient
import flare.kindling as fk
from flare.kindling import ReplayBuffer
from typing import Optional, Union, Callable, List, Tuple


class DDPG(BaseQPolicyGradient):
    """
    Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
    """

    def __init__(
        self,
        env_fn: Callable,
        actorcritic: Callable = fk.FireDDPGActorCritic,
        epochs: int = 100,
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
        hparams = None
    ):

        super().__init__(
            env_fn,
            actorcritic,
            epochs=epochs,
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
            hparams=hparams
        )

    def configure_optimizers(self):
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=self.pol_lr)
        self.q_optimizer = torch.optim.Adam(self.ac.qfunc.parameters(), lr=self.q_lr)
        return self.policy_optimizer, self.q_optimizer

    def calc_pol_loss(self, states):
        q_pi = self.ac.qfunc(states, self.ac.policy(states))
        return -q_pi.mean()

    def calc_qfunc_loss(self, data):
        o, o2, a, r, d = data 

        q = self.ac.qfunc(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.qfunc(o2, self.ac_targ.policy(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(MeanQValues=q.mean().detach().numpy())

        return loss_q, loss_info

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.policy_optimizer.zero_grad()
            policy_loss = self.calc_pol_loss(batch[0])
            policy_loss.backward()
            self.policy_optimizer.step()
            log = {
                "PolicyLoss": policy_loss
            }
            loss = policy_loss

        if optimizer_idx == 1:
            # First run one gradient descent step for Q.
            self.q_optimizer.zero_grad()
            q_loss, loss_info = self.calc_qfunc_loss(batch)
            q_loss.backward()
            self.q_optimizer.step()

            # Freeze Q-network so you don't waste computational effort
            # computing gradients for it during the policy learning step.
            for p in self.ac.qfunc.parameters():
                p.requires_grad = False

            # Unfreeze Q-network so you can optimize it at next DDPG step.
            for p in self.ac.qfunc.parameters():
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

            log = dict(QLoss=q_loss, **loss_info)
            loss = q_loss

        self.tracker_dict.update(log)
        return {"loss": loss, "log": log, "progress_bar": log}

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        pass

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure = None
    ):
        optimizer.zero_grad()

def learn(
    env_name,
    epochs: Optional[int] = 100,
    batch_size: Optional[int] = None,
    steps_per_epoch: Optional[int] = 4000,
    hidden_sizes: Optional[Union[Tuple, List]] = (256, 256),
    gamma: Optional[float] = 0.99,
    hparams = None,
    seed = 0
):
    from flare.qpolgrad.base import runner 
    batch_size = 100 if batch_size is None else batch_size
    runner(
        env_name, 
        DDPG,
        fk.FireDDPGActorCritic,
        epochs=epochs, 
        bs=batch_size, 
        hidden_sizes=hidden_sizes,
        gamma=gamma,
        hparams=hparams,
        seed = seed
        )
