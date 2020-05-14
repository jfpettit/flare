import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import gym
import pybullet_envs
import time
import flare.kindling as fk
from flare.kindling import utils
from flare.kindling.mpi_tools import mpi_avg
from typing import Optional, Any, Union, Callable, Tuple, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
from flare.polgrad import BasePolicyGradient


class LitPPO(BasePolicyGradient):
    def __init__(
        self,
        env,
        ac = fk.FireActorCritic,
        hidden_sizes = (64, 64),
        steps_per_epoch = 4000,
        minibatch_size = None,
        gamma = 0.99,
        lam = 0.97,
        pol_lr = 3e-4,
        val_lr = 1e-3,
        train_iters = 80,
        clipratio = 0.2,
        maxkl = 0.01,
        seed = 0,
        hparams = None
    ):
        super().__init__(
            env,
            ac,
            hidden_sizes=hidden_sizes,
            steps_per_epoch=steps_per_epoch,
            minibatch_size=minibatch_size,
            gamma=gamma,
            lam=lam,
            pol_lr=pol_lr,
            val_lr=val_lr,
            train_iters=train_iters,
            seed = seed,
            hparams= hparams
        )

        self.clipratio = clipratio 
        self.maxkl = maxkl

    def calc_pol_loss(self, logps, logps_old, advs):
        ratio = torch.exp(logps - logps_old)
        clipped_adv = torch.clamp(ratio, 1 - self.clipratio, 1 + self.clipratio) * advs
        pol_loss = -(torch.min(ratio * advs, clipped_adv)).mean()

        kl = (logps_old - logps).mean().item()
        return pol_loss, kl

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        states, actions, advs, rets, logps_old = batch

        if optimizer_idx == 0:
            stops = 0
            stopslst = []
            policy, logps = self.ac.policy(states, actions)
            pol_loss_old, kl = self.calc_pol_loss(logps, logps_old, advs)

            for i in range(self.train_iters):
                self.policy_optimizer.zero_grad()
                policy, logps = self.ac.policy(states, actions)
                pol_loss, kl = self.calc_pol_loss(logps, logps_old, advs)
                approx_kl = fk.mpi_tools.mpi_avg(kl)
                if approx_kl > 1.5 * self.maxkl:
                    stops += 1
                    stopslst.append(i)
                    break
                pol_loss.backward()
                self.policy_optimizer.step()

            log = {
                "PolicyLoss": pol_loss_old.item(),
                "DeltaPolLoss": (pol_loss - pol_loss_old).item(),
                "KL": approx_kl,
                "Entropy": policy.entropy().mean().item(),
                "TimesEarlyStopped": stops,
                "AvgEarlyStopStep": np.mean(stopslst) if len(stopslst) > 0 else 0
            }
            loss = pol_loss_old

        elif optimizer_idx == 1:
            values_old = self.ac.value_f(states)
            val_loss_old = self.calc_val_loss(values_old, rets)
            for i in range(self.train_iters):
                self.value_optimizer.zero_grad()
                values = self.ac.value_f(states)
                val_loss = self.calc_val_loss(values, rets)
                val_loss.backward()
                self.value_optimizer.step()

            delta_val_loss = (val_loss - val_loss_old).item()
            log = {"ValueLoss": val_loss_old.item(), "DeltaValLoss": delta_val_loss}
            loss = val_loss

        self.tracker_dict.update(log)
        return {"loss": loss, "log": log, "progress_bar": log}

"""
def learn(
    env_name: str, 
    epochs: Optional[int] = 100, 
    minibatch_size: Optional[Union[int, None]] = None, 
    steps_per_epoch: Optional[int] = 4000,
    hidden_sizes: Optional[Union[Tuple, List]] = (64, 64),
    gamma: Optional[float] = 0.99,
    lam: Optional[float] = 0.97
    ):
    
    env = lambda: gym.make(env_name)
    
    agent = LitPPO(
        env,
        fk.FireActorCritic,
        hidden_sizes=hidden_sizes,
        steps_per_epoch=steps_per_epoch, 
        minibatch_size=steps_per_epoch,
        gamma=gamma,
        lam=lam
        )

    trainer = pl.Trainer(
        reload_dataloaders_every_epoch=True,
        early_stop_callback=False,
        max_epochs=epochs
    )

    trainer.fit(agent)
"""

if __name__ == '__main__':
    env_name = "HalfCheetahBulletEnv-v0"
    epochs = 100

    from flare.polgrad.base import learn
    learn(env_name, LitPPO, epochs=epochs, minibatch_size=4000)