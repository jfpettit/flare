import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import gym
import pybullet_envs
import time
import flare.kindling as fk
from flare.kindling import utils
from typing import Optional, Any, Union, Callable, Tuple, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import sys
from flare.polgrad import BasePolicyGradient

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True

except ImportError:
    XLA_AVAILABLE = False

class A2C(BasePolicyGradient):
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
            seed=seed,
            hparams=hparams
        )

    def calc_pol_loss(self, logps, advs):
        return -(logps * advs).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        states, acts, advs, rets, logps_old = batch

        if optimizer_idx == 0:
            pol_loss_old = self.calc_pol_loss(logps_old, advs)

            policy, logps = self.ac.policy(states, a=acts)
            pol_loss = self.calc_pol_loss(logps, advs)

            ent = policy.entropy().mean().item() 
            kl = (logps_old - logps).mean().item()
            delta_pol_loss = (pol_loss - pol_loss_old).item()
            log = {"PolicyLoss": pol_loss_old.item(), "DeltaPolLoss": delta_pol_loss, "Entropy": ent, "KL": kl}
            loss = pol_loss

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

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        using_native_amp = None
    ):
        if optimizer_idx == 0:
            if self.trainer.use_tpu and XLA_AVAILABLE:
                xm.optimizer_step(optimizer)
            elif isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(second_order_closure)
            else:
                optimizer.step()

            # clear gradients
            optimizer.zero_grad()

        elif optimizer_idx == 1:
            pass

    def backward(
        self,
        trainer,
        loss,
        optimizer,
        optimizer_idx
    ):
        if optimizer_idx == 0:
            if trainer.precision == 16:

                # .backward is not special on 16-bit with TPUs
                if not trainer.on_tpu:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()

        elif optimizer_idx == 1:
            pass

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
        A2C, 
        epochs=epochs, 
        minibatch_size=minibatch_size, 
        hidden_sizes=(64, 32),
        gamma=gamma,
        lam=lam,
        hparams=hparams,
        seed = seed
        )