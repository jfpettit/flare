# import needed packages
import numpy as np
import gym
import torch
from flare.kindling import utils
from flare.polgrad import BasePolicyGradient
import flare.kindling as fk
import torch.nn.functional as F


class A2C(BasePolicyGradient):
    def __init__(
        self,
        env,
        hidden_sizes=(64, 32),
        actorcritic=fk.FireActorCritic,
        gamma=0.99,
        lam=0.97,
        steps_per_epoch=4000,
        pol_lr=3e-4,
        val_lr=1e-3,
        logstd_anneal=None,
        state_preproc=None,
        state_sze=None,
        logger_dir=None,
        tensorboard=True,
        save_screen=False,
        save_states=False
    ):
        super().__init__(
            env,
            actorcritic=actorcritic,
            gamma=gamma,
            lam=lam,
            steps_per_epoch=steps_per_epoch,
            hid_sizes=hidden_sizes,
            state_sze=state_sze,
            state_preproc=state_preproc,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_screen=save_screen,
            save_states=save_states
        )

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=val_lr)

    def update(self):
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [
            torch.Tensor(x) for x in self.buffer.get()
        ]

        _, logp, _ = self.ac.policy(states, acts)
        pol_loss = -(logp * advs).mean()
        approx_ent = (-logp).mean()

        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        self.policy_optimizer.step()

        values = self.ac.value_f(states)
        val_loss_old = F.mse_loss(values, rets)
        for _ in range(80):

            values = self.ac.value_f(states)
            val_loss = F.mse_loss(values, rets)
            self.value_optimizer.zero_grad()
            val_loss.backward()
            self.value_optimizer.step()

        _, logp, _, vals = self.ac(states, a=acts)
        pol_loss_new = -(logp * advs).mean()
        val_loss_new = F.mse_loss(vals, rets)
        approx_kl = (logprobs_old - logp).mean()
        self.logger.store(
            PolicyLoss=pol_loss.detach().numpy(),
            ValueLoss=val_loss_old.detach().numpy(),
            KL=approx_kl.detach().numpy(),
            Entropy=approx_ent.detach().numpy(),
            DeltaPolLoss=(pol_loss_new - pol_loss).detach().numpy(),
            DeltaValLoss=(val_loss_new - val_loss_old).detach().numpy(),
        )
        return pol_loss, val_loss, approx_ent, approx_kl
