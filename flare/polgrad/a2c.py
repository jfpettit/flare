# import needed packages
import numpy as np
import gym
import torch
from flare.kindling import utils
from flare.polgrad import BasePolicyGradient
import flare.kindling as fk
import torch.nn.functional as F
from flare.kindling.mpi_pytorch import mpi_avg_grads, mpi_avg


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
        seed=0,
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
            seed=seed,
            state_sze=state_sze,
            state_preproc=state_preproc,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_screen=save_screen,
            save_states=save_states
        )

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=val_lr)

    def get_name(self):
        return self.__class__.__name__

    def update(self):
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [
            torch.Tensor(x) for x in self.buffer.get()
        ]
        
        _, logp, _ = self.ac.policy(states, acts)
        approx_ent = (-logp).mean()
        pol_loss_old = -(logp * advs).mean()
        
        values = self.ac.value_f(states)
        val_loss_old = F.mse_loss(values, rets)

        self.policy_optimizer.zero_grad()
        _, logp, _ = self.ac.policy(states, acts)
        kl = mpi_avg((logprobs_old - logp).mean().item())
        pol_loss = -(logp * advs).mean()
        pol_loss.backward()
        mpi_avg_grads(self.ac.policy) 
        self.policy_optimizer.step()

        for _ in range(80):
            self.value_optimizer.zero_grad()
            values = self.ac.value_f(states)
            val_loss = F.mse_loss(values, rets)
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
        return pol_loss, val_loss, approx_ent, kl
