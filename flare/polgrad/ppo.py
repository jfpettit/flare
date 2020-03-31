from flare.polgrad import BasePolicyGradient
import numpy as np
import torch
import gym
import flare.kindling.neural_nets as nets
import torch.nn.functional as F
from termcolor import cprint


class PPO(BasePolicyGradient):
    def __init__(
        self,
        env,
        hidden_sizes=(32, 32),
        actorcritic=nets.FireActorCritic,
        gamma=0.99,
        lam=0.97,
        steps_per_epoch=4000,
        epsilon=0.2,
        maxkl=0.01,
        train_steps=80,
        pol_lr=3e-4,
        val_lr=1e-3,
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
            state_preproc=state_preproc,
            state_sze=state_sze,
            logger_dir=logger_dir,
            tensorboard=tensorboard,
            save_screen=save_screen,
            save_states=save_states
        )

        self.eps = epsilon
        self.maxkl = maxkl
        self.train_steps = train_steps

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=val_lr)

    def update(self):
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [
            torch.Tensor(x) for x in self.buffer.get()
        ]
        _, logp, _ = self.ac.policy(states, acts)
        pol_ratio = (logp - logprobs_old).exp()
        min_adv = torch.where(advs > 0, (1 + self.eps) * advs, (1 - self.eps) * advs)
        pol_loss_old = -(torch.min(pol_ratio * advs, min_adv)).mean()
        approx_ent = (-logp).mean()

        for i in range(self.train_steps):
            _, logp, _ = self.ac.policy(states, acts)
            pol_ratio = (logp - logprobs_old).exp()
            min_adv = torch.where(
                advs > 0, (1 + self.eps) * advs, (1 - self.eps) * advs
            )
            pol_loss = -(torch.min(pol_ratio * advs, min_adv)).mean()

            self.policy_optimizer.zero_grad()
            pol_loss.backward()
            self.policy_optimizer.step()

            _, logp, _ = self.ac.policy(states, acts)
            kl = (logprobs_old - logp).mean()
            if kl > 1.5 * self.maxkl:
                cprint(f"Early stopping at step {i} due to reaching max kl.", "yellow")
                break

        vals = self.ac.value_f(states)
        val_loss_old = F.mse_loss(vals, rets)

        for _ in range(self.train_steps):

            vals = self.ac.value_f(states)
            val_loss = F.mse_loss(vals, rets)
            self.value_optimizer.zero_grad()
            val_loss.backward()
            self.value_optimizer.step()

        approx_kl = kl
        self.logger.store(
            PolicyLoss=pol_loss_old.detach().numpy(),
            ValueLoss=val_loss_old.detach().numpy(),
            KL=approx_kl.detach().numpy(),
            Entropy=approx_ent.detach().numpy(),
            DeltaPolLoss=(pol_loss - pol_loss_old).detach().numpy(),
            DeltaValLoss=(val_loss - val_loss_old).detach().numpy(),
        )
        return (
            pol_loss_old.detach().numpy(),
            val_loss_old.detach().numpy(),
            approx_ent.detach().numpy(),
            approx_kl.detach().numpy(),
        )
