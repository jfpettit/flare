from flare.base import BasePolicyGradient
import numpy as np
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_
import time

class PPO(BasePolicyGradient):
    def __init__(self, env, epsilon=0.2, actorcritic=nets.FireActorCritic, gamma=0.99, lam=0.97, steps_per_epoch=4000, maxkl=0.01, train_steps=80):
        super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)
        self.eps = epsilon
        self.maxkl = maxkl
        self.train_steps = train_steps
        
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=1e-3)

    def update(self):
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [torch.Tensor(x) for x in self.buffer.get()]
        _, logp, _ = self.ac.policy(states, acts)
        pol_ratio = (logp - logprobs_old).exp()
        min_adv = torch.where(advs > 0, (1 + self.eps) * advs,
                              (1 - self.eps) * advs)
        pol_loss_old = -(torch.min(pol_ratio*advs, min_adv)).mean()
        approx_ent = (-logp).mean()

        for i in range(self.train_steps):
            _, logp, _ = self.ac.policy(states, acts)
            pol_ratio = (logp - logprobs_old).exp()
            min_adv = torch.where(advs > 0, (1 + self.eps) * advs,
                              (1 - self.eps) * advs)
            pol_loss = -(torch.min(pol_ratio*advs, min_adv)).mean()

            self.policy_optimizer.zero_grad()
            pol_loss.backward()
            self.policy_optimizer.step()

            _, logp, _ = self.ac.policy(states, acts)
            kl = (logprobs_old - logp).mean()
            if kl > 1.5 * self.maxkl:
                print(
                    'Early stopping at step %d due to reaching max kl.' % i)
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
        return pol_loss_old.detach().numpy(), val_loss_old.detach().numpy(), approx_ent.detach().numpy(), approx_kl.detach().numpy()