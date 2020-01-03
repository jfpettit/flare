# import needed packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import flare.neural_nets as nets
from flare import utils
from flare.base import BasePolicyGradient
from torch.nn.utils import clip_grad_norm_
import time

class A2C(BasePolicyGradient):
    def __init__(self, env, actorcritic=nets.FireActorCritic, gamma=.99, lam=.97, steps_per_epoch=4000):
        super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)
        
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=1e-3)

    def update(self):
        self.ac.train()
        states, acts, advs, rets, logprobs_old = [torch.Tensor(x) for x in self.buffer.get()]
        
        _, logp, _ = self.ac.policy(states, acts)
        approx_ent = torch.mean(-logp)

        pol_loss = -(logp*advs).mean()

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

        approx_kl = (logprobs_old - logp).mean()

        return pol_loss, val_loss_old, approx_ent, approx_kl
        