from flare.a2c import A2C
import numpy as np
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_
import time

class PPO(A2C):
    def __init__(self, env, epsilon=0.2, actorcritic=nets.FireActorCritic, gamma=0.99, lam=0.95, steps_per_epoch=4000, maxkl=0.01, train_steps=80):
        #super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)
        self.init_shared(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)

        self.maxkl = maxkl
        self.eps = epsilon
        self.train_steps = train_steps

        self.val_loss = nn.MSELoss()

    def loss_fcns(self, **kwargs):
        #import ipdb; ipdb.set_trace()
        pol_ratio = torch.exp(kwargs['logprobs'] - kwargs['logprobs_old'])
        surrogate1 = pol_ratio * kwargs['advs']
        #surrogate2 = torch.clamp(pol_ratio * kwargs['advs'], (1.0 - self.eps), (1.0 + self.eps))
        #pol_ratio[pol_ratio > ((1.0 + self.eps)* kwargs['advs'])] = 1.0 + self.eps
        #pol_ratio[pol_ratio < ((1.0 - self.eps)* kwargs['advs'])] = 1.0 - self.eps
        #inds1 = torch.where(pol_ratio * kwargs['advs'] > (1 + self.eps))
        #inds2 = torch.where(pol_ratio * kwargs['advs'] < (1 - self.eps))
        #pol_ratio[inds1] = 1 + self.eps
        #pol_ratio[inds2] = 1 - self.eps
        surrogate2 = torch.where(kwargs['advs'] > 0, (1+self.eps) * kwargs['advs'], (1 - self.eps) * kwargs['advs'])
        #surrogate2 = pol_ratio
        pol_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        #val_loss = 0.5 * torch.mean((kwargs['rets'] - kwargs['vals_'])**2)
        val_loss = 0.5 * self.val_loss(kwargs['rets'], kwargs['vals_'])
        clipped = (pol_ratio > (1 + self.eps)) | (pol_ratio < (1 - self.eps))
        self.eps = torch.mean(clipped.float())
        return pol_loss, val_loss, pol_loss + val_loss, pol_ratio

    def update_(self):
        states, acts, advs, rets, logprobs_old = [torch.Tensor(x) for x in self.buffer.get()]
        _, logp, _ = self.ac.policy(states, acts)
        pol_ratio = torch.exp(logp - logprobs_old)
        min_adv = torch.where(advs > 0, (1 + self.eps) * advs,
                              (1 - self.eps) * advs)
        approx_ent = (-logp).mean()

        for i in range(self.train_steps):
            _, logp, _ = self.ac.policy(states, acts)
            pol_ratio = torch.exp(logp - logprobs_old)
            min_adv = torch.where(advs > 0, (1 + self.eps) * advs,
                              (1 - self.eps) * advs)
            pol_loss = -torch.min(pol_ratio*advs, min_adv).mean()

            self.policy_optimizer.zero_grad()
            pol_loss.backward()
            #grad_norm = nn.utils.clip_grad_norm_(self.ac.policy.parameters(), 1.)
            self.policy_optimizer.step()

            _, logp, _ = self.ac.policy(states, acts)
            kl = (logprobs_old - logp).mean()
            kl = np.mean(kl.item())
            #if kl > 1.5 * self.maxkl:
            #    print(
            #        'Early stopping at step %d due to reaching max kl.' % i)
            #    break

        for _ in range(self.train_steps):
            
            vals = self.ac.value_f(states)
            val_loss = F.mse_loss(vals, rets)
            self.value_optimizer.zero_grad()
            val_loss.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(
            #   self.ac.value_f.parameters(), 1.)
            self.value_optimizer.step()
            
            #logprobs_old = torch.clone(logprobs_).detach()
        #clipped = (pol_ratio > (1 + self.eps)) | (pol_ratio < (1 - self.eps))
        #self.eps = torch.mean(clipped.float())
        approx_kl = kl
        return pol_loss.detach().numpy(), val_loss.detach().numpy(), approx_ent.detach().numpy(), approx_kl