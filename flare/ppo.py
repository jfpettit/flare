from flare.a2c import A2C
import numpy as np
import torch
import gym
import torch.nn as nn
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_

class PPO(A2C):
    def __init__(self, env, epsilon=0.2, actorcritic=nets.ActorCritic, gamma=0.99, lam=0.95, steps_per_epoch=4000, maxkl=0.01, train_steps=80):
        #super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)
        self.init_shared(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)

        self.maxkl = maxkl
        self.eps = epsilon
        self.train_steps = train_steps

    def loss_fcns(self, **kwargs):
        pol_ratio = torch.exp(kwargs['logprobs'] - kwargs['logprobs_old'].detach())
        surrogate1 = pol_ratio * kwargs['advs'].detach()
        surrogate2 = torch.clamp(pol_ratio, (1.0 - self.eps), (1.0 + self.eps)) * kwargs['advs'].detach()
        pol_loss = -torch.mean(torch.min(surrogate1, surrogate2))
        val_loss = torch.mean((kwargs['rets'].detach() - kwargs['vals_'])**2)
        return pol_loss, val_loss, pol_loss + val_loss

    def update_(self):
        states, acts, advs, rets, logprobs_old, values = self.ac.gather()
        logprobs_old = torch.tensor(logprobs_old).detach()
        vals_ = torch.stack(values).squeeze()
        #logprobs_old = torch.tensor(logprobs_old)
        
        for _ in range(self.train_steps):
            vals_, logprobs_ = self.eval_actions(states, acts)
            vals_ = vals_.squeeze()
            logprobs_ = logprobs_.squeeze()
            #advs = rets - vals_
            pol_loss, val_loss, loss = self.loss_fcns(advs=advs, rets=rets, logprobs=logprobs_, vals_=vals_, logprobs_old=logprobs_old)

            approx_kl = self.approx_kl(logprobs_old, logprobs_)
            #if approx_kl > self.maxkl * 2:
            #    print(f'WARNING: maxkl reached at step {_}. Early stopping.')
            #    break
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ac.parameters(), 1.)
            self.optimizer.step()
            #logprobs_old = torch.clone(logprobs_).detach()

        approx_ent = torch.mean(-logprobs_)
        approx_kl = self.approx_kl(logprobs_old, logprobs_)

        return pol_loss.detach().numpy().mean(), val_loss.detach().numpy(), approx_ent.detach().numpy(), approx_kl.detach().numpy()