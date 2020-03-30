# import needed packages
import numpy as np
import gym
import torch
import flare.neural_nets as nets
from flare import utils
from flare.polgrad import BasePolicyGradient
import torch.nn.functional as F

class A2C(BasePolicyGradient):
    def __init__(self, env, hidden_sizes=(32, 32), actorcritic=nets.FireActorCritic, gamma=.99, lam=.97, steps_per_epoch=4000, pol_lr=3e-4, val_lr=1e-3, logstd_anneal=None, state_preproc=None, state_sze=None, logger_dir=None):
        super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch, hid_sizes=hidden_sizes, logstd_anneal=logstd_anneal, state_sze=state_sze, state_preproc=state_preproc, logger_dir=logger_dir)
        
        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=pol_lr)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=val_lr)

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

        _, _, logp, vals = self.ac(states, acts)
        pol_loss_new = -(logp*advs).mean()
        val_loss_new = F.mse_loss(vals, rets)
        approx_kl = (logprobs_old - logp).mean()
        self.logger.store(PolicyLoss=pol_loss_old.detach().numpy(), ValueLoss=val_loss_old.detach().numpy(), KL=approx_kl.detach().numpy(), Entropy=approx_ent.detach().numpy(), DeltaPolLoss=(pol_loss - pol_loss_old).detach().numpy(), DeltaValLoss=(val_loss-val_loss_old).detach().numpy())
        return pol_loss, val_loss_old, approx_ent, approx_kl
        