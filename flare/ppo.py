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
    def __init__(self, env, epsilon=0.2, actorcritic=nets.FireActorCritic, gamma=0.99, lam=0.97, steps_per_epoch=4000, maxkl=0.01, train_steps=80):
        #super().__init__(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)
        #self.init_shared(env, actorcritic=actorcritic, gamma=gamma, lam=lam, steps_per_epoch=steps_per_epoch)

        self.env = env
        self.eps = epsilon
        self.ac = actorcritic(env.observation_space.shape[0], env.action_space)
        self.gamma = gamma
        self.lam = lam 
        self.steps_per_epoch = steps_per_epoch
        self.maxkl = maxkl
        self.train_steps = train_steps

        self.buffer = utils.Buffer(self.env.observation_space.shape, env.action_space.shape, steps_per_epoch, gamma, lam)

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=1e-3)


        self.maxkl = maxkl
        self.eps = epsilon
        self.train_steps = train_steps

        self.val_loss = nn.MSELoss()

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
    '''
    def learn(self, epochs, render=False, solved_threshold=None, horizon=1000):
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0
        for i in range(epochs):
            self.ep_length = []
            self.ep_reward = []
            self.ac.eval()
            for _ in range(self.steps_per_epoch):
                action, _, logp, value = self.ac(torch.Tensor(state.reshape(1, -1)))
                if render:
                    self.env.render()
                self.buffer.store(state, action.detach().numpy(), reward, value.item(), logp.detach().numpy())
                state, reward, done, _ = self.env.step(action.detach().numpy()[0])
                episode_reward += reward
                episode_length += 1
                over = done or (episode_length == horizon)
                if over or (_ == self.steps_per_epoch - 1):
                    last_val = reward if done else self.ac.value_f(torch.Tensor(state.reshape(1, -1))).item()
                    self.buffer.finish_path(last_val)
                    if over:    
                        self.ep_length.append(episode_length)
                        self.ep_reward.append(episode_reward)
                    state = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    done = False
                    reward = 0
            pol_loss, val_loss, approx_ent, approx_kl = self.update()
            if solved_threshold and len(self.ep_reward) > 100:
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length
            print(f'\rEpoch {i} of {epochs}\n',
            f'MeanEpRet: {np.mean(self.ep_reward)}\n',
            f'StdEpRet: {np.std(self.ep_reward)}\n',
            f'MaxEpRet: {np.max(self.ep_reward)}\n',
            f'MinEpRet: {np.min(self.ep_reward)}\n',
            f'MeanEpLen: {np.mean(self.ep_length)}\n',
            f'StdEpLen: {np.std(self.ep_length)}\n',
            f'PolicyLoss: {pol_loss}\n',
            f'ValueLoss: {val_loss}\n',
            f'ApproxEntropy: {approx_ent}\n',
            f'ApproxKL: {approx_kl}\n',
            f'Env: {self.env.unwrapped.spec.id}\n')
        print('\n')
        return self.ep_reward, self.ep_length
    '''