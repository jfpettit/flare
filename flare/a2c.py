# import needed packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_
import time

class A2C:
    def __init__(self, env, actorcritic=nets.FireActorCritic, gamma=.99, lam=.97, steps_per_epoch=4000):
        
        self.init_shared(env, actorcritic, gamma, lam, steps_per_epoch)

    def init_shared(self, env, actorcritic, gamma, lam, steps_per_epoch):
        self.env = env
        
        self.ac = actorcritic(self.env.observation_space.shape[0], self.env.action_space)
        self.buffer = utils.Buffer(env.observation_space.shape, env.action_space.shape, steps_per_epoch, gamma=gamma, lam=lam)

        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=1e-3)

    def update(self):
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

    def exploit(self, state):
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities, value = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item() 