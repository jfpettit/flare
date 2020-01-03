import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_
import time
import abc

class BasePolicyGradient:
    def __init__(self, env, actorcritic=nets.FireActorCritic, gamma=.99, lam=.97, steps_per_epoch=4000):
        self.env=env
        self.ac = actorcritic(env.observation_space.shape[0], env.action_space)
        self.steps_per_epoch = steps_per_epoch

        self.buffer = utils.Buffer(env.observation_space.shape, env.action_space.shape, steps_per_epoch, gamma, lam)

    @abc.abstractmethod
    def update(self):
        """Update rule for policy gradient algo."""
        return

    def learn(self, epochs, render=False, solved_threshold=None, horizon=1000):
        if render and 'Bullet' in self.env.unwrapped.spec.id:
            self.env.render()
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0
        for i in range(epochs):
            self.ep_length = []
            self.ep_reward = []
            self.ac.eval()
            for _ in range(self.steps_per_epoch):
                action, _, logp, value = self.ac(torch.Tensor(state.reshape(1, -1)))
                if render and 'Bullet' not in self.env.unwrapped.spec.id:
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