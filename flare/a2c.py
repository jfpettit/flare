# import needed packages
import numpy as np
import gym
import torch
import torch.nn as nn
import flare.neural_nets as nets
from flare import utils
from torch.nn.utils import clip_grad_norm_

# figure whether a GPU is available. In the future, this will be changed to use torch.tensor().to(device) syntax to maximize GPU usage
use_gpu = True if torch.cuda.is_available() else False

class A2C:
    def __init__(self, env, actorcritic=nets.ActorCritic, gamma=.99, lam=.95, steps_per_epoch=4000):
        
        self.init_shared(env, actorcritic, gamma, lam, steps_per_epoch)

    def init_shared(self, env, actorcritic, gamma, lam, steps_per_epoch):
        self.env = env
        
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        act_space = env.action_space.n if not self.continuous else env.action_space.shape[0]
        self.ac = actorcritic(env.observation_space.shape[0], act_space)
        self.ac.buffer_init(steps_per_epoch, gamma=gamma, lam=lam)

        if use_gpu:
            self.ac.cuda()

        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch

        self.optimizer = torch.optim.Adam(self.ac.parameters())

    def action_choice(self, state):
        # convert state to torch tensor, dtype=float
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()
        
        if not self.continuous:
            logsoft = nn.LogSoftmax(dim=-1)
            soft = nn.Softmax(dim=-1)
            action_logits, state_value = self.ac(state)
            all_logprobs = logsoft(action_logits)
            pi = torch.multinomial(soft(action_logits), 1)
            onehot = torch.zeros((self.env.action_space.n,))
            onehot[pi] = 1
            logprobs_act = torch.sum(onehot * all_logprobs)
            action = pi.detach().numpy()[0]
        
        elif self.continuous:
            act_dim = self.env.action_space.shape[0]
            mu, state_value = self.ac(state)
            log_stds = -0.5 * torch.ones(act_dim)
            std = torch.exp(log_stds)
            pi = mu + torch.randn(mu.shape) * std
            logprobs_act = utils.gaussian_likelihood(pi, mu, log_stds)
            action = pi.detach().numpy()
        
        return action, state_value, logprobs_act    

    def update_(self):
        states, acts, advs, rets, logprobs, values = self.ac.gather()
        logprobs_ = torch.stack(logprobs)
        vals_ = torch.stack(values).squeeze()
        
        if use_gpu:
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            vals_ = vals_.cuda()

        pol_loss = -torch.mean(logprobs_ * advs)
        val_loss = 0.5 * torch.mean((rets - vals_)**2)
        
        self.optimizer.zero_grad()
        loss = pol_loss + 0.5 * val_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ac.parameters(), 1)
        self.optimizer.step()

        return pol_loss, val_loss
        
    def learn(self, epochs, render=False, solved_threshold=None):
        for i in range(epochs):
            self.ep_length = []
            self.ep_reward = []
            state, episode_reward, episode_length = self.env.reset(), 0, 0
            for _ in range(self.steps_per_epoch):
                action, value, logprob = self.action_choice(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                self.ac.store(state, action, reward, value, logprob)
                episode_reward += reward
                episode_length += 1
                over = done or (episode_length == 1000)
                if over or (_ == self.steps_per_epoch - 1):
                    last_val = reward if done else self.ac(torch.from_numpy(state).float())[1]
                    self.ac.end_traj(last_val=last_val)
                    if over:
                        state = self.env.reset()
                        self.ep_length.append(episode_length)
                        self.ep_reward.append(episode_reward)
                        episode_reward = 0
                        episode_length = 0
                        done = False
                        reward = 0
            pol_loss, val_loss = self.update_()
            self.ac.clear_mem()
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