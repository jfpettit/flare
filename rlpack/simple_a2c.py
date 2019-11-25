import pybullet_envs
import roboschool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

def gaussian_likelihood(x, mu, log_std):
    vals = -.5 * (((x - mu)/torch.exp(log_std)+1e-8))**2 + 2 * log_std + torch.log(2*torch.tensor(np.pi))
    return torch.sum(vals, axis=0)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.mus = []
        self.logps = torch.zeros((0,))
        
        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, action_dim)

        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.tanh(self.layer1(obs))
        x = self.tanh(self.layer2(x))
        mu = self.layer3(x)
        self.mus.append(mu)
        a, logp = self.get_action(mu)
        self.logps = torch.cat((self.logps, logp.reshape(-1, 1)))
        return a

    def get_action(self, x):
        a_dims = x.size()
        logstd = -0.5 * torch.ones(a_dims)
        a = x + torch.randn(x.size()) * torch.exp(logstd)
        logp = gaussian_likelihood(x, a, logstd)
        return a, logp

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.vals = torch.zeros((0,))
        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.tanh(self.layer1(obs))
        x = self.tanh(self.layer2(x))
        val = self.layer3(x)
        self.vals = torch.cat((self.vals, val.reshape(-1,1)))
        return val

def combined_shape(length, shape=None):
    """
    Adapted from OpenAI SpinningUp code
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class Memory:
    def __init__(self, size, env, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma=gamma
        self.lam=lam

        #self.states = torch.zeros(combined_shape(size, env.observation_space.shape), dtype=torch.float32)
        self.actions = torch.zeros(combined_shape(size, env.action_space.shape), dtype=torch.float32)
        self.advantage_record = torch.zeros(size, dtype=torch.float32)
        self.return_record = torch.zeros(size, dtype=torch.float32)
        self.rew_record = torch.zeros(size, dtype=torch.float32)
        self.value_record = torch.zeros(size, dtype=torch.float32)

        self.point_idx, self.start_idx = 0, 0

    def push(self, action, reward, value):
        assert self.point_idx < self.size

        #self.states[self.point_idx] = state
        self.actions[self.point_idx] = action
        self.rew_record[self.point_idx] = reward
        self.value_record[self.point_idx] = value

        self.point_idx += 1

    def discount_cumulative_sum(self, x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def end_trajectory(self, last_value=0):
        traj_slice = slice(self.start_idx, self.point_idx)

        rews = torch.cat((self.rew_record[traj_slice], torch.tensor(last_value)))
        vals = torch.cat((self.value_record[traj_slice], torch.tensor(last_value)))

        rews_np, vals_np = rews.detach().numpy(), vals.detach().numpy()
        deltas = rews_np[:-1] + self.gamma * vals_np[1:] - vals_np[:-1]

        self.advantage_record[traj_slice] = torch.tensor(list(self.discount_cumulative_sum(
            deltas, self.gamma * self.lam)))

        self.return_record[traj_slice] = torch.tensor(list(self.discount_cumulative_sum(
            rews_np, self.gamma)[:-1]))

        self.start_idx = self.point_idx

    def gather(self):
        assert self.point_idx == self.size, 'Buffer has to be full before you can gather.'

        self.point_idx, self.start_idx = 0, 0
        advantage_mean, advantage_std = torch.mean(self.advantage_record), torch.std(self.advantage_record)
        self.advantage_record = (self.advantage_record - advantage_mean)/(advantage_std + 1e-8)

        return [self.advantage_record, self.return_record]

def actor_loss(logps, advs):
    return -torch.mean(logps*advs)

def critic_loss(returns, vals):
    return torch.mean((returns - vals)**2)

def update(logps, advs, returns, vals, act_opt, val_opt):
    pi_loss = actor_loss(logps, advs)
    val_loss = critic_loss(returns, vals)

    act_opt.zero_grad()
    pi_loss.backward()
    act_opt.step()

    for _ in range(80):
        val_opt.zero_grad()
        val_loss.backward(retain_graph=True)
        val_opt.step()

    return pi_loss, val_loss

def learn(env, epochs, actor, critic, act_opt, val_opt, horizon=250):
    memory = Memory(4000, env, gamma=0.99)
    obs, rew, done, ep_return, ep_length = env.reset(), 0, False, 0, 0

    for epoch in range(epochs):
        epochrew = []
        epochlen = []
        for step in range(4000):
            act = actor(torch.tensor(obs).float())
            value = critic(torch.tensor(obs).float())

            memory.push(act, rew, value)

            obs, rew, done, _ = env.step(act.detach().numpy())

            ep_return += rew
            ep_length += 1

            over = done or (ep_length == horizon)
            if over or (step == 4000 - 1):
                last_value = (rew if done else critic(torch.tensor(obs).float()),)
                memory.end_trajectory(last_value=last_value)
                if over:
                    epochlen.append(ep_length)
                    epochrew.append(ep_return)
                        
                    obs, rew, done, ep_return, ep_length = (
                        env.reset(),
                        0,
                        False,
                        0,
                        0,
                    )
        advs, returns = memory.gather()
        update(actor.logps, advs, returns, critic.vals[1:], act_opt, val_opt)
        actor.logps, critic.vals = torch.zeros((0,)), torch.zeros((0,))

        print(f"Epoch: {epoch}\n",
            f"MeanEpReturn: {np.mean(epochrew)}\n",
            f"MeanEpLen: {np.mean(epochlen)}")
        
if __name__ == '__main__':
    env = gym.make("RoboschoolInvertedPendulum-v1")
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic = Critic(env.observation_space.shape[0])

    act_opt = torch.optim.Adam(actor.parameters())
    val_opt = torch.optim.Adam(critic.parameters())

    learn(env, 50, actor, critic, act_opt, val_opt)
