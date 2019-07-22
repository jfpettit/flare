import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal

class simple_policy_training:
    def __init__(self, gamma, env, policy, optimizer):
        self.gamma = gamma
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
    
    def end_episode_(self):
        return_ = 0
        policy_loss = []
        returns = []
        for reward in self.policy.save_rewards[::-1]:
            return_ = reward + self.gamma * return_
            returns.insert(0, return_)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / returns.std()
        for log_prob, return_ in zip(self.policy.save_log_probs, returns):
            policy_loss.append(-log_prob * return_)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.save_rewards[:]
        del self.policy.save_log_probs[:]

    def action_choice(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probabilities = self.policy(state)
        m_ = torch.distributions.Categorical(action_probabilities)
        choice = m_.sample()
        self.policy.save_log_probs.append(m_.log_prob(choice))
        return choice.item()
    
    def train_loop_(self, render, epochs):
        running_reward = 0
        self.ep_length = []
        self.ep_reward = []
        for i in range(epochs):
            state, episode_reward = self.env.reset(), 0
            for s in range(1, 10000):
                action = self.action_choice(state)
                state, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                self.policy.save_rewards.append(reward)
                episode_reward += reward
                if done:
                    self.ep_length.append(s)
                    self.ep_reward.append(episode_reward)
                    break
                
            running_reward += 0.05 * episode_reward  + (1-0.05) * running_reward
            self.end_episode()
            self.env.close()
            
class GeneralizedAdvantageEstimation:
    def __init__(self, size, gamma=.99, lam=.95):
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.logprob_buf = np.zeros(size, dtype=np.float32)
        self.lam, self.gamma = lam, gamma
        self.point, self.path_start_index, self.max_size = 0, 0, size

    def store_(self, reward, value, logprob):
        assert self.point < self.max_size
        self.reward_buf[self.point] = reward
        self.value_buf[self.point] = value
        self.logprob_buf[self.point] = logprob
        self.point += 1

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def end_path_(self, last_value=0):
        slice_path = slice(self.path_start_index, self.point)
        rewards = np.append(self.reward_buf[slice_path], last_value)
        values = np.append(self.value_buf[slice_path], last_value)

        delts = [:-1] + self.gamma * values[1:] - values[-1:]
        self.adv_buf[slice_path] = self.discount_cumsum(delts, self.gamma * self.lam)

        self.return_buf[slice_path] = self.discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_index = self.point

    def scaler(self, x):
        x = np.asarray(x)
        mean = x.mean()
        std = np.sqrt((np.sum(x-mean)**2)/len(x))
        return mean, std

    def grab(self):
        assert self.point == self.max_size

        self.point, self.path_start_index = 0, 0
        advantage_mean, advantage_std = self.scaler(self.adv_buf)
        self.adv_buf = (self.adv_buf - advantage_mean)/advantage_std

        return [self.adv_buf, self.return_buf, self.logprob_buf]


class MathUtils:
    def __init__(self):
        super(MathUtils, self).__init__()
        self.EPS = 1e-8

    def reduce_sum(self, x, axis=None):
        x = np.asarray(x)
        if axis is None:
            return x.sum()
        return x.sum(axis=axis)

    def reduce_mean(self, x, axis=None):
        x = np.asarray(x)
        if axis is None:
            return x.mean()
        return x.mean(axis=axis )

    def one_hot_encoder(self, vec, depth):
        vec = np.asarray(vec)
        encoding = np.copy(vec)
        vec_imax = vec[i].max()
        encoding[encoding < vec_imax] = 0
        encoding[encoding == vec_imax] = 1
        return encoding[:, :depth]

    def gaussian_likelihood(self, x, mu, log_std):
        vals = -.5 * (((x - mu)/np.exp(log_std)+self.EPS))**2 + 2 * log_std + np.log(2*np.pi)
        return self.reduce_sum(vals, axis=1)




class MlpPolicyUtils:
    def __init__(self, mlp, action_space):
        self.mlp = mlp
        self.action_space = action_space
        self._math = MathUtils()

    def categorical_mlp(self, observations, actions):
        act_dim = self.action_space.n

        logits = mlp(observations)
        logprob_all = F.log_softmax(logits)
        policy = torch.squeeze(torch.distributions.multinomial.Multinomial(logits=logits), dim=1)
        logprobs = self._math.reduce_sum(self._math.one_hot_encoder(actions, depth=act_dim) * logprob_all, axis=1)
        logprobs_policy = self._math.reduce_sum(self._math.one_hot_encoder(policy.numpy(), depth=act_dim) * logprob_all, axis=1)
        return policy. logprobs, logprobs_policy

    def gaussian_mlp(self, observations, actions):
        act_dim = actions.shape.as_list()[-1]
        mu = self.mlp(observations)
        mu_ = mu.numpy()
        log_std = -.5 * np.ones(act_dim, dtype=np.float32)
        std = np.exp(log_std)
        policy = mu_ * np.random.normal(size=mu_) * std
        logprobs = self._math.gaussian_likelihood(actions, mu_, log_std)
        logprobs_policy = self._math.gaussian_likelihood(policy, mu_, log_std)
        return policy, logprobs, logprobs_policy

    def actor_critic_mlp(self, observations, actions):
        if isinstance(self.action_space, Box):
            policy = self.gaussian_mlp
        elif isinstance(self.action_space, Discrete):
            policy = self.categorical_mlp

        policy, logprobs, logprobs_policy = policy(observations, actions)
        values = torch.squeeze(self.mlp(observations), axis=1)
        return policy, logprobs, logprobs_policy, values



            
