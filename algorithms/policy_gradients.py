import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import GeneralizedAdvantageEstimation, MlpPolicyUtils


class simple_policy_net(nn.Module):
    def __init__(self, size):
        super(simple_policy_net, self).__init__()
        self.size=size
        self.save_log_probs = []
        self.save_rewards = []
        
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, self.size)
        
    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        return x
    
class simple_policy_training:
    def __init__(self, gamma, env, policy, optimizer):
        self.gamma = gamma
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
    
    def end_episode(self):
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
    
    def train_loop(self, render, epochs):
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


class VanillaPolicyGradient:
    def __init__(self, env, network, steps_per_epoch, epochs, gamma=.99, policy_lr=3e-4, value_lr=1e-3, value_train_iters=80, lam=.97, max_ep_len=1000, save_freq=10):
        self.env = env
        self.network = network
        self.epochs=epochs
        self.gamma, self.lam = gamma, lam
        self.policy_lr, self.value_lr = policy_lr, value_lr
        self.val_train_iters = value_train_iters
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.steps_per_epoch = steps_per_epoch

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        self.mlp_utils = MlpPolicyUtils(network, self.env.action_space)

        self.policy, self.logprobs, self.logprobs_policy, self.values = self.mlp_utils.actor_critic_mlp(np.empty(self.obs_dim), np.empty(self.act_dim))

        self.grabbable = [self.policy, self.values, self.logprobs_policy]

        self.       gae = GeneralizedAdvantageEstimation(self.steps_per_epoch, self.gamma, self.lam)




        


    