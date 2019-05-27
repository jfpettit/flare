import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
            
            
