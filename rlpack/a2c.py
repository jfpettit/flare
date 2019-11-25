# import needed packages
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn as nn
from rlpack.utils import AdvantageEstimatorsUtils as aeu
from rlpack.utils import Buffer
from gym.spaces import Box, Discrete
from rlpack import utils

use_gpu = True if torch.cuda.is_available() else False

class A2C:
    def __init__(self, env, pol_model, val_model, adv_fn=None, gamma=.99, lam=.97, steps_per_epoch=4000, optimizer=optim.Adam,
        standardize_rewards=True, value_train_iters=80, val_loss=nn.MSELoss(), verbose=True):
        self.env = env
        self.policy_net = pol_model
        self.value_net = val_model
        if use_gpu:
            self.policy_net.cuda()
            self.valu_net.cuda()

        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.standardize_rewards = standardize_rewards

        self.policy_optimizer = optimizer(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optimizer(self.value_net.parameters(), lr=1e-3)

        # advantage estimator setup
        if adv_fn is not None:
            self.adv_fn = adv_fn
        else:
            self.aeu_ = aeu(self.gamma, self.lam)
            self.adv_fn = self.aeu_.basic_adv_estimator

        self.value_train_iters = value_train_iters
        self.val_loss = val_loss
        self.verbose = verbose
        self.continuous = True if type(env.action_space) is gym.spaces.box.Box else False

        if isinstance(env.action_space, Box):
            action_space = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_space = 1
        self.buffer = Buffer(env.observation_space.shape[0], action_space, steps_per_epoch, gamma=gamma, lam=lam)

        self.logprobs = []

    def action_choice(self, state):
        # convert state to torch tensor, dtype=float
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        state_value = self.value_net(state)
        # same as with REINFORCE, sample action from categorical distribution paramaterizes by the network's output of action probabilities
        if isinstance(self.env.action_space, Box):
            act_dims = list(self.env.action_space.shape)[-1]
            mu = self.policy_net(state)
            log_std = -0.5 * torch.ones(act_dims)
            std = torch.exp(log_std)
            action = mu + torch.randn(mu.size()) * std
            lp = utils.gaussian_likelihood(action, mu, log_std)

        elif isinstance(self.env.action_space, Discrete):
            action_probabilities = self.policy_net(state)
            m_ = torch.distributions.Categorical(action_probabilities)
            action = m_.sample()
            lp = m_.log_prob(action)

        self.approx_entropy = -lp.mean()
        self.logprobs.append(lp)
        return action, lp.mean(), state_value

    def update_(self):
        # update function is more or less the same as REINFORCE. I'll highlight the important differences.

        states, actions, advs, returns, logprobs_ = self.buffer.gather()

        if use_gpu:
            logprobs_ = logprobs_.cuda()
            returns = returns.cuda()
            advs = advs.cuda()

        #action_mu, action_logstd = self.policy_net(states.detach())
        #logprobs_ = self.calc_log_probs(action_mu.detach(), action_logstd.detach(), actions.detach())

        self.policy_optimizer.zero_grad()
        pol_loss = -(torch.tensor(logprobs_) * advs).mean()
        pol_loss.backward()
        self.policy_optimizer.step()

        # estimate advantage and policy and value loss for each sample in the batch
        for _ in range(self.value_train_iters):
            vals = self.value_net(states)
            val_loss = (self.val_loss(torch.squeeze(returns), torch.squeeze(vals)))

            self.value_optimizer.zero_grad()
            loss = val_loss
            loss.backward()
            self.value_optimizer.step()

        return pol_loss.detach(), val_loss.detach()

    def learn(self, epochs, solved_threshold=None, render_epochs=None, render_frames=1000):
        # functionality and stuff happening in the learn function is the same as REINFORCE
        self.ep_length = []
        self.ep_reward = []
        state, episode_reward, episode_len = self.env.reset(), 0, 0
        for i in range(epochs):
            epochrew = []
            epochlen = []
            for s in range(self.steps_per_epoch):

                action, logprob, value = self.action_choice(state)
                state, reward, done, _ = self.env.step(action.detach().numpy())

                self.buffer.push(state, action, reward, value.detach().numpy(), logprob)
                episode_reward += reward
                episode_len += 1

                if done or s == self.steps_per_epoch-1:
                    state = torch.from_numpy(state).float()
                    last_value = reward if done else self.value_net(state).detach().numpy()
                    self.buffer.end_trajectory(last_value)
                    if done:
                        self.ep_length.append(episode_len)
                        self.ep_reward.append(episode_reward)
                        epochlen.append(episode_len)
                        epochrew.append(episode_reward)

                    state, reward, done, episode_reward, episode_len = self.env.reset(), 0, False, 0, 0

            pol_loss, val_loss = self.update_()
            self.buffer.reset()
            self.logprobs = []

            if solved_threshold and len(self.ep_reward) > 100:
                if np.mean(self.ep_reward[i-100:i]) >= solved_threshold:
                    print('\r Environment solved in {} steps. Ending training.'.format(i))
                    return self.ep_reward, self.ep_length

            if self.verbose:
                print(
                    '-------------------------------\n'
                    'Epoch {} of {}\n'.format(i+1, epochs),
                    'EpRewardMean: {}\n'.format(np.mean(epochrew)),
                    'EpRewardStdDev: {}\n'.format(np.std(epochrew)),
                    'EpRewardMax: {}\n'.format(np.max(epochrew)),
                    'EpRewardMin: {}\n'.format(np.min(epochrew)),
                    'EpLenMean: {}\n'.format(np.mean(epochlen)),
                    'PolicyEntropy: {}\n'.format(self.approx_entropy),
                    'PolicyLoss: {}\n'.format(pol_loss),
                    'ValueLoss: {}\n'.format(val_loss),
                    '\n', end='')
            if render_epochs is not None and i in render_epochs:
                state = self.env.reset()
                for i in range(render_frames):
                    action, logprob, value = self.action_choice(state)
                    state, reward, done, _ = self.env.step(action.detach().numpy())
                    self.env.render()
                    if done:
                        state = self.env.reset()
                self.env.close()
        print('\n')
        return self.ep_reward, self.ep_length

    def exploit(self, state):
        # this is also the same as REINFORCE
        state = np.asarray(state)
        state = torch.from_numpy(state).float()
        if use_gpu:
            state = state.cuda()

        action_probabilities, value = self.model(state)
        action = torch.argmax(action_probabilities)
        return action.item()
