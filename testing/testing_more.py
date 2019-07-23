from rlpack.algorithms import PPO, ActorCritic, REINFORCE
from rlpack.neural_nets import SimpleActorCritic, SimplePolicyNet
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rlpack.utils import AdvantageEstimatorsUtils
from itertools import count


cartpole = gym.make('CartPole-v0')
acro_env = gym.make('Acrobot-v1')
#biped = gym.make('BipedalWalker-v2')
lander = gym.make('LunarLander-v2')
snek = gym.make('gym_snake_rl:BasicSnake-big_vector-16-v0')

class model1(nn.Module):
	def __init__(self, in_size, out_size):
		super(model1, self).__init__()
		self.save_log_probs = []
		self.save_rewards = []
		self.save_values = []

		self.fc1 = nn.Linear(in_size, 20)
		self.fc2 = nn.Linear(20, out_size)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = self.fc2(x)
		#return F.softmax(x, dim=-1)
		return x

class model2(nn.Module):
	def __init__(self, in_size, out_size):
		super(model2, self).__init__()
		self.save_log_probs = []
		self.save_rewards = []
		self.save_values = []

		self.fc1 = nn.Linear(in_size, 20)
		self.fc2 = nn.Linear(20, 20)
		self.act = nn.Linear(20, out_size)
		self.val = nn.Linear(20, 1)


	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		action = self.act(x)
		value = self.val(x)
		return F.softmax(action, dim=-1), value
		#return action


if __name__ == '__main__':
	env = gym.make('CartPole-v0')

	m1 = model1(4, 2)
	m2 = model2(4, 2)
	landerm1 = model1(8, 4)
	landerm2 = model2(8, 4)

	#model = SimpleActorCritic(4, 2)
	model = SimplePolicyNet(4, 2)

	tr = REINFORCE(env, model)
	#tr = ActorCritic(env, model)
	tr.train_loop_(False, 1000)

	

	obs = env.reset()
	for i in range(1000):
		action = tr.action_choice(torch.tensor(obs))
		obs, reward, done, _ = env.step(int(action))
		env.render()
		if done:
			obs = env.reset()
	env.close()




	'''
	episodes = [acro_env, lander]
	out_sizes = [3, 4]
	for i in range(len(episodes)):
		AE = AdvantageEstimatorsUtils(0.98, 0.96)

		#m1 = model1(6, 3)
		#m2 = model2(6, 3)

		ac = SimpleActorCritic(*episodes[i].observation_space.shape, out_sizes[i])
		#pol = SimplePolicyNet(6, 3)

		vpg = VanillaPolicyGradient(episodes[i], ac, AE.gae_lambda, gamma=.99, lam=.95, steps_per_epoch=100)
		#spt = SimplePolicyTraining(0.98, acro_env, pol)

		vpg_reward, vpg_length = vpg.train_loop_(False, 10000)
		#spt_reward, spt_length = spt.train_loop_(False, 1000)


		window = 10
		avg_ac_reward = [np.mean(vpg_reward[i:i+1]) for i in np.arange(0, len(vpg_reward), window)]
		#avg_r_reward = [np.mean(spt_reward[i:i+1]) for i in np.arange(0, len(spt_reward), window)]
		plt.plot(range(len(avg_ac_reward)), avg_ac_reward, label='GAE-lambda estimation')
		#plt.plot(range(len(avg_r_reward)), avg_r_reward, label='No Advantage estimation')
		plt.title('Averaged Reward over training, window size = 10')
		plt.xlabel('epochs')
		plt.ylabel('Average Reward')
		plt.legend()
		plt.show()

		plt.plot(range(len(vpg_reward)), vpg_reward, label='GAE-lambda estimation')
		#plt.plot(range(len(spt_reward)), spt_reward, label='No Advantage estimation')
		plt.title('Reward over training')
		plt.xlabel('epochs')
		plt.ylabel('Reward')
		plt.legend()
		plt.show()

		
		avg_ep_len = [np.mean(vpg_length[i:i+1]) for i in np.arange(0, len(vpg_length), window)]
		plt.plot(range(len(avg_ep_len)), avg_ep_len)
		plt.title('Average episode length over training')
		plt.xlabel("Epochs")
		plt.ylabel('length')
		plt.show()
		

		watch_agent = True
		env = episodes[i]
		if watch_agent:
			obs, rew = env.reset(), 0
			for i in range(1000):
				action = vpg.action_choice(obs)
				obs, rew, done, _ = env.step(action)
				env.render()
				if done:
					obs, rew = env.reset(), 0

			env.close()

	
			obs, rew = acro_env.reset(), 0
			for i in range(1000):
				action = spt.action_choice(obs)
				obs, rew, done, _ = acro_env.step(action)
				acro_env.render()
				if done:
					obs, rew = acro_env.reset(), 0
			
	'''
