# import required packages
import argparse

from rlpack.algorithms import REINFORCE
from rlpack.neural_nets import PolicyNet
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

# make env 
env = gym.make('CartPole-v1')
# set up network. REINFORCE only needs a network to paramaterize the policy. This is defined in rlpack/neural_nets.py
network = PolicyNet(4, 2)

# set up argparser. --watch is a boolean whether or not to watch the agent in the env after training
#					--plot is bool, whether or not to plot rewards earned over training
parser = argparse.ArgumentParser(description='Get args for running REINFORCE agent on CartPole')
parser.add_argument('--watch', action='store_true', help='choose whether to watch trained agent')
parser.add_argument('--plot', action='store_true', help='choose whether to view plots of reward over training')

# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
	# initialize training object. defined in rlpack/algorithms.py
	trainer = REINFORCE(env, network)
	# train network for 500 episodes, it stops early if the mean reward over last 100 episodes exceeds the solved_threshold
	# criterion for solving from this gym leaderboard: https://github.com/openai/gym/wiki/Leaderboard
	rew, leng = trainer.train_loop_(500, solved_threshold=400)

	# watch agent interact with environment
	if args.watch:
		obs = env.reset()
		for i in range(1000):
			#action = trainer.action_choice(torch.tensor(obs))
			action = trainer.exploit(torch.tensor(obs))
			obs, reward, done, _ = env.step(int(action))
			env.render()
			if done:
				obs = env.reset()
		env.close()

	# plot reward earned per episode over training
	if args.plot:
		plt.plot(rew)
		plt.title('REINFORCE reward on CartPole-v1')
		plt.xlabel('Training steps')
		plt.ylabel('Reward')
		plt.show()
