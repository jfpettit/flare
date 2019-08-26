# import required packages
import argparse

from rlpack.algorithms import A2C
from rlpack.neural_nets import ActorCritic
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

# make env 
env = gym.make('Acrobot-v1')
# set up network. A2C is an actor-critic method and requires paramaterizations of the policy and value function. 
# This is defined in rlpack/neural_nets.py
network = ActorCritic(env.observation_space.shape[0], env.action_space.n)

# set up argparser. --watch is a boolean whether or not to watch the agent in the env after training
#                   --plot is bool, whether or not to plot rewards earned over training
parser = argparse.ArgumentParser(description='Get args for running REINFORCE agent on CartPole')
parser.add_argument('--watch', action='store_true', help='choose whether to watch trained agent')
parser.add_argument('--plot', action='store_true', help='choose whether to view plots of reward over training')

# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
    # initialize training object. defined in rlpack/algorithms.py
    trainer = A2C(env, network)
    # According to the gym leaderboard below, Acrobot-v1 is considered an unsolved task, so there is no reward threshold at which it is solved.
    # gym leaderboard: https://github.com/openai/gym/wiki/Leaderboard
    rew, leng = trainer.learn(1000)

    # watch agent interact with environment
    if args.watch:
        obs = env.reset()
        for i in range(10000):
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
        plt.title('A2C reward on Acrobot-v1')
        plt.xlabel('Training steps')
        plt.ylabel('Reward')
        plt.show()
