# import required packages
import argparse

from flare.a2c import A2C
from flare.neural_nets import ActorCritic
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers

# make env 
env = gym.make('LunarLander-v2')
# set up network. A2C is an actor-critic method and requires paramaterizations of the policy and value function. 
# This is defined in flare/neural_nets.py
network = ActorCritic(env.observation_space.shape[0], env.action_space.n)

# set up argparser. --watch is a boolean whether or not to watch the agent in the env after training
#                   --plot is bool, whether or not to plot rewards earned over training
parser = argparse.ArgumentParser()
parser.add_argument('--watch', type=bool, help='choose whether to watch trained agent', default=False)
parser.add_argument('--plot', type=bool, help='choose whether to view plots of reward over training', default=True)
parser.add_argument('--save_mv', type=bool, help='choose whether to save a mp4 of the agent acting', default=False)

# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
    # initialize training object. defined in flare/algorithms.py
    trainer = A2C(env)
    # According to the gym leaderboard below, Acrobot-v1 is considered an unsolved task, so there is no reward threshold at which it is solved.
    # gym leaderboard: https://github.com/openai/gym/wiki/Leaderboard
    rew, leng = trainer.learn(1000)

    # watch agent interact with environment
    if args.watch:
        if args.save_mv:
            env = wrappers.Monitor(env, 'a2c_solving_acrobot', video_callable=lambda episode_id: True, force=True)
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
