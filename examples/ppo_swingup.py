# import required packages
import argparse

from rlpack.algorithms import PPO
from rlpack.neural_nets import ActorCritic
from rlpack.utils import save_frames_as_gif
import gym
import cartpole_swingup_envs
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers

# make env 
env = gym.make('CartPoleSwingUpDiscrete-v0')
# set up network. PPO is an actor-critic method and needs paramaterizations of the policy and value function. 
# This is defined in rlpack/neural_nets.py
network = ActorCritic(env.observation_space.shape[0], env.action_space.n)

# set up argparser. --watch is a boolean whether or not to watch the agent in the env after training
#                   --plot is bool, whether or not to plot rewards earned over training
parser = argparse.ArgumentParser(description='Get args for running REINFORCE agent on CartPole')
parser.add_argument('--watch', action='store_true', help='choose whether to watch trained agent')
parser.add_argument('--plot', action='store_true', help='choose whether to view plots of reward over training')
parser.add_argument('--save_mv', action='store_true', help='choose whether to save a mp4 of the agent acting')

# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
    # initialize training object. defined in rlpack/algorithms.py
    trainer = PPO(env, network)
    # train network for 1000 episodes, it stops early if the mean reward over last 100 episodes exceeds the solved_threshold
    # criterion for solving from this gym leaderboard: https://github.com/openai/gym/wiki/Leaderboard
    rew, leng = trainer.learn(250, solved_threshold=200)

    # watch agent interact with environment
    frames = []
    if args.watch:
        if args.save_mv:
            env = wrappers.Monitor(env, 'ppo_solving_swingup', video_callable=lambda episode_id: True, force=True)
        obs = env.reset()
        for i in range(5000):
            action = trainer.exploit(torch.tensor(obs))
            obs, reward, done, _ = env.step(int(action))
            env.render()
            if done:
                obs = env.reset()
        env.close()

    # plot reward earned per episode over training
    if args.plot:
        plt.plot(rew)
        plt.title('PPO reward on CartPoleSwingUpDiscrete-v0')
        plt.xlabel('Training steps')
        plt.ylabel('Reward')
        plt.show()
