# import required packages
import argparse

from flare.a2c import A2C
from flare.ppo import PPO
from flare.neural_nets import ActorCritic
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers
import pybullet_envs

# set up argparser.
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, help='Algorithm to use', default='PPO')
parser.add_argument('--env', type=str, help='Env to train in', default='LunarLander-v2')
parser.add_argument('--watch', type=bool, help='choose whether to watch trained agent', default=False)
parser.add_argument('--plot', type=bool, help='choose whether to view plots of reward over training', default=False)
parser.add_argument('--save_mv', type=bool, help='choose whether to save a mp4 of the agent acting', default=False)
parser.add_argument('--epochs', type=int, help='Number of epochs to train for', default=100)
parser.add_argument('--horizon', type=int, help='Horizon length of each episode', default=1000)
parser.add_argument('--render', type=bool, help='Whether to render agent during training.', default=False)
parser.add_argument('--gamma', type=float, help='Discount factor for GAE-lambda advantage calculation', default=.999)
parser.add_argument('--lam', type=float, help='Lambda for GAE-lambda advantage calculation', default=.97)
parser.add_argument('--layers', nargs='+', help='MLP hidden layer sizes. Enter like this: --layers 64 64. Makes MLP w/ 2 hidden layers w/ 64 nodes each.', default=[32, 32])
parser.add_argument('--std_anneal', nargs='+', help='Whether or not to anneal policy log standard deviation over training', default=None)

# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
    # initialize training object. defined in flare/algorithms.py
    hids = [int(i) for i in args.layers]
    logstds_anneal = [float(i) for i in args.std_anneal]
    env = gym.make(args.env)
    if args.alg == 'PPO':
        trainer = PPO(env, gamma=args.gamma, lam=args.lam, hidden_sizes=hids)
    elif args.alg == 'A2C':
        trainer = A2C(env, gamma=args.gamma, lam=args.lam, hidden_sizes=hids)
    rew, leng = trainer.learn(args.epochs, horizon=args.horizon, render=args.render, logstd_anneal=logstds_anneal)

    # watch agent interact with environment
    if args.watch:
        if args.save_mv:
            env = wrappers.Monitor(env, args.alg+'_on_'+env.unwrapped.spec.id, video_callable=lambda episode_id: True, force=True)
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
        plt.title(args.alg+' returns on '+env.unwrapped.spec.id)
        plt.xlabel('Training steps')
        plt.ylabel('Return')
        plt.show()