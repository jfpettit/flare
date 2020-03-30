# import required packages
import argparse

import flare.polgrad as pg
from flare.neural_nets import ActorCritic
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers
import pybullet_envs

# set up argparser.
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alg', type=str, help='Algorithm to use', default='PPO')
parser.add_argument('-e', '--env', type=str, help='Env to train in', default='LunarLander-v2')
parser.add_argument('-w', '--watch', type=bool, help='choose whether to watch trained agent', default=False)
parser.add_argument('-p', '--plot', type=bool, help='choose whether to view plots of reward over training', default=False)
parser.add_argument('-sm', '--save_mv', type=bool, help='choose whether to save a mp4 of the agent acting', default=False)
parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs to train for', default=100)
parser.add_argument('-hor', '--horizon', type=int, help='Horizon length of each episode', default=1000)
parser.add_argument('-r', '--render', type=bool, help='Whether to render agent during training.', default=False)
parser.add_argument('-g', '--gamma', type=float, help='Discount factor for GAE-lambda advantage calculation', default=.999)
parser.add_argument('-lam', '--lam', type=float, help='Lambda for GAE-lambda advantage calculation', default=.97)
parser.add_argument('-l', '--layers', nargs='+', help='MLP hidden layer sizes. Enter like this: --layers 64 64. Makes MLP w/ 2 hidden layers w/ 64 nodes each.', default=[32, 32])
parser.add_argument('-la', '--logstd_anneal', nargs='+', help='Whether or not to anneal policy log standard deviation over training', default=None)
parser.add_argument('-sst', '--save_states', type=bool, help='Whether or not to save states over training. Saves as pickled list of NumPy arrays.', default=False)
parser.add_argument('-ssc', '--save_screen', type=bool, help='Whether to save the screens over training. Saves as pickled list of NumPy arrays.', default=False)
parser.add_argument('-nac', '--n_anneal_cycles', type=int, help='If using std annealing, how many cycles to anneal over. Default: 0', default=0)
parser.add_argument('-f', '--folder', help='Folder to log training output to.', default=None)
# get args from argparser
args = parser.parse_args()

if __name__ == '__main__':
    # initialize training object. defined in flare/algorithms.py
    hids = [int(i) for i in args.layers]
    logstds_anneal = [float(i) for i in args.logstd_anneal] if args.logstd_anneal is not None else None
    env = gym.make(args.env)
    if args.alg == 'PPO':
        trainer = pg.PPO(env, gamma=args.gamma, lam=args.lam, hidden_sizes=hids, logger_dir=args.folder)
    elif args.alg == 'A2C':
        trainer = pg.A2C(env, gamma=args.gamma, lam=args.lam, hidden_sizes=hids, logger_dir=args.folder)
    rew, leng = trainer.learn(args.epochs, horizon=args.horizon, render=args.render, logstd_anneal=logstds_anneal, save_screen=args.save_screen, n_anneal_cycles=args.n_anneal_cycles, save_states=args.save_states)

    # watch agent interact with environment
    if args.watch:
        env = gym.make(args.env)
        if args.save_mv:
            env = wrappers.Monitor(env, args.alg+'_on_'+env.unwrapped.spec.id, video_callable=lambda episode_id: True, force=True)
        if 'Bullet' in env.unwrapped.spec.id:
            env.render()
        obs = env.reset()
        for i in range(10000):
            action, _, _, _ = trainer.ac(torch.Tensor(obs.reshape(1, -1)))
            obs, reward, done, _ = env.step(action.detach().numpy()[0])
            if "Bullet" not in env.unwrapped.spec.id:
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