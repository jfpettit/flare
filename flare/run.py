# import required packages
import argparse
import flare.polgrad as pg
import flare.qpolgrad as qpg
import flare.kindling as fk
import numpy as np
import pybullet_envs
from typing import Optional, Union, Tuple, List

# set up argparser.
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--alg", type=str, help="Algorithm to use", default="PPO")
parser.add_argument(
    "-e", "--env", type=str, help="Env to train in", default="LunarLander-v2"
)
parser.add_argument(
    "-ep", "--epochs", type=int, help="Number of epochs to train for", default=100
)
parser.add_argument(
    "-hor", "--horizon", type=int, help="Horizon length of each episode", default=1000
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    help="Discount factor for GAE-lambda advantage calculation",
    default=0.99,
)
parser.add_argument(
    "-lam",
    "--lam",
    type=float,
    help="Lambda for GAE-lambda advantage calculation",
    default=0.97,
)
parser.add_argument(
    "-l",
    "--layers",
    nargs="+",
    help="MLP hidden layer sizes. Enter like this: --layers 64 64. Makes MLP w/ 2 hidden layers w/ 64 nodes each.",
    default=[64, 32],
)
parser.add_argument(
    "-spe",
    "--steps_per_epoch",
    type=int,
    help="How many env interactions per epoch",
    default=4000,
)
parser.add_argument(
    "-mbs",
    "--minibatch_size",
    type=int,
    help="Minibatch size for training. Defaults to 4000, same as steps per epoch. Should be a multiple of steps per epoch.",
    default=4000
)
parser.add_argument(
    "-seed",
    "--seed",
    type=int,
    help="Seed for agent and environment.",
    default=0
)
# get args from argparser
args = parser.parse_args()


if __name__ == "__main__":
    # initialize training object. defined in flare/algorithms.py
    hids = tuple(int(i) for i in args.layers)
    if args.alg == "REINFORCE":
        pg.reinforce.learn(
            args.env,
            args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            minibatch_size=args.minibatch_size,
            hidden_sizes=hids,
            gamma=args.gamma,
            lam=args.lam,
            seed=args.seed,
            hparams=args
        )
    if args.alg == "PPO":
        pg.ppo.learn(
            args.env,
            args.epochs,
            minibatch_size=args.minibatch_size,
            steps_per_epoch=args.steps_per_epoch,
            hidden_sizes=hids,
            gamma=args.gamma,
            lam=args.lam,
            seed=args.seed,
            hparams=args
        )
    elif args.alg == "A2C":
        pg.a2c.learn(
            args.env,
            args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            minibatch_size=args.minibatch_size,
            hidden_sizes=hids,
            gamma=args.gamma,
            lam=args.lam,
            seed=args.seed,
            hparams=args
        )
    elif args.alg == "DDPG":
       trainer = qpg.DDPG(
            env,
            gamma=args.gamma,
            hidden_sizes=hids,
            logger_dir=args.folder,
            save_screen=args.save_screen,
            save_states=args.save_states,
            steps_per_epoch=args.steps_per_epoch,
        ) 
    elif args.alg == "TD3":
        trainer = qpg.TD3(
            env,
            gamma=args.gamma,
            hidden_sizes=hids,
            logger_dir=args.folder,
            save_screen=args.save_screen,
            save_states=args.save_states,
            steps_per_epoch=args.steps_per_epoch,
        )
    elif args.alg == "SAC":
        trainer = qpg.SAC(
            env,
            gamma=args.gamma,
            hidden_sizes=hids,
            logger_dir=args.folder,
            save_screen=args.save_screen,
            save_states=args.save_states,
            steps_per_epoch=args.steps_per_epoch,
        )
    rew, leng = trainer.learn(
        args.epochs,
        horizon=args.horizon,
        render=args.render,
    )

