from rlpack import algorithms as algs
import numpy as np
import matplotlib.pyplot as plt
from rlpack import neural_nets as nets
import torch
import roboschool
import gym
import cartpole_swingup_envs

if __name__ == '__main__':
    env = gym.make('RoboschoolInvertedPendulum-v1')

    policy = nets.ContinuousPolicyNet(env.observation_space.shape[0], env.action_space.shape[0])
    value_f = nets.ValueNet(env.observation_space.shape[0])

    ppo = algs.PPO(env, policy, value_f, steps_per_epoch=4000)
    rewp, lengp = ppo.learn(1000, solved_threshold=850)