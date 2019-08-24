from rlpack import algorithms as algs
import gym
import numpy as np
import matplotlib.pyplot as plt
from rlpack import neural_nets as nets
import torch


re_net = nets.PolicyNet
ppo_net = nets.ActorCritic
a2c_net = nets.ActorCritic

envnames = ['CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']

if __name__ == '__main__':
    env = gym.make(envnames[2])
    #an = a2c_net(env.observation_space.shape[0], env.action_space.n)
    #a2c = algs.A2C(env, an, steps_per_epoch=2048)
    #rewa, lenga = a2c.learn(500)
    pn = ppo_net(env.observation_space.shape[0], env.action_space.n)
    ppo = algs.PPO(env, pn, steps_per_epoch=2048)
    rewp, lengp = ppo.learn(1000)
    #rn = re_net(env.observation_space.shape[0], env.action_space.n)
    #reinforce = algs.REINFORCE(env, rn)
    #rew, leng = reinforce.learn(500)

    #plt.plot(rewa, label='a2c')
    plt.plot(rewp, label='ppo')
    #plt.plot(rew, label='reinforce')
    plt.legend()
    plt.show()
    '''
    reward_results = {}
    for envname in envnames:
        env = gym.make(envname)
        rn = re_net(env.observation_space.shape[0], env.action_space.n)
        pn = ppo_net(env.observation_space.shape[0], env.action_space.n)
        an = a2c_net(env.observation_space.shape[0], env.action_space.n)

        reinforce = algs.REINFORCE(env, rn)
        ppo = algs.PPO(env, pn)
        a2c = algs.A2C(env, an, policy_train_iters=40)

        #re_rew, re_len = reinforce.learn(1000, solved_threshold=195)
        #reward_results['reinforce_'+envname] = re_rew
        #p_rew, p_len = ppo.learn(1000, solved_threshold=195)
        #reward_results['ppo_'+envname] = p_rew
        a_rew, a_len = a2c.learn(1000, solved_threshold=195)
        reward_results['a2c'+envname] = a_rew

        #plt.plot(re_rew, label='reinforce on '+envname)
        #plt.plot(p_rew, label='ppo on '+envname)
        plt.plot(a_rew, label='a2c on '+envname)
        plt.legend()
        plt.title('Algorithms on '+envname)
        plt.ylabel('Reward')
        plt.xlabel('Epochs')
        plt.show()
    '''
