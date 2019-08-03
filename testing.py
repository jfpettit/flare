from rlpack import algorithms as algs
import gym
import numpy as np
import matplotlib.pyplot as plt
from rlpack import neural_nets as nets

re_net = nets.SimplePolicyNet
ppo_net = nets.SimpleActorCritic
a2c_net = nets.SimpleActorCritic

envnames = ['CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']

if __name__ == '__main__':
    reward_results = {}
    for envname in envnames:
        env = gym.make(envname)
        rn = re_net(env.observation_space.shape[0], env.action_space.n)
        pn = ppo_net(env.observation_space.shape[0], env.action_space.n)
        an = a2c_net(env.observation_space.shape[0], env.action_space.n)

        reinforce = algs.REINFORCE(env, rn)
        ppo = algs.PPO(env, pn)
        a2c = algs.A2C(env, an, policy_train_iters=40)

        re_rew, re_len = reinforce.train_loop_(False, 1000)
        reward_results['reinforce_'+envname] = re_rew
        p_rew, p_len = ppo.train_loop_(1000, solved_threshold=195)
        reward_results['ppo_'+envname] = p_rew
        a_rew, a_len = a2c.train_loop_(100, solved_threshold=195)
        reward_results['a2c'+envname] = a_rew

        #plt.plot(re_rew, label='reinforce on '+envname)
        #plt.plot(p_rew, label='ppo on '+envname)
        plt.plot(a_rew, label='a2c on '+envname)
        plt.legend()
        plt.title('Algorithms on '+envname)
        plt.ylabel('Reward')
        plt.xlabel('Epochs')
        plt.show()

