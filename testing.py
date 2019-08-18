from rlpack import algorithms as algs
import roboschool
import gym
import numpy as np
import matplotlib.pyplot as plt
from rlpack import neural_nets as nets
from rlpack.evolve import ES
import torch


re_net = nets.PolicyNet
ppo_net = nets.ActorCritic
a2c_net = nets.ActorCritic
dqn = nets.FullyConnectedDQN

envnames = ['CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']

if __name__ == '__main__':
    #env = gym.make('RoboschoolInvertedPendulum-v1')
    env = gym.make('CartPole-v0')
    #an = a2c_net(env.observation_space.shape[0], env.action_space.shape[0], continuous=True)
    #a2c = algs.A2C(env, an)
    #rewa, lenga = a2c.train_loop_(1000)
    #pn = ppo_net(env.observation_space.shape[0], env.action_space.shape[0], continuous=True)
    #ppo = algs.PPO(env, pn, target_kl=0.05)
    #rewp, lengp = ppo.train_loop_(250)
    #rn = re_net(env.observation_space.shape[0], env.action_space.shape[0], continuous=True)
    #es = ES(env, mean=0.0, population_size=50)
    #es.train_loop_(re_net, wstd=1.0, generations=50, EPOCHS=25, standardize_fits=True, anneal_std=True, solved_threshold=195)
    #sd = es.get_best_state_dict()
    #rn = re_net(env.observation_space.shape[0], env.action_space.n)
    #rn.load_state_dict(sd)
    #reinforce = algs.REINFORCE(env, rn)
    #rew, leng = reinforce.train_loop_(1000, solved_threshold=195)
    dqn = dqn(env.observation_space.shape[0], env.action_space.n)
    qlearner = algs.DQNtraining(env, dqn)
    rew, leng = qlearner.train_loop_(500)

    #plt.plot(rewa, label='a2c')
    #plt.plot(rewp, label='ppo')
    plt.plot(rew, label='reinforce')
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

        #re_rew, re_len = reinforce.train_loop_(1000, solved_threshold=195)
        #reward_results['reinforce_'+envname] = re_rew
        #p_rew, p_len = ppo.train_loop_(1000, solved_threshold=195)
        #reward_results['ppo_'+envname] = p_rew
        a_rew, a_len = a2c.train_loop_(1000, solved_threshold=195)
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
