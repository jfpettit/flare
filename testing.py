from rlpack import algorithms as algs
import numpy as np
import matplotlib.pyplot as plt
from rlpack import neural_nets as nets
import torch
import roboschool
import gym
import cartpole_swingup_envs


re_net = nets.PolicyNet
ppo_net = nets.ActorCritic
a2c_net = nets.ActorCritic
re_net_con = nets.ContinuousPolicyNet
dqnet = nets.FullyConnectedDQN

envnames = ['CartPole-v0', 'Acrobot-v1', 'LunarLander-v2']

if __name__ == '__main__':
    env = gym.make(envnames[0])
    network = dqnet(env.observation_space.shape[0], env.action_space.n)
    trainer = algs.DQNtraining(env, network, buffer_size=1000)
    r, l = trainer.learn(5000, n=100)
    plt.plot(r)
    plt.show()
    #rn = re_net(env.observation_space.shape[0], env.action_space.n)
    #reinforce = algs.REINFORCE(env, rn)
    #rew, leng = reinforce.learn(500)
    #plt.plot(rew, label='reinforce')
    #plt.legend()
    #plt.show()
    
    #an = a2c_net(env.observation_space.shape[0], env.action_space.n)
    #a2c = algs.A2C(env, an)
    #rewa, lenga = a2c.learn(1000, solved_threshold=850)
    #pn = ppo_net(env.observation_space.shape[0], env.action_space.n)
    #ppo = algs.PPO(env, pn)
    #rewp, lengp = ppo.learn(1000, solved_threshold=850)
    #rn = re_net(env.observation_space.shape[0], env.action_space.n)
    #reinforce = algs.REINFORCE(env, rn)
    #rew, leng = reinforce.learn(1000, solved_threshold=850)

    #plt.plot(rewa, label='a2c', alpha=0.6)
    #plt.plot(rewp, label='ppo', alpha=0.6)
    #plt.plot(rew, label='reinforce', alpha=0.6)
    #plt.legend()
    #plt.ylabel('Reward')
    #plt.xlabel('Episodes')
    #plt.title('Performance on CartPoleSwingUpDiscrete-v0')
    #plt.savefig('CartPoleSwingUpDiscrete-v0performance.png')
    #plt.show()

    '''
    for envs in envnames:
    
    '''
    #env = gym.make('RoboschoolInvertedPendulum-v1')
    #con_renet = re_net_con(env.observation_space.shape[0], env.action_space.shape[0])
    #con_reinforce = algs.REINFORCE(env, con_renet)
    #rew, leng = con_reinforce.learn(3000, solved_threshold=200)
    #plt.plot(rew, label='reinforce')
    #net = nets.ContinuousActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    #cona2c = algs.A2C(env, net)
    #rew, leng = cona2c.learn(3000, solved_threshold=200)
    #plt.plot(rew)
    #plt.legend()
    #plt.show()
