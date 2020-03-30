# import required packages
from flare.polgrad import PPO
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import wrappers

render = False
watch = True
save_mv = False

# make env 
env = gym.make('LunarLanderContinuous-v2')

if __name__ == '__main__':
    # initialize PPO trainer object and give actor-critic net different hidden layer sizes than default.
    trainer = PPO(env, hidden_sizes=(64, 32))
    # train network for 1000 epochs, render during training if that variable is set to True. This is an example of setting log standard deviation annealing over training and setting it up so that two cycles of annealing through linspace(start, end) occur
    trainer.learn(500, render=render, logstd_anneal=(-0.7, -1.6), n_anneal_cycles=2)

    # watch agent interact with environment
    if watch:
        if save_mv:
            env = wrappers.Monitor(env, 'ppo_solving_'+env.unwrapped.spec.id, video_callable=lambda episode_id: True, force=True)
        obs = env.reset()
        for i in range(5000):
            action = trainer.exploit(torch.tensor(obs.reshape(-1, 1)))
            obs, reward, done, _ = env.step(action.detach().numpy()[0])
            env.render()
            if done:
                obs = env.reset()
        env.close()
