# rl-pack

**This repository is actively under development and is not stable.**

## Installation

Clone the repository, cd into it: 

```
git clone https://github.com/jfpettit/rl-pack.git
cd rl-pack
pip install -e .
```
## Examples

There are a few example files demonstrating training REINFORCE on CartPole, A2C on Acrobot, and PPO on LunarLander and CartPoleSwingup.
With the repository installed:
```
cd rl-pack/examples
python [PATH_TO_EXAMPLE_PYTHON_FILE] --watch --plot --save_mv
```

The ```--watch```, ```--plot``` and ```--save_mv``` tags are optional. They represent whether you want to watch the agent in the environment after it is done training, whether you want to plot the reward earned over training after training is done, and if you'd like to save a video of your trained agent, respectively.

## Details

This repository is intended to be a learning resource for anyone trying to find beginner-level code covering some RL algorithms. Some of my code isn't very well documented or clear, but I'm working on improving that. My learning path has followed Sutton and Barto's [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) and more recently, OpenAI's [SpinningUp](https://spinningup.openai.com/en/latest/). Many of the algorithms contained here are based on exercises and recommendations from those resources. The traditional RL stuff is inspired by Sutton and Barto, while much of the Deep RL stuff is inspired by SpinningUp.


Currently, all of the algorithms here are only implemented for Gym environments with discrete action spaces.

A2C, REINFORCE, and PPO are tested on GPUs. 

No further attention is being given to the classical RL algorithms here unless somebody lists an issue saying one of them doesn't work or something similar.

Each of the Deep RL algorithms needs to be more extensively tested, so far they've only been tested on problems from the "Classic Control" suite of Gym environments. 

Algorithms currently implemented are:
- Temporal Difference (TD) learning:
	- This algorithm only works with the tic-tac-toe environment I custom-built specifically for the purpose of training a tic-tac-toe agent using TD learning, and so you can look at how to quickly run this code [here](https://jfpettit.github.io/TicTacToeInterface/). That'll get you set up playing tic-tac-toe against the agent. 
- Dynamic Programming policy iteration:
	- This is a typical Dynamic Programming approach to policy iteration, and it works on a gridworld environment.
- REINFORCE
	- Simple, direct, REINFORCE policy gradient. Should work with any gym environment.
	- A simple NN architecture is built into the file here, but the training utility is built to take in any Pytorch NN.
- A2C
	- This is an implementation of the Advantage Actor Critic algorithm. This should also work with any gym environment.
	- Again, it'll take any Pytorch NN and train it. 
- PPO
	- This is an implementation of the PPO-clip variant, it checks approximate KL divergence throughout training. It has been tested on a couple of environments.
	- This also takes any Pytorch NN and works with any gym environment. 

## Miscellaneous

There is also a utils file containing some different advantage function estimators and math utilities and other stuff. Some of it is used, other stuff I thought might just come in handy at some point and wrote it anyway.

I've added a folder containing some trained models. These models were saved using ```torch.save(model.cpu().state_dict(), PATH)``` so you can load them by initializing an architecture identical to the one used to produce the model, loading the saved model file and setting your new networks state_dict to the saved one. To make this easier, I'll list which architectures produced which files:
- 'gpu_ppo_acrobot_run1' used the ActorCritic network in neural_nets.py
- 'gpu_reinforce_cartpole_run1' used the PolicyNet also in neural_nets.py
- 'reallygoodCartPolePPOrun' (wasn't actually that good) I think also used the ActorCritic network, but I'm not certain.

## More to come!
- MPI parallelization
- Implement DQN, DDPG and SAC (maybe?)
- Comment code to make it clearer
- Blog post
- Extend alglrithms to continuous action spaces
