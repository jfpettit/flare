# flare

(old) Blog post about this repository [here](https://jfpettit.svbtle.com/rlpack).

## Installation

Clone the repository, cd into it: 

```
git clone https://github.com/jfpettit/flare.git
cd flare
pip install -e .
```

## Running from command line

Presently, A2C and PPO are implemented and working. Run from the command line with:
```
cd flare/examples
python runner.py -h
```
The ```-h``` tag will show you the available optional arguments for the file. To use the defaults, just do: ```python runner.py```.

## Running in a Python file

Import required packages:

```python
import gym
from flare.a2c import A2C

env = gym.make('CartPole-v0') # or other gym env
agent = A2C(env)
agent.learn(100)
```

The above snippet will train an agent on the [CartPole environment](http://gym.openai.com/envs/CartPole-v1/) for 100 epochs. 

## Details

This repository is intended to be a lightweight and simple to use RL framework, while still getting good performance.

Algorithms will be listed here as they are implemented: 

- A2C
	- This is an implementation of the Advantage Actor Critic algorithm. It works with any Gym environment.
- PPO
	- This is an implementation of the Proximal Policy Optimization algorithm, it also works with any Gym environment.

**This architecture is deprecated and no longer works with the code. I'll be updating the base actor critic class accordingly.**

An actor critic network is [included](https://github.com/jfpettit/flare/blob/aad21963f7f67f78be1ea3ae7238b2ff7ca86e9e/flare/neural_nets.py#L11). But, if you'd like to define your own architecture, just subclass this one and write your forward pass to return action logits and state values. Make sure your ```__init__``` call takes the same arguments as the ActorCritic class does. For example:

```python
from flare.neural_nets import ActorCritic

class CustomNet(ActorCritic):
	def __init__(self, observation_size, action_size):
		### your architecture here ###

	def forward(self, x):
		### your forward pass here ###
		return logits, value
```

## More to come!
- Improve features (i.e. automatic result plotting/saving, etc.)
- Implement DQN, DDPG and SAC (maybe?)
- Comment code to make it clearer
- Test algorithm performance
- Parallelize (maybe)
