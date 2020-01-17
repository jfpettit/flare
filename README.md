# flare
![/src/lunarlandercontinuous.gif]

flare is a small reinforcement learning library. Currently, the use case for this library is small-scale RL experimentation/research. 

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
python -m flare.run
```

This will run [PPO](https://arxiv.org/abs/1707.06347) on [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) with default arguments. If you want to change the algorithm to A2C, run on a different env, or otherwise change some defaults with this command line interface, then do ```python -m flare.run -h``` to see the available optional arguments.

## Running in a Python file

Import required packages:

```python
import gym
from flare.a2c import A2C

env = gym.make('CartPole-v0') # or other gym env
agent = A2C(env)
rew, leng = agent.learn(100)
```

The above snippet will train an agent on the [CartPole environment](http://gym.openai.com/envs/CartPole-v1/) for 100 epochs. 

## Details

This repository is intended to be a lightweight and simple to use RL framework, while still getting good performance.

Algorithms will be listed here as they are implemented: 

- A2C
	- This is an implementation of the Advantage Actor Critic algorithm. It works with any Gym environment.
- PPO
	- This is an implementation of the Proximal Policy Optimization algorithm, it also works with any Gym environment.

You may alter the architecture of your actor-critic network by passing in a tuple of hidden layer sizes to your agent initialization. i.e.:

```python
from flare.ppo import PPO
agent = PPO(env, hidden_sizes=(64, 32))
rew, leng = agent.learn(100)
```

If you wish to build your own actor-critic from scratch, then it is recommended to use the [FireActorCritic](https://github.com/jfpettit/flare/blob/master/flare/neural_nets.py#L72) as a template.

## References
- [OpenAI SpinningUp](https://spinningup.openai.com/en/latest/)
- [FiredUp](https://github.com/kashif/firedup)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [A3C paper](https://arxiv.org/abs/1602.01783)
- [Pytorch RL examples](https://github.com/pytorch/examples/tree/master/reinforcement_learning)

## More to come!
- Improve features (i.e. automatic result plotting/saving, etc.)
- Implement DQN, DDPG and SAC (maybe?)
- Comment code to make it clearer
- Test algorithm performance
- Parallelize (maybe)
