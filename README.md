# rlpack

**This repository is actively under development and is not stable.**

## Details

This repository is intended to be a learning resource for anyone trying to find beginner-level code covering some RL algorithms. Some of my code isn't very well documented or clear, but I'm working on improving that. My learning path has followed Sutton and Barto's Reinforcement Learning: An Introduction [book](http://incompleteideas.net/book/the-book.html) and more recently, OpenAI's [SpinningUp](https://spinningup.openai.com/en/latest/). Many of the algorithms contained here are based on resources and exercises and recommendations made from those resources. The traditional RL stuff is inspired by Sutton and Barto, while much of the Deep RL stuff is inspired by SpinningUp.


Currently, all of the algorithms here are only implemented for Gym environments with discrete action spaces. It's on my to-do list to extend them (except DQN) to continuous action spaces.

Algorithms currently implemented are:
- Temporal Difference (TD) learning:
	- This algorithm only works with the tic-tac-toe environment I custom-built specifically for the purpose of training a tic-tac-toe agent using TD learning, and so you can look at how to quickly run this code [here](https://jfpettit.github.io/TicTacToeInterface/). That'll get you set up playing tic-tac-toe against the agent. 
- Dynamic Programming policy iteration:
	- This is a typical Dynamic Programming approach to policy iteration, and it works on a gridworld environment.
- REINFORCE
	- Simple, direct, REINFORCE policy gradient. Should work with any gym environment.
	- A simple NN architecture is built into the file here, but the training utility is built to take in any Pytorch NN.
- DQN
	- The Deep Q Network from DeepMind's [Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (fair warning, that link is to a PDF). I implemented the architecture from their paper, as well as a fully connected flavor of that architecture. Again, though, the training utility will take any Pytorch NN.
	- This will also work with any (discrete action) gym environment. 
- A2C
	- This is an implementation of the Advantage Actor Critic algorithm. This should also work with any gym environment.
	- Again, it'll take any Pytorch NN and train it. 

There is also a utils file containing some different advantage function estimators and math utilities and other stuff. Some of it is used, other stuff I thought might just come in handy at some point and wrote it anyway.


## More to come!
- A2C and REINFORCE will be extended to continuous action spaces
- Implement PPO
- Implement DDPG
