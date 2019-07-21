# reinforcement-learning

**This repository is actively under development and is not stable.**

This repository will house my reinforcement learning code and projects as I develop them.

## [Temporal Difference Learning tic-tac-toe agent](https://github.com/jfpettit/reinforcement-learning/tree/master/TD_tictactoe)
This folder contains files for the tic-tac-toe playing agent and for the tic-tac-toe game. I wrote this code while reading through [Sutton and Barto's "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book.html). Chapter 1, section 5 discusses some of the details of developing a tic-tac-toe playing agent, and my code is based off of that section. I tried to write the agent in a general way so that anyone can use it. Full demo of using the code to train and play against the agent coming soon.

## [Policy Iteration with Dynamic Programming to solve Gridworld](https://github.com/jfpettit/reinforcement-learning/tree/master/DynamicProgramming_gridworld)
This folder contains files for the gridworld environment (developed by Denny Britz, see his code [here](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py)) and for policy iteration using dynamic programming. Chapter 4 of "Reinforcement Learning: An Introduction" discusses performing policy iteration using dynamic programming, and one example throughout that chapter is gridworld. In gridworld, the agent starts in a random position on an N by N grid and has to navigate to a terminal state. The reward at every step the agent takes to travel to the terminal state is -1, so the agent is incentivized to reach the terminal state as quickly as possible. The code contains a class implementing policy iteration, prints out the learned policy and value function, and renders the agent as it moves through gridworld.

## Things actively in progress
- REINFORCE has been implemented, testing to make sure it works properly across a range of environments.
- DQN training has been implemented and is also being tested.

## Next plan:
- A2C is coming up next.

More to come!
