
Usage
-----

Running from command line
~~~~~~~~~~~~~~~~~~~~~~~~~

Presently, A2C and PPO are implemented and working. Run from the command
line with:

::

   python -m flare.run

This will run `PPO <https://arxiv.org/abs/1707.06347>`__ on
`LunarLander-v2 <https://gym.openai.com/envs/LunarLander-v2/>`__ with
default arguments. If you want to change the algorithm to A2C, run on a
different env, or otherwise change some defaults with this command line
interface, then do ``python -m flare.run -h`` to see the available
optional arguments.

Running in a Python file
~~~~~~~~~~~~~~~~~~~~~~~~

Import required packages:

.. code:: python

   import gym
   from flare.polgrad import A2C

   env = gym.make('CartPole-v0') # or other gym env
   agent = A2C(env)
   rew, leng = agent.learn(100)

The above snippet will train an agent on the `CartPole
environment <http://gym.openai.com/envs/CartPole-v1/>`__ for 100 epochs.

You may alter the architecture of your actor-critic network by passing
in a tuple of hidden layer sizes to your agent initialization. i.e.:

.. code:: python

   from flare.polgrad import PPO
   agent = PPO(env, hidden_sizes=(64, 32))
   rew, leng = agent.learn(100)

For a more detailed example using PPO, see the example file at:
`examples/ppo_example.py <https://github.com/jfpettit/flare/blob/master/examples/ppo_example.py>`__.

Details
~~~~~~~

This repository is intended to be a lightweight and simple to use RL
framework, while still getting good performance.

Algorithms will be listed here as they are implemented:

-  `Advantage Actor Critic (A2C) <https://arxiv.org/abs/1602.01783>`__
-  `Proximal Policy Optimization
   (PPO) <https://arxiv.org/abs/1707.06347>`__
-  `Deep Deterministic Policy Gradients
   (DDPG) <https://arxiv.org/abs/1509.02971>`__
-  `Twin Delayed Deep Deterministic Policy Gradients
   (TD3) <https://arxiv.org/abs/1802.09477>`__
-  `Soft Actor Critic (SAC) <https://arxiv.org/abs/1801.01290>`__

The policy gradient algorithms (A2C, PPO), support running on multiple
CPUs via MPI. The Q Policy Gradient algorithms (SAC, DDPG, TD3) do not
yet support MPI parallelization.

If you wish to build your own actor-critic from scratch, then it is
recommended to use the
`FireActorCritic <https://github.com/jfpettit/flare/blob/master/flare/neural_nets.py#L72>`__
as a template.

Flare now automatically logs run metrics to
`TensorBoard <https://www.tensorflow.org/tensorboard>`__. View these by
running ``tensorboard --logdir flare_runs`` in a terminal.