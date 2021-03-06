flare
=====

.. figure:: /src/lunarlandercontinuous.gif
   :alt: **PPO agent on LunarLanderContinuous-v2**

   **PPO agent on LunarLanderContinuous-v2**

*PPO agent trained to
play*\ `LunarLanderContinuous-v2 <https://gym.openai.com/envs/LunarLanderContinuous-v2/>`__\ *.
Reward per episode at this point was ~230.*

Table of Contents
-----------------

-  `Installation <#installation>`__
-  `Usage <#usage>`__
-  `Details <#details>`__
-  `Contributing <./CONTRIBUTING.md>`__
-  `References <#references>`__
-  `More to come <#more-to-come>`__

``flare`` is a small reinforcement learning library. Currently, the use
case for this library is small-scale RL experimentation/research. Much
of the code is refactored from and built off of
`SpinningUp <https://spinningup.openai.com/en/latest/>`__, so massive
thanks to them for writing quality, understandable, and performant code.

(old) Blog post about this repository
`here <https://jfpettit.svbtle.com/rlpack>`__.

Installation
------------

**Flare supports parallelization via MPI!** So, you’ll need to install
`OpenMPI <https://www.open-mpi.org/>`__ to run this code.
`SpinningUp <https://spinningup.openai.com/en/latest/user/installation.html#installing-openmpi>`__
provides the following installation instructions:

On Ubuntu:
~~~~~~~~~~

::

   sudo apt-get update && sudo apt-get install libopenmpi-dev

On Mac OS X
~~~~~~~~~~~

::

   brew install openmpi

On Windows
~~~~~~~~~~

If you’re on Windows, `here is a link to some
instructions <https://nyu-cds.github.io/python-mpi/setup/>`__.

If the Mac instructions don’t work for you, consider these
`instructions <http://www.science.smith.edu/dftwiki/index.php/Install_MPI_on_a_MacBook>`__.

It is recommended to use a virtual env before installing this, to avoid
conflicting with other installed packages. Anaconda and Python offer
virtual environment systems.

Finally, clone the repository, cd into it:

::

   git clone https://github.com/jfpettit/flare.git
   cd flare
   pip install -e .

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
-------

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

Contributing
------------

We’d love for you to contribute! Any help is welcome. See
`CONTRIBUTING.md <./CONTRIBUTING.md>`__ for contributor guidelines and
info.

References
----------

-  `OpenAI SpinningUp <https://spinningup.openai.com/en/latest/>`__
-  `FiredUp <https://github.com/kashif/firedup>`__
-  `PPO paper <https://arxiv.org/abs/1707.06347>`__
-  `A3C paper <https://arxiv.org/abs/1602.01783>`__
-  `Pytorch RL
   examples <https://github.com/pytorch/examples/tree/master/reinforcement_learning>`__

More to come!
-------------

-  Comment code to make it clearer
-  Test algorithm performance
