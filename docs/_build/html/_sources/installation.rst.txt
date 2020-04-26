Installation
============

**Flare supports parallelization via MPI!** So, you’ll need to install
`OpenMPI <https://www.open-mpi.org/>`__ to run this code.
`SpinningUp <https://spinningup.openai.com/en/latest/user/installation.html#installing-openmpi>`__
provides the following installation instructions:

Installing OpenMPI
------------------

On Ubuntu
~~~~~~~~~

::

   sudo apt-get update && sudo apt-get install libopenmpi-dev

On Mac OS X
~~~~~~~~~~~

::

   brew install openmpi

If the Mac instructions don’t work for you, consider these
`instructions <http://www.science.smith.edu/dftwiki/index.php/Install_MPI_on_a_MacBook>`__.

On Windows
~~~~~~~~~~

If you’re on Windows, `here is a link to some
instructions <https://nyu-cds.github.io/python-mpi/setup/>`__.

Installing the flare package
----------------------------

It is recommended to use a virtual env before installing this, to avoid
conflicting with other installed packages. Anaconda and Python offer
virtual environment systems.

Set up virtual env
~~~~~~~~~~~~~~~~~~

With Anaconda:

::

    conda create -n flare_env_name python=3.6


With python virtual envs:

::

    python3 -m venv /path/to/new/virtual/environment


`More info on Anaconda: <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>` and `more info on python venvs <https://docs.python.org/3/library/venv.html>`.

Install from GitHub
~~~~~~~~~~~~~~~~~~~

Finally, clone the repository, cd into it:

::

   git clone https://github.com/jfpettit/flare.git
   cd flare
   pip install -e .