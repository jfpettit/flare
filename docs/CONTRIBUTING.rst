Contributing to flare
=====================

Table of Contents
-----------------

-  `Main Design
   Values <#main-design-values-clarity-performance-and-composability>`__
-  `Contribution types <#contribution-types>`__
-  `Documentation <#documentation>`__
-  `Coding guidelines <#coding-guidelines>`__
-  `Testing <#testing>`__
-  `Other questions <#other-questions>`__

Thank you for taking the time to contribute! 

The aim of flare is to provide a reliable implementation of many common
reinforcement learning algorithms (and eventually some not so common
ones) and to unify those algorithms under a common API as much as
possible. There are two efforts to do this. The first is a very “plug
and play” style set-up, where a user can plug whatever environment they
want to train on into an algorithm and get decent results. The second,
the ``flare.kindling`` submodule, aims to expose building blocks of RL
algorithms so that when a user wants to work on a custom algorithm, they
can use the pre-built bits provided and only have to worry about
creating their custom/new algorithm from scratch.

Main Design values: Clarity, Performance, and Composability
-----------------------------------------------------------

We want to accelerate the rate of progress for RL researchers and reduce
the time from idea to experimentation. This means that everything
written needs to be understandable, must work well, and must be highly
composable.

`PytorchLightning <https://github.com/PyTorchLightning/pytorch-lightning>`__
has excellent values and `contributor
guidelines <https://github.com/PytorchLightning/pytorch-lightning/blob/master/.github/CONTRIBUTING.md>`__.
We take much inspiration from them, and any overlap between their
guidelines and ours is probably intentional on our part. 

-  Clarity:

   -  Things like clear, concise variable names, thorough commenting and
      documentation, and consistency in code structure.
   -  As PytorchLightning says, many users won’t be engineers. Clear
      code is much more important than fanciness and slick moves.
   -  We want to keep the API as simple as possible. The external API
      should allow a user to import an algorithm, set up their RL
      environment, and train that algorithm simply by calling a
      ``learn()`` function. In ``flare.kindling``, the API should be
      consistent and flat, without obfuscating division between
      functional groups.

-  Performance:

   -  High quality performance without compromising clarity, and
      faithful implementations of algorithms.
   -  Established best-practice hyperparameters so that a user can get
      good results without worrying about their hyperparameter setup.

-  Composability:

   -  The components in ``flare.kindling`` must be easily composed into
      full algorithms.

These values are by no means things that have already been achieved, but
are rather goals to work towards and strive for, moving forward.

Contribution types
------------------

Presently, we are looking for people to help with: 

- Implementing new algorithms 
- Bug fixes 
- Unifying the codebase API 
- Writing unit tests and continuous integration 
- Writing documentation 
- General mechanics of the project

For most things: 

1. Submit an issue on GitHub and we can discuss the bug fix/new feature/new algorithm/contribution. 
2. For bugs, try fixing it yourself or recommend a fix! 
3. After discussion, submit a pull request! Of course please update the docs and tests.

Use your best judgment to add relevant tags to your issues/PRs.

Documentation
-------------

We are using Sphinx to automatically generate documentation. For docstring formatting, we are using the Napoleon extension with Google style.

Coding guidelines
-----------------

-  Use f-strings for output formatting.
-  Use `black <https://pypi.org/project/black/>`__ or
   `flake8 <http://flake8.pycqa.org/en/latest/index.html#quickstart>`__
   to format the code. The code so far has just been formatted with
   black on default settings, so it is OK to do the same.

Testing
-------

We need to (and need help with!) build unit tests and implement
continuous integration for the code.

Other questions
---------------

Feel free to open an issue with your question! Maybe use a tag like
``general-question`` or something similar.
