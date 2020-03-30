
# flare.kindling

The ```kindling``` submodule (named for the bits of twigs and leaves and other small fuel used to help start a fire), is the broken down components used in the algorithms in the main ```flare``` module. This will accelerate algorithm experimentation, by enabling the user to only import what pre-built components they want to use and combine them with their custom-built components. 

## What's included here

This will be updated as things are added.

- buffers
    - PGBuffer: an experience buffer for Policy Gradient algorithms.
    - ReplayBuffer: an experience replay buffer for off-policy agents.
- logging
    - Logger: a general-purpose logger
    - EpochLogger: A variant of Logger tailored for tracking average values over epochs.
- tblog
    - TensorBoardWriter: a specialized writer for pulling values from an epoch dictionary and piping them into TensorBoard.
- neural_nets
    - MLP: a Multi-Layer Perceptron
    - CategoricalPolicy: A policy class for environments with discrete action spaces
    - GaussianPolicy: A policy class for environments with continuous action spaces
    - FireActorCritic: An Actor-Critic module for discrete and continuous action spaces
    - FireQActorCritic: (in dev) A Q Actor-Critic for continuous action spaces
- utils: really contains a random collection of (maybe useful, often not) functions
    - gaussian_likelihood: compute log-probability of a value drawn from a Gaussian
    - NetworkUtils:
        - Compute output size of a Conv2D layer
        - Compute squared error loss
    - save_frames_as_gif: Save a list of frames as a gif
    - NormalizedActions: a Gym action normalizing wrapper for a continuous policy
