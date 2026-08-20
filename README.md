This repository contains the implementation of accelerated algorithms for Planning and Reinforcement Learning, utilizing JAX for high-performance tensor operations.

# Project Structure
The codebase is organized into two primary modules:

Planning/: Accelerated algorithms designed for planning problems where the model dynamics are known.

Learning/: Accelerated algorithms for learning problems, specifically categorized into synchronous and asynchronous (forthcoming) approaches.

🛠️ Core Library: jaxdp
jaxdp/: The core utility library. It contains Markov Decision Process (MDP) environments and foundational RL computation formulas. The jaxdp directory serves as an internal "add-on" or utility package that powers both planning and learning modules. It includes:

1. MDP Environments
Standard and custom environments implemented in JAX for massive parallelism, such as:

Garnet MDPs

Forest Management

Graph MDP

Healthcare MDP

Grid-World MDP

2. Computing Formulas
Optimized JAX implementations of fundamental RL operators:

Bellman Optimality/Expectation Operators.

Value/Policy Evaluation kernels.

Transition matrix manipulations.

# Modules
1. Planning
Located in /Planning, these scripts solve MDPs using known dynamics.

Algorithms: Accelerated versions of Value Iteration, Policy Iteration, and more.

Main Script: Entry point for running planning experiments.

2. Learning (Synchronous)
Located in /Learning/Synchronous, these scripts address the RL setting (unknown or sampled dynamics).

Algorithms: Includes Q-learning, and accelerated methods on QL.

Policies: Definitions for exploration strategies (e.g., Epsilon-greedy, Softmax).

This repository is part of a Master's Thesis focused on Accelerated Methods in Reinforcement Learning. The goal is to compare the convergence rates and computational efficiency of standard RL algorithms when ported to functional, JIT-compiled architectures.