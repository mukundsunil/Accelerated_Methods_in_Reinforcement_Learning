<div align="center">

# ⚡ Acceleration of Q-Learning via Second-Order Methods

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](#prerequisites--installation)
[![Framework - JAX](https://img.shields.io/badge/Framework-JAX-red.svg?logo=google&logoColor=white)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Thesis-Completed-brightgreen.svg)]()

<p align="center">
  <b>A novel pre-conditioned matrix gain framework built upon Zap Q-learning to reduce computational bottlenecks and achieve faster asymptotic convergence in Markov Decision Processes.</b>
</p>

[Overview](#-overview) •
[Key Highlights](#-key-highlights) •
[Repository Structure](#-repository-structure) •
[Installation](#-prerequisites--installation) •
[Quickstart](#-quickstart--usage) •
[Citation](#-citation)

---

</div>

## 📌 Overview

Standard Q-learning exhibits slow empirical convergence, requiring substantial transition samples to isolate optimal policies in complex environments. While second-order stochastic approximation methods—such as **Zap Q-learning**—accelerate learning, they introduce severe computational overhead per iteration.

This repository provides the official implementation of a **pre-conditioned matrix gain framework** that:
- Solves the high-gain computational bottleneck of Zap Q-learning.
- Reduces per-iteration complexity down to $\mathcal{O}(n^2 m^2)$.
- Ensures stable asymptotic convergence to the true optimal value function $Q^*(s, a)$.
- Delivers empirical convergence gains across multiple MDP topologies (e.g., Garnet MDPs, GridWorlds).

---

## ✨ Key Highlights

| Feature | Description |
| :--- | :--- |
| 🚀 **Second-Order Speedup** | Drastically cuts down sample complexity compared to standard Q-learning baselines. |
| 🧮 **Optimized Complexity** | Lowers the per-step matrix update costs to $\mathcal{O}(n^2 m^2)$ via pre-conditioning. |
| ⚡ **JAX-Powered Sampling** | Uses the internal `jaxtor` engines for vectorized environment interactions. |
| 📊 **Deep Evaluation Tools** | Built-in scripts for generating parameter heatmaps, Bellman error plots, and policy evaluations. |

---

## 📂 Repository Structure

```text
├── Data/                   # Raw and processed evaluation data
├── Images/                 # Saved performance plots and comparison charts
├── plots/                  # Directory for standalone figure outputs
│
├── jaxtor/                 # JAX-accelerated sampling & environment engine
├── jaxdp/                  # Dynamic programming utilities and MDP definitions
│
├── Algorithms.py           # Core RL implementations (Zap Q-learning, Preconditioned methods)
├── Main_Script.py          # Primary training and evaluation pipeline
│
├── Heat_Map.py             # Hyperparameter performance & heatmap generation
│
├── Policies.py             # Exploration policies (epsilon-greedy, softmax, etc.)
├── tabular_env.py          # Tabular MDP environment wrappers
├── Tools.py                # Matrix manipulation and helper functions
├── utils.py                # Data visualization and curve plotting routines
