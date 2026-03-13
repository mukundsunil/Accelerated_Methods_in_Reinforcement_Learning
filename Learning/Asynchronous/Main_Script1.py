from __future__ import annotations
from typing import Any


import dataclasses
import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax 
import tyro
from chex import dataclass
from rich import print
from rich.progress import track

from jaxtor.env.tabular import garnet
from jaxtor.eval.tabular import Eval as Evaluator, optimal_q
from jaxtor.sampler import Imc, Mc
from Algorithms import q_learning_async
@dataclass 
class Config:
    """Training configuration for tyro CLI"""

    garnet: garnet.Config = dataclasses.field(default_factory=garnet.Config) # I want to make this an input. Ex: alg_MDP is the input, it gets the MDP from here
    n_steps: int = 1_000_000
    alpha_init: float = 0.5
    alpha_power: float = 0.25 # what is this
    alpha_period: float = 10_000 # what is this
    gamma: float = 0.99
    epsilon: float = 0.1
    eval_freq: int = 10_000 
    seed: int = 0

@dataclass
class Agent:
    """Agent is the policy. \epsilon-greedy tabular agent with an (A, S) array."""

    epsilon: float # decaying value

    @dataclass
    class State:
        key: chex.Array
        alg_state: Any

    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array:
        """Q-values for given state indices."""
        return state.alg_state.q_vals[:, obs]
    
    def act(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[chex.Array, Agent.State]:
        """ε-greedy action selection."""
        key, act_key, explore_key = jrd.split(state.key, 3)
        greedy = jnp.argmax(self.q_vals(state, obs))
        random = jrd.randint(act_key, (), 0, state.alg_state.q_vals.shape[0])
        action = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return action, state.replace(key=key)
    
@jax.jit
def train_step(state: Imc.State, k: int) -> Imc.State:
    """One transition + Q-learning update with decaying step size."""
 
    trans, state = imc.sample(state)
    alpha = cfg.alpha_init / (1.0 + k / cfg.alpha_period) ** cfg.alpha_power
    new_alg_state = q_learning_async.update(state.agent.alg_state, trans, alpha)
    return state.replace(agent=state.agent.replace(alg_state=new_alg_state))

cfg = tyro.cli(Config)
S, A = cfg.garnet.state_size, cfg.garnet.action_size

agent = Agent(epsilon=cfg.epsilon)
imc = Imc(
    agent=agent,
    mc=Mc(
        max_episode_len=cfg.garnet.max_episode_len,
        queue_size=20,
        env=garnet.make(cfg.garnet),
    ),
)

key = jrd.PRNGKey(cfg.seed)
key, env_key, agent_key = jrd.split(key, 3)
env_state = imc.mc.env.init(env_key)

opt_q = optimal_q(env_state.mdp, cfg.gamma)
opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))

evaluator = Evaluator(mdp=env_state.mdp, gamma=cfg.gamma, agent=agent)
jit_eval = jax.jit(evaluator.metric)
agent_state = Agent.State(key=agent_key, alg_state=q_learning_async.init(env_state.mdp, cfg.gamma))
imc_state = imc.init(mc=imc.mc.init(agent_key, env_state), agent=agent_state)
eval_state = evaluator.init(agent_state)

print(f"[bold green]Q-learning on Garnet[/bold green] ({S}S, {A}A)")

t0 = time.time()
for k in track(range(cfg.n_steps), description="Training"):
    imc_state = train_step(imc_state, k)

    if (k + 1) % cfg.eval_freq == 0:
        m, eval_state = jit_eval(eval_state, opt_q, imc_state.agent)
        print(
            f"  step {k + 1:6d}"
            f"  bellman={float(m.bellman_linf):.4f}"
            f"  value={float(m.value_norm):.4f}"
            f"  ρ(π)={float(m.pi_eval_rho):.3f}"
        )

elapsed = time.time() - t0
print(
    f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
    f"  value_norm={float(m.value_norm):.6f}"
    f"  bellman_linf={float(m.bellman_linf):.6f}"
    f"  ρ*(π)={opt_rho:.3f}"
)