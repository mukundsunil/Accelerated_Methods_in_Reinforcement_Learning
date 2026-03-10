from typing import Any

import click
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxtor.env import tabular
from jaxdp.typehints import F, PiType, StaticMeta
from Tools import e_greedy_policy as e_g_policy
from Tools import exp_decay as exp_decay

from jaxtor.sampler.mc import Mc

from Algorithms import q_learning_async
# REQUIRES MORE IMPORTING PLEASE CHECK THIS LATER

jax.config.update("jax_enable_x64", True)

# Environment Generation
key = jrd.PRNGKey(0)
config = tabular.garnet.Config(state_size=10,
    action_size=4,
    max_episode_len=50)
env = tabular.garnet.make(config=config)

# Sampler Generation
sampler = Mc(max_episode_len=100, queue_size=10, env=env)
env_state = env.init(key)
mc_state = sampler.init(key, env_state)

# Loop
key= jrd.PRNGKey(42)
init_state = q_learning_async.init(mdp=env, key=key, gamma=0.9)

def step(carry,_):
    
    q_learning_async.alg_State, mc_state, last_obs, key, iter = carry
    key, act_key = jrd.split(key)
    policy = e_g_policy.q(q_learning_async.alg_State.q_val, 0.7) # Needs changes
    action = policy[:,last_obs]
    act_loc = jnp.argmax(action)
    trans, mc_state = sampler.sample(act_loc, mc_state)
    alpha = 1/(1+iter)
    next_state = q_learning_async.update(q_learning_async.alg_State, trans, alpha)
    return (next_state, mc_state, last_obs, key, iter+1), None

final_carry,_ = jax.lax.scan(step, (init_state, mc_state, mc_state.last_obs, key, 0), length=100)
final_state, final_mc_state, last_obs, key, iter = final_carry


# class metrics(metaclass=StaticMeta):
#     """
#     ◈─────────────────────────────────────────────────────────────────────────◈
#     Metrics Namespace

#     Computes and stores training metrics.
#     ◈─────────────────────────────────────────────────────────────────────────◈
#     """

#     @struct.dataclass
#     class State:
#         """Metrics collected during training"""

#         l1: jnp.ndarray # L1 norm of Q-value change (scalar)
#         l2: jnp.ndarray # L2 norm of Q-value change (scalar)
#         linf: jnp.ndarray # L-infinity norm of Q-value change (scalar)
#         bellman_res: jnp.ndarray # Bellman error (scalar)
#         expected_policy_eval: jnp.ndarray
#         max_value_diff: jnp.ndarray
#         expected_value: jnp.ndarray
#         iteration: jnp.ndarray # Iteration number (scalar)
        
#     def compute(
#         prev: "loop.State",
#         new: "loop.State",
#         args: "loop.Args",
#         step: int,
#         dones: F["..."],
#         eval_results: "loop.EvalResult",
#     ) -> "metrics.State":
#         """Compute metrics for the current iteration

#         Args:
#             prev: Previous loop state
#             new: New loop state
#             args: Loop arguments
#             step: Current step number
#             dones: Boolean array indicating which environments completed episodes [n_envs]
#             eval_results: Evaluation results (loop.EvalResult dataclass)

#         Returns:
#             Metrics state with NaN values where no information is available
#         """
#         prev_alg = prev.alg_state
#         new_alg = new.alg_state

#         diff = new_alg.q_vals - prev_alg.q_vals
#         l1 = jnp.sum(jnp.abs(diff))
#         l2 = jnp.sqrt(jnp.sum(diff**2))
#         linf = jnp.max(jnp.abs(diff))

#         bellman_target = bellman_op.q(args.mdp, prev_alg.q_vals, prev_alg.gamma)
#         bellman_err = jnp.max(jnp.abs(prev_alg.q_vals - bellman_target))

#         ep_return = jnp.where(dones, new.last_return, jnp.nan)
#         ep_len = jnp.where(dones, prev.ep_step, jnp.nan)

#         return metrics.State(
#             l1=l1,
#             l2=l2,
#             linf=linf,
#             bellman_err=bellman_err,
#             iteration=step,
#             ep_return=ep_return,
#             ep_len=ep_len,
#             eval_mean_return=eval_results.mean_return,
#             eval_std_return=eval_results.std_return,
#         )









from __future__ import annotations

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


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    garnet: garnet.Config = dataclasses.field(default_factory=garnet.Config)
    n_steps: int = 1_000_000
    alpha_init: float = 0.5
    alpha_power: float = 0.25
    alpha_period: float = 10_000.0
    gamma: float = 0.99
    epsilon: float = 0.1
    eval_freq: int = 10_000
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """ε-greedy tabular Q-learning agent with an (A, S) Q-table."""

    epsilon: float

    @dataclass
    class State:
        key: chex.Array
        q_vals: chex.Array  # (A, S)

    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array:
        """Q-values for given state indices."""
        return state.q_vals[:, obs]
    
    def act(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[chex.Array, Agent.State]:
        """ε-greedy action selection."""
        key, act_key, explore_key = jrd.split(state.key, 3)
        greedy = jnp.argmax(state.q_vals[:, obs])
        random = jrd.randint(act_key, (), 0, state.q_vals.shape[0])
        action = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return action, state.replace(key=key)

    


@jax.jit
def train_step(state: Imc.State, k: int) -> Imc.State:
    """One transition + Q-learning update with decaying step size."""
    trans, state = imc.sample(state)
    q_vals = state.agent.q_vals
    alpha = cfg.alpha_init / (1.0 + k / cfg.alpha_period) ** cfg.alpha_power
    discount = jnp.where(trans.term, 0.0, cfg.gamma)
    td = rlax.q_learning(
        q_vals[:, trans.obs], trans.act, trans.rew, discount, q_vals[:, trans.nobs]
    )
    new_q = q_vals.at[trans.act, trans.obs].add(alpha * td)
    return state.replace(agent=state.agent.replace(q_vals=new_q))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

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
agent_state = Agent.State(key=agent_key, q_vals=jnp.zeros((A, S)))
imc_state = imc.init(mc=imc.mc.init(agent_key, env_state), agent=agent_state)
eval_state = evaluator.init(agent_state)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

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