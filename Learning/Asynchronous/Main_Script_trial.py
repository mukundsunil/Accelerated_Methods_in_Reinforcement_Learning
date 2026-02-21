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