import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Any
import chex
from chex import dataclass
import rlax 
import tyro

from jaxtor.env.tabular import TabularEnv
from jaxdp.typehints import F, QType, StaticMeta

class q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray

    def init(env_state: TabularEnv.State, gamma: jnp.ndarray) -> "q_learning_async.alg_State":
        
        q_vals = jnp.zeros((env_state.mdp.action_size, env_state.mdp.state_size))

        return q_learning_async.alg_State(q_vals=q_vals, gamma=gamma)
    
    def update(state: "q_learning_async.alg_State", trans: Any, alpha)  -> "q_learning_async.alg_State": 
        
        discount = jnp.where(trans.term, 0.0, state.gamma)
        td_err = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        next_q = state.q_vals.at[trans.act, trans.obs].add(alpha * td_err)

        return state.replace(q_vals=next_q)