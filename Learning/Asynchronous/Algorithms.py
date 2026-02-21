import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
import chex
from chex import dataclass

from jaxtor.env.tabular import TabularEnv
from jaxdp.typehints import F, QType, StaticMeta

@struct.dataclass
class Transition:
    """
    Dataclass for MDP Transition
    
    Contains information regarding one single transition 
    
    Attributes:
            obs: Current observation.
            act: Action taken.
            rew: Reward received.
            term: Terminal flag (episode ended naturally).
            trun: Truncated flag (episode ended due to time limit).
            nobs: Next observation.
            
    """
    obs: chex.Array
    act: chex.Array
    rew: chex.Array
    term: chex.Array
    trun: chex.Array
    nobs: chex.Array

class q_learning_async(metaclass=StaticMeta):

    @struct.dataclass
    class alg_State:
        q_val: QType
        gamma: jnp.ndarray

    def init(mdp: TabularEnv, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "q_learning_async.alg_State":
        
        q_val = jnp.zeros((mdp.act_space.shape[0], mdp.obs_space.shape[0]))

        return q_learning_async.alg_State(q_val=q_val, gamma=gamma)
    
    def update(state: "q_learning_async.alg_State", trans: Transition, alpha)  -> "q_learning_async.alg_State": 
        
        td_error = trans.rew + state.gamma*jnp.max(state.q_val[:, trans.nobs]) - state.q_val[trans.act, trans.obs] 
        next_q = state.q_val.at[trans.act, trans.obs].set(state.q_val[trans.act, trans.obs] + alpha * td_error)

        return state.replace(q_val=next_q)