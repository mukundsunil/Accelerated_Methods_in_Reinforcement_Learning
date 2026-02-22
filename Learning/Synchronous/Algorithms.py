import jax 
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from jaxdp.mdp import MDP
from jaxdp.typehints import F, QType, StaticMeta

@struct.dataclass
class Transition:
    """"
    Protocol dataclass for MDP transitions.

    Contains information regading one single transition (s, a, r, s^+, done).
    """
    next_state: F["S"] # next state reached (one-hot vector)
    reward: F[""] # Reward received (scalar)
    terminal: F[""] # Terminal flag (scalar). This is to signify the value of the terminal state which has 0.
    state: F["S"] # Current state (one-hot vector)
    action: F["A"] # Current action (one-hot vector)

@struct.dataclass
class SyncSample:
    """"
    Protocol dataclass for MDP transitions for synchronous updates.

    Contains information regading one single transition (s, a, r, s^+, done).

    """   

    reward: F["AS"] # Reward received 
    next_state: F["ASS"] # next state reached (one-hot vector in third direction)
    terminal: F["AS"] # Terminal flag. This is to signify the value of the terminal state which has 0. 

class q_learning_sync(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "q_learning_sync.State":
        
        q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        
        return q_learning_sync.State(q_vals = q_vals, gamma = gamma)

    def q_target(next_state: F["S"], reward: F[""], terminal: F[""], q_vals: F["AS"], gamma: F[""]) -> F[""]:

        q_next = jnp.einsum("s,as->a", next_state, q_vals)
        max_q_next = jnp.max((1.0 - terminal) * q_next)
        target = reward + gamma * max_q_next

        return target

    def update(state: "q_learning_sync.State", sample: SyncSample, alpha, step) -> "q_learning_sync.State":
 
        batch_q_target = jax.vmap(jax.vmap(q_learning_sync.q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
        target_values = batch_q_target(sample.next_state, sample.reward, sample.terminal, state.q_vals, state.gamma)
        next_q = state.q_vals + (target_values - state.q_vals) * alpha
        
        return state.replace(q_vals=next_q)

class sql_sync(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "sql_sync.State":

        q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)

        return sql_sync.State(q_vals=q_vals, prev_q=q_vals, gamma=gamma) # q_0 and q_-1 are initialized to the same value
    
    def q_target(next_state: F["S"], reward: jnp.ndarray, terminal: jnp.ndarray, q_vals: QType,
                  gamma: jnp.ndarray) -> jnp.ndarray:
        
        q_next = jnp.einsum("s,as->a", next_state, q_vals)
        max_q = jnp.max((1.0-terminal) * q_next)

        return reward + gamma * max_q
    
    def update(state: "sql_sync.State", sample: "SyncSample", alpha, step) -> "sql_sync.State":
        
        batched_q_target = jax.vmap(jax.vmap(sql_sync.q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
        prev_q_batched_target = batched_q_target(sample.next_state, sample.reward, sample.terminal, state.prev_q,
                                          state.gamma)
        q_batched_target = batched_q_target(sample.next_state, sample.reward, sample.terminal, state.q_vals,
                                          state.gamma)
        next_q = state.q_vals + alpha*(prev_q_batched_target - state.q_vals) + (1-alpha)*(q_batched_target - prev_q_batched_target)

        return state.replace(q_vals = next_q, prev_q = state.q_vals)

class momentumq_sync(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "momentumq_sync.State":

        q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)

        return momentumq_sync.State(q_vals=q_vals, prev_q=q_vals, gamma=gamma) # q_0 and q_-1 are initialized to the same value
    
    def q_target(next_state: F["S"], reward: jnp.ndarray, terminal: jnp.ndarray, q_vals: QType,
                  gamma: jnp.ndarray) -> jnp.ndarray:
        
        q_next = jnp.einsum("s,as->a", next_state, q_vals)
        max_q = jnp.max((1.0-terminal) * q_next)

        return reward + gamma * max_q
    
    def update(state: "momentumq_sync.State", sample: "SyncSample", alpha, step) -> "momentumq_sync.State":
        
        m = 1.12 # Value from paper. For m greater than 10, \alpha formaulation changes
        b = step - m - 1
        c = (-step**2 + (m+1)*step +1)/(step+1)
        batched_q_target = jax.vmap(jax.vmap(sql_sync.q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
        prev_q_batched_target = batched_q_target(sample.next_state, sample.reward, sample.terminal, state.prev_q,
                                          state.gamma)
        q_batched_target = batched_q_target(sample.next_state, sample.reward, sample.terminal, state.q_vals,
                                          state.gamma)
        S = (1-alpha) * state.prev_q + alpha * prev_q_batched_target
        P = (1-alpha) * state.q_vals + alpha * q_batched_target
        next_q = P + b * (P -S) + c* (state.q_vals - state.prev_q)

        return state.replace(q_vals = next_q, prev_q = state.q_vals)
    
class zap_ql_sync(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_vals: QType
        gamma:jnp.ndarray
        matrix_gain: F["AS AS"]

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "zap_ql_sync.State":
        
        matrix_gain = jnp.eye(mdp.action_size*mdp.state_size)
        q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)

        return zap_ql_sync.State(q_vals=q_vals, matrix_gain=matrix_gain, gamma=gamma)
    
    def q_target(next_state: F["S"], reward: jnp.ndarray, terminal: jnp.ndarray, q_vals: QType,
                  gamma: jnp.ndarray) -> jnp.ndarray:
        
        q_next = jnp.einsum("s,as->a", next_state, q_vals)
        max_q = jnp.max((1.0-terminal) * q_next)

        return reward + gamma * max_q
    
    def update(state: "zap_ql_sync.State", sample: "SyncSample", alpha, step) -> "zap_ql_sync.State":

        act_size, state_size = state.q_vals.shape
        batched_q_target = jax.vmap(jax.vmap(sql_sync.q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
        delta = batched_q_target(sample.next_state, sample.reward, sample.terminal, state.q_vals, state.gamma) - state.q_vals

        next_action = jax.nn.one_hot(jnp.argmax(jnp.einsum("asx,ux->asu", sample.next_state, state.q_vals), axis=-1), act_size)
        step_matrix_gain = (jnp.eye(act_size*state_size) - state.gamma*jnp.einsum("asx,asu->asux", sample.next_state, next_action
                                           ).reshape(act_size * state_size, act_size * state_size)) # (I -\gamma P) is being estimated as a whole
        
        next_matrix_gain = state.matrix_gain + (1/(2+step)) * (step_matrix_gain - state.matrix_gain)
        next_q = state.q_vals + alpha * (jnp.linalg.inv(next_matrix_gain) @ delta.flatten()).reshape(act_size, state_size)
        return state.replace(q_vals = next_q, matrix_gain=next_matrix_gain)
    
class r1_ql_sync(metaclass=StaticMeta):

    @struct.dataclass
    class State:
        q_vals: QType
        gamma:jnp.ndarray
        pdf: F["A S"]

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "r1_ql_sync.State":
        
        pdf_init = jnp.ones((mdp.action_size*mdp.state_size))/(mdp.action_size*mdp.state_size)
        pdf = pdf_init.reshape((mdp.action_size, mdp.state_size))
        q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)

        return r1_ql_sync.State(q_vals=q_vals, pdf=pdf, gamma=gamma)
    
    def q_target(next_state: F["S"], reward: jnp.ndarray, terminal: jnp.ndarray, q_vals: QType,
                  gamma: jnp.ndarray) -> jnp.ndarray:
        
        q_next = jnp.einsum("s,as->a", next_state, q_vals)
        max_q = jnp.max((1.0-terminal) * q_next)

        return reward + gamma * max_q
    
    def update(state: "r1_ql_sync.State", sample: "SyncSample", alpha, step) -> "r1_ql_sync.State":

        act_size, state_size = state.q_vals.shape
        batched_target = jax.vmap(jax.vmap(sql_sync.q_target, (0, 0, 0, None, None)), (0, 0, 0, None, None))
        q_batched_target = batched_target(sample.next_state, sample.reward, sample.terminal, state.q_vals, state.gamma)
        delta = q_batched_target - state.q_vals
        next_action = jax.nn.one_hot(jnp.argmax(jnp.einsum("asx,ux->asu", sample.next_state, state.q_vals), axis=-1), act_size)
        pdf_approx = jnp.einsum("asx,asu,as->ux", sample.next_state, next_action, state.pdf) 
        prev_pdf = state.pdf
        pdf = state.pdf + alpha * (pdf_approx - state.pdf) # what is beta
        lamb = (state.gamma * alpha)/(1-state.gamma) * jnp.einsum("as,as->", delta, pdf)
        next_q = state.q_vals + (delta + lamb) * alpha

        return state.replace(q_vals = next_q, pdf=pdf)


        