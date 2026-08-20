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
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "q_learning_async.alg_State":
        
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, c=c)
    
    def update(state: "q_learning_async.alg_State", trans: Any, alpha, k:int)  -> "q_learning_async.alg_State": 
        
        c_new = state.c.at[trans.act, trans.obs].add(1)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        td_err = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        next_q = state.q_vals.at[trans.act, trans.obs].add(alpha * td_err)

        return state.replace(q_vals=next_q, c=c_new)
    
class zap_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray
        matrix_gain: Any
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "zap_q_learning_async.alg_State":
        
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        matrix_gain = jnp.eye(mdp.action_size*mdp.state_size)
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return zap_q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, matrix_gain=matrix_gain, c=c)
    
    def update(state: "zap_q_learning_async.alg_State", trans: Any, alpha, k:int)  -> "zap_q_learning_async.alg_State": 
        
        c_new = state.c.at[trans.act, trans.obs].add(1)
        beta = (1/(2+k)**0.6)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        nact = jnp.argmax(state.q_vals[:, trans.nobs])

        # \hat{A_{k+1}} approximation. Here \hat{A_{k+1}} is (I - \gamma P_hat) 
        A, S = state.q_vals.shape
        E_zeros = jnp.zeros([A, S, A, S]) # 4-D Tensor
        E = E_zeros.at[trans.act, trans.obs, trans.act, trans.obs].set(1) # setting one at (a,s,a,s)
        E_reshape = E.reshape(A*S, A*S) # reshaping it to (as, as)
        P_hat = jnp.zeros((A, S, A, S))
        P_hat_upd = P_hat.at[trans.act, trans.obs, nact, trans.nobs].set(1.0) # setting one at (a,s,a',s^+)
        step_matrix_gain = (E_reshape - state.gamma*(P_hat_upd.reshape(A*S, A*S))) # (E - \gamma P)
        next_matrix_gain = state.matrix_gain + beta * (step_matrix_gain - state.matrix_gain) # SA update of (I - \gamma P) and shape is (as, as)

        td_error = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        g= jnp.zeros((A, S))
        g_flat = (g.at[trans.act, trans.obs].set(td_error)).flatten()
        pre_td_err = jnp.linalg.solve(next_matrix_gain, g_flat).reshape(A, S)
        new_q = state.q_vals + alpha * pre_td_err

        return state.replace(q_vals=new_q, matrix_gain=next_matrix_gain, c=c_new)

class pre_cond_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray
        matrix_gain: Any
        P_hat: Any
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "pre_cond_q_learning_async.alg_State":
        
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s) 
        matrix_gain = jnp.eye(mdp.action_size*mdp.state_size) # \hat{A}_{0}^{-1} (as, as)
        P_hat = jnp.zeros((mdp.action_size*mdp.state_size, mdp.action_size*mdp.state_size)) # (as, as)
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return pre_cond_q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, matrix_gain=matrix_gain, P_hat=P_hat,
                                                   c=c)
    
    def update(state: "pre_cond_q_learning_async.alg_State", trans: Any, alpha, k:int)  -> "pre_cond_q_learning_async.alg_State": 

        discount = jnp.where(trans.term, 0.0, state.gamma)
        nact = jnp.argmax(state.q_vals[:, trans.nobs]) 
        c_new = (state.c.at[trans.act, trans.obs].add(1))
        
        # Approximating \hat{P} 
        A, S = state.q_vals.shape
        E_zeros = jnp.zeros([A, S]) # (a,s)
        E_k = E_zeros.at[trans.act, trans.obs].set(1) # setting 1 at (a,s)
        E_k_next = E_zeros.at[nact, trans.nobs].set(1) # setting 1 at (a',s^+)
        e_k = E_k.flatten() # reshaping it to (as, )
        e_k_next = E_k_next.flatten() # reshaping it to (as, )
        P_reshape = state.P_hat.reshape(A, S, A, S) # 4-D Tensor
        pdf_tens = P_reshape[trans.act, trans.obs, :, :] # 2-D Tensor
        pdf = pdf_tens.reshape(A*S) # \hat{P_{k}}((s,a), :)
        diff = e_k_next - pdf # (e_{k+1} - p_k(s,a))
        P_diff = jnp.einsum("i,j->ij", e_k, diff) # e_{k} * (e_{k+1}^T - p_k(s,a)^T)
        beta = 1/(c_new[trans.act, trans.obs]**0.6)
        upd = beta * P_diff
        P_hat_new = state.P_hat + upd
        
        # Approximating \hat{A}_{k+1}^{-1} = (I - \gamma \hat{P}_{k+1})^{-1}
        u = -state.gamma * beta * e_k  # Shape (AS,)
        inv_u = jnp.dot(state.matrix_gain, u)      # (AS,)
        v_inv = jnp.dot(diff, state.matrix_gain)      # (AS,)
        den = 1.0 + jnp.dot(diff, inv_u) 
        num = jnp.outer(inv_u, v_inv)
        matrix_gain_new = state.matrix_gain - (num / den) # Woodsbury inversion formula

        td_error = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        td= jnp.zeros([A, S])
        td_flat = (td.at[trans.act, trans.obs].set(td_error)).flatten()
        gain = (jnp.einsum("ij,j->i", matrix_gain_new, td_flat)).reshape(A, S)
        new_q = state.q_vals + alpha * gain

        return state.replace(q_vals=new_q, matrix_gain=matrix_gain_new, P_hat=P_hat_new, c=c_new)