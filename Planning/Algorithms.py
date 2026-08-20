import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from Tools import Bellman_Optimality_Operator as T
from Tools import Bellman_Policy_Operator as T_pi
from Tools import greedy_policy as g
from Tools import policy_evaluation as pe
from Tools import _markov_chain_pi as mc_pi
from Tools import to_state_action_value as q_sa



from jaxdp.mdp import MDP
from jaxdp.typehints import F, VType, QType, PiType, StaticMeta

####################################### Planning Algorithms ############################################

class vi(metaclass=StaticMeta):
    r"""
    Value Iteration

    Initialized with random values. Update equation follows:
    .. math::
            v(s) = T v(s)
    """
    
    @struct.dataclass
    class State:
        v_vals: VType
        gamma: jnp.ndarray

    # INITIALIZATION
    def init(mdp: MDP, key:jrd.PRNGKey, gamma: jnp.ndarray) -> "vi.State":
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        
        return vi.State(v_vals = v_vals, gamma = gamma)
    
    # UPDATE
    def update(state: "vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "vi.State":
        next_v = T.v(mdp, state.v_vals, state.gamma)
        return state.replace(v_vals = next_v)
    

class gs_vi(metaclass=StaticMeta):
    r"""
    Gauss_Seidel Value Iteration

    Initialized with random values. Follows asynchronous VI update equation
    """
    
    @struct.dataclass
    class State:
        v_vals: VType
        gamma: jnp.ndarray

    # INITIALIZATION
    def init(mdp: MDP, key:jrd.PRNGKey, gamma: jnp.ndarray) -> "vi.State":
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        
        return vi.State(v_vals = v_vals, gamma = gamma)
    
    # UPDATE
    def update(state: "gs_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "gs_vi.State":
        prev_v = state.v_vals
        state_indices = jnp.arange(mdp.state_size)

        def scan_body(carry, state_index):
            p_j = mdp.transition[:,:,state_index]
            exp_reward = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
            r_j = exp_reward[:,state_index]

            exp_v = jnp.einsum("as,s->a", p_j, carry)

            new_v_j = jnp.max(r_j + state.gamma*exp_v)
            new_carry_v = carry.at[state_index].set(new_v_j)

            return new_carry_v, None
         
        next_v, _ = jax.lax.scan(scan_body, prev_v, state_indices)  
        return state.replace(v_vals = next_v)
    
class pi(metaclass = StaticMeta):
    r"""
    Policy Iteration

    Initialized with random values. Update equation follows:
    
    Policy Improvement: 
           \pi_k = greedy(v(s))

    Policy Evaluation:
            v_k(s) = T_{\pi_k}
    """

    @struct.dataclass
    class State:
        v_vals: VType
        gamma: jnp.ndarray

    # INITIALIZATION
    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "pi.State":
        
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        return pi.State(v_vals = v_vals, gamma=gamma)
    
    # UPDATE 
    def update(state: "pi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "pi.State":
        
        # Policy Improvement
        policy = g.v(mdp, state.v_vals, state.gamma)

        # Policy Evaluation
        next_v = pe.v(mdp, policy, state.gamma)
        return state.replace(v_vals = next_v)
    
    # UPDATE (Modified PI)
    def update_m(state: "pi.State", mdp: MDP, step: int, theta: jnp.ndarray) -> "pi.State":
        
        # Policy Improvement
        policy = g.v(mdp, state.v_vals, state.gamma)

        # Policy Evaluation
        def cond_fun(carry):
            i, v_prev = carry
            v_cur = T_pi.v(mdp, policy, v_prev, state.gamma)
            diff = jnp.max(jnp.abs(v_cur - v_prev)) 
            return jnp.logical_and(i < step, diff >= theta)
              
        def body_fun(carry):
            i, v_prev = carry
            v_cur = T_pi.v(mdp, policy, v_prev, state.gamma)
            return i + 1, v_cur

        _, v_final = jax.lax.while_loop(cond_fun,
                                        body_fun,
                                        (0, state.v_vals))
            
        return state.replace(v_vals=v_final)
    

class pi_q(metaclass = StaticMeta): 
    r"""
    Policy Iteration for Q-function
    """

    @struct.dataclass
    class State:
        q_vals: QType
        v_vals: VType
        gamma: jnp.ndarray

    # INITIALIZATION
    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "pi_q.State":
        
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        v_vals = jnp.zeros((mdp.state_size))
        return pi_q.State(q_vals = q_vals, gamma=gamma, v_vals=v_vals)
    
    # UPDATE 
    def update(state: "pi_q.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "pi_q.State":
        
        # Policy Improvement
        policy = g.q(state.q_vals)

        # Policy Evaluation
        next_q = pe.q(mdp, policy, state.gamma)
        v_vals = jnp.max(next_q, axis = 0)
        return state.replace(q_vals = next_q, v_vals = v_vals)

    
class r_vi(metaclass=StaticMeta):
    r"""
    Relaxed VI. The update equation is taken from the paper: "A FIRST-ORDER APPROACH TO ACCELERATED VALUE
    ITERATION"
    https://arxiv.org/abs/1905.09963
    """
    
    @struct.dataclass
    class State:
        v_vals: VType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "r_vi.State":
        v_vals = jrd.uniform(key, mdp.state_size, dtype = 'float', minval=0.0, maxval=1.0)
        return r_vi.State(v_vals=v_vals, gamma=gamma)
    
    def update(state: "r_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "r_vi.State":
        alpha = 0.9
        next_v = state.v_vals - alpha * (state.v_vals - T.v(mdp, state.v_vals, state.gamma))
        return state.replace(v_vals = next_v)
        
class a_vi(metaclass=StaticMeta):
    r"""
    Accelerated VI. The update equation is taken from the paper: "A FIRST-ORDER APPROACH TO ACCELERATED VALUE
    ITERATION". This is implemented in the form of General Policy Iteration.
    https://arxiv.org/abs/1905.09963
    """
    @struct.dataclass
    class State:
        v_vals: VType
        prev_v: VType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "a_vi.State":
        v_vals = jrd.uniform(key, mdp.state_size, dtype = 'float', minval=0.0, maxval=1.0)
        return a_vi.State(v_vals=v_vals, prev_v = v_vals.copy(), gamma=gamma)
    
    def update(state: "a_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "a_vi.State": 
        alpha = 1/(1+state.gamma)
        lamb = (1-jnp.sqrt(1-state.gamma**2))/state.gamma
        v_vals = state.v_vals
        prev_v = state.prev_v
        policy = g.v(mdp, v_vals, state.gamma)
        step = 1000

        def cond_fun(carry):
            i, v_vals, prev_v = carry
            h = v_vals + lamb * (v_vals - prev_v)
            next_v = h - alpha * (h - T_pi.v(mdp, policy, h, state.gamma))
            diff = jnp.max(jnp.abs(next_v - v_vals)) 
            return jnp.logical_and(i < step, diff >= theta)
              
        def body_fun(carry):
            i, v_vals, prev_v = carry
            h = v_vals + lamb * (v_vals - prev_v)
            next_v = h - alpha * (h - T_pi.v(mdp, policy, h, state.gamma))
            return i + 1, next_v, v_vals

        _, next_v, v_vals = jax.lax.while_loop(cond_fun,
                                        body_fun,
                                        (0, v_vals, prev_v))
        
        return state.replace(v_vals = next_v, prev_v = v_vals)

class m_vi(metaclass=StaticMeta):
    r"""
    Momentum VI. The update equation is taken from the paper: "A FIRST-ORDER APPROACH TO ACCELERATED VALUE
    ITERATION". This is implemented in the form of General Policy Iteration.
    https://arxiv.org/abs/1905.09963
    """
    @struct.dataclass
    class State:
        v_vals: VType
        prev_v:VType
        gamma: jnp.ndarray
    
    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "m_vi.State":
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        return m_vi.State(v_vals=v_vals, prev_v=v_vals.copy(), gamma=gamma)
    
    def update(state: "m_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "m_vi.State":
        alpha = 2/(1 + jnp.sqrt(1 - state.gamma**2))
        beta = (1 - jnp.sqrt(1 - state.gamma**2))/(1 + jnp.sqrt(1 - state.gamma**2))
        v_vals = state.v_vals
        prev_v = state.prev_v
        policy = g.v(mdp, v_vals, state.gamma)
        step = 1000

        def cond_fun(carry):
            i, v_vals, prev_v, diff = carry
            return jnp.logical_and(i < step, diff >= theta)
              
        def body_fun(carry):
            i, v_vals, prev_v, diff = carry
            next_v = v_vals - alpha * (v_vals - T_pi.v(mdp, policy, v_vals, state.gamma)) + beta * (v_vals - prev_v)
            diff = jnp.max(jnp.abs(next_v - v_vals))
            return i + 1, next_v, v_vals, diff

        _, next_v, v_vals, diff = jax.lax.while_loop(cond_fun,
                                        body_fun,
                                        (0, v_vals, prev_v, 1e-3))
        
        return state.replace(v_vals = next_v, prev_v = v_vals)
    
class sa_vi(metaclass=StaticMeta): 
    r"""
    Safe Accelerated VI or Nesterov VI. The update equation is taken from the paper: "A FIRST-ORDER APPROACH TO ACCELERATED VALUE
    ITERATION". Makes use of safe gaurding againt VI updates.
    https://arxiv.org/abs/1905.09963
    """

    @struct.dataclass
    class State:
        v_vals: VType
        v_init: VType
        prev_v: VType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "sa_vi.State":
        v_init= jrd.uniform(key, mdp.state_size, dtype = 'float', minval=0.0, maxval=1.0)
        return sa_vi.State(v_init=v_init, v_vals = v_init, prev_v = v_init.copy(), gamma=gamma)
    
    def update(state: "sa_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "sa_vi.State":
        alpha = 1/(1+state.gamma)
        lamb = (1-jnp.sqrt(1-state.gamma**2))/state.gamma
        h = state.v_vals + lamb * (state.v_vals - state.prev_v)
        dummy_v = h - alpha * (h - T.v(mdp, h, state.gamma))

        # Safe-Gaurding

        lhs = jnp.max(jnp.abs(dummy_v - T.v(mdp, dummy_v, state.gamma)))
        rhs = (state.gamma**iter) * jnp.max(jnp.abs(state.v_init - T.v(mdp, state.v_init, state.gamma)))

        def true_branch(_):
            return state.replace(v_vals = dummy_v, prev_v = state.v_vals)

        def false_branch(_):
            return state.replace(v_vals = T.v(mdp, state.v_vals, state.gamma), prev_v = state.v_vals)

        new_state = jax.lax.cond(lhs<=rhs, true_branch, false_branch, operand = None)
        
        return new_state

class anc_vi(metaclass = StaticMeta):
    r"""
    Anchoring VI. The update equation is taken from the paper: "Accelerating Value Iteration with Anchoring". 
    https://arxiv.org/abs/2305.16569
    """

    @struct.dataclass
    class State:
        v_init: VType
        v_vals: VType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "anc_vi.State":
        v_init = jrd.uniform(key, mdp.state_size, dtype= 'float', minval=0.0, maxval=1.0)
        return anc_vi.State(v_init = v_init, v_vals = v_init, gamma = gamma)
    
    def update(state: "anc_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "anc_vi.State":
        v_init = state.v_init
        def sum(gamma, iter):           
            def body_func(i, acc):
                return acc + gamma**(-2*i)
            return jax.lax.fori_loop(0, iter+1, body_func, 0.0) 
        beta = 1/sum(state.gamma, iter)
        next_v = beta * v_init + (1 - beta) * T.v(mdp, state.v_vals, state.gamma)
        return state.replace(v_vals = next_v)

class and_vi(metaclass=StaticMeta):
    r"""
    Anderson VI. The update equation is taken from the paper: "Anderson Acceleration for Reinforcement Learning". 
    This makes use of safe-gaurding w.r.t VI. The anderson window is chosen to have a size of 5
    https://arxiv.org/abs/1809.09501
    """
    @struct.dataclass
    class State:
        v_vals: VType
        v_init: VType
        p_max: jnp.ndarray
        gamma: jnp.ndarray
        Delta: VType
        delta: VType
        Tvals: VType

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "and_vi.State":
        v_init = jrd.uniform(key, mdp.state_size, dtype= 'float', minval=0.0, maxval=1.0)
        v_vals= T.v(mdp, v_init, gamma)
        delta = v_vals - v_init
        p_max = 4 # As jax is not mutable, creating Delta size bigger than Anderson window size
        init_Delta = jnp.zeros((mdp.state_size, p_max))
        Tvals = jnp.zeros((mdp.state_size, p_max))
        init_Tvals = Tvals.at[:,0].set(v_vals)
        return and_vi.State(v_init = v_init, v_vals = v_vals, delta = delta, gamma = gamma, Delta = init_Delta, p_max = p_max, Tvals = init_Tvals)    
    
    def update(state: "and_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "and_vi.State":
        p = 4 # Anderson Window
        p_k = jnp.minimum(p, iter)
        one_p_max = jnp.ones(p)
        ones = one_p_max[p_k]
        BO = T.v(mdp, state.v_vals, state.gamma)
        init_delta = state.delta
        init_Delta = state.Delta.at[:,0].set(init_delta)
        init_Tvals = state.Tvals

        def true_fn(operand): # iter < p
            init_Delta, BO, iter, p_k = operand
            inter_delta = BO - state.v_vals          
            inter_Delta = init_Delta.at[:,iter].set(inter_delta) # Array to be passed on for next iteration
            new_Delta = inter_Delta[:,:p] # Internal Loop computation Array
            return inter_Delta, new_Delta

        def false_fn(operand):
            init_Delta, BO, iter, p_k = operand
            inter_delta = BO - state.v_vals
            rem_first_col_Delta = init_Delta[:,1:]
            inter_Delta = jnp.concatenate([rem_first_col_Delta, inter_delta[:, None]], axis = 1) # Array to be passed on for next iteration
            new_Delta = inter_Delta[:,:p] # Internal Loop computation Array
            return inter_Delta, new_Delta

        inter_Delta, new_Delta = jax.lax.cond(iter<p, true_fn, false_fn, (init_Delta, BO, iter, p_k))

        DTD_inv = jnp.linalg.pinv(new_Delta.T @ new_Delta) 
        ones = jnp.ones(DTD_inv.shape[0])
        num = DTD_inv @ ones
        den = ones.T @ num
        alpha = num/den 

        def case_equal(operand): # iter < p
            init_Tvals, BO, iter, p_k = operand          
            inter_Tvals = init_Tvals.at[:,iter].set(BO) # Array to be passed on for next iteration
            new_Tvals = inter_Tvals[:,:p] # Internal Loop computation Array
            return inter_Tvals, new_Tvals

        def case_not_equal(operand):
            init_Tvals, BO, iter, p_k = operand
            rem_first_col_Tvals = init_Tvals[:,1:]
            inter_Tvals = jnp.concatenate([rem_first_col_Tvals, BO[:, None]], axis = 1) # Array to be passed on for next iteration
            new_Tvals = inter_Tvals[:,:p] # Internal Loop computation Array
            return inter_Tvals, new_Tvals

        inter_Tvals, new_Tvals = jax.lax.cond(iter<p, case_equal, case_not_equal, (init_Tvals, BO, iter, p_k))

        inter_v = new_Tvals @ alpha 

        # Safe-Gaurding
        lhs = jnp.max(jnp.abs(inter_v - T.v(mdp, inter_v, state.gamma)))
        rhs = (state.gamma ** iter) * jnp.max(jnp.abs(state.v_init - T.v(mdp, state.v_init, state.gamma)))

        def true_fn(_):
            return inter_v
        
        def false_fn(_):
            return T.v(mdp, state.v_vals, state.gamma)
        
        new_v_vals = jax.lax.cond(lhs<rhs, true_fn, false_fn, operand = None)
        

        return state.replace(v_vals = new_v_vals, Delta = inter_Delta, Tvals = inter_Tvals)
    
class qpi(metaclass = StaticMeta):
    r"""
    Quasi Policy Iteration. The update equation is taken from the paper: "From Optimization to Control:
    Quasi Policy Iteration". 
    This makes use of safe-gaurding w.r.t VI. 
    https://arxiv.org/abs/2311.11166
    """

    @struct.dataclass
    class State:
        v_vals: VType
        v_init: VType
        gamma: jnp.ndarray
        Tvals: VType

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "qpi.State":
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        Tvals = T.v(mdp, v_vals, gamma)
        return qpi.State(v_vals = v_vals, v_init = v_vals, Tvals = Tvals, gamma=gamma)
    
    def update(state: "qpi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "qpi.State":

        # Policy Improvement
        policy = g.v(mdp, state.v_vals, state.gamma)
        transition_pi, reward = mc_pi(mdp, policy) # P_\pi_(v_k) = P_k and r_\pi_(v_k) = r_k
        reward_pi = jnp.einsum("sx, sx->s", transition_pi.T, reward) # Reward obtained from MC is (n,n), it must be (n,)

        # Policy Evaluation
        con_z = (jnp.ones((mdp.state_size)).T @ reward_pi)/mdp.state_size
        z = reward_pi - (con_z * jnp.ones((mdp.state_size)))
        gk = state.v_vals - T.v(mdp, state.v_vals, state.gamma)
        con_y = (jnp.ones((mdp.state_size)).T @ gk)/mdp.state_size
        y = gk - (con_y * jnp.ones((mdp.state_size)))
        den = (state.v_vals).T @ (y+z)

        def true_fn(den):
            den = den
            return 0.0
        
        def false_fn(den):
            den = den
            return (state.v_vals.T @ y)/den
        
        delta = jax.lax.cond(den == 0, true_fn, false_fn, den)
        p1 = state.gamma/(mdp.state_size * (1 - state.gamma))
        p2 = jnp.ones((mdp.state_size)).T
        p3 = ((delta - 1) * gk) + (delta * reward_pi)
        lambd = p1 * (p2 @ p3)
        inter_v = ((1 - delta) * T.v(mdp, state.v_vals, state.gamma)) + (delta * reward_pi) + (lambd * jnp.ones((mdp.state_size)))
        
        # Safe-Gaurding
        lhs = jnp.max(jnp.abs(inter_v - T.v(mdp, inter_v, state.gamma)))
        rhs = (state.gamma ** iter) * jnp.max(jnp.abs(state.v_init - state.Tvals))

        def true_fn(_):
            return inter_v
        
        def false_fn(_):
            return T.v(mdp, state.v_vals, state.gamma)
        
        next_vals = jax.lax.cond(lhs<rhs, true_fn, false_fn, operand = None)

        return state.replace(v_vals = next_vals)
        
class r1_vi(metaclass=StaticMeta):
    r"""
    Rank-1 VI. The update equation is taken from the paper: "RANK-ONE MODIFIED VALUE ITERATION".
    https://arxiv.org/abs/2505.01828
    """

    @struct.dataclass
    class State:
        v_vals: VType
        v_init: VType
        gamma: jnp.ndarray
        pdf: jnp.ndarray
        P: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "r1_vi.State":
        v_vals = jrd.uniform(key, (mdp.state_size), 
                             dtype = 'float', minval =0.0, maxval = 1.0)
        pdf_init = jnp.ones((mdp.state_size))/mdp.state_size
        P = jnp.zeros((mdp.state_size, mdp.state_size))
        return r1_vi.State(v_vals = v_vals, v_init = v_vals, gamma=gamma, pdf = pdf_init, P = P)
    
    def update(state: "r1_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "r1_vi.State":
        
        # Policy Extraction
        policy =g.v(mdp, state.v_vals, state.gamma) 
        Tv = T.v(mdp, state.v_vals, state.gamma)
        transition_pi, reward_pi = mc_pi(mdp, policy)
        
        # Power method
        step = 10
        def scan_body(d, x):
            f = transition_pi @ d
            d = f/jnp.linalg.norm(f, ord=1)
            return d, d
        pdf, history = jax.lax.scan(scan_body, state.pdf, xs = jnp.arange(step))

        p1 = state.gamma/(1-state.gamma)
        p2 = pdf.T @ (Tv - state.v_vals)
        next_val = Tv + p1*p2*jnp.ones((mdp.state_size))

        return state.replace(v_vals = next_val, pdf = pdf)


class pid_vi(metaclass=StaticMeta):
    r"""
    PID VI. The update equation is taken from the paper: "PID Accelerated Value Iteration Algorithm". Initial value for P, I and D requires tuning. 
    https://arxiv.org/html/2407.08803v2
    """
    @struct.dataclass
    class State:
        v_vals: VType
        prev_v: VType
        prev1_v: VType
        gamma: jnp.ndarray
        z: VType
        prev_z: VType
        alpha: jnp.ndarray
        beta: jnp.ndarray
        k_p: jnp.ndarray
        k_d: jnp.ndarray
        k_i: jnp.ndarray

    # INITIALIZATION
    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "pid_vi.State":
        
        v_vals = jnp.zeros((mdp.state_size))
        z = jnp.zeros((mdp.state_size))
        alpha = 0.05
        beta = 0.95
        k_p = 1.0
        k_i = 0.75
        k_d = 0.4
        return pid_vi.State(v_vals = v_vals, prev_v = v_vals, prev1_v = v_vals, gamma=gamma, 
                             z =z, prev_z = z, alpha =alpha, beta = beta,
                               k_p = k_p, k_i = k_i, k_d = k_d)
    
    # UPDATE 
    def update(state: "pid_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "pid_vi.State":
        
        policy = g.v(mdp, state.v_vals, state.gamma) # Greedy policy 
        BR = T.v(mdp, state.v_vals, state.gamma) - state.v_vals
        transition_pi, reward = mc_pi(mdp, policy)
        p1 = jnp.eye((mdp.state_size)) - state.gamma*transition_pi
        
        def true_fn(p1, BR, eta, eps, prev_v, z, prev1_v, prev_z):
            alpha = state.alpha
            beta = state.beta
            k_p = state.k_p
            k_d = state.k_d
            k_i = state.k_i
            return alpha, beta, k_p, k_i, k_d

        def false_fn(p1, BR, eta, eps, prev_v, z, prev1_v, prev_z):

            # k_p calculation
            grad_k_p = -p1 @ (T.v(mdp, prev_v, state.gamma) - prev_v)
            contr_states_k_p = jnp.einsum("s,s->", BR, grad_k_p)
            k_p = state.k_p - (eta * contr_states_k_p)/(jnp.linalg.norm(T.v(mdp, prev_v, state.gamma) - prev_v)**2 + eps)

            # k_d calculation
            grad_k_d = -p1 @ (prev_v - prev1_v)
            contr_states_k_d = jnp.einsum("s,s->", BR, grad_k_d)
            k_d = state.k_d - (eta * contr_states_k_d)/(jnp.linalg.norm(T.v(mdp, prev_v, state.gamma) - prev_v)**2 + eps)

            # k_i calculation
            grad_k_i = -p1 @ z
            contr_states_k_i = jnp.einsum("s,s->", BR, grad_k_i)
            k_i = state.k_i - (eta * contr_states_k_i)/(jnp.linalg.norm(T.v(mdp, prev_v, state.gamma) - prev_v)**2 + eps)

            # alpha calculation (Paper suggested that the following two values are to be kept constant)
            grad_alpha = -k_i*p1 @ (T.v(mdp, prev_v, state.gamma) - prev_v)
            contr_states_alpha = jnp.einsum("s,s->", BR, grad_alpha)
            alpha = state.alpha - (eta * contr_states_alpha)/(jnp.linalg.norm(T.v(mdp, prev_v, state.gamma) - prev_v)**2 + eps)

            # beta calculation
            grad_beta = -k_i*p1 @ prev_z
            contr_states_beta = jnp.einsum("s,s->", BR, grad_beta)
            beta = state.beta - (eta * contr_states_beta)/(jnp.linalg.norm(T.v(mdp, prev_v, state.gamma) - prev_v)**2 + eps)

            return alpha, beta, k_p, k_i, k_d

        eta = 0.001
        eps = 1e-20
        operands = (p1, BR, eta, eps, state.prev_v, state.z, state.prev1_v, state.prev_z)
        alpha, beta, k_p, k_i, k_d = jax.lax.cond(iter<3, true_fn, false_fn, *operands)

        z_vals = beta*state.z + alpha*BR
        next_v = (1-k_p)*state.v_vals + k_p*T.v(mdp, state.v_vals, state.gamma) + k_i*z_vals + k_d*(state.v_vals - state.prev_v)
        
        return state.replace(v_vals = next_v, prev_v = state.v_vals,
                              prev1_v = state.prev_v, 
                              z =z_vals, alpha =alpha, beta = beta,
                               k_p = k_p, k_i = k_i, k_d = k_d)
    
class gso_vi(metaclass=StaticMeta):
    r"""
    Generalized Second Order VI. The update equation is taken from the paper: "Generalized Second Order Value Iteration in
    Markov Decision Processes".  
    https://arxiv.org/abs/1905.03927
    """

    @struct.dataclass
    class State:
        q_vals: QType
        gamma: jnp.ndarray

    def init(mdp: MDP, key: jrd.PRNGKey, gamma: jnp.ndarray) -> "gso_vi.State":
        q_vals = jrd.uniform(key, (mdp.action_size, mdp.state_size), dtype = 'float',
                             minval = 0.0, maxval = 1.0)
        return gso_vi.State(q_vals=q_vals, gamma=gamma)

    def update(state: "gso_vi.State", mdp: MDP, iter: int, theta: jnp.ndarray) -> "gso_vi.State":
        w = 1.1
        N = 50
        q_vals = state.q_vals # q_k R^{mxn}
        q_vals_flat = q_vals.reshape(-1) # R^{mn}

        # Uq(k) calculation
        exp_rewards = jnp.einsum("asx,axs->as", mdp.reward, mdp.transition)
        max_approx = jax.nn.logsumexp(N * q_vals, axis=0) / N
        target_value = jnp.einsum("axs,x->as", mdp.transition, max_approx)
        u_q_vals = w*(exp_rewards + state.gamma*target_value) + (1-w)*max_approx 
        u_q_vals_flat = u_q_vals.reshape(-1) # uq_k is a vector(nm,)

        # Matrix J calculation
        pol = jax.nn.softmax(N * q_vals, axis=0)
        prob = mdp.transition.transpose(0, 2, 1) # P has the size (A, S^+, S), needs transpose
        term1 = w*state.gamma*jnp.einsum("aik,ck->aick", prob, pol)
        id_tensor = jnp.eye(mdp.action_size*mdp.state_size).reshape(mdp.action_size, mdp.state_size, mdp.action_size, mdp.state_size)
        term2 = (1-w)*jnp.einsum("ck,aick->aick", pol, id_tensor)
        J_tensor = term1 + term2
        J = J_tensor.reshape(mdp.action_size*mdp.state_size,mdp.action_size*mdp.state_size)
        
        # Update 
        diff = q_vals_flat - u_q_vals_flat
        y = jnp.eye(mdp.action_size*mdp.state_size) - J
        inv_J_diff = jnp.linalg.solve(y, diff)
        next_q_flat = q_vals_flat - inv_J_diff
        next_q = next_q_flat.reshape(mdp.action_size, mdp.state_size)
        return state.replace(q_vals=next_q)
