from typing import Callable, Dict, Union, List, Tuple, NamedTuple, Type, Any
import argparse
from dataclasses import dataclass
from jaxdp.typehints import F, PiType, StaticMeta

import click
import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct

from Tools import Bellman_Optimality_Operator as T
from Tools import policy_evaluation as PE
from Tools import greedy_policy as gp
from Tools import sync_sample as sync
from Tools import exp_decay as exp_decay
from Tools import opt_val

from jaxdp.mdp import MDP
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.forest_mdp import forest_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp.healthcare_mdp import healthcare_mdp

from Algorithms import Transition, SyncSample, q_learning_sync, sql_sync, momentumq_sync, zap_ql_sync, r1_ql_sync
from jaxdp.typehints import F, QType, VType, PiType, StaticMeta
from utils import log_results_sync, plot_results, benchmark_log_results_sync, benchmark_plot_results, benchmark_plot_val_err_results

jax.config.update("jax_enable_x64", True) # Enables 64-bit floating-point precision (double precision)

class metrics(metaclass=StaticMeta):
    """
    Computes and store Training metrics.
    """
    @struct.dataclass
    class State:
        """Metrics collected during training"""
       
        l1: jnp.ndarray
        l2: jnp.ndarray
        linf: jnp.ndarray
        bellman_res: jnp.ndarray
        expected_policy_eval: jnp.ndarray
        max_value_diff: jnp.ndarray
        expected_value: jnp.ndarray
        iteration: jnp.ndarray
        val_err: jnp.ndarray
 
    def compute(prev: "loop.State", new : "loop.State", args: "loop.Args",
            step: int, eval_results: "loop.EvalResult", q_star: QType) -> "metrics.State":
        """
        Compute metrics for the current iteration
       
        Args:
            prev: Previous loop state
            new: new loop state
            args: Loop arguments
            step: Current step number
            eval_results: Evaluation result (loop.EvalResult dataclass)
 
            Returns:
                Metrics state values or NaN where no information is available.      
       
        """
       
        prev_alg = prev.alg_state
        new_alg = new.alg_state
        policy = gp.q(prev_alg.q_vals)

        diff = new_alg.q_vals - prev_alg.q_vals
        l1 = jnp.linalg.norm(diff, 1)
        l2 = jnp.linalg.norm(diff)
        linf = jnp.max(jnp.abs(diff))
 
        bellman_target = T.q(args.mdp, new_alg.q_vals, new_alg.gamma)
        bellman_res = jnp.max(jnp.abs(new_alg.q_vals - bellman_target))
        expected_value = jnp.einsum("as,as,s->", policy, new_alg.q_vals, args.mdp.initial) # Total Expected Value of a policy at the start of an episode
        expected_policy_eval = eval_results.policy_value
        value_error = jnp.max(jnp.abs(new_alg.q_vals - q_star))/jnp.max(jnp.abs(q_star)) # Computation doubt
       
        return metrics.State(l1=l1, l2=l2, linf=linf,
                             bellman_res=bellman_res, iteration=step, expected_policy_eval=expected_policy_eval,
                             expected_value=expected_value, max_value_diff=linf, val_err=value_error
                             )


class loop(metaclass=StaticMeta):
    """
    ◈─────────────────────────────────────────────────────────────────────────◈
    Training Loop for Value-based RL Algorithms

    Provides init(), train(), and evaluate() functions for managing the
    training loop state and running value-based RL algorithms with various
    exploration policies.
    ◈─────────────────────────────────────────────────────────────────────────◈
    """
    
    @struct.dataclass
    class State:
        """Training loop state - manages MDP interaction and episode tracking"""

        alg_state: Any  # q_learning.State

    @struct.dataclass
    class Args:
        """Training loop arguments - static configuration"""

        value_fn: Any  # Algorithm namespace (e.g., q_learning)
        mdp: MDP
        seed: int
        n_steps: int
        eval_period: int  # Evaluate every N steps (0 = no evaluation)
        n_seed: int   
        gamma: jnp.ndarray

    @struct.dataclass
    class EvalResult:
        """Evaluation results"""

        policy_value: QType # Passed to compute metrics function
        

    def init(alg_state: Any, args: "loop.Args") -> "loop.State":
        """Initialize loop state with algorithm and policy states

        Args:
            alg_state: Initialized algorithm state (e.g., q_learning.State)

        Returns:
            Initialized loop state with environment states and episode tracking
        """


        return loop.State(
            alg_state=alg_state
        )
        
    def train(state: "loop.State", args: "loop.Args", q_star: QType) -> tuple["loop.State", Any]:
        """Run training loop for n_steps with specified learning method

        Args:
            state: Initial loop state
            args: Loop arguments (includes value_fn, mdp)

        Returns:
            Final loop state and all metrics collected during training
        """

        def step_fn(state: loop.State, step_and_keys):
            step, keys = step_and_keys
            prev = state
            reward, next_state, terminal = sync(args.mdp, keys)
            sample = SyncSample(reward, next_state, terminal)
            alpha = exp_decay(step)
            new_alg_state = args.value_fn.update(state.alg_state, sample, alpha, step)
            new_state = state.replace(alg_state=new_alg_state)

            eval_condn = (args.eval_period > 0) & ((step + 1) % args.eval_period == 0)
            eval_results = jax.lax.cond(eval_condn, lambda s: loop.evaluate(s, args), 
                                        lambda s: loop.EvalResult(policy_value=jnp.nan),
                                          new_state)
            
            metrics_state = metrics.compute(prev, new_state, args, step, eval_results, q_star)
            return new_state, metrics_state
        
        keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps)
        keys = keys.reshape(args.n_steps, -1)
        steps_and_keys = (jnp.arange(args.n_steps), keys)
        final_state, all_metrics = jax.lax.scan(step_fn, state, steps_and_keys)
        
        return final_state, all_metrics   
    
    def train_vmap(state: "loop.State", args: "loop.Args", q_star: QType) -> tuple["loop.State", Any]:
        """Run training loop for n_steps with specified learning method

        Args:
            state: Initial loop state
            args: Loop arguments (includes value_fn, mdp)

        Returns:
            Final loop state and all metrics collected during training
        """

        def step_fn(state: loop.State, step_and_keys):
            step, keys = step_and_keys
            prev = state
            reward, next_state, terminal = sync(args.mdp, keys)
            sample = SyncSample(reward, next_state, terminal)
            alpha = exp_decay(step)

            vmap_upd = jax.vmap(args.value_fn.update, in_axes = (0, None, None, None))
            new_alg_state = vmap_upd(state.alg_state, sample, alpha, step)
            new_state = state.replace(alg_state=new_alg_state)

            vmap_eval = jax.vmap(loop.evaluate, in_axes = (0, None))
            eval_condn = (args.eval_period > 0) & ((step + 1) % args.eval_period == 0)
            eval_results = jax.lax.cond(eval_condn, lambda s: vmap_eval(s, args), 
                                        lambda s: loop.EvalResult(policy_value=jnp.full((args.n_seed,), jnp.nan)),
                                          new_state)
            
            vmap_metrics = jax.vmap(metrics.compute, in_axes = (0, 0, None, None, None, 0))
            metrics_state = vmap_metrics(prev, new_state, args, step, eval_results, q_star)
            return new_state, metrics_state
        
        keys = jrd.split(jrd.PRNGKey(args.seed), args.n_steps)
        keys = keys.reshape(args.n_steps, -1)
        steps_and_keys = (jnp.arange(args.n_steps), keys)
        final_state, all_metrics = jax.lax.scan(step_fn, state, steps_and_keys)
        
        return final_state, all_metrics  

    def evaluate(state: "loop.State", args: "loop.Args") -> "loop.EvalResult":
        """Evaluate the learned policy (greedy, no exploration)

        Args:
            state: Current loop state with learned Q-values
            args: Loop arguments (includes mdp, max_ep_len, n_eval_episodes, eval_seed)

        Returns:
            loop.EvalResult dataclass with evaluation statistics
        """

        policy = gp.q(state.alg_state.q_vals)
        policy_value = (PE.q(args.mdp, policy, state.alg_state.gamma) * args.mdp.initial).sum()
        
        return loop.EvalResult(policy_value=policy_value)


########################################## MDP ###########################################

def grid_mdp_create() -> MDP:

    """" Grid MDP is created """
    
    board = [
    "########",
    "#  @#  #",
    "#  ## +#",
    "#      #",
    "#X     #",
    "#+   P #",
    "#     X#",
    "########"
    ]
    
    p_slip = 0.25

    return grid_world(board, p_slip)


def garnet_MDP_create(state_size: int, action_size: int,
                       branch_size: int, key: jrd.PRNGKey)-> MDP:
    """ Garnet MDP is created"""
    return garnet_mdp(state_size=state_size, action_size=action_size,
                      branch_size=branch_size, key=key)


def forest_MDP_create(rotation) -> MDP:
    return forest_mdp(rotation)

def graph_MDP_create() -> MDP:
    return graph_mdp()

def healthcare_MDP_create() -> MDP:
    return healthcare_mdp()

######################################################################################

####################################### LOOP INPUTS ##################################

hyperparams = {
    "seed": 42,
    "n_steps": 5000, 
    "eval_period": 1, 
    "n_seed": 5, 
    "gamma": 0.9
}
######################################################################################

################################ ALGORITHM IMPLEMENTATION ############################

def alg_implementation(alg_display_name, mdp_name, state, loop_args):
    q_star = opt_val.q(mdp, hyperparams["gamma"])
    final_state, metrics = loop.train(state, loop_args, q_star)
    
    # Store and visualize results
    results = {f"{mdp_name}": (metrics, final_state.alg_state.q_vals)}
    log_results_sync(results, alg_display_name)
    plot_results(results, alg_display_name)
    return final_state.alg_state.q_vals

######################################################################################

################################ BENCHMARK IMPLEMENTATION ############################

def benchmark_alg_implementation(mdp_name, mdp, hyperparams): # only pass mdp_name and mdp
    
    q_star = opt_val.q(mdp, hyperparams["gamma"])
    q_star = jnp.broadcast_to(q_star, (hyperparams["n_seed"], mdp.action_size, mdp.state_size))
    
    alg_map = {
        q_learning_sync: "Q-Learning sync",
        sql_sync: "Speedy Q-Learning sync",
        momentumq_sync: "Momentum Q sync",
        zap_ql_sync: "Zap Q-Learning sync",
        r1_ql_sync: "Rank 1 Q-Learning sync",
    }
    
    avg_results = {}
    all_results = {}

    for alg_module, alg_display_name in alg_map.items():

        loop_args = loop.Args(value_fn = alg_module, mdp = mdp, **hyperparams)
        keys = jrd.split(jrd.PRNGKey(loop_args.seed), loop_args.n_seed)

        vmap_init = jax.vmap(loop_args.value_fn.init, in_axes = (None, 0, None))
        init_state = vmap_init(mdp, keys, loop_args.gamma)
        state = loop.init(init_state, loop_args)        
        final_state, metrics = loop.train_vmap(state, loop_args, q_star)

        avg_metrics = jax.tree.map(
            lambda x: jnp.mean(x, axis=1) if x.ndim > 1 else x,
            metrics
        )
        avg_results[alg_display_name] = avg_metrics
        print(f"Completed for {alg_display_name}")
        all_results[alg_display_name] = metrics
 
    settings = {"name":mdp_name,"gamma": loop_args.gamma}
    benchmark_log_results_sync(avg_results, settings)
    benchmark_plot_results(all_results, settings)
    benchmark_plot_val_err_results(all_results, settings)
    return avg_results

######################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX RL Benchmarks")
    parser.add_argument("benchmark_type", 
                        choices=[
                "q_learning_grid", "q_learning_garnet", "q_learning_forest", "q_learning_graph", "q_learning_healthcare",
                "sql_sync_grid", "sql_sync_garnet", "sql_sync_forest", "sql_sync_graph", "sql_sync_healthcare",
                "momentumq_sync_grid", "momentumq_sync_garnet", "momentumq_sync_forest", "momentumq_sync_graph", "momentumq_sync_healthcare",
                "zap_ql_sync_grid", "zap_ql_sync_garnet", "zap_ql_sync_forest", "zap_ql_sync_graph", "zap_ql_sync_healthcare",
                "r1_ql_sync_grid", "r1_ql_sync_garnet", "r1_ql_sync_forest", "r1_ql_sync_graph", "r1_ql_sync_healthcare",
                "benchmarkalg_grid", "benchmarkalg_garnet", "benchmark_alg_grid", "benchmark_alg_garnet"],
        help="Type of benchmark to run")
    args = parser.parse_args()

    mdp_map = {
        "grid": (grid_mdp_create, {}),
        "garnet": (garnet_MDP_create, {"state_size": 50, "action_size": 5, "branch_size":10, "key": jrd.PRNGKey(42)}),
        "forest": (forest_MDP_create, {"rotation": 25}),
        "graph": (graph_MDP_create, {}),
        "healthcare": (healthcare_MDP_create, {})
    }

    alg_map = {
        "q_learning": (q_learning_sync, "Q-Learning sync"),
        "sql_sync": (sql_sync, "Speedy Q-Learning sync"),
        "momentumq_sync": (momentumq_sync, "Momentum Q sync"),
        "zap_ql_sync": (zap_ql_sync, "Zap Q-Learning sync"),
        "r1_ql_sync": (r1_ql_sync, "Rank 1 Q-Learning sync"),
    }

    selected_alg_module = None
    selected_mdp_type = None

    if args.benchmark_type.startswith("benchmarkalg"):

        mdp_type = args.benchmark_type[len("benchmarkalg")+1:]
        mdp_creator, mdp_params = mdp_map[mdp_type]
        mdp = mdp_creator(**mdp_params)
        mdp_name = mdp_type.capitalize() 
        benchmark_alg_implementation(mdp_name, mdp, hyperparams)
        
    else:
        
        for prefix in alg_map:
            if args.benchmark_type.startswith(prefix):
                alg_module, alg_display_name = alg_map[prefix]
                # Extract the environment name by stripping the algorithm prefix and the connecting underscore
                mdp_type = args.benchmark_type[len(prefix)+1:]
                break
        else:
            raise ValueError(f"Unknown Benchmark type: {args.benchmark_type}")

        mdp_creator, mdp_params = mdp_map[mdp_type]
        mdp = mdp_creator(**mdp_params)
        mdp_name = mdp_type.capitalize()
        loop_args = loop.Args(value_fn = alg_module, mdp = mdp, **hyperparams)

        init_state = loop_args.value_fn.init(mdp, jrd.PRNGKey(loop_args.seed), loop_args.gamma)
        state = loop.init(init_state, loop_args)

        alg_implementation(alg_display_name, mdp_name, state, loop_args) 

