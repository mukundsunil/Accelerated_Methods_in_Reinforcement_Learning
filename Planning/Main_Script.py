import argparse
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrd
from Algorithms import vi, gs_vi, pi, r_vi, a_vi, m_vi, sa_vi, anc_vi, and_vi, pi_q, qpi, r1_vi, pid_vi, gso_vi
from flax import struct
from utils import log_results, plot_results, log_comprehensive_benchmark_alg, plot_results_alg, plot_results_val_err_alg, plot_results_mulseed_garnet_alg, plot_results_mulseed_garnet_val_err_alg
from Tools import Bellman_Optimality_Operator as T
from Tools import Bellman_Policy_Operator as Tp

from jaxdp.mdp import MDP
from jaxdp.mdp.grid_world import grid_world
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.forest_mdp import forest_mdp
from jaxdp.mdp.simple_graph import graph_mdp
from jaxdp.mdp.healthcare_mdp import healthcare_mdp

from jaxdp.typehints import F, VType, QType, PiType, StaticMeta


jax.config.update("jax_enable_x64", True)

@struct.dataclass
class Metrics:
    l1: jnp.ndarray
    l2: jnp.ndarray
    linf: jnp.ndarray
    bellman_res: jnp.ndarray
    log_bellman_res: jnp.ndarray
    iteration: jnp.ndarray
    value_error: jnp.ndarray

def compute_metrics(prev_state, new_state, mdp, step):
    """ Compute metrics for the current iteration """
    new_v = new_state.v_vals
    prev_v = prev_state.v_vals
    gamma = prev_state.gamma

    diff = new_v - prev_v
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = T.v(mdp, prev_v, gamma)
    bellman_res = jnp.max(jnp.abs(prev_v - bellman_target))
    log_bellman_res = jnp.log(jnp.max(jnp.abs(prev_v - bellman_target)))
    value_error = 0

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_res=bellman_res,
        log_bellman_res=log_bellman_res,
        iteration=step, value_error=value_error
    )

def compute_metrics_benchmark(prev_state, new_state, v_star, mdp, step):
    """ Compute metrics for the current iteration """
    new_v = new_state.v_vals
    prev_v = prev_state.v_vals
    gamma = prev_state.gamma

    diff = new_v - prev_v
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = T.v(mdp, prev_v, gamma)
    bellman_res = jnp.max(jnp.abs(prev_v - bellman_target))
    log_bellman_res = jnp.log(jnp.max(jnp.abs(prev_v - bellman_target)))

    value_error = jnp.max(jnp.abs(prev_v - v_star))

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_res=bellman_res,
        log_bellman_res=log_bellman_res,
        iteration=step, value_error=value_error
    )

def compute_metrics_q(prev_state, new_state, q_star, mdp, step):
    """ Compute metrics for the current iteration """
    new_q = new_state.q_vals
    prev_q = prev_state.q_vals
    gamma = prev_state.gamma

    diff = new_q - prev_q
    l1 = jnp.sum(jnp.abs(diff))
    l2 = jnp.sqrt(jnp.sum(diff**2))
    linf = jnp.max(jnp.abs(diff))

    bellman_target = T.q(mdp, prev_q, gamma)
    bellman_res = jnp.max(jnp.abs(prev_q - bellman_target))
    log_bellman_res = jnp.log(jnp.max(jnp.abs(prev_q - bellman_target)))
    value_error = jnp.max(jnp.abs(prev_q - q_star))

    return Metrics(
        l1=l1,
        l2=l2,
        linf=linf,
        bellman_res=bellman_res,
        log_bellman_res=log_bellman_res,
        iteration=step, value_error=value_error 
    )

@dataclass(frozen=True)
class LoopArgs:
    seed: int
    n_iters: int
    theta: jnp.ndarray
    gamma: jnp.ndarray
    n_seed: int = 1


def loop(mdp: MDP,
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, int, jnp.ndarray, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, VType, MDP, jnp.ndarray], Any],
         callback: Callable[[int, Any], None] | None = None,
         ) -> tuple[Any, Any]:
    """
    Run the loop for a fixed number of iterations, updating the algorithm state and computing metrics.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        callback (Callable[[int, Any], None] | None): Optional callback function to be called at each iteration

    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    theta = args.theta

    def scan_body(state: Any, iter_idx: int) -> tuple[Any, Any]:
        prev_state = state
        #new_state = update_fn(state, mdp, iter_idx)
        new_state = update_fn(state, mdp, iter_idx, theta)


        metrics = metrics_fn(prev_state, new_state, mdp, iter_idx)

        if callback is not None:
            jax.debug.callback(callback, iter_idx, metrics)

        return new_state, metrics

    final_state, all_metrics = jax.lax.scan(
        scan_body,
        alg_state,
        jnp.arange(args.n_iters),
        )

    return final_state, all_metrics

def loop_stop(mdp:MDP, 
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, MDP, jnp.ndarray], Any],
         ) -> tuple[Any, Any]:
    """"
    Loop is run for fixed number of iterations, the algorithm state is updates and metrics are computed.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        
    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    
    step = args.n_iters
    theta = args.theta

   
    def scan_body(state: Any, i) -> tuple[Any, Any]:

        prev_state = state
        upd_state = update_fn(prev_state, mdp, i+1, theta)

        metrics = metrics_fn(prev_state, upd_state, mdp, i+1)

        upd_state = jax.lax.cond(metrics.bellman_res < theta,
                                 lambda _: prev_state,
                                 lambda _:upd_state,
                                 None) # State frozen after convergence
        
        
        return upd_state, metrics
        
    final_state, all_metrics = jax.lax.scan(scan_body, alg_state,
                                            xs = jnp.arange(step))
         
    return final_state, all_metrics 

def loop_stop_vmap(mdp:MDP, 
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, VType, MDP, jnp.ndarray], Any],
         v_star
         ) -> tuple[Any, Any]:
    """"
    Loop is run for fixed number of iterations, the algorithm state is updates and metrics are computed.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        
    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    
    step = args.n_iters
    theta = args.theta
   
    def scan_body(state: Any, i) -> tuple[Any, Any, Any]:

        prev_state = state
        upd_state = update_fn(prev_state, mdp, i+1, theta)

        metrics = metrics_fn(prev_state, upd_state, v_star, mdp, i+1)

        is_converged = metrics.bellman_res < theta

        upd_state = jax.tree_util.tree_map(
                                lambda next_s, prev_s: jnp.where(is_converged, prev_s, next_s),
                                upd_state,
                                prev_state
                                ) # State frozen after convergence
        
        return upd_state, metrics 
        
    final_state, all_metrics = jax.lax.scan(scan_body, alg_state,
                                            xs = jnp.arange(step))
         
    return final_state, all_metrics  

def loop_mdp_vmap(mdp:MDP, 
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, MDP, jnp.ndarray], Any],
         ) -> tuple[Any, Any]: # loop_stop_vmap without v_star and stop
    """"
    Loop is run for fixed number of iterations, the algorithm state is updates and metrics are computed.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        
    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    
    step = 50
    theta = args.theta
   
    def scan_body(state: Any, i) -> tuple[Any, Any, Any]:

        prev_state = state
        upd_state = update_fn(prev_state, mdp, i+1, theta)

        metrics = metrics_fn(prev_state, upd_state, mdp, i+1)
        
        return upd_state, metrics 
        
    final_state, all_metrics = jax.lax.scan(scan_body, alg_state,
                                            xs = jnp.arange(step))
         
    return final_state, all_metrics 

def loop_stop_mdp_vmap(mdp:MDP, 
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, VType, MDP, jnp.ndarray], Any],
         v_star
         ) -> tuple[Any, Any]:
    """"
    Loop is run for fixed number of iterations, the algorithm state is updates and metrics are computed.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        
    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    
    step = args.n_iters
    theta = args.theta
   
    def scan_body(state: Any, i) -> tuple[Any, Any, Any]:

        prev_state = state
        upd_state = update_fn(prev_state, mdp, i+1, theta)

        metrics = metrics_fn(prev_state, upd_state, v_star, mdp, i+1)

        is_converged = metrics.bellman_res < theta

        upd_state = jax.tree_util.tree_map(
                                lambda next_s, prev_s: jnp.where(
                                is_converged.reshape(is_converged.shape + (1,) * (next_s.ndim - 1)), 
                                prev_s, 
                                next_s
                                ),
                                    upd_state,
                                    prev_state) # State frozen after convergence
        
        return upd_state, metrics 
        
    final_state, all_metrics = jax.lax.scan(scan_body, alg_state,
                                            xs = jnp.arange(step))
         
    return final_state, all_metrics 


def loop_stop_mulseed_vmap(mdp:MDP, 
         alg_state: Any,
         args: LoopArgs,
         update_fn: Callable[[Any, MDP, jnp.ndarray], Any],
         metrics_fn: Callable[[Any, Any, MDP, jnp.ndarray], Any],
         ) -> tuple[Any, Any]:
    """"
    Loop is run for fixed number of iterations, the algorithm state is updates and metrics are computed.

    Args:
        mdp (MDP): Markov Decision Process
        alg_state (Any): Initial state of the algorithm
        args (LoopArgs): Loop arguments
        update_fn (Callable[[Any, MDP, jnp.ndarray], Any]): Function to update the algorithm state
        metrics_fn (Callable[[Any, Any, MDP, jnp.ndarray], Any]): Function to compute metrics
        
    Returns:
        tuple[Any, Any]: Final algorithm state and all metrics collected during the loop
    """
    
    step = args.n_iters
    theta = args.theta

   
    def scan_body(state: Any, i) -> tuple[Any, Any]:

        prev_state = state
        upd_state = update_fn(prev_state, mdp, i+1, theta)

        metrics = metrics_fn(prev_state, upd_state, mdp, i+1)

        is_converged = metrics.bellman_res < theta

        mask = is_converged[:, jnp.newaxis]

        frozen_v_vals = jnp.where(mask, prev_state.v_vals, upd_state.v_vals)
        
        upd_state = upd_state.replace(v_vals=frozen_v_vals)
        
        
        return upd_state, metrics
        
    final_state, all_metrics = jax.lax.scan(scan_body, alg_state,
                                            xs = jnp.arange(step))
         
    return final_state, all_metrics

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

##################################### LOOP INPUTS ####################################

loop_args = LoopArgs(seed=42, n_iters=10000, theta=1e-6, gamma = 0.999, n_seed=10) # n_seed must be changed when implementing multi-seed garnet else 1

######################################################################################

################################ ALGORITHM IMPLEMENTATION ############################

def alg_implementation(mdp, alg_name, mdp_name, init_state, update_fn): # Change when v and q functions are used

    final_state, metrics = loop_stop(mdp, 
                                alg_state = init_state,
                                args = loop_args,
                                update_fn = update_fn,
                                metrics_fn = compute_metrics
                                )
    
    result = {f"{mdp_name}":(metrics, final_state.v_vals)}
    log_results(result, alg_name)
    plot_results(result, alg_name)
    return final_state.v_vals

######################################################################################

################################ BENCHMARK IMPLEMENTATION ############################

def benchmark_vi(mdp,mdp_name) -> dict:

    # Applicable only for Grid, Graph, Forest and Healthcare MDP
    init_state = pi.init(mdp, jrd.PRNGKey(loop_args.seed), loop_args.gamma)
    update_fn = pi.update
    final_state, metrics = loop_stop(mdp, 
                                alg_state = init_state,
                                args = loop_args,
                                update_fn = update_fn,
                                metrics_fn = compute_metrics
                                )
    
    v_star = final_state.v_vals # PI values
    algs = {
            vi: "Value Iteration",
            gs_vi: "Gauss Seidel Value Iteration",
            pi: "Policy Iteration",
            r_vi: "Relaxed Value Iteration",
            sa_vi: "Safe Accelerated Value Iteration",
            anc_vi: "Anchor value Iteration",
            and_vi: "Anderson Value Iteration",
            qpi: "Quasi PI",
            r1_vi: "Rank 1",
            pid_vi: "PID Value Iteration"            
            }
    
    all_results ={}
    
    for alg_module, alg_name in algs.items():

        keys = jrd.split(jrd.PRNGKey(loop_args.seed), loop_args.n_seed)

        vmap_init = jax.vmap(alg_module.init, in_axes=(None, 0, None))
        init_states = vmap_init(mdp, keys, loop_args.gamma)

        update_fn = alg_module.update
        vmap_update = jax.vmap(update_fn, in_axes = (0, None, None, None))

        vmap_metrics = jax.vmap(compute_metrics_benchmark, in_axes = (0, 0, None, None, None))

        final_state, metrics = loop_stop_vmap(mdp, 
                                alg_state = init_states,
                                args = loop_args,
                                update_fn = vmap_update,
                                metrics_fn = vmap_metrics,
                                v_star=v_star
                                ) 
        avg_metrics = jax.tree.map(
            lambda x: jnp.mean(x, axis=1) if x.ndim > 1 else x,
            metrics
        )

        all_results[alg_name] = avg_metrics
    
    settings = {"name":mdp_name,"gamma": loop_args.gamma}
    log_comprehensive_benchmark_alg(all_results, settings)
    plot_results_alg(all_results, settings)
    plot_results_val_err_alg(all_results, settings)
    return all_results   

######################################################################################

################################ MULTISEED IMPLEMENTATION ############################

def benchmark_garnet_comparison(state_size, action_size, branch_size, loop_args):

    # vampping MDP
    keys_mdp = jrd.split(jrd.PRNGKey(loop_args.seed), loop_args.n_seed)
    vmap_mdp = jax.vmap(garnet_MDP_create, in_axes=(None, None, None, 0))
    mdp = vmap_mdp(state_size, action_size, branch_size, keys_mdp)
    mdpname = "garnet"
    mdp_name = mdpname.capitalize()

    # v_star computation
    vmap_init = jax.vmap(pi.init, in_axes=(0, None, None))
    init_states = vmap_init(mdp, jrd.PRNGKey(loop_args.seed), loop_args.gamma)

    update_fn = pi.update
    vmap_update = jax.vmap(update_fn, in_axes = (0, 0, None, None))

    vmap_metrics = jax.vmap(compute_metrics, in_axes = (0, 0, 0, None))

    final_state, metrics = loop_mdp_vmap(mdp, 
                                alg_state = init_states,
                                args = loop_args,
                                update_fn = vmap_update,
                                metrics_fn = vmap_metrics
                                )
    v_star = final_state.v_vals
    # print(v_star)
   
    algs = {
            vi: "Value Iteration",
            gs_vi: "Gauss Seidel Value Iteration",
            pi: "Policy Iteration",
            r_vi: "Relaxed Value Iteration",
            sa_vi: "Safe Accelerated Value Iteration",
            anc_vi: "Anchor value Iteration",
            and_vi: "Anderson Value Iteration",
            qpi: "Quasi PI",
            r1_vi: "Rank 1",
            pid_vi: "PID Value Iteration"            
            }
    
    avg_all_results ={}
    all_results = {}
    for alg_module, alg_name in algs.items():

        key = jrd.PRNGKey(loop_args.seed)

        vmap_init = jax.vmap(alg_module.init, in_axes=(0, None, None))
        init_states = vmap_init(mdp, key, loop_args.gamma)

        update_fn = alg_module.update
        vmap_update = jax.vmap(update_fn, in_axes = (0, 0, None, None))

        vmap_metrics = jax.vmap(compute_metrics_benchmark, in_axes = (0, 0, 0, 0, None))

        final_state, metrics = loop_stop_mdp_vmap(mdp, 
                                alg_state = init_states,
                                args = loop_args,
                                update_fn = vmap_update,
                                metrics_fn = vmap_metrics,
                                v_star=v_star
                                ) 
        avg_metrics = jax.tree.map(
            lambda x: jnp.mean(x, axis=1) if x.ndim > 1 else x,
            metrics
        )

        avg_all_results[alg_name] = avg_metrics
        all_results[alg_name] = metrics
    
    settings = {"name":mdp_name,"gamma": loop_args.gamma, "S": state_size, "A": action_size, "b": branch_size}
    log_comprehensive_benchmark_alg(avg_all_results, settings)
    plot_results_mulseed_garnet_alg(all_results, settings)
    plot_results_mulseed_garnet_val_err_alg(all_results, settings)
    return avg_all_results   


######################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX DP benchmarks")
    parser.add_argument(
        "benchmark_type",
        choices=[
                "vi_grid", "vi_garnet", "vi_forest", "vi_graph", "vi_healthcare",
                "gs_vi_grid", "gs_vi_garnet", "gs_vi_forest", "gs_vi_graph", "gs_vi_healthcare",
                "pi_grid", "pi_garnet", "pi_forest", "pi_graph", "pi_healthcare",
                "r_vi_grid", "r_vi_garnet", "r_vi_forest", "r_vi_graph", "r_vi_healthcare",
                "a_vi_grid", "a_vi_garnet", "a_vi_forest", "a_vi_graph", "a_vi_healthcare",
                "m_vi_grid", "m_vi_garnet", "m_vi_forest", "m_vi_graph", "m_vi_healthcare",
                "sa_vi_grid", "sa_vi_garnet", "sa_vi_forest", "sa_vi_graph", "sa_vi_healthcare",
                "anc_vi_grid", "anc_vi_garnet", "anc_vi_forest", "anc_vi_graph", "anc_vi_healthcare",
                "and_vi_grid", "and_vi_garnet", "and_vi_forest", "and_vi_graph", "and_vi_healthcare",
                "q_pi_grid", "q_pi_garnet", "q_pi_forest", "q_pi_graph", "q_pi_healthcare",
                "qpi_grid", "qpi_garnet", "qpi_forest", "qpi_graph", "qpi_healthcare",
                "r1_vi_grid", "r1_vi_garnet", "r1_vi_forest", "r1_vi_graph", "r1_vi_healthcare",
                "ct_vi_grid", "ct_vi_garnet", "ct_vi_forest", "ct_vi_graph", "ct_vi_healthcare",
                "gso_vi_grid", "gso_vi_garnet", "gso_vi_forest", "gso_vi_graph", "gso_vi_healthcare",
                "benchmark_alg_grid", "benchmark_alg_garnet", "benchmark_alg_forest", "benchmark_alg_graph", "benchmark_alg_healthcare",
                "mulseed_vi_grid", "mulseed_vi_garnet", "mulseed_vi_forest", "mulseed_vi_graph", "mulseed_vi_healthcare",
                "mulseed_pi_grid", "mulseed_pi_garnet", "mulseed_pi_forest", "mulseed_pi_graph", "mulseed_pi_healthcare",
                "mulseed_r1_vi_grid", "mulseed_r1_vi_garnet", "mulseed_r1_vi_forest", "mulseed_r1_vi_graph", "mulseed_r1_vi_healthcare",
                "mulseed_and_vi_grid", "mulseed_and_vi_garnet", "mulseed_and_vi_forest", "mulseed_and_vi_graph", "mulseed_and_vi_healthcare",
                "multiseed_benchmark","benchmark_garnet"
                ],
        help="Type of benchmark to run"
    )

    args = parser.parse_args()

    mdp_map = {
        "grid": (grid_mdp_create, {}),
        "garnet": (garnet_MDP_create, {"state_size": 30, "action_size": 5, "branch_size":5, "key": jrd.PRNGKey(42)}),
        "forest": (forest_MDP_create, {"rotation": 25}),
        "graph": (graph_MDP_create, {}),
        "healthcare": (healthcare_MDP_create, {})
    }

    alg_map = {
        "vi": (vi, "Value Iteration"),
        "gs_vi": (gs_vi, "Gauss Seidel Value Iteration"),
        "pi": (pi, "Policy Iteration"),
        "r_vi": (r_vi, "Relaxed Value Iteration"),
        "a_vi": (a_vi, "Accelerated Value Iteration"),
        "m_vi": (m_vi, "Momentum Value Iteration"),
        "sa_vi": (sa_vi, "Safe Accelerated Value Iteration"),
        "anc_vi": (anc_vi, "Anchor value Iteration"),
        "and_vi": (and_vi, "Anderson Value Iteration"),
        "q_pi": (pi_q, "Q PI"),
        "qpi": (qpi, "Quasi PI"),
        "r1_vi": (r1_vi, "Rank 1 Value Iteration"),
        "ct_vi": (pid_vi, "PID Value Iteration"),
        "gso_vi": (gso_vi, "General Second Order Value Iteration")
    }

    benchmark_map = {
        "benchmark_alg":benchmark_vi
    }

    if args.benchmark_type.startswith("benchmark_alg"):

        mdp_type = args.benchmark_type[len("benchmark_alg")+1:]
        mdp_creator, mdp_params = mdp_map[mdp_type]
        mdp = mdp_creator(**mdp_params)
        mdp_name = mdp_type.capitalize() 
        benchmark_vi(mdp,mdp_name)

    elif args.benchmark_type.startswith("benchmark_garnet"):
        benchmark_garnet_comparison(state_size=50, action_size=5, branch_size=15, loop_args=loop_args)
    
    else:

        for prefix in alg_map:
            if args.benchmark_type.startswith(prefix):
                #print(prefix)
                alg_module, alg_display_name = alg_map[prefix]
                mdp_type = args.benchmark_type[len(prefix)+1:]
                break
        else:
            raise ValueError(f"Unknown Benchmark type: {args.benchmark_type}") 

        mdp_creator, mdp_params = mdp_map[mdp_type]
        mdp = mdp_creator(**mdp_params)
        mdp_name = mdp_type.capitalize() 

        alg_name = f"{alg_display_name} {mdp_name}" 
        seed_key = "n_seed" if "_" in prefix else "seed"
        init_state = alg_module.init(mdp, jrd.PRNGKey(getattr(loop_args, seed_key)), loop_args.gamma)
        update_fn = alg_module.update
        alg_implementation(mdp, alg_name, mdp_name, init_state, update_fn)        