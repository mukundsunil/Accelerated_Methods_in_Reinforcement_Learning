from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import jax.numpy as jnp


def log_results_sync(results, alg_name):
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()}[/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white",
    )
    table.add_column("MDP", style="bold white", width=20)
    table.add_column("Bellman Residual", justify="right", style="white")
    table.add_column("Max L-inf", justify="right", style="white")
    table.add_column("Iterations", justify="right", style="white")
    

    for mdp_name, (metrics, q_vals) in results.items():
        bellman_res = float(metrics.bellman_res[-1])
        max_linf = float(jnp.max(metrics.linf[-1]))
        iters = (metrics.iteration[-1])

        table.add_row(
            mdp_name,
            f"{bellman_res:.6f}",
            f"{max_linf:.6f}",
            f"{iters:}"
        )

    console.print()
    console.print(table)
    console.print()

def plot_results(results, alg_name):
    for mdp_name, (metrics, q_vals) in results.items():
        bellman_res = (metrics.bellman_res)
        iters = (metrics.iteration)
        plt.plot(iters, bellman_res)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.grid
    plt.savefig(f"Images/{alg_name}_{mdp_name}_plot.png", dpi=300)
    plt.close()

def benchmark_log_results_sync(avg_results, settings):
    mdp_name = settings["name"]
    
    console = Console()
    table = Table(title=mdp_name)

    table.add_column("Algorithm", style="bold white", width=18)
    table.add_column("Bellman Residual", justify="right")
    table.add_column("Max L-Inf", justify="right")
    table.add_column("Iterations", justify="right")

    for alg_name, alg_metrics in avg_results.items():
        # IMPORTANT: Use float() or int() to convert JAX/NumPy types to Python scalars
        # Access the last element [-1] directly
        bellman_res = float(alg_metrics.bellman_res[-1])
        linf = float(jnp.max(alg_metrics.linf[-1]))
        it = int(alg_metrics.iteration[-1])
        
        # Pass strings, not lists, to add_row
        table.add_row(
            alg_name, 
            f"{bellman_res:.6f}", 
            f"{linf:.6f}", 
            f"{it:}"
        )

    console.print(table)

def benchmark_plot_results(avg_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    plt.figure(figsize=(10, 6))

    for alg_name, alg_metrics in avg_results.items():
        bellman_res = alg_metrics.bellman_res
        iters = alg_metrics.iteration
        plt.plot(iters, bellman_res, label = alg_name)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}")
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"Images/Benchmark_alg_{mdp_name}_{gamma}_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close('all') # Explicitly close all figure objects
    print(f"Plot saved to {save_path}")















def log_comp_results_sync(multi_seed_results, alg_name):
    """
    Args:
        multi_seed_results: Dict mapping mdp_name -> list of (metrics, q_vals)
        alg_name: String name of the algorithm
    """
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()} (AGGREGATED RESULTS)[/bold]",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    
    table.add_column("MDP", style="bold white", width=15)
    table.add_column("Bellman Residual (Mean ± Std)", justify="right")
    table.add_column("Max L-inf (Mean ± Std)", justify="right")
    table.add_column("Avg Iterations", justify="right")

    for mdp_name, runs in multi_seed_results.items():
        # runs is a list of tuples: [(metrics_seed0, q0), (metrics_seed1, q1), ...]
        
        # Extract final values from each seed
        bellman_finals = [float(m.bellman_res[-1]) for m, _ in runs]
        linf_finals = [float(jnp.max(m.linf[-1])) for m, _ in runs]
        iters = [int(m.iteration[-1]) for m, _ in runs]

        # Calculate statistics
        b_mean, b_std = np.mean(bellman_finals), np.std(bellman_finals)
        l_mean, l_std = np.mean(linf_finals), np.std(linf_finals)
        i_mean = np.mean(iters)

        table.add_row(
            mdp_name,
            f"{b_mean:.6f} ± {b_std:.6e}",
            f"{l_mean:.6f} ± {l_std:.6e}",
            f"{i_mean:.1f}"
        )

    console.print(table)
    
    
def plot_comparative_results(all_results, mdp_name):
    plt.figure(figsize=(10, 6))
    
    for alg_name, metrics_list in all_results.items():
        # Convert list of metrics to a 2D numpy array: (seeds, steps)
        # Adjust 'returns' to whatever key your metrics object uses
        data = jnp.array([m.returns for m in metrics_list]) 
        
        mean_vals = jnp.mean(data, axis=0)
        std_vals = jnp.std(data, axis=0)
        steps = jnp.arange(len(mean_vals))

        # 1. Plot individual runs in light color
        for i in range(data.shape[0]):
            plt.plot(steps, data[i], alpha=0.15, color=None) # color=None lets it cycle
            
        # 2. Plot Mean with a solid line
        line, = plt.plot(steps, mean_vals, label=f"{alg_name}", linewidth=2)
        
        # 3. Shaded area for Standard Deviation
        plt.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, 
                         color=line.get_color(), alpha=0.2)

    plt.title(f"Performance Comparison on {mdp_name.capitalize()}")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward / Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()