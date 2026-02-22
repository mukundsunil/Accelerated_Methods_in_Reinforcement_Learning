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
    table.add_column("Value Error", justify="right")
    table.add_column("Max L-Inf", justify="right")
    table.add_column("Iterations", justify="right")
    

    for alg_name, alg_metrics in avg_results.items():
        # IMPORTANT: Use float() or int() to convert JAX/NumPy types to Python scalars
        # Access the last element [-1] directly
        bellman_res = float(alg_metrics.bellman_res[-1])
        val_err = float(alg_metrics.val_err[-1])
        linf = float(jnp.max(alg_metrics.linf[-1]))
        it = int(alg_metrics.iteration[-1])
        
        # Pass strings, not lists, to add_row
        table.add_row(
            alg_name, 
            f"{bellman_res:.6f}", 
            f"{val_err:.6f}", 
            f"{linf:.6f}", 
            f"{it:}"
        )

    console.print(table)

def benchmark_plot_results(avg_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, (alg_name, alg_metrics) in enumerate(avg_results.items()):
        color = cmap(i%20)
        bellman_res = alg_metrics.bellman_res
        iters = alg_metrics.iteration[:,0]
        plt.plot(iters, bellman_res, color=color, alpha=0.5, linewidth=0.1)

        mean_res = jnp.mean(bellman_res, axis=1)
        plt.plot(iters, mean_res, color=color, linewidth=1, label=f"{alg_name}")

        std_res = jnp.std(bellman_res, axis=1)
        lower_bound = mean_res - std_res 
        upper_bound = mean_res + std_res
        plt.fill_between(iters, lower_bound, upper_bound, color=color, alpha=0.2)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}")
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"Images/Benchmark_alg_BE_{mdp_name}_{gamma}_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close('all') # Explicitly close all figure objects
    print(f"Plot saved to {save_path}")

def benchmark_plot_val_err_results(avg_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, (alg_name, alg_metrics) in enumerate(avg_results.items()):
        color = cmap(i%20)
        val_err = alg_metrics.val_err
        iters = alg_metrics.iteration[:,0]
        plt.plot(iters, val_err, color=color, alpha=0.5, linewidth=0.1)

        mean_res = jnp.mean(val_err, axis=1)
        plt.plot(iters, mean_res, color=color, linewidth=1, label=f"{alg_name}")

        std_res = jnp.std(val_err, axis=1)
        lower_bound = mean_res - std_res 
        upper_bound = mean_res + std_res
        plt.fill_between(iters, lower_bound, upper_bound, color=color, alpha=0.2)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}")
    plt.xlabel("Iterations")
    plt.ylabel("Value Error")
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"Images/Benchmark_alg_VE_{mdp_name}_{gamma}_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close('all') # Explicitly close all figure objects
    print(f"Plot saved to {save_path}")