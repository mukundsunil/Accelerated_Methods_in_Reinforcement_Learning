from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def log_results(results, alg_name):
    console = Console()

    table = Table(
        title=f"[bold]{alg_name.upper()} [/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    table.add_column("MDP", style="bold white", width=20)
    table.add_column("Bellman Residual", justify="right", style="white")
    table.add_column("Log Bellman Residual", justify="right", style="white")
    table.add_column("Iterations", justify="right", style="white")

    for mdp_name, (metrics, _) in results.items():
        bellman_res = float(metrics.bellman_res[-1])
        log_bellman_res = float(metrics.log_bellman_res[-1])
        iters = int(metrics.iteration[-1])

        table.add_row(
            mdp_name,
            f"{bellman_res:.6f}",
            f"{log_bellman_res:.6f}", 
            f"{iters:}"
            )

    console.print()
    console.print(table)
    console.print()

def log_comprehensive_benchmark_alg(all_results, settings):
    console = Console()

    table = Table(
        title="[bold] Mean Metrics [/bold]",
        show_header=True,
        header_style="bold white",
        border_style="white",
        title_style="bold white"
    )
    mdp_name = settings["name"]
    table.add_column("Algorithm", style="bold white", width=18)
    table.add_column("Bellman Residual", justify="right", style="white")
    table.add_column("Log Bellman Residual", justify="right", style="white")
    table.add_column("Value Error", justify="right", style="white")
    table.add_column("Iterations", justify="right", style="white")

    for alg_name, alg_metrics in all_results.items():
        row_data = [alg_name]
        bellman = []
        log_bellman = []
        val_err = []
        iters = []
        bellman_res = float(alg_metrics.bellman_res[-1])
        log_bellman_res = float(alg_metrics.log_bellman_res[-1])
        val_err = float(alg_metrics.value_error[-1])
        it = int(alg_metrics.iteration[-1])
        bellman.append(bellman_res)
        log_bellman.append(log_bellman_res)
        iters.append(it)
        row_data.append(f"{bellman_res:.6f}")
        row_data.append(f"{log_bellman_res:.6f}")
        row_data.append(f"{val_err:.6f}")
        row_data.append(f"{iters:}")
        table.add_row(*row_data)

    console.print()
    console.print(table)
    console.print()

def plot_results(results, alg_name):
  
    for mdp_name, (metrics, _) in results.items():
        bellman_res = (metrics.bellman_res)
        iters = (metrics.iteration)
        plt.plot(iters, bellman_res)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(alg_name)
    plt.xlabel("Iterations")
    plt.ylabel("Bellman Residual")
    plt.ylim(bottom=1e-6, top = 1e3)
    plt.grid
    plt.savefig(f"Images/{alg_name}_plot.png", dpi=300)
    plt.close()

def plot_results_alg(all_results, settings):
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    plt.rcParams.update({'font.size': 12})    
    plt.figure(figsize=(10, 6))
    for alg_name, alg_metrics in all_results.items():
        bellman_res = alg_metrics.bellman_res
        iters = alg_metrics.iteration
        plt.plot(iters, bellman_res, label=alg_name, linewidth=2)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for $\gamma$={gamma}", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Bellman Residual", fontsize=14)
    plt.ylim(bottom=1e-6, top=1e6)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)  
    plt.tight_layout()
    plt.savefig(f"Images/Benchmark_alg_BE_{mdp_name}_{gamma}_plot.png", dpi=300)
    plt.close()

def plot_results_val_err_alg(all_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))

    for alg_name, alg_metrics in all_results.items():
        bellman_res = (alg_metrics.value_error)
        iters = (alg_metrics.iteration)
        plt.plot(iters, bellman_res, label = alg_name)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Value Error", fontsize=14)
    plt.ylim(bottom=1e-6, top = 1e6)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f"Images/Benchmark_alg_Value_error_{mdp_name}_{gamma}_plot.png", dpi=300)
    plt.close()

def plot_results_mulseed_garnet_alg(all_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    S = settings["S"]
    A = settings["A"]
    b = settings["b"]
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, (alg_name, alg_metrics) in enumerate(all_results.items()):
        color = cmap(i%20)
        bellman_res = (alg_metrics.bellman_res)
        iters = (alg_metrics.iteration[:, 0])
        
        mean_res = jnp.mean(bellman_res, axis=1)
        plt.plot(iters, mean_res, color=color, linewidth=1)

        std_res = jnp.std(bellman_res, axis=1)
        lower_bound = mean_res - std_res 
        upper_bound = mean_res + std_res
        plt.fill_between(iters, lower_bound, upper_bound, color=color, alpha=0.5)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Bellman Residual", fontsize=14)
    plt.ylim(bottom=1e-6, top = 1e6) 
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    plt.savefig(f"Images/Benchmark_alg_BE_MS_Garnet_{S}_{A}_{b}_{gamma}_plot.png", dpi=300)
    plt.close()

def plot_results_mulseed_garnet_val_err_alg(all_results, settings):
  
    mdp_name = settings["name"]
    gamma = settings["gamma"]
    S = settings["S"]
    A = settings["A"]
    b = settings["b"]
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')

    for i, (alg_name, alg_metrics) in enumerate(all_results.items()):
        color = cmap(i%20)
        val_err = (alg_metrics.value_error)
        iters = (alg_metrics.iteration[:, 0])

        mean_res = jnp.mean(val_err, axis=1)
        plt.plot(iters, mean_res, color=color, linewidth=1)

        std_res = jnp.std(val_err, axis=1)
        lower_bound = mean_res - std_res
        upper_bound = mean_res + std_res
        plt.fill_between(iters, lower_bound, upper_bound, color=color, alpha=0.5)
       
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Benchmark {mdp_name} for gamma={gamma}", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Value Error", fontsize=14)
    plt.ylim(bottom=1e-6, top = 1e6) 
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    plt.savefig(f"Images/Benchmark_alg_Value_Error_MS_Garnet_{S}_{A}_{b}_{gamma}_plot.png", dpi=300)
    # plt.savefig(f"Images/Benchmark_alg_Value_Error_MS_Garnet_G_{gamma}_plot.png", dpi=300)
    plt.close()
