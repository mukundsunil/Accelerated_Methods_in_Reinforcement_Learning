import pickle
import jax
import numpy as np
import matplotlib.pyplot as plt

# Load your uploaded pickle file
file_path = "ZAP_Q-Learning_(Random)_0.999_Trial_1_results.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

steps = np.array(data['steps'])
steps = np.where(steps <= 0, 1, steps)
bellman_error = data['metrics']['bellman_linf']

plt.figure(figsize=(9, 5.5))

# Loop over each seed inside the array and add a distinct label
num_seeds = bellman_error.shape[0]
for i in range(num_seeds):
    plt.plot(steps, bellman_error[i], alpha=0.7, linewidth=1.2, label=f"Seed {i+1}")

plt.yscale("log")  
plt.xscale("log")

plt.title("Zap Q-Learning: Bellman Error Convergence Across Seeds ($\gamma = 0.999$)", fontsize=12, fontweight="bold")
plt.xlabel("Iteration Steps (Log Scale)", fontsize=11)
plt.ylabel("Bellman Error ($L_\infty$ Norm, Log Scale)", fontsize=11)

# Clean up layout, grids, and display the legend for your seeds
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
plt.savefig("Comparison_Plot.png", dpi=300)
plt.show()