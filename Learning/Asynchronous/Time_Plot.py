import matplotlib.pyplot as plt

# Define the data from the table
mdp_labels = [50, 98, 200, 312, 450]

# Time for iteration values (in seconds)
q_learning = [93.0946, 93.6714, 96.7785, 97.9071, 99.5971]
zap_ql = [137.9392, 245.1411, 845.2697, 2628.3625, 10935.5351]
pre_cond_ql = [108.0419, 120.4623, 277.9409, 441.4880, 685.1778]

# Plot lines with distinct markers for readability
plt.plot(mdp_labels, q_learning, marker='o', linewidth=2, label='Q Learning', color='#ff7f0e')
plt.plot(mdp_labels, zap_ql, marker='s', linewidth=2, label='Zap QL', color='#2ca02c')
plt.plot(mdp_labels, pre_cond_ql, marker='^', linewidth=2, label='Pre Cond QL', color='#1f77b4')

# Customize labels and styling
plt.title('Time per Iteration across Different MDP Size', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('MDP Size', fontsize=12, labelpad=10)
plt.ylabel('Time for Iteration (seconds)', fontsize=12, labelpad=10)

# Add grid lines for easier value tracking
plt.grid(True, linestyle='--', alpha=0.6)

# Place the legend
plt.legend(fontsize=11)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the plot
plt.savefig('Time_per_iteration.png')