import numpy as np
import matplotlib.pyplot as plt

# Define the classification with velocity and depth ranges
classification = {
    0: {"name": "Banks", "velocity": (0, 1.2), "depth": (0, 0.05), "color": "grey", "alpha": 0.5},
    1: {"name": "Banks", "velocity": (0, 0.05), "depth": (0.05, 1), "color": "grey", "alpha": 0.5},
    2: {"name": "Pool", "velocity": (0.05, 0.50), "depth": (0.35, 1.0), "color": "blue", "alpha": 0.5},
    3: {"name": "Slackwater", "velocity": (0.05, 0.15), "depth": (0.05, 0.35), "color": "green", "alpha": 0.5},
    4: {"name": "Glide", "velocity": (0.15, 0.50), "depth": (0.05, 0.35), "color": "orange", "alpha": 1.0},
    5: {"name": "Riffle", "velocity": (0.50, 1.2), "depth": (0.05, 0.35), "color": "red", "alpha": 0.5},
    6: {"name": "Run", "velocity": (0.50, 1.2), "depth": (0.35, 1.0), "color": "purple", "alpha": 0.5}
}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Iterate through classification and plot each region
for label, bounds in classification.items():
    velocity_min, velocity_max = bounds["velocity"]
    depth_min, depth_max = bounds["depth"]

    # Create a filled region for each classification
    ax.fill_betweenx(
        np.linspace(depth_min, depth_max, 100), velocity_min, velocity_max,
        color=bounds["color"], alpha=bounds["alpha"], label=f"{label} ({bounds['name']})"
    )

# Customize the plot
ax.set_xlabel("Velocity (m/s)", fontsize=14)
ax.set_ylabel("Water Depth (m)", fontsize=14)
ax.set_title("Velocity / Water Depth thresholds", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xticks(np.arange(0, 1.2 + 0.2, 0.2))
ax.set_yticks(np.arange(0, 1.0 + 0.2, 0.2))

# Place the legend outside the plot
fig.subplots_adjust(right=0.75)
ax.legend(title="Morphological Units", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True)

plt.tight_layout()
plt.show()