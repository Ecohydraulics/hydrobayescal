import numpy as np
import matplotlib.pyplot as plt

# Define the classification conditions with explicit exclusions
velocity_raster_name = "velocity-beforeflush"
wdepth_raster_name = "wdepth-beforeflush"

classification = {
    "1 (Pool)": {"velocity": [0.05, 0.80], "depth": [0.40, 1.0], "color": "blue", "alpha": 0.5},
    "3 (Glide)": {"velocity": [0.60, 1.0], "depth": [0.20, 0.60], "color": "orange", "alpha": 1.0},
    "2 (Slackwater)": {"velocity": [0.05, 0.60], "depth": [0.05, 0.40], "color": "green", "alpha": 0.5},
    "4 (Riffle)": {"velocity": [0.60, 1.6], "depth": [0.05, 0.20], "color": "red", "alpha": 0.5},
    "5 (Run)": {"velocity": [0.80, 1.6], "depth": [0.20, 1.0], "color": "purple", "alpha": 0.5},
    "0 (Banks)": {"velocity": [0, 1.6], "depth": [0, 1], "color": "grey", "alpha": 0.5},
}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize an empty string for QGIS Raster Calculator expression
qgis_expression = ""

# Iterate through classification ensuring proper order
for label, bounds in classification.items():
    velocity_min, velocity_max = bounds["velocity"]
    depth_min, depth_max = bounds["depth"]

    qgis_expression += f"""(
        "{velocity_raster_name}@1" >= {velocity_min} AND "{velocity_raster_name}@1" <= {velocity_max} AND 
        "{wdepth_raster_name}@1" >= {depth_min} AND "{wdepth_raster_name}@1" <= {depth_max}
    ) * {label.split()[0]} +\n"""

    ax.fill_betweenx(
        np.linspace(depth_min, depth_max, 100), velocity_min, velocity_max,
        color=bounds["color"], alpha=bounds["alpha"], label=label
    )

# Remove trailing "+"
qgis_expression = qgis_expression.rstrip(" +\n")

# Customize the plot
ax.set_xlabel("Velocity (m/s)", fontsize=14)
ax.set_ylabel("Water Depth (m)", fontsize=14)
ax.set_title("Velocity / Water Depth Classification", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xticks(np.arange(0, 1.8 + 0.2, 0.2))
ax.set_yticks(np.arange(0, 1.0 + 0.2, 0.2))

# Place the legend outside the plot
fig.subplots_adjust(right=0.75)
ax.legend(title="Morphological Units", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, frameon=True)

plt.tight_layout()
plt.show()

# Print the fixed QGIS Raster Calculator expression
print("QGIS Raster Calculator Expression:\n")
print(qgis_expression)
# Print a message to confirm the file save location
#print(f"Graph saved to {output_path}")
