import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Define the classification conditions
velocity_raster_name="MU-interpolated-scalar-velocity-initial-model-ref"
wdepth_raster_name="MU-interpolated-waterdepth-initial-model-ref"
classification = {
    "0 (Banks)": {"velocity": [0, 1.8], "depth": [0, 1], "color": "grey"},
    "1 (Pool)": {"velocity": [0.05, 0.60], "depth": [0.40, 1.0], "color": "blue"},
    "1 (Pool Extension)": {"velocity": [0.60, 0.80], "depth": [0.60, 1.0], "color": "blue"},
    "2 (Slackwater)": {"velocity": [0.05, 0.60], "depth": [0.05, 0.40], "color": "green"},
    "3 (Glide)": {"velocity": [0.60, 1.0], "depth": [0.20, 0.60], "color": "orange"},
    "4 (Riffle)": {"velocity": [0.60, 1.8], "depth": [0.05, 0.20], "color": "red"},
    "5 (Run)": {"velocity": [0.80, 1.0], "depth": [0.60, 1.0], "color": "purple"},
    "5 (Run - Extended)": {"velocity": [1.0, 1.80], "depth": [0.20, 1.0], "color": "purple"},
}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize an empty string for QGIS Raster Calculator expression
qgis_expression = ""

# Plot each classification as a region and generate QGIS expression
for label, bounds in classification.items():
    # Create the range of velocities and depths
    velocity_range = np.linspace(bounds["velocity"][0], bounds["velocity"][1], 100)
    depth_min, depth_max = bounds["depth"]
    depth_range = np.linspace(depth_min, depth_max, 100)

    # Fill the region with the specified color
    ax.fill_betweenx(
        depth_range, velocity_range[0], velocity_range[-1],
        color=bounds["color"], alpha=0.3, label=label
    )

    # Append the classification to the QGIS expression
    qgis_expression += f"""(
        "{velocity_raster_name}@1" >= {bounds["velocity"][0]} AND "{velocity_raster_name}@1" <= {bounds["velocity"][1]} AND 
        "{wdepth_raster_name}@1" >= {bounds["depth"][0]} AND "{wdepth_raster_name}@1" <= {bounds["depth"][1]}
    ) * {label.split()[0]} +\n"""

# Remove trailing "+" from the final QGIS expression
qgis_expression = qgis_expression.rstrip(" +\n")

# Customize the plot
ax.set_xlabel("Velocity (m/s)", fontsize=12)
ax.set_ylabel("Water Depth (m)", fontsize=12)
ax.set_title("Morphological Units Classification based on simulated velocity and water depth values ", fontsize=14)
ax.legend(title="Classification", loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xticks(np.arange(0, 1.5 + 0.2, 0.2))  # X-axis (velocity): from 0 to 1.5 with step 0.2
ax.set_yticks(np.arange(0, 1.0 + 0.2, 0.2))  # Y-axis (water depth): from 0 to 1.0 with step 0.2

# Display the plot
plt.tight_layout()
plt.show()

# Print the QGIS Raster Calculator expression
print("QGIS Raster Calculator Expression:\n")
print(qgis_expression)

# Print a message to confirm the file save location
#print(f"Graph saved to {output_path}")
