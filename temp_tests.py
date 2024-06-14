
import numpy as np

# Example data for illustration
mc_size_AL = 1000
n_points = 3
al_unique_index = [0, 1, 2]
iAL = 1  # Current index

# Hypothetical surrogate model predictions and standard deviations (adjusted for example)
surrogate_prediction = np.array([
    [1.0, 1.5, 2.0],
    [1.2, 1.6, 2.1]
])
surrogate_std = np.array([
    [0.1, 0.2, 0.15],
    [0.2, 0.25, 0.1]
])

# Ensure the selection matches the correct shape
selected_std = surrogate_std[:, al_unique_index[iAL]].reshape(-1, 1)
selected_prediction = surrogate_prediction[:, al_unique_index[iAL]].reshape(-1, 1)

# Generate AL exploration samples
al_exploration = np.random.normal(size=(mc_size_AL, selected_std.shape[0])) * selected_std.T #+ selected_prediction.T

print("AL Exploration Samples:")
print(al_exploration)