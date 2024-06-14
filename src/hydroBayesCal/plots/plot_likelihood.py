import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Example data
measured_data = np.array([2.5, 3.0, 2.8, 3.2, 2.9])
n = len(measured_data)

# Example modeled outputs for p sets of parameters
modeled_outputs = [
    np.array([2.4, 2.9, 2.7, 3.1, 2.8]),  # Parameters set 1
    np.array([2.5, 3.1, 2.8, 3.3, 3.0]),  # Parameters set 2
    np.array([2.6, 3.0, 2.9, 3.2, 2.9])   # Parameters set 3
]

# Assume a uniform prior over the parameter sets
p = len(modeled_outputs)
prior = np.ones(p) / p

# Assume a known standard deviation for the measurement errors
sigma = 0.1

# Function to calculate likelihood for each parameter set
def likelihood(measured_data, modeled_data, sigma):
    return np.prod(norm.pdf(measured_data, loc=modeled_data, scale=sigma))

# Calculate the likelihood for each parameter set
likelihoods = np.array([likelihood(measured_data, modeled_output, sigma) for modeled_output in modeled_outputs])

# Calculate the unnormalized posterior
unnormalized_posterior = prior * likelihoods

# Normalize the posterior
posterior = unnormalized_posterior / np.sum(unnormalized_posterior)

# Plotting the prior and posterior distributions
parameter_indices = np.arange(1, p + 1)

plt.figure(figsize=(12, 6))

# Plot prior
plt.subplot(1, 2, 1)
plt.bar(parameter_indices, prior, color='blue', alpha=0.6)
plt.xlabel('Parameter Set')
plt.ylabel('Probability')
plt.title('Prior Distribution')

# Plot posterior
plt.subplot(1, 2, 2)
plt.bar(parameter_indices, posterior, color='green', alpha=0.6)
plt.xlabel('Parameter Set')
plt.ylabel('Probability')
plt.title('Posterior Distribution')

plt.tight_layout()
plt.show()

# Print the posterior probabilities
for i, prob in enumerate(posterior):
    print(f"Posterior probability for parameter set {i + 1}: {prob:.4f}")

# Identify the best parameter set
best_index = np.argmax(posterior)
print(f"The best parameter set is set {best_index + 1} with posterior probability {posterior[best_index]:.4f}")