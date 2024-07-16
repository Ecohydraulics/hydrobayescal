#code 1
import numpy as np

# Example initialization
N_locations = 4       # Number of locations
N_variables = 2       # Number of variables
N_realizations = 10  # Number of realizations

# Example data
errors = np.array([[0.1, 0.2], [0.15, 0.25], [0.12, 0.18], [0.2, 0.3]])  # Shape: (N_locations, N_variables)
means_vect = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])    # Shape: (N_variables, N_locations)
model_predictions = np.zeros((N_variables, N_realizations, N_locations))

model_predictions = np.array([[[0.7, 1.4, 2.8, 1.1, 2.4, 3.0, 0.9, 2.0, 3.1, 1.0],
                               [0.9, 1.3, 2.6, 1.0, 2.3, 2.9, 1.2, 2.4, 3.3, 0.8],
                               [0.8, 2.0, 3.1, 1.2, 2.2, 3.4, 0.7, 2.1, 2.9, 1.1],
                               [1.2, 2.2, 2.7, 0.7, 1.7, 3.7, 0.6, 1.4, 2.7, 1.2]],
                              [[0.8, 1.3, 2.9, 1.0, 2.3, 2.9, 1.0, 2.1, 3.2, 0.9],
                               [0.6, 1.6, 3.2, 1.3, 2.6, 3.3, 1.1, 2.3, 3.5, 0.7],
                               [0.7, 2.1, 3.2, 1.1, 2.1, 3.5, 0.8, 2.0, 2.8, 1.0],
                               [1.0, 2.0, 3.3, 0.9, 1.6, 3.6, 0.9, 1.3, 2.6, 1.1]]])  #
# Example data
#errors = np.array([[0.1], [0.15], [0.12]])  # Shape: (N_locations, N_variables)
# means_vect = np.array([[1.0, 2.0, 3.0]])  # Shape: (N_variables, N_locations)
# model_predictions = np.array([[[0.8, 1.2, 2.5], [1.1, 2.0, 3.2], [0.9, 2.1, 3.0], [1.0, 2.2, 2.8]]])  # Shape: (N_variables, N_realizations, N_locations)

cov_mats = np.array([np.diag(errors[:, var]) for var in range(N_variables)])  # Shape: (N_variables, N_locations, N_locations)

invRs = np.linalg.inv(cov_mats)

# # Print mean errors and covariance matrix for verification
# print("Covariance Matrix:\n", cov_mats)

# Vectorize errors (if necessary)
# if errors.ndim == 2 and errors.shape[1] == 1:
#     errors = errors.reshape(-1)  # Reshape to (N_locations, N_variables)

# Reshape means_vect for broadcasting
means_vect_new = means_vect[:, np.newaxis]  # Shape: (N_variables, 1, N_locations)

# Reshape model_predictions if needed
if model_predictions.ndim == 3 and model_predictions.shape[0] == 1:
    model_predictions_new = model_predictions.reshape(N_variables,N_realizations, N_locations)
else:
    model_predictions_new = model_predictions
likelihoods = np.zeros((N_realizations, model_predictions_new.shape[0]))
log_likelihoods = np.zeros((N_realizations, model_predictions_new.shape[0]))
for var in range(model_predictions_new.shape[0]):
    # Calculate differences
    # print(means_vect_new[var].shape)
    # print(model_predictions_new[var].shape)
    means_vect = means_vect_new[var].reshape(1, 1, N_locations)
    model_predictions = model_predictions_new[var].reshape(1, N_realizations, N_locations)
    diff_new = means_vect - model_predictions  # Shape: (N_variables, N_realizations, N_locations)
    #diff_new = diff_new.transpose(1, 2, 0)  # Reshape to (N_realizations, N_locations, N_variables)

    # Add the following lines:
    diff_4d = diff_new[:, :, np.newaxis]  # Shape: (N_realizations, N_locations, 1, N_variables)
    transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)  # Shape: (N_realizations, N_locations, N_variables, 1)

    # Calculate values inside the exponent using vectorized operations
    inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invRs[var])  # Shape: (N_realizations, N_locations, 1, N_locations)
    inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)  # Shape: (N_realizations, N_locations)

    total_inside_exponent = inside_2.transpose(2,1,0)  # Sum over locations
    total_inside_exponent = np.reshape(total_inside_exponent,
                                       (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))
    total_inside_exponent = total_inside_exponent.squeeze()


    # Calculate likelihood
    likelihoods[:, var] = np.exp(-0.5 * total_inside_exponent)
    log_likelihoods[:, var] = -0.5 * total_inside_exponent
print("Likelihoods (Vectorized Code with Two Variables):", likelihoods)
print("log_Likelihoods (Vectorized Code with Two Variables):", log_likelihoods)
# Calculate overall likelihood for each realization by summing the logarithms of the likelihoods across variables
# Calculate overall likelihood for each realization by summing the logarithms of the likelihoods across variables
if likelihoods.shape[1] > 1:
    # Converting likelihoods to a single normalized likelihood for each realization
    combined_likelihoods = np.mean(likelihoods, axis=1)  # Shape: (N_realizations,)
    # normalized_likelihoods = combined_likelihoods / np.sum(combined_likelihoods)  # Normalize to sum to 1

    # Converting log likelihoods to a single normalized log likelihood for each realization
    combined_log_likelihoods = np.mean(log_likelihoods, axis=1)  # Shape: (N_realizations,)
    # normalized_log_likelihoods = combined_log_likelihoods - np.max(combined_log_likelihoods)
else:
    normalized_likelihoods = likelihoods.flatten()  # Shape: (N_realizations,)
    normalized_log_likelihoods = log_likelihoods.flatten()  # Shape: (N_realizations,)

print("Likelihoods (Vectorized Code with Two Variables):", combined_likelihoods)
print("Log_Likelihoods (Vectorized Code with Two Variables):", combined_log_likelihoods)

# Code 2
# Example initialization
N_locations = 4  # Number of locations
N_variables = 1  # Number of variables (changed to 1)
N_realizations = 10  # Number of realizations

# Example data
errors = np.array([[0.1], [0.15], [0.12], [0.2]])  # Shape: (N_locations, N_variables)
means_vect = np.array([[1.0, 2.0, 3.0, 4.0]])  # Shape: (N_variables, N_locations)
model_predictions = np.zeros((N_variables, N_realizations, N_locations))

# Generate random model predictions for the single variable
model_predictions = np.array([[[0.7, 1.4, 2.8, 1.1, 2.4, 3.0, 0.9, 2.0, 3.1, 1.0],
                               [0.9, 1.3, 2.6, 1.0, 2.3, 2.9, 1.2, 2.4, 3.3, 0.8],
                               [0.8, 2.0, 3.1, 1.2, 2.2, 3.4, 0.7, 2.1, 2.9, 1.1],
                               [1.2, 2.2, 2.7, 0.7, 1.7, 3.7, 0.6, 1.4, 2.7, 1.2]]])

# Calculate covariance matrix for the single variable
cov_mats = np.array(
    [np.diag(errors[:, var]) for var in range(N_variables)])  # Shape: (N_variables, N_locations, N_locations)

# Invert the covariance matrix
invRs = np.linalg.inv(cov_mats)

# # Print covariance matrix for verification
# print("Covariance Matrix:\n", cov_mats)

# Reshape means_vect for broadcasting
means_vect_new = means_vect[:, np.newaxis]  # Shape: (N_variables, 1, N_locations)

# Reshape model_predictions if needed
if model_predictions.ndim == 3 and model_predictions.shape[0] == 1:
    model_predictions_new = model_predictions.reshape(N_variables, N_realizations, N_locations)
else:
    model_predictions_new = model_predictions

likelihoods = np.zeros((N_realizations, model_predictions_new.shape[0]))
log_likelihoods = np.zeros((N_realizations, model_predictions_new.shape[0]))

for var in range(model_predictions_new.shape[0]):
    # Calculate differences
    means_vect = means_vect_new[var].reshape(1, 1, N_locations)
    model_predictions = model_predictions_new[var].reshape(1, N_realizations, N_locations)
    diff_new = means_vect - model_predictions  # Shape: (N_variables, N_realizations, N_locations)

    # Add the following lines:
    diff_4d = diff_new[:, :, np.newaxis]  # Shape: (N_realizations, N_locations, 1, N_variables)
    transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)  # Shape: (N_realizations, N_locations, N_variables, 1)

    # Calculate values inside the exponent using vectorized operations
    inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invRs[var])  # Shape: (N_realizations, N_locations, 1, N_locations)
    inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)  # Shape: (N_realizations, N_locations)

    total_inside_exponent = inside_2.transpose(2, 1, 0)  # Sum over locations
    total_inside_exponent = np.reshape(total_inside_exponent,
                                       (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))
    total_inside_exponent = total_inside_exponent.squeeze()

    # Calculate likelihood
    likelihoods[:, var] = np.exp(-0.5 * total_inside_exponent)
    log_likelihoods[:, var] = -0.5 * total_inside_exponent

# Calculate overall likelihood for each realization by summing the logarithms of the likelihoods across variables
if likelihoods.shape[1] > 1:
    # Converting likelihoods to a single normalized likelihood for each realization
    combined_likelihoods = np.mean(likelihoods, axis=1)  # Shape: (N_realizations,)
    # normalized_likelihoods = combined_likelihoods / np.sum(combined_likelihoods)  # Normalize to sum to 1

    # Converting log likelihoods to a single normalized log likelihood for each realization
    combined_log_likelihoods = np.mean(log_likelihoods, axis=1)  # Shape: (N_realizations,)
    normalized_log_likelihoods = combined_log_likelihoods - np.max(combined_log_likelihoods)
else:
    normalized_likelihoods = likelihoods.flatten()  # Shape: (N_realizations,)
    normalized_log_likelihoods = log_likelihoods.flatten()  # Shape: (N_realizations,)

# Print normalized likelihoods and log likelihoods for each realization
print("Normalized Likelihoods:", normalized_likelihoods)
print("Normalized Log Likelihoods:", normalized_log_likelihoods)

# # code 2
# # vectorize means:
# if errors.ndim == 2 and errors.shape[1] == 1:
#     errors = errors.reshape(-1)  # Reshape to (2,)
#
# cov_mat = np.diag(mean_errors)
# invR = np.linalg.inv(cov_mat)
#
# means_vect = means_vect[:, np.newaxis]  # ############
# if model_predictions.ndim == 3 and model_predictions.shape[0] == 1:
#     model_predictions = model_predictions.reshape(N_realizations, N_locations)
# else:
#     print("The condition model_predictions.shape[0] == 1 is not met.")
# # else:
# diff = means_vect - model_predictions  # Shape: # means
# diff_4d = diff[:, :, np.newaxis]
# transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)
#
# # Calculate values inside the exponent
# inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
# inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
# total_inside_exponent = inside_2.transpose(2, 1, 0)
# total_inside_exponent = np.reshape(total_inside_exponent,
#                                    (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))
#
# # likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)
# likelihood_1 = np.exp(-0.5 * total_inside_exponent)
# log_likelihood_1 = -0.5 * total_inside_exponent
# print(likelihood_1)
#
# # Convert likelihoods to vector:
# if log_likelihood_1.shape[1] == 1:
#     likelihood = likelihood_1[:, 0]
#     log_likelihood = log_likelihood_1[:, 0]
# log_likelihood_1 = log_likelihood_1
# likelihood_1 = likelihood_1
# print(likelihood_1)
