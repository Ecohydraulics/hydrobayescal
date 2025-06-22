import pandas as pd
import numpy as np

def heuristic_error(model_csv_path, obs_csv_path,
                    sigma_meas_velocity=None, sigma_meas_depth=None, sigma_meas_cbe=None,
                    output_csv_path="heuristic_errors.csv"):
    """
    Computes heuristic model error per location for velocity, water depth, and cumulative bed evolution (CBE),
    and saves the result as a CSV with three columns: velocity_error, depth_error, cbe_error.

    Parameters:
    - model_csv_path: str
        Path to CSV with model outputs (header included)
    - obs_csv_path: str
        Path to CSV with measured data (header included)
    - sigma_meas_velocity: array-like or None
        Optional measurement error for velocities
    - sigma_meas_depth: array-like or None
        Optional measurement error for depths
    - sigma_meas_cbe: array-like or None
        Optional measurement error for CBE
    - output_csv_path: str
        Path to save the output CSV with computed errors
    """

    # Read model outputs and observations
    model_array = pd.read_csv(model_csv_path, header=0).values
    obs_array = pd.read_csv(obs_csv_path, header=0).values.flatten()

    # Extract by assuming data column order: depth, velocity, cbe
    depth_model = model_array[:, ::3]
    velocity_model = model_array[:, 1::3]
    cbe_model = model_array[:, 2::3]

    depth_obs = obs_array[::3]
    velocity_obs = obs_array[1::3]
    cbe_obs = obs_array[2::3]

    n_locations = len(velocity_obs)

    # Default measurement errors
    if sigma_meas_velocity is None:
        sigma_meas_velocity = np.zeros(n_locations)
    if sigma_meas_depth is None:
        sigma_meas_depth = np.zeros(n_locations)
    if sigma_meas_cbe is None:
        sigma_meas_cbe = np.zeros(n_locations)

    # Heuristic model error (RMSE per location)
    sigma_model_velocity = np.sqrt(np.mean((velocity_model - velocity_obs) ** 2, axis=0))
    sigma_model_depth = np.sqrt(np.mean((depth_model - depth_obs) ** 2, axis=0))
    sigma_model_cbe = np.sqrt(np.mean((cbe_model - cbe_obs) ** 2, axis=0))

    # Total error (including measurement error)
    sigma_total_velocity = np.sqrt(sigma_model_velocity ** 2 + sigma_meas_velocity ** 2)
    sigma_total_depth = np.sqrt(sigma_model_depth ** 2 + sigma_meas_depth ** 2)
    sigma_total_cbe = np.sqrt(sigma_model_cbe ** 2 + sigma_meas_cbe ** 2)

    # Create DataFrame and save
    error_df = pd.DataFrame({
        "velocity_error": sigma_total_velocity,
        "depth_error": sigma_total_depth,
        "cbe_error": sigma_total_cbe
    })

    error_df.to_csv(output_csv_path, index=False)
    print(f"Heuristic errors saved to: {output_csv_path}")


# Example usage
if __name__ == "__main__":
    model_outputs_file = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/runs-outputs.csv"
    measured_data_file = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/measured-data.csv"
    output_file = "/home/IWS/hidalgo/Documents/hydrobayescal/examples/ering-data/simulation_folder_telemac/heuristic error per location.csv"

    heuristic_error(
        model_csv_path=model_outputs_file,
        obs_csv_path=measured_data_file,
        output_csv_path=output_file
    )
