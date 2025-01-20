import sys
import os
import pythomac as ptc
examples_dir = os.getcwd()
simulation_dir = os.path.join(examples_dir, "inn")
telemac_cas = "steady2d-wet.cas"
print(simulation_dir)

# extract fluxes across boundaries
fluxes_df = ptc.extract_fluxes(
    model_directory=simulation_dir,
    cas_name=telemac_cas,
    plotting=True
)
print(fluxes_df.index.values)
print(fluxes_df.index.values)
# back-calculate Telemac timestep size
timestep_in_cas = int(max(fluxes_df.index.values) / (len(fluxes_df.index.values) - 1))

# plot convergence
iota_t = ptc.calculate_convergence(
    series_1=fluxes_df["Fluxes Boundary 1"][1:],  # remove first zero-entry
    series_2=fluxes_df["Fluxes Boundary 2"][1:],  # remove first zero-entry
    cas_timestep=timestep_in_cas,
    plot_dir=simulation_dir
)

# write the result to a CSV file
iota_t.to_csv(os.path.join(simulation_dir, "convergence-rate-unsteady.csv"))

# identify the timestep at which convergence was reached at a desired precision
convergence_time_iteration = ptc.get_convergence_time(
    convergence_rate=iota_t["Convergence rate"],
    convergence_precision=1.0E-6
)

if not("nan" in str(convergence_time_iteration).lower()):
    print("The simulation converged after {0} simulation seconds ({1}th printout).".format(
        str(timestep_in_cas * convergence_time_iteration), str(convergence_time_iteration)))