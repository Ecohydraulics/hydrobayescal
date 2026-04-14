import numpy as np

# --- inputs from ADV and flume geometry ---
u_prime_x = np.array([...])  # u' time series from ADV [m/s]
u_prime_y = np.array([...])  # v' time series from ADV [m/s]
u_prime_z = np.array([...])  # w' time series from ADV [m/s]
g         = 9.81  # [m/s^2]
R_h       = 0.1153   # hydraulic radius [m]
S_0       = 0.0025   # flume bed slope [-]
z_meas    = 0.09   # ADV measurement height above bed [m]
nu        = 1e-6  # kinematic viscosity water [m^2/s]
kappa     = 0.41  # von Karman constant
k_s       = 0.005   # Nikuradse roughness height [m]
Delta_proposed = np.nan        # DEFINE target cell size to evaluate [m]

# --- TKE from ADV ---
k_tke = 0.5 * (np.var(u_prime_x) +
               np.var(u_prime_y) +
               np.var(u_prime_z))
print(f"TKE k          = {k_tke:.4f} m^2/s^2")

# --- friction velocity from flume slope ---
u_tau = np.sqrt(g * R_h * S_0)
print(f"u_tau          = {u_tau:.4f} m/s")

# --- dissipation rate (log-layer equilibrium assumption) ---
epsilon = u_tau**3 / (kappa * z_meas)
print(f"epsilon        = {epsilon:.4e} m^2/s^3")

# --- integral length scale ---
L_int = k_tke**1.5 / epsilon
print(f"L_int          = {L_int:.4f} m")

# --- Kolmogorov scale ---
eta = (nu**3 / epsilon)**0.25
print(f"eta (Kolmog.)  = {eta:.4e} m")

# --- required LES cell size ---
Delta_conservative = 0.1 * L_int
Delta_practical    = 0.2 * L_int
print(f"Delta max (conservative, Pope >80%) = {Delta_conservative:.4f} m")
print(f"Delta max (practical,    Pope >75%) = {Delta_practical:.4f} m")

# --- resolved TKE fraction for proposed Delta ---
if Delta_proposed <= 0:
    raise ValueError("Delta_proposed must be a positive float in metres.")
if Delta_proposed >= L_int:
    print("WARNING: Delta_proposed >= L_int — filter is outside inertial subrange.")

k_sgs_fraction = (Delta_proposed / L_int) ** (2/3)
k_res_fraction = 1.0 - k_sgs_fraction
print(f"\nProposed Delta        = {Delta_proposed:.4f} m")
print(f"Resolved TKE fraction = {k_res_fraction:.2%}")
print(f"Pope criterion (>80%) = {'PASS' if k_res_fraction > 0.8 else 'FAIL'}")

# --- wall-normal placement constraints ---
z1_smooth_min = 30  * nu / u_tau
z1_smooth_max = 200 * nu / u_tau
z1_rough_min  = 3.0 * k_s
print(f"\nWall-normal z1 (smooth log-layer): {z1_smooth_min:.5f} – {z1_smooth_max:.5f} m")
print(f"Wall-normal z1 (rough, > 3*k_s):   {z1_rough_min:.5f} m")
