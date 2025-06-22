import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "T": [0, 5000, 5055, 5475, 5715, 5955, 6315, 6675, 7035, 7155, 7455, 7755, 8055, 8355, 8655,
          8955, 10180, 11405, 12255, 14055, 15855, 17655, 19455, 21255, 23055, 24855, 26655,
          28455, 29655, 30255, 30855, 31455, 32055, 32655, 45000, 50000],
    "Q": [2, 2, 2, 4, 5.5, 6.3, 8, 9.5, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7,
          9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7, 9.7,
          9.7, 9.7, 9.5, 7.5, 5.5, 3.5, 2, 2, 2],
    "QG": [0, 0, 0, 5.645, 7.31, 8.065, 10.83, 18.215, 19.4, 19.4, 19.4,
           19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4,
           19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 18.215, 10.205,
           7.31, 4.615, 0, 0, 0],
    "TR": [0, 0, 0, 0.35, 0.5, 1.25, 2.3, 3.2, 3.1, 3.23, 3.03, 3.15, 2.82,
           2.22, 1.94, 1.63, 1.21, 1.21, 0.85, 0.85, 0.52, 0.48, 0.43,
           0.37, 0.31, 0.3, 0.35, 0.27, 0.22, 0.2, 0.19, 0.15, 0.13, 0, 0, 0]
}
df = pd.DataFrame(data)
df['T_hours'] = df['T'] / 3600

# Identify constant Q = 9.7 region
q97 = df[df["Q"] == 9.7]
t_start = q97["T"].iloc[0]
t_end = q97["T"].iloc[-1]

# Plot setup
fig, ax1 = plt.subplots(figsize=(12, 6))

# Primary y-axis: Q only
ax1.plot(df["T"], df["Q"], label="Q (m³/s)", color="blue")
ax1.set_ylabel("Q (m³/s)", color="black")
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(2, df["Q"].max() + 1)

# Vertical lines for flood period
q_mid = (df["Q"].max() + 2) / 2
ax1.vlines([t_start, t_end], ymin=2, ymax=df["Q"].max(), colors="black", linestyles="--")
ax1.text(t_start, q_mid, "Flood start", rotation=90, va='center', ha='right', fontsize=12)
ax1.text(t_end, q_mid, "Flood end", rotation=90, va='center', ha='right', fontsize=12)

# Secondary y-axis: QG and TR
ax2 = ax1.twinx()
ax2.plot(df["T"], df["QG"], label="QG (kg/s)", color="green", linestyle="--")
ax2.plot(df["T"], df["TR"], label="TR (g/L)", color="red")
ax2.set_ylabel("QG (kg/s) & TR (g/L)")
ax2.tick_params(axis='y')
ax2.set_ylim(0, max(df["QG"].max(), df["TR"].max()) + 1)

# Secondary x-axis in hours
def secax(x): return x / 3600
def secax_inv(x): return x * 3600
secax_obj = ax1.secondary_xaxis('top', functions=(secax, secax_inv))
secax_obj.set_xlabel("Time [hours]")

# Legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Axes labels
ax1.set_xlabel("Time [s]")
ax1.set_title("Simulated Hydrograph")
ax1.grid(True)

plt.tight_layout()
plt.show()