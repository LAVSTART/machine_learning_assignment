import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Read your combined data
df = pd.read_csv("D_summary_all.csv")

# Boltzmann constant in eV/K
kB = 8.617333262e-5

for label, group in df.groupby("label"):
    T = group["Temperature(K)"]
    D = group["D_m2_s"]
    invT = 1.0 / T
    lnD = np.log(D)

    slope, intercept, r, p, se = linregress(invT, lnD)
    Ea = -slope * kB  # eV

    print(f"{label}: Ea = {Ea:.3f} eV (R={r:.3f})")

    # Plot Arrhenius
    plt.plot(invT, lnD, 'o', label=f"{label} data")
    plt.plot(invT, intercept + slope*invT, '--', label=f"{label} fit (Ea={Ea:.3f} eV)")

plt.xlabel("1 / Temperature (1/K)")
plt.ylabel("ln(D) [ln(m²/s)]")
plt.title("Arrhenius Plot: Diffusion vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Arrhenius_Plot.png", dpi=300)
plt.show()
