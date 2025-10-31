import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the summary data
df = pd.read_csv("thermo_summary.csv")

# Sort by temperature
df = df.sort_values("Temperature(K)").reset_index(drop=True)
print(df)

# ===== Plot Volume vs Temperature =====
plt.figure(figsize=(6,4))
plt.plot(df["Temperature(K)"], df["Avg_Volume"], 'o-', label='Avg Volume')
plt.xlabel("Temperature (K)")
plt.ylabel("Average Volume (Å³)")
plt.title("Volume vs Temperature (CaTiO₃)")
plt.grid(True)
plt.tight_layout()
plt.savefig("volume_vs_T.png", dpi=300)
plt.show()

# ===== Plot Potential Energy vs Temperature =====
plt.figure(figsize=(6,4))
plt.plot(df["Temperature(K)"], df["Avg_PotEng"], 'o-', color='orange', label='Potential Energy')
plt.xlabel("Temperature (K)")
plt.ylabel("Average Potential Energy (eV)")
plt.title("Potential Energy vs Temperature (CaTiO₃)")
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_vs_T.png", dpi=300)
plt.show()

# ===== Compute Thermal Expansion Coefficient =====
# Linear fit: V = V0 * (1 + αΔT)
x = df["Temperature(K)"]
y = df["Avg_Volume"]

slope, intercept, r, p, stderr = linregress(x, y)
alpha = slope / intercept  # Approximation near linear region

print(f"\n📈 Linear fit: V = {intercept:.3f} + {slope:.5f} * T")
print(f"✅ Approx. Thermal Expansion Coefficient α = {alpha:.4e} K⁻¹")
print(f"R² = {r**2:.4f}")

with open("thermal_expansion_results.txt", "w") as f:
    f.write(f"Linear Fit: V = {intercept:.3f} + {slope:.5f} * T\n")
    f.write(f"Thermal Expansion Coefficient (α): {alpha:.4e} K⁻¹\n")
    f.write(f"R²: {r**2:.4f}\n")

print("\n📂 Saved plots: volume_vs_T.png, energy_vs_T.png")
print("📄 Saved results: thermal_expansion_results.txt")
