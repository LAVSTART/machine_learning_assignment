import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import os

# ===========================
# Load combined diffusion data
# ===========================
df = pd.read_csv("combined_D_summary.csv")  # file created earlier by combine_D_summaries.py

print("\n📊 Loaded Diffusion Dataset:")
print(df.head())

# Create directory for saving results
os.makedirs("rdf_results", exist_ok=True)

# Loop through each species (Ca, Ti, O)
results = []

for species in df["label"].unique():
    sub = df[df["label"] == species].sort_values("Temperature(K)")
    T = sub["Temperature(K)"].values.reshape(-1, 1)
    D = sub["D_m2_s"].values
    
    if len(T) < 2:
        print(f"⚠️ Skipping {species}: not enough data points")
        continue

    # Arrhenius form: ln(D) = ln(D0) - Ea/(kB*T)
    kB = 8.617333e-5  # eV/K
    invT = 1 / T
    lnD = np.log(D)

    lin_model = LinearRegression()
    lin_model.fit(invT, lnD)
    lnD_pred = lin_model.predict(invT)

    Ea = -lin_model.coef_[0] * kB
    D0 = np.exp(lin_model.intercept_)

    r2_lin = r2_score(lnD, lnD_pred)
    rmse_lin = np.sqrt(mean_squared_error(lnD, lnD_pred))

    # Polynomial regression on (T, D)
    poly = PolynomialFeatures(degree=2)
    T_poly = poly.fit_transform(T)
    poly_model = LinearRegression()
    poly_model.fit(T_poly, D)
    D_pred_poly = poly_model.predict(T_poly)
    r2_poly = r2_score(D, D_pred_poly)

    results.append([species, Ea, D0, r2_lin, r2_poly])

    # ----------------------
    # Plot both fits
    # ----------------------
    plt.figure(figsize=(7,5))
    plt.scatter(1/T, lnD, color='blue', label='ln(D) data', s=60)
    plt.plot(1/T, lnD_pred, color='red', label='Arrhenius Fit', lw=2)
    plt.xlabel('1/T (1/K)')
    plt.ylabel('ln(D)')
    plt.title(f'Arrhenius Fit for {species}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"rdf_results/Arrhenius_fit_{species}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(T, D, color='green', label='D data', s=60)
    T_pred = np.linspace(min(T), max(T), 100).reshape(-1, 1)
    D_pred = poly_model.predict(poly.transform(T_pred))
    plt.plot(T_pred, D_pred, color='orange', lw=2, label='Polynomial ML Fit')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Diffusion Coefficient (m²/s)')
    plt.title(f'Polynomial ML Fit for {species}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"rdf_results/Diffusion_poly_fit_{species}.png", dpi=300)
    plt.close()

# ===========================
# Save summary
# ===========================
res_df = pd.DataFrame(results, columns=["Species", "Ea_eV", "D0_m2_s", "R2_Arrhenius", "R2_Poly"])
res_df.to_csv("rdf_results/Diffusion_ML_summary.csv", index=False)

print("\n✅ ML Diffusion Modeling Complete.")
print(res_df)
print("\n📂 Saved plots in rdf_results/ folder.")
