import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("thermo_summary.csv")
df = df.dropna(subset=["Avg_PotEng"])  # clean any missing data

print("\n📊 Loaded Data:")
print(df)

# Prepare features and target
X = df[["Temperature(K)"]].values
y = df["Avg_PotEng"].values

# Polynomial regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predictions
X_pred = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_pred = model.predict(poly.transform(X_pred))

# Evaluate
y_fit = model.predict(X_poly)
r2 = r2_score(y, y_fit)
rmse = np.sqrt(mean_squared_error(y, y_fit))

print("\n📈 Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Equation
coef = model.coef_
intercept = model.intercept_
print(f"\n🔹 Equation: E = {intercept:.3f} + {coef[1]:.3f}*T + {coef[2]:.6f}*T²")

# Plot
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=70)
plt.plot(X_pred, y_pred, color='red', label='Polynomial Fit', linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel("Average Potential Energy (eV)")
plt.title("Polynomial Fit: Potential Energy vs Temperature")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rdf_results/ML_poly_thermo_fit.png", dpi=300)
plt.show()

print("\n✅ Saved polynomial ML plot as rdf_results/ML_poly_thermo_fit.png")
