import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# 1️⃣ Load dataset
# ============================
df = pd.read_csv("thermo_summary.csv")

print("\n📊 Dataset:")
print(df)

# Drop missing values
df = df.dropna(subset=["Avg_Press", "Avg_PotEng", "Temperature(K)"])

if df.empty:
    raise ValueError("❌ Dataset is empty after dropping NaN values. Check thermo_summary.csv")

# ============================
# 2️⃣ Prepare features & target
# ============================
X = df[["Temperature(K)"]]   # Feature (Temperature)
y = df["Avg_PotEng"]         # Target (Potential Energy)

# ============================
# 3️⃣ Train Linear Regression model
# ============================
model = LinearRegression()
model.fit(X, y)

# Predict on same data (since dataset is small)
y_pred = model.predict(X)

# ============================
# 4️⃣ Evaluate performance
# ============================
print("\n📈 Model Performance:")
try:
    r2 = r2_score(y, y_pred)
except Exception:
    r2 = np.nan
print(f"R² Score: {r2:.4f}" if not np.isnan(r2) else "R² Score: nan")

rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse:.4f}")

# ============================
# 5️⃣ Predict for new synthetic temperatures
# ============================
T_new = np.linspace(300, 1200, 10).reshape(-1, 1)
E_pred = model.predict(T_new)

pred_df = pd.DataFrame({
    "Temperature(K)": T_new.flatten(),
    "Predicted_PotEng": E_pred
})

print("\n📄 Predicted Potential Energy (synthetic range):")
print(pred_df)

# ============================
# 6️⃣ Plot the fit
# ============================
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=80)
plt.plot(T_new, E_pred, color='red', linestyle='--', label='Linear Fit (Predicted)')
plt.xlabel("Temperature (K)")
plt.ylabel("Potential Energy (eV)")
plt.title("Thermo ML Prediction — Potential Energy vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ml_thermo_fit.png", dpi=300)
plt.show()

print("\n✅ Plot saved as ml_thermo_fit.png")
