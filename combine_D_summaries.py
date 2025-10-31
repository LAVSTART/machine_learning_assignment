# combine_D_summaries.py
import pandas as pd
import glob
import os
import re

# Match only folders that have temperature, like rdf_results_300K/
summary_files = sorted(glob.glob("rdf_results_*K/D_summary.csv"))

if not summary_files:
    print("❌ No temperature summary files found (expected folders like rdf_results_300K/).")
    exit()

all_data = []

for f in summary_files:
    # Extract numeric temperature from path using regex
    match = re.search(r"_(\d+)K", f)
    if not match:
        print(f"⚠️ Skipping {f} (no temperature found in name)")
        continue
    temp = int(match.group(1))

    df = pd.read_csv(f)
    df["Temperature(K)"] = temp
    all_data.append(df)

if not all_data:
    print("❌ No valid temperature data found.")
    exit()

combined = pd.concat(all_data)
combined = combined[["Temperature(K)", "label", "slope", "intercept", "D_A2_ps", "D_m2_s", "r"]]
combined.sort_values(by=["label", "Temperature(K)"], inplace=True)

outpath = "D_summary_all.csv"
combined.to_csv(outpath, index=False)
print(f"✅ Combined summary saved as {outpath}")

# Optional preview
print("\nPreview:")
print(combined.head(10))

# Optional plotting
import matplotlib.pyplot as plt
for label, group in combined.groupby("label"):
    plt.plot(group["Temperature(K)"], group["D_m2_s"], 'o-', label=label)

plt.xlabel("Temperature (K)")
plt.ylabel("Diffusion Coefficient D (m²/s)")
plt.title("Diffusion Coefficient vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Diffusion_vs_Temperature.png", dpi=300)
plt.show()
