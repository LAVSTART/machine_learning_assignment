import pandas as pd

labels = ['Ca', 'Ti', 'O']
data = {}

for label in labels:
    try:
        df = pd.read_csv(f"rdf_results_900K/{label}_msd.dat", comment='#', sep='\s+', names=['Time', 'MSD'])
        # Simple linear fit to find diffusion coefficient (D = slope / 6)
        fit = pd.Series(df['MSD']).rolling(20).mean().dropna()
        slope = (fit.iloc[-1] - fit.iloc[0]) / (df['Time'].iloc[-1] - df['Time'].iloc[0])
        D = slope / 6
        data[label] = D
    except Exception as e:
        print(f"⚠️ Could not process {label}: {e}")

# Save summary
pd.DataFrame(list(data.items()), columns=['Element', 'Diffusion_Coefficient']).to_csv('rdf_results_900K/D_summary.csv', index=False)
print("✅ D_summary.csv created successfully!")
