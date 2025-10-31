import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_property(df, x, y, ylabel, title, filename):
    plt.figure(figsize=(8,5))
    plt.plot(df[x], df[y], lw=1.5)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LAMMPS thermo output")
    parser.add_argument("--file", required=True, help="Path to thermo.csv file")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.file)

    print(f"✅ Loaded {len(df)} records from {args.file}")
    print("Columns:", list(df.columns))

    # Common columns in LAMMPS output
    if "Step" in df.columns and "Temp" in df.columns:
        plot_property(df, "Step", "Temp", "Temperature (K)", "Temperature vs Step", f"{args.outdir}/temp_vs_step.png")

    if "Step" in df.columns and "PotEng" in df.columns:
        plot_property(df, "Step", "PotEng", "Potential Energy (eV)", "Potential Energy vs Step", f"{args.outdir}/poteng_vs_step.png")

    if "Step" in df.columns and "Volume" in df.columns:
        plot_property(df, "Step", "Volume", "Volume (Å³)", "Volume vs Step", f"{args.outdir}/volume_vs_step.png")

    if "Step" in df.columns and "Press" in df.columns:
        plot_property(df, "Step", "Press", "Pressure (bar)", "Pressure vs Step", f"{args.outdir}/press_vs_step.png")

    print(f"✅ Plots saved to '{args.outdir}' folder.")
