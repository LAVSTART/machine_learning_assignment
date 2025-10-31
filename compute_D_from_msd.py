# compute_D_from_msd.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob, os, csv

def fit_linear_region(t, msd, tmin=None, tmax=None):
    # choose default region: skip first 10% and fit between 10% and 80% of range
    n = len(t)
    i0 = int(n * 0.10) if tmin is None else (np.abs(t - tmin)).argmin()
    i1 = int(n * 0.80) if tmax is None else (np.abs(t - tmax)).argmin()
    x = t[i0:i1]
    y = msd[i0:i1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, (i0, i1)

def process_file(fn, outdir):
    label = os.path.splitext(os.path.basename(fn))[0].replace("_msd","")
    t, msd = np.loadtxt(fn, unpack=True)
    slope, intercept, r, (i0,i1) = fit_linear_region(t, msd)

    # Einstein relation in 3D: MSD = 6 D t  => D = slope / 6
    D_A2_per_ps = slope / 6.0
    D_m2_per_s = D_A2_per_ps * 1e-8  # 1 Å^2/ps = 1e-8 m^2/s

    # Save text summary
    txt = f"{label} slope={slope:.6e} Å²/ps intercept={intercept:.6e} Å² D={D_A2_per_ps:.6e} Å²/ps = {D_m2_per_s:.6e} m²/s R={r:.4f}\n"
    print(txt)
    with open(os.path.join(outdir, f"{label}_D.txt"), "w") as fw:
        fw.write(txt)

    # Plot with fit
    plt.figure()
    plt.plot(t, msd, label=f"{label} MSD")
    xx = t[i0:i1]
    plt.plot(xx, intercept + slope*xx, "--", label="linear fit")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å²)")
    plt.title(f"{label} MSD and linear fit (slope={slope:.3e})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_msd_fit.png"), dpi=300)
    plt.close()

    return {"label": label, "slope": slope, "intercept": intercept,
            "D_A2_ps": D_A2_per_ps, "D_m2_s": D_m2_per_s, "r": r}

if __name__ == "__main__":
    # Find all *_msd.dat files recursively (handles 300K, 600K, 900K, etc.)
    all_files = sorted(glob.glob("rdf_results*/*_msd.dat"))

    # Group files by their parent folder
    from collections import defaultdict
    folder_map = defaultdict(list)
    for f in all_files:
        folder = os.path.dirname(f)
        folder_map[folder].append(f)

    for folder, files in folder_map.items():
        rows = []
        print(f"\n📁 Processing folder: {folder}")
        for f in files:
            rows.append(process_file(f, outdir=folder))
        # Write summary CSV in the same folder
        csv_path = os.path.join(folder, "D_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"✅ Wrote {csv_path}")
