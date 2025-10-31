import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def compute_rdf(u, type1, type2, outdir, label):
    print(f"📊 Computing RDF between type {type1} and {type2} ({label})...")

    atoms1 = u.select_atoms(f"type {type1}")
    atoms2 = u.select_atoms(f"type {type2}")

    if len(atoms1) == 0 or len(atoms2) == 0:
        print(f"⚠️ Warning: One of the atom selections is empty! Skipping {label}.")
        return

    rdf_analysis = rdf.InterRDF(atoms1, atoms2, range=(0.0, 10.0), nbins=200)
    rdf_analysis.run()

    r = rdf_analysis.results.bins
    g_r = rdf_analysis.results.rdf

    np.savetxt(f"{outdir}/rdf_{label}.dat", np.column_stack((r, g_r)),
               header="r (Å)\tg(r)")

    plt.figure(figsize=(6, 4))
    plt.plot(r, g_r, label=f"{label}", linewidth=2)
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/rdf_{label}.png", dpi=300)
    plt.close()

    print(f"✅ RDF saved to {outdir}/rdf_{label}.dat and rdf_{label}.png\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RDF analysis for LAMMPS trajectory")
    parser.add_argument("--traj", required=True, help="LAMMPS trajectory file")
    parser.add_argument("--data", required=True, help="LAMMPS data file")
    parser.add_argument("--outdir", default="rdf_results", help="Output directory for RDF data and plots")
    args = parser.parse_args()

    print("📂 Loading trajectory...")

    os.makedirs(args.outdir, exist_ok=True)

    # ✅ OVITO-generated file looks like 'id type charge x y z'
    atom_style = "id type charge x y z"

    u = mda.Universe(
        args.data,
        args.traj,
        topology_format="DATA",
        format="LAMMPSDUMP",
        atom_style=atom_style
    )

    print(f"📘 Parsed {len(u.atoms)} atoms successfully.")

    compute_rdf(u, 1, 3, args.outdir, "Ca_O")
    compute_rdf(u, 2, 3, args.outdir, "Ti_O")

    print(f"🎉 RDF analysis complete! Results saved in: {args.outdir}")

