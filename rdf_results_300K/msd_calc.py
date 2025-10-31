import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def compute_msd(u, select="type 1", dt=1.0, outdir=".", label="msd"):
    """
    Compute Mean Square Displacement (MSD) for selected atoms
    """
    print(f"📈 Computing MSD for selection: {select} ({label})")

    atoms = u.select_atoms(select)
    n_atoms = len(atoms)

    if n_atoms == 0:
        print("❌ No atoms found for selection:", select)
        return

    n_frames = len(u.trajectory)
    ref_positions = atoms.positions.copy()

    msd = np.zeros(n_frames)

    for i, ts in enumerate(u.trajectory):
        displacement = atoms.positions - ref_positions
        squared_disp = np.sum(displacement ** 2, axis=1)
        msd[i] = np.mean(squared_disp)

    time = np.arange(n_frames) * dt

    # Save data
    outfile = os.path.join(outdir, f"{label}_msd.dat")
    np.savetxt(outfile, np.column_stack((time, msd)), header="Time(ps) MSD(A^2)")

    # Plot MSD
    plt.figure()
    plt.plot(time, msd, label=label, color='b')
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å²)")
    plt.title(f"MSD for {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_msd.png"))
    plt.close()

    print(f"✅ MSD data saved: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MSD from LAMMPS trajectory")
    parser.add_argument("-d", "--data", required=True, help="LAMMPS data file (e.g., catio3_supercell.data)")
    parser.add_argument("-t", "--traj", required=True, help="LAMMPS trajectory file (e.g., traj_prod.lammpstrj)")
    parser.add_argument("-s", "--select", default="type 1", help="Atom selection string (default: 'type 1')")
    parser.add_argument("-dt", "--timestep", type=float, default=1.0, help="Timestep between frames (ps)")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory")
    parser.add_argument("-l", "--label", default="msd", help="Label for output files")
    args = parser.parse_args()

    u = mda.Universe(args.data, args.traj, atom_style='id type charge x y z', format='LAMMPSDUMP')
    compute_msd(u, select=args.select, dt=args.timestep, outdir=args.outdir, label=args.label)
