import pandas as pd
import argparse
import io

def parse_lammps_log(filename):
    with open(filename) as f:
        lines = f.readlines()

    start, end = None, None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step"):
            start = i
        elif line.strip().startswith("Loop time"):
            end = i
            break

    if start is None or end is None:
        raise ValueError("Could not find thermo data in log file.")

    header = lines[start].split()
    data = lines[start+1:end]
    data_str = "".join(data)
    df = pd.read_csv(io.StringIO(data_str), delim_whitespace=True, names=header)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse LAMMPS log file to CSV")
    parser.add_argument("--log", required=True, help="Path to log.lammps file")
    parser.add_argument("--out", required=True, help="Output CSV file name")
    args = parser.parse_args()

    df = parse_lammps_log(args.log)
    df.to_csv(args.out, index=False)
    print(f"✅ Parsed data saved to {args.out}")

