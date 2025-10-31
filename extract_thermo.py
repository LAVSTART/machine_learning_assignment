import pandas as pd
import re
import glob
from io import StringIO

def parse_lammps_log(filename):
    with open(filename, "r") as f:
        text = f.read()

    # Find all "Step ..." to "Loop time" blocks
    blocks = re.findall(r"(Step.*?Loop time)", text, flags=re.S)
    if not blocks:
        print(f"⚠️  No thermo blocks found in {filename}")
        return None

    all_dfs = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        header = lines[0].split()
        data = "\n".join(lines[1:-1])  # skip 'Loop time' line
        try:
            df = pd.read_csv(StringIO(data), sep=r"\s+", names=header, engine="python")
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Could not parse block in {filename}: {e}")
            continue

    if not all_dfs:
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    return df_all


data = []
for logfile in sorted(glob.glob("log_*.lammps")):
    temp_match = re.search(r"log_(\d+)K", logfile)
    if not temp_match:
        continue
    temp = int(temp_match.group(1))

    df = parse_lammps_log(logfile)
    if df is None:
        continue

    # Check which thermo columns exist
    cols = df.columns
    row = {"Temperature(K)": temp}
    for key in ["Temp", "Press", "PotEng", "Volume"]:
        if key in cols:
            row[f"Avg_{key}"] = df[key].mean()
        else:
            row[f"Avg_{key}"] = None

    data.append(row)

out = pd.DataFrame(data)
out.to_csv("thermo_summary.csv", index=False)
print("✅ Saved thermo_summary.csv")
print(out)
