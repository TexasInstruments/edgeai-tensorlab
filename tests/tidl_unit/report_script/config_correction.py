#!/usr/bin/env python3
"""
clean_headers.py
----------------
• Recursively finds **all** .csv files below the current directory.
• For every file:
  1. Replace “_” with a space in each column name.
  2. Convert column names to lower-case.
  3. Drop columns whose (already-cleaned) name is
        - exactly “dataset number”
        - exactly “output shape”,  OR
        - ends with “file”  (e.g. “input file”, “somefile”, “abc file”)
• Saves the file **in-place** (no extra copy).  Make a backup first if you need one!
"""

from pathlib import Path
import pandas as pd

# ---------------- configuration ----------------
ROOT = Path("configs").resolve()          # current directory
CSV_GLOB = "*.csv"                  # match pattern
# ------------------------------------------------

DROP_EXACT = {"dataset number", "output shape"}
SUFFIX = "file"                     # 4-char suffix to match

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1 & 2: normalise headers
    df.columns = [c.replace("_", " ").lower() for c in df.columns]

    # Step 3: build list of columns to delete
    drop_cols = [
        c for c in df.columns
        if c in DROP_EXACT or c.endswith(SUFFIX)
    ]
    return df.drop(columns=drop_cols, errors="ignore")

def process_file(csv_path: Path) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[SKIP] {csv_path}: cannot read ({exc})")
        return

    df = clean_columns(df)

    try:
        df.to_csv(csv_path, index=False)
        print(f"[OK]   {csv_path}")
    except Exception as exc:
        print(f"[FAIL] {csv_path}: cannot write ({exc})")

def main() -> None:
    csv_files = list(ROOT.rglob(CSV_GLOB))
    if not csv_files:
        print("No CSV files found under", ROOT)
        return

    for csv in csv_files:
        process_file(csv)

if __name__ == "__main__":
    main()