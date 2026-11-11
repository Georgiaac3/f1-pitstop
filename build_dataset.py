"""Download every race of the chosen seasons from OpenF1 and save one lap table.

Usage:
    python build_dataset.py                       # 2023, 2024, 2025
    python build_dataset.py --years 2024 2025

Writes data/laps.pkl (one row per driver and lap, columns from openf1.FEATURES
plus LapStart, Circuit, Year and SessionKey). Races that fail to load are skipped and listed.
"""

import argparse
from pathlib import Path

import pandas as pd

from openf1 import load_session, race_sessions

OUT = Path(__file__).parent / "data" / "laps.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()

    tables, failed = [], []
    for year in args.years:
        sessions = race_sessions(year)
        for i, s in sessions.iterrows():
            label = f"{year} {s['location']}"
            try:
                laps = load_session(s)
            except Exception as e:  # missing endpoint data, API hiccup, etc.
                failed.append((label, repr(e)[:120]))
                print(f"  FAILED {label}: {e!r}"[:160], flush=True)
                continue
            laps["Circuit"] = s["circuit_short_name"]
            laps["Year"] = year
            laps["SessionKey"] = int(s["session_key"])
            tables.append(laps)
            print(f"{label}: {len(laps)} rows, {laps['Pit'].sum()} pits", flush=True)

    if not tables:  # every race failed: a code bug, not rate limiting; keep the old file
        raise SystemExit(f"no race loaded, {OUT} left unchanged; first error: {failed[0][1]}")
    OUT.parent.mkdir(exist_ok=True)
    df = pd.concat(tables, ignore_index=True)
    df.to_pickle(OUT)
    print(f"\nsaved {len(df)} rows from {df['Event'].nunique()} races to {OUT}")
    if failed:
        print("failed races:")
        for label, err in failed:
            print(f"  {label}: {err}")


if __name__ == "__main__":
    main()
