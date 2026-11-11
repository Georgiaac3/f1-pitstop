"""Download every race of the chosen seasons from OpenF1 and save one lap table.

Usage:
    python build_dataset.py                       # 2023, 2024, 2025
    python build_dataset.py --years 2024 2025

Writes data/laps.pkl (one row per driver and lap, columns from openf1.FEATURES
plus Circuit, Year and LapStart). Races that fail to load are skipped and listed.
"""

import argparse
from pathlib import Path

import pandas as pd

from openf1 import get, load_session, race_sessions

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
            # Lap start times are needed for the outcome; load_session drops them.
            raw = get("laps", session_key=int(s["session_key"]))
            names = get("drivers", session_key=int(s["session_key"]))
            starts = (raw[["driver_number", "lap_number", "date_start"]]
                      .merge(names[["driver_number", "name_acronym"]]
                             .drop_duplicates("driver_number"), on="driver_number")
                      .rename(columns={"name_acronym": "Driver", "lap_number": "LapNumber"}))
            starts["LapStart"] = pd.to_datetime(starts["date_start"], format="ISO8601")
            laps = laps.merge(starts[["Driver", "LapNumber", "LapStart"]]
                              .drop_duplicates(["Driver", "LapNumber"]),
                              on=["Driver", "LapNumber"], how="left")
            laps["Circuit"] = s["circuit_short_name"]
            laps["Year"] = year
            laps["SessionKey"] = int(s["session_key"])
            tables.append(laps)
            print(f"{label}: {len(laps)} rows, {laps['Pit'].sum()} pits", flush=True)

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
