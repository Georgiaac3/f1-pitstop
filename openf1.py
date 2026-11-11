"""Load historical race data from the OpenF1 API into one row per (driver, lap).

OpenF1 (https://openf1.org) is free for historical data from 2023 onward and
needs no account. Responses are cached in ./cache/openf1, so each race is
downloaded once.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.openf1.org/v1"
CACHE_DIR = Path(__file__).parent / "cache" / "openf1"
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Every column of the lap table: (role in the proposal, description).
FEATURES = {
    "Event":       ("id",      "Race (year and location)"),
    "Driver":      ("id",      "Driver code, e.g. VER"),
    "Team":        ("context", "Team name"),
    "LapNumber":   ("context", "Lap number, starting at 1"),
    "TotalLaps":   ("context", "Laps completed by the winner"),
    "RaceFraction": ("context", "LapNumber / TotalLaps"),
    "Position":    ("context", "Running position at the end of the lap"),
    "GapAhead":    ("context", "Seconds to the car ahead at the end of the lap"),
    "GapBehind":   ("context", "Seconds to the car behind at the end of the lap"),
    "Compound":    ("context", "Tyre compound on this lap"),
    "Stint":       ("context", "Stint number (1 = starting tyres)"),
    "TyreLife":    ("context", "Laps on this set of tyres, incl. laps from earlier sessions"),
    "SafetyCar":   ("context", "1 if the safety car was out during the lap"),
    "VSC":         ("context", "1 if a virtual safety car was active during the lap"),
    "AirTemp":     ("context", "Air temperature, °C"),
    "TrackTemp":   ("context", "Track temperature, °C"),
    "Rainfall":    ("context", "1 if rain was reported during the lap"),
    "Pit":         ("action",  "1 if the driver pitted at the end of this lap"),
    "PitOutLap":   ("outcome", "1 if the lap started in the pit lane (slow lap)"),
    "LapTime":     ("outcome", "Lap time, seconds"),
    "Sector1":     ("outcome", "Sector 1 time, seconds"),
    "Sector2":     ("outcome", "Sector 2 time, seconds"),
    "Sector3":     ("outcome", "Sector 3 time, seconds"),
}


def get(endpoint, **params):
    """GET an OpenF1 endpoint as a DataFrame, cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = endpoint + "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))

    for attempt in range(5):
        try:
            resp = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429:  # rate limited: back off and retry
            time.sleep(2 ** attempt)
            continue
        break
    if resp.status_code == 404:  # OpenF1 answers 404 for "no results"
        data = []
    else:
        resp.raise_for_status()
        data = resp.json()
    # Write then rename, so an interrupted run never leaves a truncated cache file.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    time.sleep(0.4)  # stay under the free rate limit
    return pd.DataFrame(data)


def race_sessions(year):
    """All (non-cancelled) races of a season, in calendar order."""
    sessions = get("sessions", year=year, session_name="Race")
    sessions = sessions[~sessions["is_cancelled"].fillna(False).astype(bool)]
    sessions = sessions.sort_values("date_start").reset_index(drop=True)
    sessions["label"] = sessions["country_name"] + " – " + sessions["location"]
    return sessions


def find_race(year, name):
    sessions = race_sessions(year)
    name = name.lower()
    match = sessions[
        sessions["country_name"].str.lower().str.contains(name)
        | sessions["location"].str.lower().str.contains(name)
        | sessions["circuit_short_name"].str.lower().str.contains(name)
    ]
    if match.empty:
        raise ValueError(f"no {year} race matching '{name}'")
    return match.iloc[0]


def track_status_by_lap(race_control, total_laps):
    """Per-lap safety car / VSC flags (arrays indexed by lap) from race-control messages.

    A period covers the laps from its deployment to its end, inclusive. Besides the
    explicit "SAFETY CAR IN THIS LAP" / "VIRTUAL SAFETY CAR ENDING", a period also
    ends at a red flag, a VSC ends when it is upgraded to a full safety car, and a
    period still open at the flag runs to the last lap. (Without these rules about
    one race in seven loses SC/VSC laps, e.g. 2023 Melbourne's two red-flagged SCs.)
    """
    flags = {"sc": np.zeros(total_laps + 1, dtype=int),
             "vsc": np.zeros(total_laps + 1, dtype=int)}
    if race_control.empty:
        return flags["sc"], flags["vsc"]

    start = {"sc": None, "vsc": None}

    def open_(kind, lap):
        if start[kind] is None:  # a repeated "DEPLOYED" keeps the earlier start
            start[kind] = lap

    def close(kind, lap):
        if start[kind] is not None:
            flags[kind][start[kind]:lap + 1] = 1
            start[kind] = None

    msgs = race_control[race_control["category"].isin(["SafetyCar", "Flag"])
                        & race_control["lap_number"].notna()].sort_values("date")
    for text, lap in zip(msgs["message"].str.upper(), msgs["lap_number"].astype(int)):
        lap = min(lap, total_laps)
        if "VIRTUAL SAFETY CAR DEPLOYED" in text:
            open_("vsc", lap)
        elif "VIRTUAL SAFETY CAR ENDING" in text:
            close("vsc", lap)
        elif "SAFETY CAR DEPLOYED" in text:
            close("vsc", lap)  # VSC upgraded to a full safety car
            open_("sc", lap)
        elif "SAFETY CAR IN THIS LAP" in text:
            close("sc", lap)
        elif "RED FLAG" in text:
            close("sc", lap)
            close("vsc", lap)
    close("sc", total_laps)
    close("vsc", total_laps)
    return flags["sc"], flags["vsc"]


def _tyres_by_lap(stints):
    """Compound, tyre age and pit decision per (driver, lap) from the stint table.

    A pit stop happens at the end of the last lap of every stint but the final one.
    """
    rows = []
    last = stints.groupby("driver_number")["stint_number"].transform("max")
    for (_, s), last_stint in zip(stints.iterrows(), last):
        if pd.isna(s["lap_start"]) or pd.isna(s["lap_end"]):
            continue
        for lap in range(int(s["lap_start"]), int(s["lap_end"]) + 1):
            rows.append({
                "driver_number": s["driver_number"], "LapNumber": lap,
                "Stint": s["stint_number"], "Compound": s["compound"],
                "TyreLife": s["tyre_age_at_start"] + lap - s["lap_start"],
                "Pit": int(lap == s["lap_end"] and s["stint_number"] != last_stint),
            })
    return pd.DataFrame(rows).drop_duplicates(["driver_number", "LapNumber"], keep="last")


def _asof(laps, table, value_col, out, by="driver_number"):
    """Attach the latest value of `value_col` before each lap's end time."""
    if table.empty:
        laps[out] = np.nan
        return laps
    t = table.copy()
    # Match LapEnd's resolution: pandas >= 3 may parse the two at different units
    # (ns vs us), and merge_asof refuses mismatched keys.
    t["date"] = pd.to_datetime(t["date"], format="ISO8601").astype(laps["LapEnd"].dtype)
    t[out] = pd.to_numeric(t[value_col], errors="coerce")  # "+1 LAP" -> NaN
    cols = ["date", out] + ([by] if by else [])
    return pd.merge_asof(laps.sort_values("LapEnd"), t[cols].sort_values("date"),
                         left_on="LapEnd", right_on="date", by=by,
                         direction="backward").drop(columns="date")


def load_session(session):
    """One row per (driver, lap) for a race session (a row of race_sessions())."""
    key = int(session["session_key"])
    laps = get("laps", session_key=key)
    stints = get("stints", session_key=key)
    drivers = get("drivers", session_key=key)
    positions = get("position", session_key=key)
    intervals = get("intervals", session_key=key)
    race_control = get("race_control", session_key=key)
    weather = get("weather", session_key=key)

    laps = laps.rename(columns={
        "lap_number": "LapNumber", "lap_duration": "LapTime",
        "duration_sector_1": "Sector1", "duration_sector_2": "Sector2",
        "duration_sector_3": "Sector3", "is_pit_out_lap": "PitOutLap"})
    laps["LapStart"] = pd.to_datetime(laps["date_start"], format="ISO8601")
    laps["LapEnd"] = laps["LapStart"] + pd.to_timedelta(laps["LapTime"], unit="s")
    laps = laps.dropna(subset=["LapEnd"])
    total_laps = int(laps["LapNumber"].max())

    laps = laps.merge(_tyres_by_lap(stints), on=["driver_number", "LapNumber"], how="left")
    laps = _asof(laps, positions, "position", "Position")
    laps = _asof(laps, intervals, "interval", "GapAhead")
    laps = _asof(laps, weather, "air_temperature", "AirTemp", by=None)
    laps = _asof(laps, weather, "track_temperature", "TrackTemp", by=None)
    laps = _asof(laps, weather, "rainfall", "Rainfall", by=None)

    # Gap behind = gap ahead of the car one position back, on the same lap. Position
    # feeds can briefly give two cars the same position; keep one so the merge
    # cannot duplicate laps.
    behind = (laps[["LapNumber", "Position", "GapAhead"]].dropna(subset=["Position"])
              .drop_duplicates(["LapNumber", "Position"]))
    behind["Position"] -= 1
    laps = laps.merge(behind.rename(columns={"GapAhead": "GapBehind"}),
                      on=["LapNumber", "Position"], how="left")

    names = drivers[["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")
    laps = laps.merge(names, on="driver_number", how="left").rename(
        columns={"name_acronym": "Driver", "team_name": "Team"})

    sc, vsc = track_status_by_lap(race_control, total_laps)
    lap_idx = laps["LapNumber"].clip(upper=total_laps).astype(int)
    laps["SafetyCar"], laps["VSC"] = sc[lap_idx], vsc[lap_idx]
    laps["Event"] = f"{session['year']} {session['location']}"
    laps["TotalLaps"] = total_laps
    laps["RaceFraction"] = laps["LapNumber"] / total_laps
    laps["Rainfall"] = (laps["Rainfall"] > 0).astype(int)
    laps["PitOutLap"] = laps["PitOutLap"].astype("boolean").fillna(False).astype(int)
    laps["Pit"] = laps["Pit"].fillna(0).astype(int)

    # LapStart is not a feature but the outcome (elapsed time over K laps) needs it.
    return (laps[list(FEATURES) + ["LapStart"]]
            .sort_values(["Driver", "LapNumber"]).reset_index(drop=True))


def load_race(year, name):
    return load_session(find_race(year, name))
