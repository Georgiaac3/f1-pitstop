"""Train on 2023-24, evaluate off-policy on 2025, write results/ for the dashboard.

Usage:
    python run_experiment.py              # main run + sensitivity analyses
    python run_experiment.py --quick      # skip sensitivity analyses
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from policy import (TREE_FEATURES, PolicyTree, bootstrap_gain, build_rows,
                    estimate_pit_loss, estimate_stop_cost, nuisance_predictions, policy_value, rewards)

DATA = Path(__file__).parent / "data" / "laps.pkl"
OUT = Path(__file__).parent / "results"
TRAIN_YEARS, TEST_YEAR = [2023, 2024], 2025
STREET = {"Monte Carlo", "Baku", "Singapore", "Jeddah", "Las Vegas", "Miami", "Melbourne"}


def prepare(laps, K=10, stop_debt=True, trim=0.01, clip=0.01, drop_sc=False,
            debt_pitlane=False, reference="local"):
    """Rows, cross-fitted nuisances and rewards for train and test."""
    train_laps = laps[laps["Year"].isin(TRAIN_YEARS)]
    # Circuit constants are nuisances too: estimate them on training races only.
    pit_loss = estimate_pit_loss(train_laps)
    stop_cost = estimate_stop_cost(build_rows(train_laps, K=K, stop_debt=False,
                                              pit_loss=pit_loss, stop_cost=({}, 0.0),
                                              reference=reference))
    if debt_pitlane:  # first attempt: charge the full pit-lane loss (biased to pitting)
        stop_cost = pit_loss
    rows = build_rows(laps, K=K, stop_debt=stop_debt, pit_loss=pit_loss, stop_cost=stop_cost,
                      reference=reference)
    if drop_sc:
        rows = rows[(rows["SafetyCar"] == 0) & (rows["VSC"] == 0)]
    train = rows[rows["Year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = rows[rows["Year"] == TEST_YEAR].reset_index(drop=True)
    train, test = nuisance_predictions(train, test)
    info = {"rows_train": len(train), "rows_test": len(test)}
    # Common support: drop rows whose pit propensity is essentially 0 or 1.
    keep_tr = train["p1"].between(trim, 1 - trim)
    keep_te = test["p1"].between(trim, 1 - trim)
    info.update(dropped_train=int((~keep_tr).sum()), dropped_test=int((~keep_te).sum()),
                dropped_pits_train=int(train.loc[~keep_tr, "T"].sum()),
                dropped_pits_test=int(test.loc[~keep_te, "T"].sum()))
    train, test = train[keep_tr].reset_index(drop=True), test[keep_te].reset_index(drop=True)
    for df in (train, test):
        dm, dr = rewards(df, clip=clip)
        df[["dm0", "dm1"]], df[["dr0", "dr1"]] = dm, dr
    info["stop_cost_overall"] = stop_cost[1]
    return train, test, info, pit_loss


def gamma(df, kind="dr"):
    return df[[f"{kind}0", f"{kind}1"]].to_numpy()


def tau(df, kind="dr"):
    return (df[f"{kind}1"] - df[f"{kind}0"]).to_numpy()


def X(df):
    return df[TREE_FEATURES].to_numpy(dtype=float)


def fit_threshold(train):
    """Pit when tyre age >= a; a tuned on training DR rewards."""
    t = tau(train)
    grid = range(5, 51)
    vals = [t[train["TyreLife"].to_numpy() >= a].sum() for a in grid]
    return list(grid)[int(np.argmax(vals))]


def cv_depth(train, depths=(1, 2, 3), n_folds=5):
    """Grouped (by race) CV of the DR tree's value; Gamma is already cross-fitted."""
    scores = {d: [] for d in depths}
    for fit_idx, val_idx in GroupKFold(n_splits=n_folds).split(train, groups=train["Event"]):
        tr, va = train.iloc[fit_idx], train.iloc[val_idx]
        base = va["y"].mean()
        for d in depths:
            tree = PolicyTree(depth=d).fit(X(tr), tau(tr))
            scores[d].append(policy_value(gamma(va), tree.predict(X(va))).mean() - base)
    return {d: float(np.mean(s)) for d, s in scores.items()}


def evaluate(test, actions):
    """DR gain over historical behaviour on the test races, with race-bootstrap CI."""
    per_row = policy_value(gamma(test), actions) - test["y"].to_numpy()
    est, lo, hi = bootstrap_gain(per_row, test["Event"].to_numpy())
    return {"gain": est, "lo": lo, "hi": hi, "pit_rate": float(np.mean(actions))}


def fit_policies(train, depth):
    pol = {}
    pol["Never pit"] = lambda df: np.zeros(len(df), dtype=int)
    a = fit_threshold(train)
    pol[f"Threshold rule (tyre age >= {a})"] = lambda df: (df["TyreLife"] >= a).to_numpy().astype(int)
    trees = {}
    for d in (1, 2, 3):
        trees[("dr", d)] = PolicyTree(depth=d).fit(X(train), tau(train))
    trees[("dm", depth)] = PolicyTree(depth=depth).fit(X(train), tau(train, "dm"))
    pol[f"DM tree (depth {depth})"] = lambda df: trees[("dm", depth)].predict(X(df))
    for d in (1, 2, 3):
        pol[f"DR tree (depth {d})"] = (lambda dd: lambda df: trees[("dr", dd)].predict(X(df)))(d)
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                        min_samples_leaf=100, random_state=0)
    gbm.fit(X(train), tau(train))
    pol["Black box (gradient boosting on Γ)"] = lambda df: (gbm.predict(X(df)) > 0).astype(int)
    return pol, trees, a


def sanity_checks(train, test, laps):
    out = {}
    # Nuisance quality on held-out races.
    out["propensity_auc_test"] = float(roc_auc_score(test["T"], test["p1"]))
    out["propensity_brier_test"] = float(brier_score_loss(test["T"], test["p1"]))
    out["pit_rate_test"] = float(test["T"].mean())
    for t in (0, 1):
        m = test["T"] == t
        out[f"outcome_r2_test_T{t}"] = float(r2_score(test.loc[m, "y"], test.loc[m, f"f{t}"]))
    # Known fact 1: a stop under SC/VSC is cheaper (DM and DR effect of pitting).
    neutral = (test["SafetyCar"] + test["VSC"]) > 0
    out["effect_by_track_status"] = [
        {"status": name, "rows": int(m.sum()), "pits": int(test.loc[m, "T"].sum()),
         "dm_effect": float(tau(test[m], "dm").mean()), "dr_effect": float(tau(test[m]).mean())}
        for name, m in [("Green flag", ~neutral), ("Safety car / VSC", neutral)]]
    # Known fact 2: softer compounds degrade faster (within-stint lap-time slope).
    clean = laps[(laps["Pit"] == 0) & (laps["PitOutLap"] == 0) & (laps["SafetyCar"] == 0)
                 & (laps["VSC"] == 0) & (laps["LapNumber"] > 1) & (laps["Rainfall"] == 0)
                 & laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    clean = clean[clean["LapTime"] < 1.07 * clean.groupby("Event")["LapTime"].transform("median")]
    slopes = []
    for (ev, drv, st), g in clean.groupby(["Event", "Driver", "Stint"]):
        if len(g) >= 8:
            slopes.append((g["Compound"].iloc[0], np.polyfit(g["TyreLife"], g["LapTime"], 1)[0]))
    s = pd.DataFrame(slopes, columns=["Compound", "Slope"])
    out["degradation"] = [{"compound": c, "stints": int(len(g)),
                           "median_slope_s_per_lap": float(g["Slope"].median())}
                          for c, g in s.groupby("Compound")]
    # Effect of pitting by tyre age (DR), the core dose-response.
    bins = pd.cut(test["TyreLife"], [4, 10, 15, 20, 25, 30, 40, 80])
    out["effect_by_tyre_age"] = [
        {"tyre_age": str(b), "rows": int(len(g)), "pits": int(g["T"].sum()),
         "dr_effect": float(tau(g).mean()), "dm_effect": float(tau(g, "dm").mean())}
        for b, g in test.groupby(bins, observed=True)]
    return out


def overlap_table(df):
    """Propensity distribution by action and race phase (for the overlap plot)."""
    phase = pd.cut(df["RaceFraction"], [0, 0.33, 0.67, 1.0], labels=["Early", "Middle", "Late"])
    b = pd.cut(df["p1"], np.linspace(0, 1, 41))
    t = (df.assign(Phase=phase, Bin=b.apply(lambda x: round(x.mid, 4)),
                   Action=df["T"].map({0: "Stayed out", 1: "Pitted"}))
         .groupby(["Phase", "Action", "Bin"], observed=True).size().rename("Rows").reset_index())
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--K", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    laps = pd.read_pickle(DATA)
    laps = laps[laps["Year"].isin(TRAIN_YEARS + [TEST_YEAR])]
    print(f"laps: {len(laps)} from {laps['Event'].nunique()} races")

    # ---- Main run -----------------------------------------------------------
    train_all, test_all, info, pit_loss = prepare(laps, K=args.K, trim=0.0)
    overlap = overlap_table(pd.concat([train_all.assign(Split="train"), test_all.assign(Split="test")]))
    overlap.to_csv(OUT / "overlap.csv", index=False)

    train, test, info, _ = prepare(laps, K=args.K)
    print(json.dumps(info))
    cv = cv_depth(train)
    depth = max(cv, key=cv.get)
    print("CV gain by depth:", cv, "-> depth", depth)

    policies, trees, threshold = fit_policies(train, depth)
    table = []
    for name, pol in policies.items():
        r = evaluate(test, pol(test))
        r["policy"] = name
        r["train_gain"] = float((policy_value(gamma(train), pol(train)) - train["y"]).mean())
        table.append(r)
        print(f"{name:40s} test gain {r['gain']:+.2f} s [{r['lo']:+.2f}, {r['hi']:+.2f}]  "
              f"pit rate {r['pit_rate']:.3f}")
    table.insert(0, {"policy": "Historical (teams)", "gain": 0.0, "lo": 0.0, "hi": 0.0,
                     "pit_rate": float(test["T"].mean()), "train_gain": 0.0})

    best = trees[("dr", depth)]
    test["TreeAction"] = best.predict(X(test))
    test["Street"] = test["Circuit"].isin(STREET)
    per_row = policy_value(gamma(test), test["TreeAction"].to_numpy()) - test["y"]
    by_race = (test.assign(G=per_row).groupby(["Event", "Circuit", "Street"])
               .agg(gain=("G", "mean"), rows=("G", "size"), pits=("T", "sum"),
                    tree_pits=("TreeAction", "sum")).reset_index())
    by_race.to_csv(OUT / "by_race.csv", index=False)
    by_type = []
    for street, g in test.assign(G=per_row).groupby("Street"):
        est, lo, hi = bootstrap_gain(g["G"].to_numpy(), g["Event"].to_numpy())
        by_type.append({"type": "Street" if street else "Permanent", "gain": est, "lo": lo,
                        "hi": hi, "races": int(g["Event"].nunique())})

    keep = ["Event", "Circuit", "Driver", "Team", "LapNumber", "TotalLaps", "Compound",
            "TyreLife", "Position", "GapAhead", "GapBehind", "SafetyCar", "VSC", "T",
            "TreeAction", "p1", "f0", "f1", "y"]
    test.assign(tau_dr=tau(test))[keep + ["tau_dr"]].to_csv(OUT / "test_rows.csv", index=False)

    results = {
        "setup": {"K": args.K, "train_years": TRAIN_YEARS, "test_year": TEST_YEAR,
                  "trim": 0.01, "clip": 0.01, "stop_debt": True,
                  "races_train": int(train["Event"].nunique()),
                  "races_test": int(test["Event"].nunique()),
                  "pit_loss_by_circuit": pit_loss[0], "pit_loss_overall": pit_loss[1], **info},
        "cv_by_depth": cv, "depth": depth, "threshold": threshold,
        "tree": best.tree_, "rules": best.rules(),
        "trees_by_depth": {d: trees[("dr", d)].rules() for d in (1, 2, 3)},
        "dm_tree_rules": trees[("dm", depth)].rules(),
        "policies": table, "by_type": by_type,
        "sanity": sanity_checks(train, test, laps),
    }

    # ---- Sensitivity (depth-2 DR tree, full pipeline re-run each time) -------
    if not args.quick:
        sens = []
        def run(label, setting, **kw):
            tr, te, inf, _ = prepare(laps, **kw)
            tree = PolicyTree(depth=2).fit(X(tr), tau(tr))
            r = evaluate(te, tree.predict(X(te)))
            r.update(analysis=label, setting=setting, rows_test=len(te),
                     dropped_test=inf["dropped_test"], rule=" | ".join(
                         f"{x['rule']} → {x['action']}" for x in tree.rules()))
            print(f"[{label}={setting}] gain {r['gain']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
            sens.append(r)
        for K in (5, 10, 15, 20):
            run("Horizon K (laps)", str(K), K=K)
        for c in (0.005, 0.02, 0.05):
            run("Weight clip (no trimming)", str(c), trim=0.0, clip=c)
        run("Safety-car rows", "excluded", drop_sc=True)
        run("Outcome reference", "whole-field median", reference="field")
        run("Stop debt", "off (pure K-lap outcome)", stop_debt=False)
        run("Stop debt", "full pit-lane loss (biased)", debt_pitlane=True)
        results["sensitivity"] = sens

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT}/results.json")
    print("Chosen tree:")
    for r in best.rules():
        print(f"  if {r['rule']}: {r['action']}  (n={r['n']})")


if __name__ == "__main__":
    main()
