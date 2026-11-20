"""P1 study: make the analysis credible (see report, "Next steps", P1).

  1. Tree features without circuit stand-ins: PitLoss and TrackTemp stay in the
     nuisance models only (policy.TREE_FEATURES_P1). Depth is reselected by CV.
  2. DryStint (index among dry stints) replaces Stint, so a wet start does not shift it.
     The executable evaluation's strata use it too.
  3. Pit-lap outcome model: T-learner (v1, P0) vs S-learner vs "offset" learner
     (f1 = f0 + a shallow model of the pit rows' residual), chosen on cross-fitted
     training metrics, then checked on test: R^2 of f1, SC sign under DM, and the
     DM-vs-DR gap (calibration of f1).
  4. Driver effects in the nuisances (outcome models, propensity), each kept only if it
     improves the cross-fitted fit on the training races.
  5. Nested cross-fitting in model selection. In P0 the tree's training rewards came
     from nuisances cross-fitted over *all* training races, so for CV fold k they were
     built by models that had seen fold k's races. Here, for each fold k, the nuisances
     are cross-fitted on the other folds only, and fold k's validation stints are
     weighted by a propensity fit without fold k. P0-style ("plain") CV is kept for
     comparison: the difference is the optimism of the old selection.
  6. Second holdout: train 2023, test 2024, next to train 2023-24, test 2025.

Evaluation as in P0: executable value (one stop per stint), rest-of-race outcome,
grace 2 laps, stint weights truncated at 20. Writes results_p1/.

Usage:
    python run_p1.py            # full run
    python run_p1.py --quick    # small grid, 10 bootstrap resamples (smoke test)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from policy import (NUISANCE_FEATURES_P1, TREE_FEATURES, TREE_FEATURES_P1, NuisanceSpec,
                    PolicyTree, nuisance_predictions, rewards)
from run_experiment import DATA, TRIM, build_split, trim_support
from run_p0 import GRACE, N_FOLDS, SEED, W_MAX, ci, fit_threshold_exec, literals, \
    race_folds, resample_races, rule_mask, rule_str
from stopping import Stints

OUT = Path(__file__).parent / "results_p1"
HOLDOUTS = {"2023-24 -> 2025": ([2023, 2024], 2025), "2023 -> 2024": ([2023], 2024)}
OUTCOMES = {"K10": 10, "race": "race"}
FEATURE_SETS = {"P1": TREE_FEATURES_P1, "v1": TREE_FEATURES}
STRATUM = "DryStint"


def X(df, feats):
    return df[list(feats)].to_numpy(dtype=float)


def stints(rows, w_max=W_MAX):
    return Stints(rows, w_max=w_max, stratum=STRATUM)


def tau_dr(df, clip):
    _, dr = rewards(df, clip=clip)
    return dr[:, 1] - dr[:, 0]


# --------------------------------------------------------------------------
# Nuisance models: candidates and diagnostics
# --------------------------------------------------------------------------

def p1_spec(outcome, driver_outcome, driver_prop):
    return NuisanceSpec(features=tuple(NUISANCE_FEATURES_P1),
                        categorical=("Team", "Circuit") + (("Driver",) if driver_outcome else ()),
                        outcome=outcome, propensity="gbm",
                        prop_categorical=("Driver",) if driver_prop else ())


LEARNERS = {"t": "T-learner", "s": "S-learner", "offset": "offset"}


def candidate_specs():
    specs = {"v1: logit, T-learner, v1 features": NuisanceSpec(),
             "P0: gbm, T-learner, v1 features": NuisanceSpec(propensity="gbm")}
    for o, name in LEARNERS.items():
        for drv in (False, True):
            specs[f"P1: {name}{' + driver' if drv else ''}"] = p1_spec(o, drv, drv)
    return specs


def diagnostics(train, test, race):
    """train: cross-fitted predictions; test: from the all-train fit. Untrimmed rows."""
    out = {}
    for split, df in (("cv", train), ("test", test)):
        T, y = df["T"].to_numpy(), df["y"].to_numpy()
        f0, f1, p = df["f0"].to_numpy(), df["f1"].to_numpy(), df["p1"].to_numpy()
        out[f"r2_f0_{split}"] = float(r2_score(y[T == 0], f0[T == 0]))
        out[f"r2_f1_{split}"] = float(r2_score(y[T == 1], f1[T == 1]))
        out[f"f1_bias_{split}"] = float((y[T == 1] - f1[T == 1]).mean())
        out[f"logloss_{split}"] = float(log_loss(T, np.clip(p, 1e-6, 1 - 1e-6)))
        out[f"auc_{split}"] = float(roc_auc_score(T, p))
        dm, dr = rewards(df, clip=0.01)
        tdm, tdr = dm[:, 1] - dm[:, 0], dr[:, 1] - dr[:, 0]
        sc = ((df["SafetyCar"] + df["VSC"]) > 0).to_numpy()
        # Known fact: pitting is cheaper under SC/VSC, so this should be > 0.
        out[f"sc_dm_{split}"] = float(tdm[sc].mean() - tdm[~sc].mean())
        out[f"sc_dr_{split}"] = float(tdr[sc].mean() - tdr[~sc].mean())
        # Calibration of the outcome model against DR: mean DR - DM pit effect.
        out[f"dr_minus_dm_{split}"] = float(tdr.mean() - tdm.mean())
        if race:  # propensity calibration over whole stints (report, Table "meanw")
            st = stints(df, w_max=None)
            for a in (15, 20, 25):
                r = st.evaluate((df["TyreLife"] >= a).to_numpy(), n_boot=1)
                out[f"mean_w_a{a}_{split}"] = r["mean_w"]
    return out


def choose_spec(diag):
    """Pick the P1 nuisance spec on cross-fitted *training* metrics only.
    Outcome learner and driver-in-outcome: highest mean (R^2 f0 + R^2 f1) over the two
    outcomes. Driver-in-propensity: lower cross-fitted log-loss on the race rows."""
    d = pd.DataFrame(diag)
    d = d[d["spec"].str.startswith("P1")]
    score = (d["r2_f0_cv"] + d["r2_f1_cv"]).groupby(d["spec"]).mean()
    best = score.idxmax()
    learner = next(o for o, n in LEARNERS.items() if n in best)
    drv_out = "driver" in best
    race = d[d["outcome"] == "race"].set_index("spec")["logloss_cv"]
    drv_prop = bool(race["P1: T-learner + driver"] < race["P1: T-learner"])
    return p1_spec(learner, drv_out, drv_prop), {
        "outcome_score_by_spec": score.to_dict(), "chosen_outcome": best,
        "driver_in_propensity": drv_prop,
        "race_logloss_cv": race[["P1: T-learner", "P1: T-learner + driver"]].to_dict()}


# --------------------------------------------------------------------------
# Tree selection by CV (nested and plain)
# --------------------------------------------------------------------------

def _grid_task(Xtr, tau, pits, Xval, st_val, mins, depths):
    out = []
    for mp in mins:
        for d in depths:
            tree = PolicyTree(depth=d, min_leaf_pits=mp).fit(Xtr, tau, pits=pits)
            out.append((mp, d, st_val.value(tree.predict(Xval), GRACE)[0]))
    return out


def cv_trees(rows, D, spec, folds, clips, mins, depths, n_jobs):
    """rows: {outcome: untrimmed training rows without nuisances}; D: full cross-fit.
    Returns one record per (mode, outcome, features, clip, min_pits, depth)."""
    fold_of = lambda df: df["Event"].map(folds).to_numpy()
    tasks, keys = [], []
    for k in range(N_FOLDS):
        # nested: nuisances refit without fold k, for both the rewards and the validation
        nested = {}
        for oc, r in rows.items():
            fk = fold_of(r)
            inner, val = nuisance_predictions(r[fk != k].reset_index(drop=True),
                                              r[fk == k].reset_index(drop=True), spec=spec)
            nested[oc] = (inner[inner["p1"].between(TRIM, 1 - TRIM)].reset_index(drop=True), val)
        val_nested = nested["race"][1]
        plain_race = D["race"]["train"]
        val_plain = plain_race[fold_of(plain_race) == k].reset_index(drop=True)
        for mode, val_rows in (("nested", val_nested), ("plain", val_plain)):
            st_val = stints(val_rows)
            for oc in rows:
                if mode == "nested":
                    trn = nested[oc][0]
                else:
                    t = D[oc]["train_trim"]
                    trn = t[fold_of(t) != k].reset_index(drop=True)
                for clip in clips:
                    tau = tau_dr(trn, clip)
                    for fs, feats in FEATURE_SETS.items():
                        tasks.append(delayed(_grid_task)(
                            X(trn, feats), tau, trn["T"].to_numpy(float), X(val_rows, feats),
                            st_val, mins, depths))
                        keys.append((k, mode, oc, clip, fs))
        print(f"  fold {k + 1}/{N_FOLDS} nuisances done", flush=True)
    res = Parallel(n_jobs=n_jobs)(tasks)
    recs = {}
    for (k, mode, oc, clip, fs), out in zip(keys, res):
        for mp, d, g in out:
            recs.setdefault((mode, oc, fs, clip, mp, d), [None] * N_FOLDS)[k] = g
    return [{"mode": m, "outcome": oc, "features": fs, "clip": c, "min_pits": mp,
             "depth": d, "cv_gain": float(np.nanmean(g)), "cv_folds": g}
            for (m, oc, fs, c, mp, d), g in recs.items()]


def rule_text(tree):
    return " | ".join(f"{r['rule']} -> {r['action']}" for r in tree.rules())


def fit_cfg(D, cfg):
    trn = D[cfg["outcome"]]["train_trim"]
    feats = FEATURE_SETS[cfg["features"]]
    return PolicyTree(depth=cfg["depth"], min_leaf_pits=cfg["min_pits"], features=feats).fit(
        X(trn, feats), tau_dr(trn, cfg["clip"]), pits=trn["T"].to_numpy())


def _stab_task(D, cfg, draw, Rte, st_test):
    trn = resample_races(D[cfg["outcome"]]["train_trim"], draw)
    feats = FEATURE_SETS[cfg["features"]]
    tree = PolicyTree(depth=cfg["depth"], min_leaf_pits=cfg["min_pits"], features=feats).fit(
        X(trn, feats), tau_dr(trn, cfg["clip"]), pits=trn["T"].to_numpy())
    a = tree.predict(X(Rte, feats))
    return {"root": tree.tree_.get("feature", "(no split)"),
            "root_thr": tree.tree_.get("threshold"), "rule": rule_text(tree),
            "lap_pit_rate": float(a.mean()), "test_exec_gain": st_test.value(a, GRACE)[0]}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_holdout(laps, name, train_years, test_year, args, t0):
    tag = name.replace(" ", "").replace(">", "")
    clips = (0.01, 0.05) if args.quick else (0.01, 0.05, 0.10)
    mins = (0, 30) if args.quick else (0, 15, 30, 60)
    depths = (1, 2) if args.quick else (1, 2, 3)
    n_boot = 10 if args.quick else args.boot
    res = {"train_years": train_years, "test_year": test_year}

    rows = {}
    for oc, K in OUTCOMES.items():
        tr, te, info, _ = build_split(laps, K=K, train_years=train_years, test_year=test_year)
        rows[oc] = (tr, te)
    print(f"[{name}] rows: " + ", ".join(f"{oc} {len(r[0])}/{len(r[1])}" for oc, r in rows.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)

    # ---- Nuisance comparison (items 2-4) ----------------------------------------
    diag = []
    for sname, spec in candidate_specs().items():
        for oc, (tr, te) in rows.items():
            a, b = nuisance_predictions(tr, te, spec=spec)
            diag.append({"spec": sname, "outcome": oc, **diagnostics(a, b, oc == "race")})
        last = [d for d in diag if d["spec"] == sname]
        print(f"  {sname:38s} " + "  ".join(
            f"{d['outcome']}: R2 f0 {d['r2_f0_cv']:+.3f}/{d['r2_f0_test']:+.3f} "
            f"f1 {d['r2_f1_cv']:+.3f}/{d['r2_f1_test']:+.3f} SC(DM) {d['sc_dm_test']:+.2f}"
            for d in last), flush=True)
    spec, why = choose_spec(diag)
    res["nuisance_diagnostics"] = diag
    res["nuisance_choice"] = {**why, "spec": spec.__dict__}
    pd.DataFrame(diag).to_csv(OUT / f"nuisances_{tag}.csv", index=False)
    print(f"  chosen nuisance spec: {spec}", flush=True)

    # ---- Full cross-fit with the chosen spec ------------------------------------
    D = {}
    for oc, (tr, te) in rows.items():
        a, b = nuisance_predictions(tr, te, spec=spec)
        at, bt, dropped = trim_support(a, b, TRIM)
        D[oc] = {"train": a, "test": b, "train_trim": at, "test_trim": bt, "dropped": dropped}
    Rtr, Rte = D["race"]["train"], D["race"]["test"]
    st_train, st_test = stints(Rtr), stints(Rte)
    res["setup"] = {"races_train": int(Rtr["Event"].nunique()),
                    "races_test": int(Rte["Event"].nunique()),
                    "stints_train": st_train.n, "stints_test": st_test.n,
                    "dropped": {oc: D[oc]["dropped"] for oc in D}}
    print(f"  stints train {st_train.n} test {st_test.n}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- Tree selection: nested vs plain CV (items 1, 5) ------------------------
    folds = race_folds(Rtr["Event"])
    cv = cv_trees({oc: r[0] for oc, r in rows.items()}, D, spec, folds, clips, mins, depths,
                  args.jobs)
    for c in cv:
        c["holdout"] = name
    res["cv"] = cv
    pd.DataFrame([{k: v for k, v in c.items() if k != "cv_folds"} for c in cv]).to_csv(
        OUT / f"cv_{tag}.csv", index=False)
    cvd = pd.DataFrame(cv)
    piv = cvd.pivot_table(index=["outcome", "features", "clip", "min_pits", "depth"],
                          columns="mode", values="cv_gain")
    res["cv_optimism"] = {"mean_plain_minus_nested": float((piv["plain"] - piv["nested"]).mean())}
    cfg_keys = ("outcome", "features", "clip", "min_pits", "depth")
    chosen = {}
    for mode in ("nested", "plain"):
        for fs in FEATURE_SETS:
            sub = cvd[(cvd["mode"] == mode) & (cvd["features"] == fs)]
            b = sub.loc[sub["cv_gain"].idxmax()]
            cfg = {k: (b[k].item() if hasattr(b[k], "item") else b[k]) for k in cfg_keys}
            other = "plain" if mode == "nested" else "nested"
            chosen[f"{fs} features, {mode} CV"] = {
                **cfg, "cv_gain": float(b["cv_gain"]),
                f"cv_gain_{other}": float(piv.loc[tuple(cfg[k] for k in cfg_keys), other])}
    res["cv_selected"] = chosen
    for k, v in chosen.items():
        print(f"  CV pick [{k}]: {v}", flush=True)
    res["cv_by_depth"] = (cvd[cvd["mode"] == "nested"].groupby(["features", "depth"])["cv_gain"]
                          .max().reset_index().to_dict("records"))

    # ---- Final fits, evaluated on the held-out season ---------------------------
    final, actions = [], {}

    def add(pname, a, extra):
        r = st_test.evaluate(a)
        r.update(policy=pname, lap_pit_rate=float(np.mean(a)), **extra)
        final.append(r)
        actions[pname] = a
        print(f"  TEST {pname:42s} {ci(r)}  fol {r['followers']} "
              f"ESS {[round(e) for e in r['ess_by_stratum']]} fire age {r['fire_tyre_age']}",
              flush=True)

    trees = {}
    for key in ("P1 features, nested CV", "v1 features, nested CV", "v1 features, plain CV"):
        cfg = chosen[key]
        tree = fit_cfg(D, cfg)
        trees[key] = tree
        add(f"DR tree ({key})", tree.predict(X(Rte, FEATURE_SETS[cfg["features"]])),
            {"rule": rule_text(tree), "config": {k: cfg[k] for k in cfg_keys}})
    lits = literals(Rtr)
    _, thr = fit_threshold_exec(st_train, Rtr, lits)
    if thr:
        add("Tyre age threshold (exec-tuned)", rule_mask(Rte, thr), {"rule": rule_str(thr)})
    res["final"] = final
    pd.DataFrame(final).to_csv(OUT / f"final_{tag}.csv", index=False)

    dose = []
    for a in range(8, 41, 2):
        r = st_test.evaluate((Rte["TyreLife"] >= a).to_numpy(), n_boot=300)
        dose.append({"tyre_age": a, **{k: r[k] for k in ("gain", "lo", "hi", "followers")}})
    res["dose_response_test"] = dose
    pd.DataFrame(dose).to_csv(OUT / f"dose_{tag}.csv", index=False)
    print("  dose-response (test):", [(d["tyre_age"], round(d["gain"], 1)) for d in dose],
          flush=True)

    # ---- Stability over bootstrap resamples of the training races ---------------
    rng = np.random.default_rng(SEED)
    events = np.array(sorted(Rtr["Event"].unique()))
    draws = [rng.choice(events, size=len(events), replace=True) for _ in range(n_boot)]
    stab = {}
    for key in ("P1 features, nested CV", "v1 features, nested CV"):
        cfg = chosen[key]
        lst = Parallel(n_jobs=args.jobs)(delayed(_stab_task)(D, cfg, d, Rte, st_test)
                                         for d in draws)
        df = pd.DataFrame(lst)
        df.to_csv(OUT / f"stability_{tag}_{key.split()[0]}.csv", index=False)
        stab[key] = {
            "root_freq": {k: v / len(df) for k, v in Counter(df["root"]).most_common()},
            "top_rules": [{"rule": k, "freq": v / len(df)}
                          for k, v in Counter(df["rule"]).most_common(5)],
            "lap_pit_rate_median": float(df["lap_pit_rate"].median()),
            "test_exec_gain_median": float(df["test_exec_gain"].median()),
            "test_exec_gain_iqr": [float(df["test_exec_gain"].quantile(q)) for q in (.25, .75)]}
        print(f"  stability [{key}]: roots {dict(list(stab[key]['root_freq'].items())[:4])} "
              f"top rule {stab[key]['top_rules'][0]['freq']:.2f}  ({time.time() - t0:.0f}s)",
              flush=True)
    res["stability"] = stab
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--holdout", choices=list(HOLDOUTS), default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    laps = pd.read_pickle(DATA)
    results = {"setup": {"grace": GRACE, "w_max": W_MAX, "trim": TRIM, "folds": N_FOLDS,
                         "stratum": STRATUM, "quick": args.quick,
                         "tree_features": FEATURE_SETS}}
    for name, (ty, te) in HOLDOUTS.items():
        if args.holdout and name != args.holdout:
            continue
        results[name] = run_holdout(laps, name, ty, te, args, t0)
        (OUT / "p1.json").write_text(json.dumps(results, indent=2, default=str))
    results["runtime_s"] = time.time() - t0
    (OUT / "p1.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}/p1.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
