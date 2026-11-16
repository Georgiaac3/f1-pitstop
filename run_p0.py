"""P0 study: make the learned rule meaningful (see report, "Next steps", P0).

  1. Executable evaluation (stopping.py): every rule is scored as "one stop per stint,
     at the first lap the rule says PIT (grace window g)", off-policy by
     clone-censor-weight, on the rest-of-race outcome. This is the yardstick for
     everything below, including model selection.
  2. Stop debt: the rest-of-race outcome (build_rows(K="race")) pays a deferred stop
     inside the outcome, so no debt constant is needed. Check: lap-level "never pit".
     The pit propensity is gradient boosted (better calibrated over whole stints
     than the v1 logistic model) and stint weights are truncated at W_MAX.
  3. Variance control: minimum number of actual stops per leaf, clip level and depth
     are chosen by grouped (race) CV on 2023-24 with the executable value.
     A second learner searches simple rules (one condition, or two joined by AND/OR)
     directly on the executable value.
  4. Stability: refit on bootstrap resamples of the training races.

Writes results_p0/. The v1 experiment (run_experiment.py, first_results/) is unchanged.

Usage:
    python run_p0.py            # full run (~20 min)
    python run_p0.py --quick    # small grid, 10 bootstrap resamples (smoke test)
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from policy import TREE_FEATURES, PolicyTree, bootstrap_gain, policy_value, rewards
from run_experiment import (DATA, TEST_YEAR, TRAIN_YEARS, TRIM, X, fit_policies, prepare,
                            trim_support)
from stopping import Stints

OUT = Path(__file__).parent / "results_p0"
GRACE = 2          # laps allowed between the rule's first PIT and the stop
W_MAX = 20         # stint weights truncated here (sensitivity: 10, 50, none)
N_FOLDS = 5
MIN_ESS = 15       # rule search: every stratum must keep this ESS where it is fit
PROPENSITY = "gbm" # see policy._GBMPropensity: calibrated over whole stints
SEED = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ci(r):
    f = lambda v: "  n/a" if v is None or not np.isfinite(v) else f"{v:+.2f}"
    return f"{f(r['gain'])} [{f(r['lo'])},{f(r['hi'])}]"


def tau_dr(df, clip):
    _, dr = rewards(df, clip=clip)
    return dr[:, 1] - dr[:, 0]


def lap_gain(test_trim, actions, clip=0.01):
    """v1 metric (lap-level DR gain over teams), for reference."""
    _, dr = rewards(test_trim, clip=clip)
    per_row = policy_value(dr, actions) - test_trim["y"].to_numpy()
    est, lo, hi = bootstrap_gain(per_row, test_trim["Event"].to_numpy())
    return {"lap_gain": est, "lap_lo": lo, "lap_hi": hi}


def race_folds(events, n_folds=N_FOLDS, seed=SEED):
    ev = np.array(sorted(set(events)))
    perm = np.random.default_rng(seed).permutation(len(ev))
    return {e: i % n_folds for e, i in zip(ev[perm], range(len(ev)))}


def resample_races(df, events_draw):
    """Rows of the drawn races; duplicated races get distinct labels."""
    parts = [df[df["Event"] == e].assign(Event=f"{e}#{k}") for k, e in enumerate(events_draw)]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Learner 1: DR policy tree with variance control
# --------------------------------------------------------------------------

def fit_tree(train_trim, depth, min_pits, clip):
    return PolicyTree(depth=depth, min_leaf_pits=min_pits).fit(
        X(train_trim), tau_dr(train_trim, clip), pits=train_trim["T"].to_numpy())


def rule_text(tree):
    return " | ".join(f"{r['rule']} -> {r['action']}" for r in tree.rules())


# --------------------------------------------------------------------------
# Learner 2: direct search of simple stopping rules on the executable value
# --------------------------------------------------------------------------

def literals(rows, n_thr=8):
    lits = []
    for f in TREE_FEATURES:
        v = rows[f].to_numpy(float)
        u = np.unique(v)
        thr = u[:-1] if len(u) <= n_thr + 1 else np.unique(
            np.quantile(v, np.linspace(0, 1, n_thr + 2)[1:-1]))
        for t in thr:
            lits += [(f, ">", float(t)), (f, "<=", float(t))]
    return lits


def lit_text(l):
    return f"{l[0]} {l[1]} {l[2]:g}"


def rule_mask(rows, rule):
    """rule = (op, [literals]) with op in {"one", "and", "or"}."""
    op, ls = rule
    ms = [(rows[f].to_numpy(float) > t) if o == ">" else (rows[f].to_numpy(float) <= t)
          for f, o, t in ls]
    if op == "one":
        return ms[0]
    return (ms[0] & ms[1]) if op == "and" else (ms[0] | ms[1])


def rule_str(rule):
    op, ls = rule
    return lit_text(ls[0]) if op == "one" else f" {op.upper()} ".join(lit_text(l) for l in ls)


def search_rules(stints, rows, lits, forms=("one", "and", "or"), grace=GRACE,
                 min_ess=MIN_ESS, restrict=None):
    """Best rule by executable value on (stints, rows). `restrict`: only literals whose
    feature is in this set (for the classic baselines)."""
    cand = [l for l in lits if restrict is None or l[0] in restrict]
    M = np.array([rule_mask(rows, ("one", [l])) for l in cand])
    best = (-np.inf, None)

    def consider(mask, rule):
        nonlocal best
        g, _, ess = stints.value(mask, grace)
        if ess >= min_ess and np.isfinite(g) and g > best[0]:
            best = (g, rule)

    if "one" in forms:
        for i, l in enumerate(cand):
            consider(M[i], ("one", [l]))
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            # AND: two features, or one feature from both sides (an interval)
            if "and" in forms and (cand[i][0] != cand[j][0] or cand[i][1] != cand[j][1]):
                consider(M[i] & M[j], ("and", [cand[i], cand[j]]))
            if "or" in forms and cand[i][0] != cand[j][0]:
                consider(M[i] | M[j], ("or", [cand[i], cand[j]]))
    return best


def fit_threshold_exec(stints, rows, lits):
    """Classic heuristic, tuned on the executable value: pit when tyre age > a."""
    tl = [l for l in lits if l[0] == "TyreLife" and l[1] == ">"]
    best = (-np.inf, None)
    for l in tl:
        g, _, ess = stints.value(rule_mask(rows, ("one", [l])), GRACE)
        if ess >= MIN_ESS and g > best[0]:
            best = (g, ("one", [l]))
    return best


def fit_sc_or_threshold(stints, rows, lits):
    """Pit under a safety car / VSC, or when tyre age > a (a tuned)."""
    tl = [l for l in lits if l[0] == "TyreLife" and l[1] == ">"]
    best = (-np.inf, None)
    for l in tl:
        m = rule_mask(rows, ("one", [l])) | (rows["SafetyCar"].to_numpy() > 0) \
            | (rows["VSC"].to_numpy() > 0)
        g, _, ess = stints.value(m, GRACE)
        if ess >= MIN_ESS and g > best[0]:
            best = (g, ("or", [("SafetyCar|VSC", ">", 0.0), l]))
    return best


def sc_or_mask(rows, rule):
    l = rule[1][1]
    return (rule_mask(rows, ("one", [l])) | (rows["SafetyCar"].to_numpy() > 0)
            | (rows["VSC"].to_numpy() > 0))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    clips = (0.01, 0.05) if args.quick else (0.01, 0.05, 0.10)
    min_pits_grid = (0, 30) if args.quick else (0, 15, 30, 60)
    depths = (1, 2) if args.quick else (1, 2, 3)
    n_boot = 10 if args.quick else args.boot

    laps = pd.read_pickle(DATA)
    laps = laps[laps["Year"].isin(TRAIN_YEARS + [TEST_YEAR])]

    # ---- Data: K=10 (v1 outcome, with stop debt) and rest-of-race --------------
    D = {}
    for name, K in (("K10", 10), ("race", "race")):
        tr, te, info, _ = prepare(laps, K=K, trim=0.0, propensity=PROPENSITY)
        trt, tet, dropped = trim_support(tr, te, TRIM)
        D[name] = {"train": tr, "test": te, "train_trim": trt, "test_trim": tet,
                   "info": {**info, **dropped}}
        print(f"[{name}] rows train {len(tr)} test {len(te)}  "
              f"(trimmed {len(trt)}/{len(tet)})  {time.time() - t0:.0f}s")
    Rtr, Rte = D["race"]["train"], D["race"]["test"]
    st_test, st_train = Stints(Rte, w_max=W_MAX), Stints(Rtr, w_max=W_MAX)
    print(f"stints: train {st_train.n}, test {st_test.n}")

    results = {"setup": {"grace": GRACE, "w_max": W_MAX, "propensity": PROPENSITY,
                         "trim": TRIM, "min_ess": MIN_ESS,
                         "folds": N_FOLDS, "boot": n_boot,
                         "data": {k: v["info"] for k, v in D.items()},
                         "races_train": int(Rtr["Event"].nunique()),
                         "races_test": int(Rte["Event"].nunique()),
                         "stints_train": st_train.n, "stints_test": st_test.n}}

    # ---- Sanity: dose-response of "pit at tyre age >= a" -------------------------
    # (The teams' own actions are not a valid check: the weights assume the rule is
    # a function of the observed context, and realised team stops are not.)
    dose = []
    for a in range(8, 41, 2):
        for split, st, rows in (("train", st_train, Rtr), ("test", st_test, Rte)):
            r = st.evaluate((rows["TyreLife"] >= a).to_numpy(), n_boot=300)
            dose.append({"tyre_age": a, "split": split, **r})
    results["dose_response"] = dose
    pd.DataFrame(dose).to_csv(OUT / "dose_response.csv", index=False)
    print("dose-response (test):", [(d["tyre_age"], round(d["gain"], 1)) for d in dose
                                    if d["split"] == "test"])

    # ---- (2) stop debt check: lap-level "never pit" -----------------------------
    results["never_pit_lap_level"] = {
        name: lap_gain(D[name]["test_trim"], np.zeros(len(D[name]["test_trim"]), int))
        for name in D}
    print("never pit, lap-level DR gain:",
          {k: round(v["lap_gain"], 2) for k, v in results["never_pit_lap_level"].items()})

    # ---- (1) v1 policies under the executable evaluation ------------------------
    # Refit exactly as in v1 (logistic propensity), then score executably.
    tr10, te10, _, _ = prepare(laps, K=10)
    v1_pol, _, _ = fit_policies(tr10, depth=1)
    v1 = []
    for name, pol in v1_pol.items():
        r = st_test.evaluate(pol(Rte))
        r.update(policy=name, lap_pit_rate=float(np.mean(pol(Rte))))
        v1.append(r)
        print(f"v1 {name:40s} exec gain {ci(r)} "
              f"followers {r['followers']} ESS {[round(e) for e in r['ess_by_stratum']]}")
    results["v1_executable"] = v1

    # ---- (3) CV on training races: trees (variance control) and rule search -----
    folds = race_folds(Rtr["Event"])
    fold_of = lambda df: df["Event"].map(folds).to_numpy()
    fold_eval = []
    for k in range(N_FOLDS):
        rows_k = Rtr[fold_of(Rtr) == k].reset_index(drop=True)
        fold_eval.append((rows_k, Stints(rows_k, w_max=W_MAX)))

    cv = []
    for oc in ("K10", "race"):
        trn = D[oc]["train_trim"]
        fk = fold_of(trn)
        for clip in clips:
            tau = tau_dr(trn, clip)
            for mp in min_pits_grid:
                for d in depths:
                    gains = []
                    for k in range(N_FOLDS):
                        m = fk != k
                        tree = PolicyTree(depth=d, min_leaf_pits=mp).fit(
                            X(trn[m]), tau[m], pits=trn["T"].to_numpy()[m])
                        rows_k, st_k = fold_eval[k]
                        gains.append(st_k.value(tree.predict(X(rows_k)), GRACE)[0])
                    cv.append({"family": "DR tree", "outcome": oc, "clip": clip,
                               "min_pits": mp, "depth": d, "cv_gain": float(np.nanmean(gains)),
                               "cv_folds": [float(g) for g in gains]})
                    print(f"cv tree {oc:4s} clip {clip:.2f} minpits {mp:2d} depth {d}: "
                          f"{cv[-1]['cv_gain']:+.2f}  ({time.time() - t0:.0f}s)")

    lits = literals(Rtr)
    print(f"rule search over {len(lits)} literals")
    searchers = {
        "Rule search (1 condition)": lambda st, rows: search_rules(st, rows, lits, ("one",)),
        "Rule search (2 conditions)": lambda st, rows: search_rules(st, rows, lits),
        "Threshold (tyre age, exec-tuned)": lambda st, rows: fit_threshold_exec(st, rows, lits),
        "SC/VSC or tyre age (exec-tuned)": lambda st, rows: fit_sc_or_threshold(st, rows, lits),
    }
    mask_of = lambda name, rows, rule: (sc_or_mask(rows, rule) if name.startswith("SC/VSC")
                                        else rule_mask(rows, rule))
    for name, fn in searchers.items():
        gains, picked = [], []
        for k in range(N_FOLDS):
            rows_fit = Rtr[fold_of(Rtr) != k].reset_index(drop=True)
            _, rule = fn(Stints(rows_fit, w_max=W_MAX), rows_fit)
            rows_k, st_k = fold_eval[k]
            gains.append(st_k.value(mask_of(name, rows_k, rule), GRACE)[0] if rule else np.nan)
            picked.append(rule_str(rule) if rule else None)
        cv.append({"family": name, "cv_gain": float(np.nanmean(gains)),
                   "cv_folds": [float(g) for g in gains], "fold_rules": picked})
        print(f"cv {name}: {cv[-1]['cv_gain']:+.2f}  rules {picked}  ({time.time() - t0:.0f}s)")
    results["cv"] = cv
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("cv_folds", "fold_rules")}
                  for c in cv]).to_csv(OUT / "cv.csv", index=False)

    # ---- Final fits on all training races, evaluated on 2025 --------------------
    trees_cv = [c for c in cv if c["family"] == "DR tree"]
    best_tree = max(trees_cv, key=lambda c: c["cv_gain"])
    best_by_outcome = {oc: max([c for c in trees_cv if c["outcome"] == oc],
                               key=lambda c: c["cv_gain"]) for oc in ("K10", "race")}
    final = []

    def add(name, actions, extra):
        r = st_test.evaluate(actions)
        r.update(policy=name, lap_pit_rate=float(np.mean(actions)), **extra)
        final.append(r)
        print(f"TEST {name:45s} {ci(r)} "
              f"fol {r['followers']} ESS {[round(e) for e in r['ess_by_stratum']]} "
              f"fire age {r['fire_tyre_age']}")

    tree_cfgs = {"DR tree (CV-selected)": best_tree}
    for oc, c in best_by_outcome.items():
        tree_cfgs[f"DR tree (best {oc} outcome)"] = c
    tree_cfgs["DR tree (v1 settings, depth 1)"] = {"outcome": "K10", "clip": 0.01,
                                                   "min_pits": 0, "depth": 1}
    final_trees = {}
    for name, c in tree_cfgs.items():
        tree = fit_tree(D[c["outcome"]]["train_trim"], c["depth"], c["min_pits"], c["clip"])
        final_trees[name] = (tree, c)
        a = tree.predict(X(Rte))
        lg = lap_gain(D["race"]["test_trim"], tree.predict(X(D["race"]["test_trim"])))
        add(name, a, {"rule": rule_text(tree), "config": {k: c[k] for k in
                      ("outcome", "clip", "min_pits", "depth")}, **lg})
    final_rules = {}
    for name, fn in searchers.items():
        _, rule = fn(st_train, Rtr)
        if rule is None:
            print(f"{name}: no rule reaches ESS {MIN_ESS}")
            continue
        final_rules[name] = rule
        add(name, mask_of(name, Rte, rule), {"rule": rule_str(rule)})
    results["final"] = final
    pd.DataFrame(final).to_csv(OUT / "final.csv", index=False)

    # Sensitivity of the test estimates to the grace window and the weight cap.
    final_actions = {n: t.predict(X(Rte)) for n, (t, _) in final_trees.items()}
    final_actions.update({n: mask_of(n, Rte, r) for n, r in final_rules.items()})
    sens = []
    for g in (0, 1, 2, 3, 5):
        for name, a in final_actions.items():
            sens.append({"analysis": "grace", "setting": g, "policy": name,
                         **st_test.evaluate(a, grace=g)})
    for cap in (10, 50, None):
        st_cap = Stints(Rte, w_max=cap)
        for name, a in final_actions.items():
            sens.append({"analysis": "w_max", "setting": str(cap), "policy": name,
                         **st_cap.evaluate(a)})
    results["sensitivity"] = sens
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)

    # ---- (4) Stability over bootstrap resamples of the training races -----------
    rng = np.random.default_rng(SEED)
    events = np.array(sorted(Rtr["Event"].unique()))
    stab_cfgs = {"v1 settings, depth 1": {"outcome": "K10", "clip": 0.01, "min_pits": 0, "depth": 1},
                 "v1 settings, depth 2": {"outcome": "K10", "clip": 0.01, "min_pits": 0, "depth": 2},
                 "v1 settings, depth 3": {"outcome": "K10", "clip": 0.01, "min_pits": 0, "depth": 3},
                 "CV-selected tree": {k: best_tree[k] for k in ("outcome", "clip", "min_pits", "depth")}}
    for oc, c in best_by_outcome.items():
        stab_cfgs[f"best {oc} tree"] = {k: c[k] for k in ("outcome", "clip", "min_pits", "depth")}
    draws = [rng.choice(events, size=len(events), replace=True) for _ in range(n_boot)]
    stab = {name: [] for name in list(stab_cfgs) + list(searchers)}
    for b, draw in enumerate(draws):
        for name, c in stab_cfgs.items():
            trn = resample_races(D[c["outcome"]]["train_trim"], draw)
            tree = fit_tree(trn, c["depth"], c["min_pits"], c["clip"])
            root = tree.tree_.get("feature", "(no split)")
            a = tree.predict(X(Rte))
            stab[name].append({"root": root, "root_thr": tree.tree_.get("threshold"),
                               "rule": rule_text(tree), "lap_pit_rate": float(a.mean()),
                               "test_exec_gain": st_test.value(a, GRACE)[0]})
        rows_b = resample_races(Rtr, draw)
        st_b = Stints(rows_b, w_max=W_MAX)
        for name, fn in searchers.items():
            _, rule = fn(st_b, rows_b)
            a = mask_of(name, Rte, rule) if rule else np.zeros(len(Rte), bool)
            stab[name].append({"root": rule_str(rule).split(" ")[0] if rule else None,
                               "rule": rule_str(rule) if rule else None,
                               "lap_pit_rate": float(a.mean()),
                               "test_exec_gain": st_test.value(a, GRACE)[0]})
        print(f"bootstrap {b + 1}/{n_boot}  ({time.time() - t0:.0f}s)")
    summary = {}
    for name, lst in stab.items():
        df = pd.DataFrame(lst)
        roots = Counter(df["root"].fillna("none"))
        rules = Counter(df["rule"].fillna("none"))
        summary[name] = {
            "root_freq": {k: v / len(df) for k, v in roots.most_common()},
            "top_rules": [{"rule": k, "freq": v / len(df)} for k, v in rules.most_common(5)],
            "lap_pit_rate_median": float(df["lap_pit_rate"].median()),
            "lap_pit_rate_iqr": [float(df["lap_pit_rate"].quantile(.25)),
                                 float(df["lap_pit_rate"].quantile(.75))],
            "test_exec_gain_median": float(df["test_exec_gain"].median()),
            "test_exec_gain_iqr": [float(df["test_exec_gain"].quantile(.25)),
                                   float(df["test_exec_gain"].quantile(.75))],
        }
        df.to_csv(OUT / f"stability_{name.replace(' ', '_').replace('/', '-')}.csv", index=False)
    results["stability"] = summary
    results["runtime_s"] = time.time() - t0
    (OUT / "p0.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}/p0.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
