"""P2 study: extensions (see report, "Next steps", P2).

  1. Compound choice (compound.py): given that the team stops, which dry compound to
     fit. Multi-action DR policy tree over {SOFT, MEDIUM, HARD}, depth chosen by nested
     CV over races, evaluated on the held-out season by DR (team keeps its stop lap).
     Run on both holdouts.
  2. Interpretable AI: OptimalTreePolicyMaximizer (iai_compare.jl) on exactly the
     rewards our solvers use, for the stay/pit tree (P1 setup) and the compound tree.
     Compares training objectives (same depth and minimum leaf size) and test values.
  3. Case studies: lap-by-lap "rule vs team" replays of four 2025 races (the race with
     the most stops under SC/VSC, a street circuit, the race that helps the rule most and
     the one that hurts it most), written as TikZ figures for the report and as CSVs
     for the dashboard.

Writes results_p2/ and report/case_*.tex.

Usage:
    python run_p2.py            # full run
    python run_p2.py --quick    # 10 bootstrap resamples, no IAI
    python run_p2.py --no-iai   # skip the Julia step
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import compound as C
from policy import TREE_FEATURES_P1, PolicyTree, bootstrap_gain, nuisance_predictions, rewards
from run_experiment import DATA, STREET, TRIM, build_split, trim_support
from run_p0 import GRACE, N_FOLDS, SEED, fit_threshold_exec, literals, race_folds, \
    resample_races, rule_mask, rule_str
from run_p1 import HOLDOUTS, X, p1_spec, rule_text, stints, tau_dr

ROOT = Path(__file__).parent
OUT = ROOT / "results_p2"
REPORT = ROOT / "report"
MAIN = "2023-24 -> 2025"
P1_SPEC = p1_spec("t", False, False)           # chosen by run_p1 on 2023-24
P1_TREE = {"clip": 0.01, "min_pits": 60, "depth": 2}  # P1 nested-CV pick (rest of race)
IAI_JULIA = os.environ.get("IAI_JULIA", str(Path.home() / ".juliaup/bin/julia"))
IAI_CHANNEL = os.environ.get("IAI_CHANNEL", "+1.11.6/IAI/3.2.2")


def ci_txt(r):
    return f"{r['gain']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]"


# --------------------------------------------------------------------------
# 1. Compound choice
# --------------------------------------------------------------------------

def compound_dr_gain(stops, G, actions):
    per = C.value(G, actions) - stops["y"].to_numpy()
    est, lo, hi = bootstrap_gain(per, stops["Event"].to_numpy())
    return {"gain": est, "lo": lo, "hi": hi,
            "share": {a: float(np.mean(np.asarray(actions) == i)) for i, a in enumerate(C.ACTIONS)}}


def compound_cv(stops, folds, grid):
    """Nested CV: for fold k, rewards for the tree come from nuisances cross-fitted on
    the other folds; fold k is scored with nuisances fit without it."""
    fk = stops["Event"].map(folds).to_numpy()
    scores = {g: [] for g in grid}
    for k in range(N_FOLDS):
        inner = stops[fk != k].reset_index(drop=True)
        val = stops[fk == k].reset_index(drop=True)
        (mi, ei), (mv, ev) = C.nuisance_predictions(inner, val)
        Gi, Gv = C.dr_rewards(inner, mi, ei), C.dr_rewards(val, mv, ev)
        for depth, mlf in grid:
            t = C.MultiActionTree(depth=depth, min_leaf_frac=mlf).fit(X(inner, TREE_FEATURES_P1), Gi)
            a = t.predict(X(val, TREE_FEATURES_P1))
            scores[(depth, mlf)].append(float((C.value(Gv, a) - val["y"]).mean()))
    return {g: float(np.mean(s)) for g, s in scores.items()}


def run_compound(laps, name, train_years, test_year, n_boot):
    tr, te, _, _ = build_split(laps, K="race", train_years=train_years, test_year=test_year)
    S_tr, S_te = C.stop_rows(tr), C.stop_rows(te)
    (mu_tr, e_tr), (mu_te, e_te) = C.nuisance_predictions(S_tr, S_te)
    G_tr, G_te = C.dr_rewards(S_tr, mu_tr, e_tr), C.dr_rewards(S_te, mu_te, e_te)
    res = {"stops_train": len(S_tr), "stops_test": len(S_te),
           "races_train": int(S_tr["Event"].nunique()), "races_test": int(S_te["Event"].nunique()),
           "compound_share_train": S_tr["NewCompound"].value_counts(normalize=True).to_dict(),
           "compound_share_test": S_te["NewCompound"].value_counts(normalize=True).to_dict(),
           # overlap: how often a compound is (almost) never chosen in that context
           "share_min_propensity_below_5pct_test": float((e_te.min(1) < 0.05).mean()),
           "propensity_of_taken_action_test_median": float(np.median(
               e_te[np.arange(len(S_te)), S_te["A"]]))}

    grid = [(d, m) for d in (1, 2) for m in (0.05, 0.10)]
    cv = compound_cv(S_tr, race_folds(S_tr["Event"]), grid)
    (d, mlf) = max(cv, key=cv.get)
    res["cv"] = [{"depth": g[0], "min_leaf_frac": g[1], "cv_gain": v} for g, v in cv.items()]
    res["chosen"] = {"depth": d, "min_leaf_frac": mlf}
    print(f"[{name}] compound: {len(S_tr)}/{len(S_te)} stops; CV {cv} -> depth {d}, leaf {mlf}",
          flush=True)

    Xtr, Xte = X(S_tr, TREE_FEATURES_P1), X(S_te, TREE_FEATURES_P1)
    tree = C.MultiActionTree(depth=d, min_leaf_frac=mlf).fit(Xtr, G_tr)
    dm = C.MultiActionTree(depth=d, min_leaf_frac=mlf).fit(Xtr, mu_tr)
    pols = {"DR tree (nested CV)": tree.predict(Xte), "DM tree (same depth)": dm.predict(Xte)}
    for i, a in enumerate(C.ACTIONS):
        pols[f"Always {a.lower()}"] = np.full(len(S_te), i)
    final = []
    for pname, a in pols.items():
        r = {"policy": pname, **compound_dr_gain(S_te, G_te, a)}
        final.append(r)
        print(f"  TEST {pname:24s} {ci_txt(r)}  shares {r['share']}", flush=True)
    res["final"] = final
    res["rules"] = tree.rules()
    res["dm_rules"] = dm.rules()
    print("  rule:", " | ".join(f"{r['rule']} -> {r['action']}" for r in tree.rules()), flush=True)

    # Known fact: a soft tyre pays off only for a short final stint. DR (and DM) value of
    # soft minus hard by laps remaining, on the test stops.
    bins = pd.cut(S_te["LapsRemaining"], [0, 15, 25, 40, 90])
    fact = []
    for b, g in S_te.groupby(bins, observed=True):
        i = g.index.to_numpy()
        fact.append({"laps_remaining": str(b), "stops": int(len(i)),
                     "soft_share": float((g["A"] == 0).mean()),
                     "dr_soft_minus_hard": float((G_te[i, 0] - G_te[i, 2]).mean()),
                     "dm_soft_minus_hard": float((mu_te[i, 0] - mu_te[i, 2]).mean())})
    res["soft_vs_hard_by_laps_remaining"] = fact
    print("  soft - hard by laps remaining:",
          [(f["laps_remaining"], round(f["dr_soft_minus_hard"], 1)) for f in fact], flush=True)

    # Stability: refit on bootstrap resamples of training races (rewards held fixed).
    rng = np.random.default_rng(SEED)
    events = np.array(sorted(S_tr["Event"].unique()))
    stab = []
    for _ in range(n_boot):
        draw = rng.choice(events, size=len(events), replace=True)
        parts = [np.flatnonzero(S_tr["Event"].to_numpy() == e) for e in draw]
        idx = np.concatenate(parts)
        t = C.MultiActionTree(depth=d, min_leaf_frac=mlf).fit(Xtr[idx], G_tr[idx])
        a = t.predict(Xte)
        stab.append({"root": t.tree_.get("feature", "(no split)"),
                     "actions": "/".join(sorted({r["action"] for r in t.rules()})),
                     "test_gain": float((C.value(G_te, a) - S_te["y"]).mean())})
    sd = pd.DataFrame(stab)
    res["stability"] = {"root_freq": {k: v / len(sd) for k, v in Counter(sd["root"]).most_common()},
                        "actions_freq": {k: v / len(sd) for k, v in Counter(sd["actions"]).most_common()},
                        "test_gain_iqr": [float(sd["test_gain"].quantile(q)) for q in (.25, .75)]}
    print(f"  stability: {res['stability']['root_freq']}", flush=True)
    return res, (S_tr, S_te, G_tr, G_te, tree, d, mlf)


# --------------------------------------------------------------------------
# 2. IAI comparison
# --------------------------------------------------------------------------

def export_problem(d, name, Xtr, G, cols, Xte, valid, minbucket, max_depth):
    pd.DataFrame(Xtr, columns=cols).to_csv(d / f"{name}_Xtrain.csv", index=False)
    pd.DataFrame(G[0], columns=G[1]).to_csv(d / f"{name}_Gtrain.csv", index=False)
    pd.DataFrame(Xte, columns=cols).to_csv(d / f"{name}_Xtest.csv", index=False)
    pd.DataFrame({"valid": valid.astype(int)}).to_csv(d / f"{name}_valid.csv", index=False)
    pd.DataFrame({"minbucket": [minbucket], "max_depth": [max_depth]}).to_csv(
        d / f"{name}_meta.csv", index=False)


def run_iai(D, st_test, comp):
    d = OUT / "iai"
    d.mkdir(exist_ok=True)
    feats = TREE_FEATURES_P1
    trn, Rte = D["train_trim"], D["test"]
    folds = race_folds(trn["Event"])
    tau = tau_dr(trn, P1_TREE["clip"])
    # stay/pit: same rewards as our tree (DR, clip 0.01), columns stay / pit
    _, Gsp = rewards(trn, clip=P1_TREE["clip"])
    mb_sp = max(1, int(0.02 * len(trn)))       # PolicyTree's default min_leaf_frac
    export_problem(d, "stay_pit", X(trn, feats), (Gsp, ["stay", "pit"]), feats, X(Rte, feats),
                   (trn["Event"].map(folds) == 0).to_numpy(), mb_sp, 3)
    S_tr, S_te, G_tr, G_te, ctree, cd, cmlf = comp
    mb_c = max(1, int(cmlf * len(S_tr)))
    cfolds = race_folds(S_tr["Event"])
    export_problem(d, "compound", X(S_tr, feats), (G_tr, C.ACTIONS), feats, X(S_te, feats),
                   (S_tr["Event"].map(cfolds) == 0).to_numpy(), mb_c, 2)

    if shutil.which(IAI_JULIA) is None and not Path(IAI_JULIA).exists():
        print("IAI: julia not found, skipping")
        return None
    t0 = time.time()
    proc = subprocess.run([IAI_JULIA, IAI_CHANNEL, str(ROOT / "iai_compare.jl"), str(d),
                           "stay_pit,compound"], capture_output=True, text=True)
    out = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines()
                    if "machine" not in l.lower() and "licen" not in l.lower())
    print(out[-2000:])
    if proc.returncode != 0:
        print("IAI failed")
        return {"error": out[-2000:]}
    res = {"runtime_s": time.time() - t0}

    # stay/pit: objective on the training rows = sum of tau over rows sent to PIT
    ptr = pd.read_csv(d / "stay_pit_iai_pred_train.csv")
    pte = pd.read_csv(d / "stay_pit_iai_pred_test.csv")
    info = pd.read_csv(d / "stay_pit_iai_info.csv").set_index("model")
    rows = []
    for depth in (1, 2, 3):
        ours = PolicyTree(depth=depth, features=feats).fit(X(trn, feats), tau,
                                                          pits=trn["T"].to_numpy())
        a_ours_tr = ours.predict(X(trn, feats))
        a_iai_tr = (ptr[f"iai_d{depth}"] == "pit").to_numpy().astype(int)
        r_o = st_test.evaluate(ours.predict(X(Rte, feats)))
        r_i = st_test.evaluate((pte[f"iai_d{depth}"] == "pit").to_numpy().astype(int))
        rows.append({"problem": "stay/pit", "depth": depth,
                     "objective_ours": float(tau[a_ours_tr == 1].sum()),
                     "objective_iai": float(tau[a_iai_tr == 1].sum()),
                     "agree_train": float((a_ours_tr == a_iai_tr).mean()),
                     "test_ours": r_o["gain"], "test_ours_ci": [r_o["lo"], r_o["hi"]],
                     "test_iai": r_i["gain"], "test_iai_ci": [r_i["lo"], r_i["hi"]],
                     "iai_seconds": float(info.loc[f"iai_d{depth}", "seconds"])})
        print(f"IAI stay/pit d{depth}: objective ours {rows[-1]['objective_ours']:.0f} "
              f"IAI {rows[-1]['objective_iai']:.0f}, agree {rows[-1]['agree_train']:.2f}, "
              f"test ours {r_o['gain']:+.2f} IAI {r_i['gain']:+.2f}", flush=True)
    r_g = st_test.evaluate((pte["iai_grid"] == "pit").to_numpy().astype(int))
    res["stay_pit_grid"] = {"params": info.loc["iai_grid", "params"], "test": r_g["gain"],
                            "test_ci": [r_g["lo"], r_g["hi"]],
                            "pit_rate_test": float((pte["iai_grid"] == "pit").mean())}

    # compound
    ctr = pd.read_csv(d / "compound_iai_pred_train.csv")
    cte = pd.read_csv(d / "compound_iai_pred_test.csv")
    cinfo = pd.read_csv(d / "compound_iai_info.csv").set_index("model")
    amap = {a: i for i, a in enumerate(C.ACTIONS)}
    Xtr, Xte = X(S_tr, feats), X(S_te, feats)
    for depth in (1, 2):
        ours = C.MultiActionTree(depth=depth, min_leaf_frac=cmlf).fit(Xtr, G_tr)
        a_o, a_i = ours.predict(Xtr), ctr[f"iai_d{depth}"].map(amap).to_numpy()
        t_o = compound_dr_gain(S_te, G_te, ours.predict(Xte))
        t_i = compound_dr_gain(S_te, G_te, cte[f"iai_d{depth}"].map(amap).to_numpy())
        rows.append({"problem": "compound", "depth": depth,
                     "objective_ours": float(C.value(G_tr, a_o).sum()),
                     "objective_iai": float(C.value(G_tr, a_i).sum()),
                     "agree_train": float((a_o == a_i).mean()),
                     "test_ours": t_o["gain"], "test_ours_ci": [t_o["lo"], t_o["hi"]],
                     "test_iai": t_i["gain"], "test_iai_ci": [t_i["lo"], t_i["hi"]],
                     "iai_seconds": float(cinfo.loc[f"iai_d{depth}", "seconds"])})
        print(f"IAI compound d{depth}: objective ours {rows[-1]['objective_ours']:.0f} "
              f"IAI {rows[-1]['objective_iai']:.0f}, agree {rows[-1]['agree_train']:.2f}, "
              f"test ours {t_o['gain']:+.2f} IAI {t_i['gain']:+.2f}", flush=True)
    t_g = compound_dr_gain(S_te, G_te, cte["iai_grid"].map(amap).to_numpy())
    res["compound_grid"] = {"params": cinfo.loc["iai_grid", "params"], "test": t_g["gain"],
                            "test_ci": [t_g["lo"], t_g["hi"]]}
    res["by_depth"] = rows
    pd.DataFrame(rows).to_csv(OUT / "iai_comparison.csv", index=False)
    return res


# --------------------------------------------------------------------------
# 3. Case studies
# --------------------------------------------------------------------------

def stint_table(st, rows, actions):
    """Per stint: identifiers, team stop lap, rule firing lap, follows, weight, Y."""
    srt = rows.reset_index(drop=True).iloc[st.order].reset_index(drop=True)
    follows, w, fires, fire_idx = st.follow(actions, GRACE)
    head = srt.iloc[st.start][["Event", "Driver", "Stint", "DryStint", "LapNumber", "TyreLife"]]
    return head.assign(TeamStopLap=np.where(st.has_obs, st.obs_lap, np.nan),
                       FireLap=np.where(fires, st.lap[fire_idx], np.nan),
                       Follows=follows, Weight=w, Y=st.Y).reset_index(drop=True)


def race_influence(st, actions):
    """Contribution of each race to the executable gain: g(all) - g(all but race r)."""
    _, w, _, _ = st.follow(actions, GRACE)
    agg = st._agg(w)
    tot = agg.sum(0)
    g = st._gain(tot)
    races = np.unique(st.race)
    return g, pd.Series([float(g - st._gain(tot - agg[r])) for r in range(len(races))],
                        index=races)


COMP_COL = {"SOFT": "softc", "MEDIUM": "medc", "HARD": "hardc",
            "INTERMEDIATE": "interc", "WET": "blue!50"}


def tikz_race(laps_r, stints_a, stints_b, title):
    """One race as a TikZ picture: a strip per driver (finishing order), coloured by
    compound, team stops as black ticks, rule firing laps as markers, SC/VSC shaded."""
    n_laps = int(laps_r["LapNumber"].max())
    W = 13.0 / n_laps
    last = laps_r.sort_values("LapNumber").groupby("Driver").tail(1)
    order = last.sort_values(["LapNumber", "Position"], ascending=[False, True])["Driver"].tolist()
    H = 0.28
    lines = [r"\begin{tikzpicture}[font=\tiny]"]
    top = 0.0
    bottom = -H * len(order)
    sc = laps_r.groupby("LapNumber")[["SafetyCar", "VSC"]].max()
    for lap, r in sc.iterrows():
        if r["SafetyCar"] or r["VSC"]:
            col = "orange!35" if r["SafetyCar"] else "orange!15"
            lines.append(rf"\fill[{col}] ({(lap - 1) * W:.3f},{bottom - 0.05:.3f}) rectangle "
                         rf"({lap * W:.3f},{top + 0.05:.3f});")
    for k, drv in enumerate(order):
        y = -H * (k + 1)
        d = laps_r[laps_r["Driver"] == drv].sort_values("LapNumber")
        lines.append(rf"\node[anchor=east] at (-0.05,{y + H / 2:.3f}) {{{drv}}};")
        seg = (d["Compound"] != d["Compound"].shift()) | (d["Stint"] != d["Stint"].shift())
        for _, g in d.groupby(seg.cumsum()):
            a, b = g["LapNumber"].min(), g["LapNumber"].max()
            col = COMP_COL.get(g["Compound"].iloc[0], "gray!30")
            lines.append(rf"\fill[{col}] ({(a - 1) * W:.3f},{y + 0.04:.3f}) rectangle "
                         rf"({b * W:.3f},{y + H - 0.04:.3f});")
        for lap in d.loc[d["Pit"] == 1, "LapNumber"]:
            lines.append(rf"\draw[thick] ({lap * W:.3f},{y:.3f}) -- ({lap * W:.3f},{y + H:.3f});")
        for tab, style in ((stints_a, r"fill=blue!70!black, regular polygon, regular polygon sides=3, rotate=180"),
                           (stints_b, r"fill=red!70!black, diamond")):
            for _, s in tab[tab["Driver"] == drv].dropna(subset=["FireLap"]).iterrows():
                x = (s["FireLap"] - 0.5) * W
                lines.append(rf"\node[{style}, inner sep=0pt, minimum size=4pt] at "
                             rf"({x:.3f},{y + H / 2:.3f}) {{}};")
    for lap in range(10, n_laps + 1, 10):
        lines.append(rf"\node[gray] at ({(lap - 0.5) * W:.3f},{bottom - 0.2:.3f}) {{{lap}}};")
    lines.append(rf"\node[anchor=south west, font=\scriptsize] at (0,{top + 0.05:.3f}) {{{title}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def run_cases(laps, D, st_test, thr_rule, tree):
    Rte = D["test"]
    a_thr = rule_mask(Rte, thr_rule)
    a_tree = tree.predict(X(Rte, TREE_FEATURES_P1))
    g, infl = race_influence(st_test, a_thr)
    test_laps = laps[laps["Year"] == 2025]
    stops_sc = (test_laps[(test_laps["Pit"] == 1) & ((test_laps["SafetyCar"] + test_laps["VSC"]) > 0)]
                .groupby("Event").size())
    picks = {}
    # only races with evaluated stints (a wet race like 2025 Melbourne has none)
    picks["Safety car race"] = stops_sc.reindex(infl.index).fillna(0).idxmax()
    street = [e for e in infl.index if test_laps.loc[test_laps["Event"] == e, "Circuit"].iloc[0]
              in STREET and e not in picks.values()]
    picks["Street circuit"] = max(street, key=lambda e: abs(infl[e]))
    rest = infl.drop(list(picks.values()))
    picks["Rule's best race"] = rest.idxmax()
    picks["Rule's worst race"] = rest.drop(picks["Rule's best race"]).idxmin()

    s_thr = stint_table(st_test, Rte, a_thr)
    s_tree = stint_table(st_test, Rte, a_tree)
    s_thr.assign(Rule=rule_str(thr_rule)).to_csv(OUT / "replay_stints_threshold.csv", index=False)
    s_tree.assign(Rule=rule_text(tree)).to_csv(OUT / "replay_stints_tree.csv", index=False)
    test_laps[["Event", "Driver", "LapNumber", "Position", "Compound", "Stint", "TyreLife",
               "Pit", "SafetyCar", "VSC"]].to_csv(OUT / "replay_laps.csv", index=False)
    pd.DataFrame({"Event": infl.index, "influence": infl.values}).to_csv(
        OUT / "race_influence.csv", index=False)

    cases = []
    for k, (kind, ev) in enumerate(picks.items()):
        lr = test_laps[test_laps["Event"] == ev]
        sa, sb = s_thr[s_thr["Event"] == ev], s_tree[s_tree["Event"] == ev]
        title = f"{ev} ({kind.lower()})"
        (REPORT / f"case_{k + 1}.tex").write_text(tikz_race(lr, sa, sb, title))
        fol = sa[sa["Follows"]]
        cases.append({"kind": kind, "event": ev, "influence_s": float(infl[ev]),
                      "stints": int(len(sa)), "followers_threshold": int(len(fol)),
                      "followers_tree": int(sb["Follows"].sum()),
                      "team_stop_age_median": float(np.nanmedian(
                          sa["TeamStopLap"] - sa["LapNumber"] + sa["TyreLife"])),
                      "fire_age_threshold_median": float(np.nanmedian(
                          sa["FireLap"] - sa["LapNumber"] + sa["TyreLife"])),
                      "fire_lap_threshold_median": float(sa["FireLap"].median()),
                      "fire_lap_tree_median": float(sb["FireLap"].median()),
                      "team_stop_lap_median": float(sa["TeamStopLap"].median()),
                      "mean_Y_followers_threshold": float(fol["Y"].mean()) if len(fol) else None,
                      "mean_Y_all": float(sa["Y"].mean())})
        print(f"case {k + 1} [{kind}] {ev}: influence {infl[ev]:+.2f} s, "
              f"{len(fol)}/{len(sa)} stints follow the threshold rule", flush=True)
    return {"threshold_rule": rule_str(thr_rule), "tree_rule": rule_text(tree),
            "gain_threshold_test": float(g), "cases": cases}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-iai", action="store_true")
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    n_boot = 10 if args.quick else args.boot
    laps = pd.read_pickle(DATA)
    results = {"setup": {"p1_spec": P1_SPEC.__dict__, "p1_tree": P1_TREE, "grace": GRACE}}

    # 1. compound choice on both holdouts
    comp_main = None
    results["compound"] = {}
    for name, (ty, tyr) in HOLDOUTS.items():
        res, objs = run_compound(laps, name, ty, tyr, n_boot)
        results["compound"][name] = res
        if name == MAIN:
            comp_main = objs
        print(f"  ({time.time() - t0:.0f}s)", flush=True)

    # Stay/pit setup of P1 on the main holdout (for IAI and the case studies)
    ty, tyr = HOLDOUTS[MAIN]
    tr, te, _, _ = build_split(laps, K="race", train_years=ty, test_year=tyr)
    a, b = nuisance_predictions(tr, te, spec=P1_SPEC)
    at, bt, _ = trim_support(a, b, TRIM)
    D = {"train": a, "test": b, "train_trim": at, "test_trim": bt}
    st_train, st_test = stints(a), stints(b)
    tree = PolicyTree(depth=P1_TREE["depth"], min_leaf_pits=P1_TREE["min_pits"],
                      features=TREE_FEATURES_P1).fit(
        X(at, TREE_FEATURES_P1), tau_dr(at, P1_TREE["clip"]), pits=at["T"].to_numpy())
    _, thr_rule = fit_threshold_exec(st_train, a, literals(a))
    print(f"stay/pit: tree {rule_text(tree)}; threshold {rule_str(thr_rule)}", flush=True)

    # 2. IAI
    if not (args.quick or args.no_iai):
        results["iai"] = run_iai(D, st_test, comp_main)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)

    # 3. case studies
    results["cases"] = run_cases(laps, D, st_test, thr_rule, tree)
    results["runtime_s"] = time.time() - t0
    (OUT / "p2.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT}/p2.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
