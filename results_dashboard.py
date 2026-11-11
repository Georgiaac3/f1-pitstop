"""Results of the pit-stop policy model (reads results/ from run_experiment.py).

Usage:
    streamlit run results_dashboard.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RES = Path(__file__).parent / "results"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#8a8984"

st.set_page_config(page_title="F1 Pit Policy Results", layout="wide")

if not (RES / "results.json").exists():
    st.error("No results yet. Run `python run_experiment.py` first.")
    st.stop()

R = json.loads((RES / "results.json").read_text())
setup, S = R["setup"], R["sanity"]
pol = pd.DataFrame(R["policies"])
by_race = pd.read_csv(RES / "by_race.csv")
rows = pd.read_csv(RES / "test_rows.csv")
overlap = pd.read_csv(RES / "overlap.csv")

# ---- Header ---------------------------------------------------------------
st.title("Interpretable pit-stop policy: results")
st.caption(
    f"Trained on {setup['train_years']} ({setup['races_train']} races), evaluated off-policy "
    f"on {setup['test_year']} ({setup['races_test']} held-out races). Outcome: seconds gained "
    f"on the field median over the next K = {setup['K']} laps, minus the circuit pit loss if "
    f"the driver still owes the two-compound stop. Gains are doubly-robust estimates "
    f"relative to what teams actually did; bands are 95% bootstrap intervals over races.")

tree_row = pol[pol["policy"] == f"DR tree (depth {R['depth']})"].iloc[0]
c = st.columns(4)
c[0].metric("Chosen policy", f"DR tree, depth {R['depth']}",
            help="Depth picked by grouped (by race) 5-fold CV on training seasons.")
c[1].metric("Est. gain vs teams (per decision)", f"{tree_row['gain']:+.2f} s",
            help="Mean over test decision points of Γ_DR(π(x)) − y, seconds over K laps.")
c[2].metric("95% CI", f"[{tree_row['lo']:+.2f}, {tree_row['hi']:+.2f}] s")
c[3].metric("Pit rate: tree vs teams",
            f"{tree_row['pit_rate']:.1%} vs {pol.iloc[0]['pit_rate']:.1%}")

# ---- The rule -------------------------------------------------------------
st.subheader("The learned rule")
rules = pd.DataFrame(R["rules"])
st.dataframe(rules.rename(columns={"rule": "If", "action": "Then", "n": "Training rows"}),
             hide_index=True, width="stretch")
with st.expander("Trees at other depths, and the DM tree"):
    for d, rs in R["trees_by_depth"].items():
        st.markdown(f"**DR tree, depth {d}** (CV gain {R['cv_by_depth'][str(d)]:+.2f} s)")
        st.dataframe(pd.DataFrame(rs), hide_index=True, width="stretch")
    st.markdown(f"**DM tree, depth {R['depth']}** (no propensity correction)")
    st.dataframe(pd.DataFrame(R["dm_tree_rules"]), hide_index=True, width="stretch")

# ---- Policy comparison ----------------------------------------------------
st.subheader("Policies compared on held-out races")
order = list(pol["policy"])
base = alt.Chart(pol).encode(y=alt.Y("policy:N", sort=order, title=None))
bars = base.mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                     height=14).encode(
    x=alt.X("gain:Q", title="Estimated gain vs historical (s over K laps, per decision)"),
    tooltip=["policy", alt.Tooltip("gain:Q", format="+.2f"), alt.Tooltip("lo:Q", format="+.2f"),
             alt.Tooltip("hi:Q", format="+.2f"), alt.Tooltip("pit_rate:Q", format=".1%"),
             alt.Tooltip("train_gain:Q", format="+.2f", title="train gain")])
err = base.mark_rule(color="#3b3b3b", strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY).encode(x="x:Q")
st.altair_chart((bars + err + zero).properties(height=36 * len(pol)), width="stretch")
st.dataframe(pol[["policy", "gain", "lo", "hi", "pit_rate", "train_gain"]].round(3),
             hide_index=True, width="stretch")
st.caption("Pit rate = share of decision points where the policy says pit. A policy that "
           "pits on many consecutive laps is not executable as-is: the evaluation is "
           "one-shot (myopic), as stated in the proposal.")

# ---- By race ---------------------------------------------------------------
left, right = st.columns([2, 1])
left.markdown("**Gain by held-out race (chosen tree)**")
left.altair_chart(alt.Chart(by_race).mark_bar(cornerRadiusTopRight=4,
                                              cornerRadiusBottomRight=4, height=10).encode(
    y=alt.Y("Event:N", sort="-x", title=None),
    x=alt.X("gain:Q", title="Mean gain per decision (s)"),
    color=alt.Color("Street:N", scale=alt.Scale(domain=[False, True], range=[BLUE, ORANGE]),
                    legend=alt.Legend(title="Street circuit", orient="top")),
    tooltip=["Event", alt.Tooltip("gain:Q", format="+.2f"), "rows", "pits", "tree_pits"]
).properties(height=max(300, 18 * len(by_race))), width="stretch")
right.markdown("**By circuit type**")
right.dataframe(pd.DataFrame(R["by_type"]).round(2), hide_index=True, width="stretch")

# ---- Sanity checks ---------------------------------------------------------
st.subheader("Sanity checks")
c = st.columns(4)
c[0].metric("Propensity AUC (test)", f"{S['propensity_auc_test']:.3f}")
c[1].metric("Propensity Brier (test)", f"{S['propensity_brier_test']:.4f}")
c[2].metric("Outcome R², stay (test)", f"{S['outcome_r2_test_T0']:.2f}")
c[3].metric("Outcome R², pit (test)", f"{S['outcome_r2_test_T1']:.2f}")

a, b = st.columns(2)
a.markdown("**Effect of pitting vs staying out, by track status** (test rows)")
a.dataframe(pd.DataFrame(S["effect_by_track_status"]).round(2), hide_index=True,
            width="stretch")
a.caption("Expect a stop to be much cheaper under a safety car or VSC.")
b.markdown("**Tyre degradation by compound** (median within-stint slope)")
b.dataframe(pd.DataFrame(S["degradation"]).round(3), hide_index=True, width="stretch")
b.caption("Expect SOFT > MEDIUM > HARD. Slopes include fuel burn-off (≈ −0.06 s/lap).")

eff = pd.DataFrame(S["effect_by_tyre_age"]).melt(
    ["tyre_age", "rows", "pits"], ["dr_effect", "dm_effect"], "Estimator", "Effect")
eff["Estimator"] = eff["Estimator"].map({"dr_effect": "DR", "dm_effect": "DM"})
st.markdown("**Estimated effect of pitting by tyre age** (s over K laps; test rows)")
st.altair_chart(alt.Chart(eff).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
    x=alt.X("tyre_age:N", sort=None, title="Tyre age (laps)"),
    y=alt.Y("Effect:Q", title="Pit minus stay (s)"),
    color=alt.Color("Estimator:N", scale=alt.Scale(domain=["DR", "DM"], range=[BLUE, ORANGE]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["tyre_age", "Estimator", alt.Tooltip("Effect:Q", format="+.2f"), "rows", "pits"]
).properties(height=280), width="stretch")

st.markdown("**Overlap: estimated pit propensity by action and race phase** (before trimming)")
split = overlap.groupby(["Phase", "Action", "Bin"], as_index=False)["Rows"].sum()
st.altair_chart(alt.Chart(split).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=alt.X("Bin:Q", title="Estimated Pr(pit | x)", scale=alt.Scale(domain=[0, 0.6])),
    y=alt.Y("Rows:Q", title="Rows", stack=None),
    color=alt.Color("Action:N", scale=alt.Scale(domain=["Stayed out", "Pitted"],
                                                range=[BLUE, ORANGE]),
                    legend=alt.Legend(orient="top", title=None)),
    opacity=alt.value(0.75),
    column=alt.Column("Phase:N", sort=["Early", "Middle", "Late"], title=None),
    tooltip=["Phase", "Action", "Bin", "Rows"]).properties(height=220, width=260))
st.caption(f"Rows with propensity < {setup['trim']} or > {1 - setup['trim']} are dropped: "
           f"{setup['dropped_train']} train rows ({setup['dropped_pits_train']} pits), "
           f"{setup['dropped_test']} test rows ({setup['dropped_pits_test']} pits).")

# ---- Sensitivity -------------------------------------------------------------
if "sensitivity" in R:
    st.subheader("Sensitivity (depth-2 DR tree, full pipeline refit)")
    sens = pd.DataFrame(R["sensitivity"])
    sens["label"] = sens["analysis"] + ": " + sens["setting"]
    st.altair_chart((alt.Chart(sens).mark_point(size=80, filled=True, color=BLUE).encode(
        y=alt.Y("label:N", sort=None, title=None), x=alt.X("gain:Q", title="Gain (s)"),
        tooltip=["label", alt.Tooltip("gain:Q", format="+.2f"), "rule"])
        + alt.Chart(sens).mark_rule(color=BLUE, strokeWidth=2).encode(
            y=alt.Y("label:N", sort=None), x="lo:Q", x2="hi:Q")
        + zero).properties(height=34 * len(sens)), width="stretch")
    st.dataframe(sens[["analysis", "setting", "gain", "lo", "hi", "pit_rate", "rule"]].round(3),
                 hide_index=True, width="stretch")
    st.caption("Gains are not comparable across K or debt settings (the outcome changes); "
               "look at sign, CI and whether the rule stays the same.")

# ---- One race ----------------------------------------------------------------
st.subheader("Tree vs team, lap by lap")
ev = st.selectbox("Held-out race", sorted(rows["Event"].unique()))
r = rows[rows["Event"] == ev]
drv = st.selectbox("Driver", sorted(r["Driver"].unique()))
d = r[r["Driver"] == drv].sort_values("LapNumber")
d = d.assign(Team_pit=d["T"].map({0: "", 1: "pitted"}))
base = alt.Chart(d).encode(x=alt.X("LapNumber:Q", title="Lap"))
line = base.mark_line(color=BLUE, strokeWidth=2).encode(
    y=alt.Y("tau_dr:Q", title="DR estimated gain from pitting (s)"),
    tooltip=["LapNumber", "Compound", "TyreLife", "Position",
             alt.Tooltip("tau_dr:Q", format="+.1f"), alt.Tooltip("p1:Q", format=".3f")])
tree_marks = base.transform_filter("datum.TreeAction == 1").mark_tick(
    color=GREEN, thickness=3, size=18).encode(y=alt.value(8))
team_marks = base.transform_filter("datum.T == 1").mark_point(
    shape="triangle-down", size=120, filled=True, color=ORANGE).encode(y=alt.value(24))
st.altair_chart((line + tree_marks + team_marks).properties(height=300), width="stretch")
st.caption("Green ticks (top): laps where the tree says pit. Orange triangles: laps the "
           "team actually pitted. Line: per-row DR score (noisy by design; only averages "
           "are meaningful). Rows exist only where a stop was a real option.")
