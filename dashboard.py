"""Minimal dashboard to explore the OpenF1 lap data before modelling.

Usage:
    streamlit run dashboard.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from openf1 import FEATURES, load_session, race_sessions

YEARS = [2023, 2024, 2025]
DEFAULT_RACES = ["Sakhir", "Barcelona", "Monza", "Silverstone", "Suzuka", "Monaco"]
# Usual F1 tyre colours (hard is white on TV; grey here so it shows on a light page).
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_COLORS = ["#e34948", "#eda100", "#8a8984", "#008300", "#2a78d6"]
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BAR_COLOR = "#2a78d6"

st.set_page_config(page_title="F1 Data Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def sessions_for(year):
    return race_sessions(year)


@st.cache_data(show_spinner=False)
def load_sample(year, session_keys):
    sessions = sessions_for(year).set_index("session_key", drop=False)
    return pd.concat([load_session(sessions.loc[k]) for k in session_keys],
                     ignore_index=True)


def feature_table(laps):
    rows = []
    for col, (role, desc) in FEATURES.items():
        s = laps[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            summary = f"{s.min():.4g} – {s.median():.4g} – {s.max():.4g}"
            kind = "numeric"
        else:
            top = s.value_counts().head(4)
            summary = ", ".join(f"{v} ({n})" for v, n in top.items())
            kind = "binary" if s.nunique() == 2 else "category"
        rows.append({"Feature": col, "Role": role, "Type": kind,
                     "Missing %": round(100 * s.isna().mean(), 1),
                     "Min – median – max / top values": summary,
                     "Description": desc})
    return pd.DataFrame(rows)


def distribution_chart(laps, col):
    s = laps[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
        return alt.Chart(laps).mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=4,
                                        cornerRadiusTopRight=4).encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40), title=col),
            y=alt.Y("count():Q", title="Laps"),
            tooltip=[alt.Tooltip(f"{col}:Q", bin=alt.Bin(maxbins=40)),
                     alt.Tooltip("count():Q", title="Laps")])
    counts = s.astype(str).value_counts().rename_axis(col).reset_index(name="Laps")
    return alt.Chart(counts).mark_bar(color=BAR_COLOR, cornerRadiusTopRight=4,
                                      cornerRadiusBottomRight=4).encode(
        y=alt.Y(f"{col}:N", sort="-x", title=None),
        x=alt.X("Laps:Q"),
        tooltip=[f"{col}:N", "Laps:Q"])


def pit_rate_chart(laps, col, min_laps=20):
    s = laps[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 60:
        key = pd.cut(s, bins=20).apply(lambda b: round(b.mid, 2)).astype(float)
    else:
        key = s
    rate = (laps.assign(_key=key).groupby("_key", observed=True)["Pit"]
            .agg(Rate="mean", Laps="count").reset_index().rename(columns={"_key": col}))
    rate = rate[rate["Laps"] >= min_laps]
    numeric = pd.api.types.is_numeric_dtype(rate[col]) and rate[col].nunique() > 2
    return alt.Chart(rate).mark_bar(color=BAR_COLOR, cornerRadiusTopLeft=4,
                                    cornerRadiusTopRight=4).encode(
        x=alt.X(f"{col}:{'O' if numeric else 'N'}", title=col,
                axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("Rate:Q", title="Share of laps ending in a pit stop",
                axis=alt.Axis(format="%")),
        tooltip=[col, alt.Tooltip("Rate:Q", format=".1%"), "Laps:Q"])


def strategy_chart(race):
    final = race.sort_values("LapNumber").groupby("Driver")["Position"].last().sort_values()
    stints = (race.groupby(["Driver", "Stint"])
              .agg(Compound=("Compound", "first"), Start=("LapNumber", "min"),
                   End=("LapNumber", "max"), StartAge=("TyreLife", "min"))
              .reset_index())
    stints["Laps"] = stints["End"] - stints["Start"] + 1
    stints["x"] = stints["Start"] - 1 + 0.15  # small gap between stints
    stints["x2"] = stints["End"] - 0.15
    stints["Mid"] = (stints["x"] + stints["x2"]) / 2
    stints["Letter"] = stints["Compound"].str[0]

    neutral = (race.groupby("LapNumber")[["SafetyCar", "VSC"]].max().reset_index()
               .melt("LapNumber", var_name="Status").query("value == 1"))
    neutral["x"], neutral["x2"] = neutral["LapNumber"] - 1, neutral["LapNumber"]

    y = alt.Y("Driver:N", sort=list(final.index), title="Driver (by finishing position)")
    shading = alt.Chart(neutral).mark_rect(opacity=0.15, color="#8a8984").encode(
        x="x:Q", x2="x2:Q",
        tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("LapNumber:Q", title="Lap")])
    bars = alt.Chart(stints).mark_bar(cornerRadius=4, height=12).encode(
        x=alt.X("x:Q", title="Lap"), x2="x2:Q", y=y,
        color=alt.Color("Compound:N", scale=alt.Scale(domain=COMPOUNDS, range=COMPOUND_COLORS),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Driver", "Stint", "Compound", "Start", "End", "Laps",
                 alt.Tooltip("StartAge:Q", title="Tyre age at start")])
    labels = alt.Chart(stints[stints["Laps"] >= 4]).mark_text(
        fontSize=9, fontWeight="bold", color="#0b0b0b").encode(
        x="Mid:Q", y=y, text="Letter:N")
    return (shading + bars + labels).properties(height=max(300, 22 * len(final)))


def lap_time_chart(race, drivers):
    d = race[race["Driver"].isin(drivers)]
    d = d[(d["PitOutLap"] == 0) & (d["Pit"] == 0) & (d["LapNumber"] > 1)]
    d = d[d["LapTime"] < 1.07 * race["LapTime"].median()]  # drop SC / slow laps
    color = alt.Color("Driver:N", scale=alt.Scale(domain=drivers,
                                                  range=SERIES_COLORS[:len(drivers)]),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(d).encode(x=alt.X("LapNumber:Q", title="Lap"),
                               y=alt.Y("LapTime:Q", title="Lap time (s)",
                                       scale=alt.Scale(zero=False)),
                               color=color)
    return (base.mark_line(strokeWidth=2)
            + base.mark_point(size=40, filled=True).encode(
                tooltip=["Driver", "LapNumber", alt.Tooltip("LapTime:Q", format=".3f"),
                         "Compound", "TyreLife"])).properties(height=320)


# ---- Sidebar: choose the sample -------------------------------------------
st.sidebar.header("Sample")
year = st.sidebar.selectbox("Season", YEARS, index=0)
sessions = sessions_for(year)
labels = dict(zip(sessions["label"], sessions["session_key"]))
default = [l for l in labels if any(r in l for r in DEFAULT_RACES)] or list(labels)[:6]
chosen = st.sidebar.multiselect("Races", list(labels), default=default)
st.sidebar.caption("Data: OpenF1 API (2023 onward). The first load of a race "
                   "downloads it; later loads come from ./cache.")

if not chosen:
    st.info("Pick at least one race in the sidebar.")
    st.stop()

with st.spinner(f"Loading {len(chosen)} races…"):
    laps = load_sample(year, tuple(labels[c] for c in chosen))

# ---- Overview ---------------------------------------------------------------
st.title("F1 race data explorer")
st.caption("One row per driver and lap. No model yet: this page is for seeing "
           "which features we have and what they look like.")

neutral_laps = laps.groupby(["Event", "LapNumber"])[["SafetyCar", "VSC"]].max().max(axis=1).sum()
c = st.columns(5)
c[0].metric("Races", laps["Event"].nunique())
c[1].metric("Drivers", laps["Driver"].nunique())
c[2].metric("Driver-laps", f"{len(laps):,}")
c[3].metric("Pit stops", int(laps["Pit"].sum()))
c[4].metric("SC / VSC laps", int(neutral_laps))

# ---- Features -----------------------------------------------------------------
st.subheader("Features")
st.caption("Role follows the proposal: context x (what the strategist sees), "
           "action T (pit or not), outcome y (built from lap times later).")
st.dataframe(feature_table(laps), hide_index=True, width="stretch",
             height=35 * (len(FEATURES) + 1) + 3)

# ---- One feature at a time ----------------------------------------------------
st.subheader("One feature at a time")
explorable = [f for f, (role, _) in FEATURES.items() if role != "id" and f != "Pit"]
col = st.selectbox("Feature", explorable, index=explorable.index("TyreLife"))
left, right = st.columns(2)
left.markdown(f"**Distribution of {col}**")
left.altair_chart(distribution_chart(laps, col).properties(height=300),
                  width="stretch")
right.markdown(f"**Pit rate by {col}**")
right.altair_chart(pit_rate_chart(laps, col).properties(height=300),
                   width="stretch")
right.caption("Values with fewer than 20 laps are hidden.")

# ---- One race -------------------------------------------------------------------
st.subheader("One race")
event = st.selectbox("Race", sorted(laps["Event"].unique()))
race = laps[laps["Event"] == event]
st.markdown("**Tyre strategies** (grey bands: safety car / VSC)")
st.altair_chart(strategy_chart(race), width="stretch")

order = race.sort_values("LapNumber").groupby("Driver")["Position"].last().sort_values()
drivers = st.multiselect("Drivers (up to 4)", list(order.index),
                         default=list(order.index[:3]), max_selections=4,
                         key=f"drivers_{event}")
if drivers:
    st.markdown("**Lap times** (pit in/out laps and laps over 107% of the median removed)")
    st.altair_chart(lap_time_chart(race, drivers), width="stretch")

# ---- Raw rows -------------------------------------------------------------------
st.subheader("Rows")
st.dataframe(race.head(200), hide_index=True, width="stretch")
st.download_button("Download the whole sample as CSV", laps.to_csv(index=False),
                   file_name=f"openf1_sample_{year}.csv", mime="text/csv")
