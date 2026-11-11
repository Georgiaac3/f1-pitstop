"""Interpretable pit-stop policies from observational OpenF1 data (15.095 proposal).

Pipeline (Section 2 of the proposal):
  1. decision rows: one per (driver, lap) where a stop is a real option
  2. outcome y: seconds gained over the next K laps on the median of the cars within
     +-3 positions, minus a "stop debt" (the circuit's typical in-window cost of a
     green-flag stop) if the driver still owes the mandatory second dry compound
     when the window ends; see build_rows
  3. nuisances, cross-fitted by race: outcome models f_t(x) and propensity p_t(x)
  4. rewards Gamma^DM and Gamma^DR (doubly robust)
  5. optimal policy tree of depth <= 3, exhaustive search over quantile splits
Evaluation (Section 3): train on 2023-24, hold out 2025, DR off-policy value with
nuisances fit on training races only, bootstrap CIs over races.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

DRY = ["SOFT", "MEDIUM", "HARD"]
BIG_GAP = 30.0  # stands in for "no car ahead/behind" or "a lap or more"

# Context the policy tree may split on (all numeric, all visible to a strategist).
TREE_FEATURES = ["TyreLife", "RaceFraction", "LapsRemaining", "CompoundCode",
                 "TwoCompoundsDone", "Stint", "Position", "GapAhead", "GapBehind",
                 "SafetyCar", "VSC", "TrackTemp", "PitLoss"]
# Nuisance models also get team and circuit effects (proposal: confounding).
CATEGORICAL = ["Team", "Circuit"]


# --------------------------------------------------------------------------
# 1-2. Decision rows and outcome
# --------------------------------------------------------------------------

def estimate_pit_loss(laps):
    """Green-flag time lost to a stop, per circuit: in-lap + out-lap minus two
    of the driver's typical green laps. Returns {circuit: seconds} and a global median."""
    laps = laps.sort_values(["Event", "Driver", "LapNumber"])
    clean = laps[(laps["Pit"] == 0) & (laps["PitOutLap"] == 0) & (laps["SafetyCar"] == 0)
                 & (laps["VSC"] == 0) & (laps["LapNumber"] > 1)]
    typical = clean.groupby(["Event", "Driver"])["LapTime"].median().rename("Typical")
    nxt = laps.groupby(["Event", "Driver"])["LapTime"].shift(-1)
    nxt_sc = laps.groupby(["Event", "Driver"])[["SafetyCar", "VSC"]].shift(-1).max(axis=1)
    stops = laps.assign(OutLap=nxt, NextNeutral=nxt_sc)
    stops = stops[(stops["Pit"] == 1) & (stops["SafetyCar"] == 0) & (stops["VSC"] == 0)
                  & (stops["NextNeutral"] == 0)]
    stops = stops.join(typical, on=["Event", "Driver"])
    stops["Loss"] = stops["LapTime"] + stops["OutLap"] - 2 * stops["Typical"]
    stops = stops[stops["Loss"].between(5, 60)]  # drop broken laps / long repairs
    return stops.groupby("Circuit")["Loss"].median().to_dict(), float(stops["Loss"].median())


def estimate_stop_cost(laps):
    """Typical net cost of a green-flag stop inside the K-lap window, per circuit:
    minus the median Gain of drivers who pitted at the end of a green lap.
    `laps` must already carry the Gain column for the K in use."""
    neutral = (laps["SafetyCar"] + laps["VSC"]) > 0
    stops = laps[(laps["Pit"] == 1) & ~neutral & laps["Gain"].notna()
                 & laps["Compound"].isin(DRY)]
    return ((-stops.groupby("Circuit")["Gain"].median()).to_dict(),
            float(-stops["Gain"].median()))


def local_median(df, value="Elapsed", window=3):
    """Median of `value` over the other cars within +-window positions on the same lap."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby(["Event", "LapNumber"]):
        pos, v = g["Position"].to_numpy(), g[value].to_numpy()
        near = (np.abs(pos[:, None] - pos[None, :]) <= window) & ~np.eye(len(g), dtype=bool)
        near &= ~np.isnan(v)[None, :]
        vals = np.where(near, v[None, :], np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(vals, axis=1) if len(g) > 1 else np.full(len(g), np.nan)
        out[g.index] = med
    return out


def build_rows(laps, K=10, min_tyre_age=5, stop_debt=True, pit_loss=None, stop_cost=None,
               reference="local"):
    """One row per (driver, lap) decision point with context x, action T, outcome y.

    y = (reference elapsed time over laps L..L+K-1) - (driver's elapsed time), where the
    reference is the median of the cars within +-3 positions at lap L ("local") or of
    the whole field ("field"). Whole-field seconds are distorted by safety cars: the
    bunching erases leaders' margins whether or not they pit.
    The in-lap (L) and out-lap (L+1) of a stop at the end of lap L are both inside.
    With stop_debt, drivers who end the window still on one dry compound (so they
    must still stop) are charged `stop_cost`: the circuit's typical net cost of a
    green-flag stop measured over the same K-lap window (pit-lane loss minus what
    fresh tyres win back inside the window). Charging the full pit-lane loss
    instead would ignore the deferred stop's own fresh-tyre benefit and bias the
    policy toward pitting; charging nothing makes every stop look ~8-20 s worse.
    `pit_loss` / `stop_cost` are ({circuit: s}, overall) and should come from
    training races; if None they are estimated from `laps`.
    """
    laps = laps.sort_values(["Event", "Driver", "LapNumber"]).copy()
    g = laps.groupby(["Event", "Driver"])

    # Elapsed race time from the start of lap L to the start of lap L+K. Rows are
    # shifted, so require the row K ahead to really be lap L+K (laps with no timing
    # are missing from the table, and a gap would silently stretch the window).
    whole_window = g["LapNumber"].shift(-K) == laps["LapNumber"] + K
    laps["Elapsed"] = ((g["LapStart"].shift(-K) - laps["LapStart"]).dt.total_seconds()
                       .where(whole_window))
    if reference == "local":  # cars within +-3 positions: field bunching (SC) cancels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            laps["FieldElapsed"] = local_median(laps)
    else:                     # whole-field median, as first written in the proposal
        laps["FieldElapsed"] = laps.groupby(["Event", "LapNumber"])["Elapsed"].transform("median")
    laps["Gain"] = laps["FieldElapsed"] - laps["Elapsed"]

    # Dry compounds used so far, and whether the rule is met by the window's end.
    dry = laps["Compound"].where(laps["Compound"].isin(DRY))
    for c in DRY:
        laps[f"Used{c}"] = (dry == c).astype(int).groupby([laps["Event"], laps["Driver"]]).cummax()
    used = laps[[f"Used{c}" for c in DRY]].sum(axis=1)
    laps["TwoCompoundsDone"] = (used >= 2).astype(int)
    laps["DoneByWindowEnd"] = g["TwoCompoundsDone"].shift(-(K - 1))

    if pit_loss is None:
        pit_loss = estimate_pit_loss(laps)
    by_circuit, overall = pit_loss
    laps["PitLoss"] = laps["Circuit"].map(by_circuit).fillna(overall)
    if stop_cost is None:
        stop_cost = estimate_stop_cost(laps)
    laps["StopCost"] = laps["Circuit"].map(stop_cost[0]).fillna(stop_cost[1])
    laps["Debt"] = (1 - laps["DoneByWindowEnd"]) * laps["StopCost"] if stop_debt else 0.0
    laps["y"] = laps["Gain"] - laps["Debt"]

    # A lap is "wet" if anyone on track is on intermediates or wets.
    wet = laps.groupby(["Event", "LapNumber"])["Compound"].transform(
        lambda s: s.isin(["INTERMEDIATE", "WET"]).any())
    wet_ahead = wet.groupby([laps["Event"], laps["Driver"]]).transform(
        lambda s: s[::-1].rolling(K, min_periods=1).max()[::-1])

    # Windows containing a red flag (race suspended) or a long repair are not
    # strategy decisions: drop windows over twice the normal K-lap time (theirs or
    # the reference's).
    normal = K * laps.groupby("Event")["LapTime"].transform("median")
    abnormal = (laps["Elapsed"] > 2 * normal) | (laps["FieldElapsed"] > 2 * normal)

    keep = (~abnormal & laps["Compound"].isin(DRY) & (laps["LapNumber"] >= 2)
            & (laps["LapNumber"] + K <= laps["TotalLaps"]) & (laps["PitOutLap"] == 0)
            & (laps["TyreLife"] >= min_tyre_age) & (wet_ahead == 0) & (laps["Rainfall"] == 0)
            & laps["y"].notna() & laps["DoneByWindowEnd"].notna())
    rows = laps[keep].copy()

    rows["LapsRemaining"] = rows["TotalLaps"] - rows["LapNumber"]
    rows["CompoundCode"] = rows["Compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2})
    rows["GapAhead"] = rows["GapAhead"].fillna(BIG_GAP).clip(upper=BIG_GAP)
    rows["GapBehind"] = rows["GapBehind"].fillna(BIG_GAP).clip(upper=BIG_GAP)
    rows["Position"] = rows["Position"].fillna(rows["Position"].median())
    for c in ["TrackTemp", "AirTemp"]:
        rows[c] = rows[c].fillna(rows.groupby("Event")[c].transform("median"))
    rows["T"] = rows["Pit"].astype(int)
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Nuisance models
# --------------------------------------------------------------------------

def _design(df):
    X = df[TREE_FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = df[c].astype(str)
    return X


def _outcome_model():
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=[len(TREE_FEATURES) + i
                                                     for i in range(len(CATEGORICAL))],
        random_state=0)


def _encode_for_hgb(X, cats):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=cats[c]).codes.astype(float)
        X.loc[X[c] < 0, c] = np.nan
    return X


def _propensity_model():
    num_smooth = ["TyreLife", "RaceFraction", "LapsRemaining"]
    num_lin = [f for f in TREE_FEATURES if f not in num_smooth]
    pre = ColumnTransformer([
        ("spline", SplineTransformer(n_knots=6, degree=3), num_smooth),
        ("lin", StandardScaler(), num_lin),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000))


def fit_nuisances(train):
    """Fit f_0, f_1 and the pit propensity on `train`; return a predict function."""
    X = _design(train)
    cats = {c: sorted(X[c].unique()) for c in CATEGORICAL}
    Xh = _encode_for_hgb(X, cats)
    f = {}
    for t in (0, 1):
        m = train["T"] == t
        f[t] = _outcome_model().fit(Xh[m], train.loc[m, "y"])
    prop = _propensity_model().fit(X, train["T"])

    def predict(df):
        Xd = _design(df)
        Xdh = _encode_for_hgb(Xd, cats)
        return f[0].predict(Xdh), f[1].predict(Xdh), prop.predict_proba(Xd)[:, 1]
    return predict


def nuisance_predictions(train, test, n_folds=5):
    """Cross-fitted (by race) predictions on train; train-only fit for test."""
    tr = np.zeros((len(train), 3))
    for fit_idx, pred_idx in GroupKFold(n_splits=n_folds).split(train, groups=train["Event"]):
        predict = fit_nuisances(train.iloc[fit_idx])
        tr[pred_idx] = np.column_stack(predict(train.iloc[pred_idx]))
    te = np.column_stack(fit_nuisances(train)(test))
    cols = ["f0", "f1", "p1"]
    return (train.assign(**dict(zip(cols, tr.T))), test.assign(**dict(zip(cols, te.T))))


# --------------------------------------------------------------------------
# 4. Rewards
# --------------------------------------------------------------------------

def rewards(df, clip=0.01):
    """Gamma^DM and Gamma^DR as (n, 2) arrays; columns = stay, pit."""
    f = df[["f0", "f1"]].to_numpy()
    p1 = df["p1"].clip(clip, 1 - clip).to_numpy()
    p = np.column_stack([1 - p1, p1])
    T = df["T"].to_numpy()
    y = df["y"].to_numpy()
    onehot = np.column_stack([T == 0, T == 1]).astype(float)
    dr = f + onehot / p * (y - f[np.arange(len(T)), T])[:, None]
    return f, dr


# --------------------------------------------------------------------------
# 5. Optimal policy tree (exhaustive search over binned splits)
# --------------------------------------------------------------------------
# With two actions, sum_i Gamma[i, pi(x_i)] = sum_i Gamma[i, stay] + sum_{pi=pit} tau_i
# with tau = Gamma[:, pit] - Gamma[:, stay], so each leaf just checks sign(sum tau).

class PolicyTree:
    def __init__(self, depth=2, n_bins=16, min_leaf_frac=0.02, features=TREE_FEATURES):
        self.depth, self.n_bins, self.min_leaf_frac = depth, n_bins, min_leaf_frac
        self.features = list(features)

    # -- binning --
    def _fit_bins(self, X):
        self.thresholds = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.n_bins:
                thr = u[:-1]
            else:
                thr = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.thresholds.append(thr)
        self.q = max(len(t) for t in self.thresholds) + 1

    def _bin(self, X):
        # bin b means x <= thr[b] (b < len(thr)), last bin is "above every threshold"
        return np.column_stack([np.searchsorted(t, X[:, j], side="left")
                                for j, t in enumerate(self.thresholds)])

    # -- search --
    def _leaf(self, tau_sum, n):
        return {"action": int(tau_sum > 0), "n": int(n), "tau_sum": float(tau_sum)}

    def _hist(self, idx):
        """Per (feature, bin): sum of tau and count, shape (p, q)."""
        p, q = self.B.shape[1], self.q
        keys = (self.B[idx] + np.arange(p) * q).ravel()
        s = np.bincount(keys, weights=np.repeat(self.tau[idx], p), minlength=p * q)
        c = np.bincount(keys, minlength=p * q)
        return s.reshape(p, q), c.reshape(p, q)

    def _best1_from_hist(self, s, c):
        """Best depth-1 value per (..., feature): returns value, bin."""
        Ls, Lc = np.cumsum(s, -1)[..., :-1], np.cumsum(c, -1)[..., :-1]
        Ts, Tc = s.sum(-1, keepdims=True), c.sum(-1, keepdims=True)
        val = np.maximum(Ls, 0) + np.maximum(Ts - Ls, 0)
        ok = (Lc >= self.min_leaf) & (Tc - Lc >= self.min_leaf)
        val = np.where(ok, val, -np.inf)
        return val.max(-1), val.argmax(-1)

    def _depth1(self, idx):
        total = self.tau[idx].sum()
        best = (max(total, 0), None)
        if len(idx) >= 2 * self.min_leaf:
            s, c = self._hist(idx)
            v, b = self._best1_from_hist(s, c)
            j = int(np.argmax(v))
            if v[j] > best[0] + 1e-9:
                best = (v[j], (j, int(b[j])))
        return best

    def _depth2(self, idx):
        """Exact best depth<=2 tree on idx: O(n p^2) via joint histograms."""
        v1, split1 = self._depth1(idx)
        best = (v1, ("d1", split1))
        if len(idx) < 2 * self.min_leaf:
            return best
        p, q = self.B.shape[1], self.q
        Bi, ti = self.B[idx], self.tau[idx]
        tot_s, tot_c = self._hist(idx)
        for j in range(p):
            keys = (Bi[:, [j]] * (p * q) + np.arange(p) * q + Bi).ravel()
            s = np.bincount(keys, weights=np.repeat(ti, p), minlength=q * p * q).reshape(q, p, q)
            c = np.bincount(keys, minlength=q * p * q).reshape(q, p, q)
            Ls, Lc = np.cumsum(s, 0)[:-1], np.cumsum(c, 0)[:-1]       # left child per root bin
            Rs, Rc = tot_s[None] - Ls, tot_c[None] - Lc
            nL, nR = Lc[:, 0].sum(-1), Rc[:, 0].sum(-1)
            sumL, sumR = Ls[:, 0].sum(-1), Rs[:, 0].sum(-1)
            bestL = np.maximum(self._best1_from_hist(Ls, Lc)[0].max(-1), np.maximum(sumL, 0))
            bestR = np.maximum(self._best1_from_hist(Rs, Rc)[0].max(-1), np.maximum(sumR, 0))
            val = np.where((nL >= self.min_leaf) & (nR >= self.min_leaf), bestL + bestR, -np.inf)
            b = int(np.argmax(val))
            if val[b] > best[0] + 1e-9:
                best = (val[b], ("d2", (j, b)))
        return best

    def _split(self, idx, j, b):
        m = self.B[idx, j] <= b
        return idx[m], idx[~m]

    def _build(self, idx, kind_split):
        kind, sp = kind_split
        if sp is None:
            return self._leaf(self.tau[idx].sum(), len(idx))
        j, b = sp
        left, right = self._split(idx, j, b)
        if kind == "d1":    # children are leaves
            lk = rk = ("d1", None)
        elif kind == "d2":  # children are best depth<=1 trees
            lk, rk = ("d1", self._depth1(left)[1]), ("d1", self._depth1(right)[1])
        else:               # "d3": children are best depth<=2 trees
            lk, rk = self._depth2(left)[1], self._depth2(right)[1]
        return {"feature": self.features[j], "threshold": float(self.thresholds[j][b]),
                "n": int(len(idx)), "left": self._build(left, lk), "right": self._build(right, rk)}

    def fit(self, X, tau):
        X = np.asarray(X, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.min_leaf = max(1, int(self.min_leaf_frac * len(X)))
        self._fit_bins(X)
        self.B = self._bin(X)
        idx = np.arange(len(X))
        if self.depth <= 1:
            best = self._depth1(idx)
            best = (best[0], ("d1", best[1]))
        elif self.depth == 2:
            best = self._depth2(idx)
        else:  # depth 3: exhaustive root split, exact depth-2 subtrees
            best = self._depth2(idx)
            p = self.B.shape[1]
            for j in range(p):
                for b in range(len(self.thresholds[j])):
                    left, right = self._split(idx, j, b)
                    if len(left) < self.min_leaf or len(right) < self.min_leaf:
                        continue
                    v = self._depth2(left)[0] + self._depth2(right)[0]
                    if v > best[0] + 1e-9:
                        best = (v, ("d3", (j, b)))
        self.tree_ = self._build(idx, best[1])
        del self.B, self.tau
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X), dtype=int)
        self._predict(self.tree_, X, np.arange(len(X)), out)
        return out

    def _predict(self, node, X, idx, out):
        if "action" in node:
            out[idx] = node["action"]
            return
        m = X[idx, self.features.index(node["feature"])] <= node["threshold"]
        self._predict(node["left"], X, idx[m], out)
        self._predict(node["right"], X, idx[~m], out)

    def rules(self, node=None, conds=()):
        """Leaves as human-readable rules."""
        node = node or self.tree_
        if "action" in node:
            return [{"rule": " and ".join(conds) or "always", "action":
                     "PIT" if node["action"] else "STAY OUT", "n": node["n"]}]
        f, t = node["feature"], node["threshold"]
        return (self.rules(node["left"], conds + (f"{f} <= {t:g}",))
                + self.rules(node["right"], conds + (f"{f} > {t:g}",)))


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def policy_value(gamma, actions):
    return gamma[np.arange(len(actions)), actions.astype(int)]


def bootstrap_gain(per_row_gain, groups, n_boot=1000, seed=0):
    """Mean gain with a 95% CI from resampling races."""
    df = pd.DataFrame({"g": per_row_gain, "e": groups})
    agg = df.groupby("e")["g"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    k = len(agg)
    draws = rng.integers(0, k, size=(n_boot, k))
    s, c = agg["sum"].to_numpy()[draws].sum(1), agg["count"].to_numpy()[draws].sum(1)
    est = s / c
    return float(per_row_gain.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))
