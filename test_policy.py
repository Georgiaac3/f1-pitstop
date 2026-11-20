"""Tests for the policy-tree solver, the DR rewards and the race-control parser.

Run with:  python -m pytest -q
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from openf1 import track_status_by_lap
from policy import PolicyTree, rewards


# --------------------------------------------------------------------------
# Policy tree: compare the fast search with brute-force enumeration
# --------------------------------------------------------------------------

def _value(tree, X, tau):
    """Objective the tree maximises: sum of tau over rows it sends to "pit"."""
    return float(tau[tree.predict(X) == 1].sum())


def _brute_force(X, tau, thresholds, min_leaf, depth, pits=None, min_pits=0):
    """Best sum over leaves of max(sum tau, 0), trying every split on the same thresholds."""
    pits = np.zeros(len(X)) if pits is None else pits

    def ok(i):
        return len(i) >= min_leaf and pits[i].sum() >= min_pits

    def best(idx, d):
        v = max(tau[idx].sum(), 0.0)  # a leaf is always allowed
        if d == 0:
            return v
        for j, thr in enumerate(thresholds):
            for t in thr:
                m = X[idx, j] <= t
                left, right = idx[m], idx[~m]
                if ok(left) and ok(right):
                    v = max(v, best(left, d - 1) + best(right, d - 1))
        return v
    return best(np.arange(len(X)), depth)


def _toy(seed, n=240, p=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, size=(n, p)).astype(float)
    # Signal on an interaction, so depth 2 genuinely beats depth 1.
    tau = np.where((X[:, 0] > 2) & (X[:, 1] <= 3), 1.0, -1.0) + rng.normal(0, 1.5, n)
    return X, tau


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("depth", [1, 2])
def test_depth2_search_is_exact(seed, depth):
    X, tau = _toy(seed)
    tree = PolicyTree(depth=depth, features=["a", "b", "c"], min_leaf_frac=0.05).fit(X, tau)
    expected = _brute_force(X, tau, tree.thresholds, tree.min_leaf, depth)
    assert _value(tree, X, tau) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(2))
def test_depth3_search_is_exact(seed):
    X, tau = _toy(seed, n=150)
    tree = PolicyTree(depth=3, features=["a", "b", "c"], min_leaf_frac=0.05).fit(X, tau)
    expected = _brute_force(X, tau, tree.thresholds, tree.min_leaf, 3)
    assert _value(tree, X, tau) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_min_stops_per_leaf_search_is_exact(seed, depth):
    X, tau = _toy(seed, n=150 if depth == 3 else 240)
    pits = (np.random.default_rng(seed + 10).random(len(X)) < 0.15).astype(float)
    tree = PolicyTree(depth=depth, features=["a", "b", "c"], min_leaf_frac=0.05,
                      min_leaf_pits=5).fit(X, tau, pits=pits)
    expected = _brute_force(X, tau, tree.thresholds, tree.min_leaf, depth, pits, 5)
    assert _value(tree, X, tau) == pytest.approx(expected)
    assert all(r["pits"] >= 5 for r in _leaves(tree.tree_)) or "action" in tree.tree_


def _leaves(node):
    return [node] if "action" in node else _leaves(node["left"]) + _leaves(node["right"])


def test_deeper_is_never_worse_on_training_data():
    X, tau = _toy(0)
    values = [_value(PolicyTree(depth=d, features=["a", "b", "c"]).fit(X, tau), X, tau)
              for d in (1, 2, 3)]
    assert values[0] <= values[1] + 1e-9 <= values[2] + 2e-9


def test_rules_partition_the_training_rows():
    X, tau = _toy(1)
    tree = PolicyTree(depth=2, features=["a", "b", "c"]).fit(X, tau)
    rules = tree.rules()
    assert sum(r["n"] for r in rules) == len(X)
    assert all(r["n"] >= tree.min_leaf for r in rules)


# --------------------------------------------------------------------------
# Doubly robust rewards
# --------------------------------------------------------------------------

def test_dr_reward_corrects_the_logged_action_only():
    df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, -1.0], "p1": [0.25, 0.5],
                       "T": [1, 0], "y": [5.0, 0.0]})
    dm, dr = rewards(df, clip=0.01)
    np.testing.assert_allclose(dm, [[1, 3], [2, -1]])
    # Row 0 pitted: pit column gets (y - f1) / p1 = (5 - 3) / 0.25 = 8 added.
    # Row 1 stayed: stay column gets (y - f0) / (1 - p1) = (0 - 2) / 0.5 = -4 added.
    np.testing.assert_allclose(dr, [[1, 11], [-2, -1]])


def test_dr_reward_equals_outcome_when_model_is_perfect():
    rng = np.random.default_rng(0)
    n = 50
    T = rng.integers(0, 2, n)
    y = rng.normal(size=n)
    df = pd.DataFrame({"f0": np.where(T == 0, y, 0.0), "f1": np.where(T == 1, y, 0.0),
                       "p1": rng.uniform(0.1, 0.9, n), "T": T, "y": y})
    dm, dr = rewards(df)
    np.testing.assert_allclose(dr, dm)


# --------------------------------------------------------------------------
# Safety car / VSC laps from race-control messages
# --------------------------------------------------------------------------

def _rc(*msgs):
    """Race-control table from (lap, message[, category]) tuples, in time order."""
    rows = []
    for i, m in enumerate(msgs):
        lap, text = m[0], m[1]
        category = m[2] if len(m) > 2 else "SafetyCar"
        rows.append({"date": f"2024-01-01T00:{i:02d}:00", "lap_number": lap,
                     "message": text, "category": category})
    return pd.DataFrame(rows)


def _laps(flags):
    return [int(i) for i in np.flatnonzero(flags)]


def test_normal_safety_car_and_vsc():
    sc, vsc = track_status_by_lap(_rc((5, "SAFETY CAR DEPLOYED"), (8, "SAFETY CAR IN THIS LAP"),
                                      (20, "VIRTUAL SAFETY CAR DEPLOYED"),
                                      (21, "VIRTUAL SAFETY CAR ENDING")), 30)
    assert _laps(sc) == [5, 6, 7, 8]
    assert _laps(vsc) == [20, 21]


def test_vsc_upgraded_to_safety_car():  # 2023 Silverstone
    sc, vsc = track_status_by_lap(_rc((33, "VIRTUAL SAFETY CAR DEPLOYED"),
                                      (33, "SAFETY CAR DEPLOYED"),
                                      (38, "SAFETY CAR IN THIS LAP")), 52)
    assert _laps(vsc) == [33]
    assert _laps(sc) == [33, 34, 35, 36, 37, 38]


def test_red_flag_ends_the_safety_car():  # 2023 Melbourne
    sc, _ = track_status_by_lap(_rc((7, "SAFETY CAR DEPLOYED"), (8, "RED FLAG", "Flag"),
                                    (54, "SAFETY CAR DEPLOYED"), (55, "RED FLAG", "Flag")), 58)
    assert _laps(sc) == [7, 8, 54, 55]


def test_race_finishing_under_safety_car():  # 2025 Montreal
    sc, _ = track_status_by_lap(_rc((67, "SAFETY CAR DEPLOYED")), 70)
    assert _laps(sc) == [67, 68, 69, 70]


def test_messages_without_a_lap_are_ignored():
    sc, _ = track_status_by_lap(_rc((None, "SAFETY CAR DEPLOYED"), (3, "SAFETY CAR DEPLOYED"),
                                    (4, "SAFETY CAR IN THIS LAP")), 10)
    assert _laps(sc) == [3, 4]


def test_no_messages():
    sc, vsc = track_status_by_lap(pd.DataFrame(), 10)
    assert sc.sum() == vsc.sum() == 0


# --------------------------------------------------------------------------
# Executable (one stop per stint) evaluation: clone-censor-weight
# --------------------------------------------------------------------------

def _sim_stints(n, rng, rule_age=None, grace=2, n_laps=30):
    """Stints of n_laps decision laps. Team hazard rises with tyre age and with a
    confounder z that also moves the best stop lap. With rule_age=None the team
    decides (logged data); otherwise the regime 'stay until age >= rule_age, then stop
    within `grace` laps as the team would, at the latest on the last' is simulated."""
    rows, values = [], []
    for s in range(n):
        z = rng.normal()
        best = 18 + 4 * z
        stop = None
        for l in range(n_laps):
            age = 5 + l
            p = 1 / (1 + np.exp(-(-6 + 0.2 * age + 1.0 * z)))
            team = rng.random() < p
            if rule_age is None:
                act = team
            else:
                fire = rule_age if rule_age >= 5 else 5
                act = (age >= fire and team) or age == fire + grace
            if rule_age is None:
                rows.append((s, l, age, p, int(act)))
            if act:
                stop = age
                break
        y = -0.05 * ((stop if stop is not None else 40) - best) ** 2 + rng.normal(0, 1)
        values.append(y)
    return rows, np.array(values)


def test_executable_estimator_recovers_simulated_value():
    from stopping import Stints
    rng = np.random.default_rng(0)
    rows, y = _sim_stints(20000, rng)
    df = pd.DataFrame(rows, columns=["Unit", "LapNumber", "TyreLife", "p1", "T"])
    df["Event"] = "race" + (df["Unit"] % 50).astype(str)
    df["Driver"] = df["Unit"].astype(str)
    df["Stint"] = 1 + df["Unit"] % 3          # spread units over the three strata
    df["y"] = y[df["Unit"]]
    stop = df[df["T"] == 1].set_index("Unit")["LapNumber"]
    df["StintStopLap"] = df["Unit"].map(stop).astype(float)
    st = Stints(df, clip=1e-6)
    for rule_age in (12, 16, 20):  # late rules (age 26) have ESS < 500: noisy, not biased
        actions = (df["TyreLife"] >= rule_age).to_numpy()
        est, _, _ = st.value(actions, grace=2)
        _, y_rule = _sim_stints(20000, np.random.default_rng(1), rule_age=rule_age)
        truth = y_rule.mean() - y.mean()
        assert est == pytest.approx(truth, abs=0.15), (rule_age, est, truth)


# --------------------------------------------------------------------------
# P1: dry-stint index, outcome learners, honest cross-fitting
# --------------------------------------------------------------------------

def test_dry_stint_ignores_wet_stints():
    from policy import dry_stint
    comp = ["INTERMEDIATE"] * 3 + ["MEDIUM"] * 4 + ["HARD"] * 3
    stint = [1] * 3 + [2] * 4 + [3] * 3
    wet = pd.DataFrame({"Event": "e", "Driver": "A", "LapNumber": range(1, 11),
                        "Stint": stint, "Compound": comp})
    dry = pd.DataFrame({"Event": "e", "Driver": "B", "LapNumber": range(1, 11),
                        "Stint": [1] * 5 + [2] * 5, "Compound": ["SOFT"] * 5 + ["HARD"] * 5})
    laps = pd.concat([wet, dry], ignore_index=True).sample(frac=1, random_state=0)
    out = dry_stint(laps)
    got = laps.assign(D=out).sort_values(["Driver", "LapNumber"])["D"].tolist()
    assert got == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2] + [1] * 5 + [2] * 5


def _nuisance_toy(seed=0, n_races=30, per_race=300):
    from policy import NUISANCE_FEATURES_P1
    rng = np.random.default_rng(seed)
    n = n_races * per_race
    df = pd.DataFrame(rng.normal(size=(n, len(NUISANCE_FEATURES_P1))), columns=NUISANCE_FEATURES_P1)
    df["TyreLife"] = rng.integers(5, 40, n).astype(float)
    df["LapsRemaining"] = rng.integers(5, 60, n).astype(float)
    df["TwoCompoundsDone"] = rng.integers(0, 2, n).astype(float)
    df["Event"] = np.repeat([f"r{i}" for i in range(n_races)], per_race)
    df["Team"] = rng.choice(list("ABCDE"), n)
    df["Circuit"] = df["Event"]
    df["Driver"] = rng.choice(list("abcdefghij"), n)
    p = 1 / (1 + np.exp(-(-4 + 0.08 * df["TyreLife"])))
    df["T"] = (rng.random(n) < p).astype(int)
    # Pit effect: -20 s, a bit less with worn tyres. Baseline depends on tyre age.
    effect = -20 + 0.2 * (df["TyreLife"] - 20)
    df["y"] = -1.0 * df["TyreLife"] + df["T"] * effect + rng.normal(0, 2, n)
    return df, effect.to_numpy()


@pytest.mark.parametrize("learner", ["t", "s", "offset"])
def test_outcome_learners_recover_the_pit_effect(learner):
    from policy import NuisanceSpec, NUISANCE_FEATURES_P1, nuisance_predictions
    df, effect = _nuisance_toy()
    spec = NuisanceSpec(features=tuple(NUISANCE_FEATURES_P1),
                        categorical=("Team", "Circuit", "Driver"), outcome=learner,
                        propensity="gbm", prop_categorical=("Driver",))
    tr, te = df[df["Event"] < "r2"], df[df["Event"] >= "r2"]
    tr, te = nuisance_predictions(tr.reset_index(drop=True), te.reset_index(drop=True), spec=spec)
    est = (te["f1"] - te["f0"]).to_numpy()
    assert est.mean() == pytest.approx(-20 + 0.2 * (te["TyreLife"].mean() - 20), abs=1.5)
    pit = te["T"] == 1
    assert r2_score_(te.loc[pit, "y"], te.loc[pit, "f1"]) > 0.7  # oracle: ~0.94


def r2_score_(y, f):
    y, f = np.asarray(y), np.asarray(f)
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def test_cross_fitted_predictions_never_see_their_own_race():
    """The property nested CV relies on: a race's cross-fitted nuisances do not change
    when that race's outcomes or actions change."""
    from policy import NuisanceSpec, NUISANCE_FEATURES_P1, nuisance_predictions
    df, _ = _nuisance_toy(n_races=10, per_race=200)
    spec = NuisanceSpec(features=tuple(NUISANCE_FEATURES_P1), outcome="offset",
                        propensity="gbm")
    a, _ = nuisance_predictions(df, df.iloc[:5], spec=spec)
    df2 = df.copy()
    m = df2["Event"] == "r3"
    df2.loc[m, "y"] += 100.0
    df2.loc[m, "T"] = 1 - df2.loc[m, "T"]
    b, _ = nuisance_predictions(df2, df2.iloc[:5], spec=spec)
    cols = ["f0", "f1", "p1"]
    np.testing.assert_allclose(a.loc[m, cols], b.loc[m, cols])
    assert not np.allclose(a.loc[~m, cols], b.loc[~m, cols])


# --------------------------------------------------------------------------
# P2: multi-action tree (compound choice)
# --------------------------------------------------------------------------

def _brute_multi(X, G, thresholds, min_leaf, depth):
    def best(idx, d):
        v = G[idx].sum(0).max()
        if d == 0:
            return v
        for j, thr in enumerate(thresholds):
            for t in thr:
                m = X[idx, j] <= t
                l, r = idx[m], idx[~m]
                if len(l) >= min_leaf and len(r) >= min_leaf:
                    v = max(v, best(l, d - 1) + best(r, d - 1))
        return v
    return best(np.arange(len(X)), depth)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("depth", [1, 2])
def test_multi_action_tree_is_exact(seed, depth):
    from compound import MultiActionTree
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(150, 3)).astype(float)
    G = rng.normal(size=(150, 3))
    G[X[:, 0] > 2, 2] += 1.5
    G[(X[:, 0] <= 2) & (X[:, 1] > 1), 0] += 1.5
    tree = MultiActionTree(depth=depth, min_leaf_frac=0.05, features=["a", "b", "c"]).fit(X, G)
    got = G[np.arange(len(X)), tree.predict(X)].sum()
    want = _brute_multi(X, G, [np.unique(X[:, j])[:-1] for j in range(3)], int(0.05 * 150), depth)
    assert got == pytest.approx(want)
    assert tree.objective_ == pytest.approx(want)


def test_compound_dr_reward_is_unbiased_with_true_propensity():
    from compound import dr_rewards
    rng = np.random.default_rng(0)
    n = 200000
    e = rng.dirichlet([2, 2, 2], size=n)
    A = (rng.random(n)[:, None] > np.cumsum(e, 1)).sum(1)
    truth = np.array([1.0, 2.0, -1.0])
    y = truth[A] + rng.normal(0, 1, n)
    mu = np.zeros((n, 3))              # deliberately wrong outcome model
    G = dr_rewards(pd.DataFrame({"A": A, "y": y}), mu, e, clip=1e-9)
    np.testing.assert_allclose(G.mean(0), truth, atol=0.05)
