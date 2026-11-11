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


def _brute_force(X, tau, thresholds, min_leaf, depth):
    """Best sum over leaves of max(sum tau, 0), trying every split on the same thresholds."""
    def best(idx, d):
        v = max(tau[idx].sum(), 0.0)  # a leaf is always allowed
        if d == 0:
            return v
        for j, thr in enumerate(thresholds):
            for t in thr:
                m = X[idx, j] <= t
                left, right = idx[m], idx[~m]
                if len(left) >= min_leaf and len(right) >= min_leaf:
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
