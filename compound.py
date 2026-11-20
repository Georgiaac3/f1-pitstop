"""Compound choice at a stop (P2): which dry tyre to fit, given that the team stops.

The stay/pit timing is a stopping problem evaluated per stint (stopping.py). The
compound is a second decision taken *at* the stop, so we treat it as its own
multi-action problem on the rows where the team pitted:

    unit       a stop: (driver, lap L) with T = 1, the new tyre being dry
    action     A in {SOFT, MEDIUM, HARD}, the compound fitted
    outcome    y = rest-of-race gain from lap L (build_rows(K="race"))
    estimand   V(pi) = E[ y(pi(x)) | stop ]: the team keeps its stop lap, the rule picks
               the compound; the gain over the teams is V(pi) - E[y | stop]

Identification needs the compound choice to be unconfounded given x among stops (the
team's private plan, e.g. one stop vs two, is the main threat) and overlap: every
compound must be a real option where the rule picks it.

Nuisances, cross-fitted by race:
    mu(x, a)  outcome model: one boosted model with the compound as a feature
              (S-learner: ~1k stops, soft has only ~200, so the actions share structure)
    e(a | x)  propensity: multinomial boosted classifier, clipped below at `clip`
DR reward:  Gamma[i, a] = mu(x_i, a) + 1{A_i = a} / e(a | x_i) * (y_i - mu(x_i, A_i)).
The policy is an exact optimal tree over the three actions (MultiActionTree).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from policy import DRY, NUISANCE_FEATURES_P1, TREE_FEATURES_P1

ACTIONS = DRY  # SOFT, MEDIUM, HARD
NUIS = NUISANCE_FEATURES_P1
CATS = ["Team", "Circuit"]


def stop_rows(rows):
    """Stops onto a dry compound, with the action index A (0 soft, 1 medium, 2 hard)."""
    s = rows[(rows["T"] == 1) & rows["NewCompound"].isin(ACTIONS)].copy()
    s["A"] = s["NewCompound"].map({c: i for i, c in enumerate(ACTIONS)}).astype(int)
    return s.reset_index(drop=True)


def _X(df, cats, with_a=None):
    X = df[NUIS].copy()
    for c in CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cats[c]).codes.astype(float)
        X.loc[X[c] < 0, c] = np.nan
    if with_a is not None:
        X.insert(len(NUIS), "A", with_a)
    return X


def fit_nuisances(train):
    cats = {c: sorted(train[c].astype(str).unique()) for c in CATS}
    n = len(NUIS)
    mu = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0,
        categorical_features=[n, n + 1, n + 2]).fit(
        _X(train, cats, train["A"].to_numpy()), train["y"])
    e = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
        l2_regularization=1.0, random_state=0,
        categorical_features=[n, n + 1]).fit(_X(train, cats), train["A"])

    def predict(df):
        m = np.column_stack([mu.predict(_X(df, cats, np.full(len(df), a)))
                             for a in range(len(ACTIONS))])
        p = np.zeros((len(df), len(ACTIONS)))
        p[:, e.classes_] = e.predict_proba(_X(df, cats))
        return m, p
    return predict


def nuisance_predictions(train, test, n_folds=5):
    """(mu, e) cross-fitted by race on train, and from the all-train fit on test."""
    k = len(ACTIONS)
    mu_tr, e_tr = np.zeros((len(train), k)), np.zeros((len(train), k))
    for fit_idx, pred_idx in GroupKFold(n_splits=n_folds).split(train, groups=train["Event"]):
        m, p = fit_nuisances(train.iloc[fit_idx])(train.iloc[pred_idx])
        mu_tr[pred_idx], e_tr[pred_idx] = m, p
    mu_te, e_te = fit_nuisances(train)(test)
    return (mu_tr, e_tr), (mu_te, e_te)


def dr_rewards(df, mu, e, clip=0.05):
    A, y = df["A"].to_numpy(), df["y"].to_numpy()
    e = np.clip(e, clip, None)
    resid = y - mu[np.arange(len(A)), A]
    onehot = np.eye(len(ACTIONS))[A]
    return mu + onehot / e * resid[:, None]


def value(gamma, actions):
    return gamma[np.arange(len(actions)), np.asarray(actions, int)]


class MultiActionTree:
    """Exact optimal policy tree of depth <= 2 over K actions, on binned splits.

    Objective: sum_i Gamma[i, pi(x_i)]; a leaf takes the action with the largest sum of
    Gamma over its rows. Each leaf holds at least min_leaf_frac of the rows.
    Depth 2 is solved exactly: every root split, then the best depth-1 subtree per side.
    """

    def __init__(self, depth=1, n_bins=16, min_leaf_frac=0.05, features=TREE_FEATURES_P1):
        self.depth, self.n_bins, self.min_leaf_frac = depth, n_bins, min_leaf_frac
        self.features = list(features)

    def _thresholds(self, X):
        out = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            out.append(u[:-1] if len(u) <= self.n_bins else np.unique(
                np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1])))
        return out

    def _leaf(self, idx):
        s = self.G[idx].sum(0)
        return float(s.max()), {"action": int(s.argmax()), "n": int(len(idx))}

    def _best1(self, idx):
        """Best depth<=1 tree on idx: (value, split or None)."""
        best_v, _ = self._leaf(idx)
        best = (best_v, None)
        if len(idx) < 2 * self.min_leaf:
            return best
        G, X = self.G[idx], self.X[idx]
        tot = G.sum(0)
        for j, thr in enumerate(self.thr):
            if not len(thr):
                continue
            b = np.searchsorted(thr, X[:, j], side="left")         # bin per row
            q = len(thr) + 1
            S = np.zeros((q, G.shape[1]))
            np.add.at(S, b, G)
            C = np.bincount(b, minlength=q)
            L, Lc = np.cumsum(S, 0)[:-1], np.cumsum(C)[:-1]          # left: x <= thr[k]
            v = L.max(1) + (tot - L).max(1)
            v[(Lc < self.min_leaf) | (len(idx) - Lc < self.min_leaf)] = -np.inf
            k = int(np.argmax(v))
            if v[k] > best[0] + 1e-9:
                best = (float(v[k]), (j, float(thr[k])))
        return best

    def _split(self, idx, sp):
        j, t = sp
        m = self.X[idx, j] <= t
        return idx[m], idx[~m]

    def _build1(self, idx, sp):
        if sp is None:
            return self._leaf(idx)[1]
        left, right = self._split(idx, sp)
        return {"feature": self.features[sp[0]], "threshold": sp[1], "n": int(len(idx)),
                "left": self._leaf(left)[1], "right": self._leaf(right)[1]}

    def fit(self, X, gamma):
        self.X, self.G = np.asarray(X, float), np.asarray(gamma, float)
        self.min_leaf = max(1, int(self.min_leaf_frac * len(self.X)))
        self.thr = self._thresholds(self.X)
        idx = np.arange(len(self.X))
        v1, sp1 = self._best1(idx)
        best = (v1, "d1", sp1)
        if self.depth >= 2:
            for j, thr in enumerate(self.thr):
                for t in thr:
                    left, right = self._split(idx, (j, float(t)))
                    if len(left) < self.min_leaf or len(right) < self.min_leaf:
                        continue
                    vl, sl = self._best1(left)
                    vr, sr = self._best1(right)
                    if vl + vr > best[0] + 1e-9:
                        best = (vl + vr, "d2", ((j, float(t)), sl, sr))
        if best[1] == "d1":
            self.tree_ = self._build1(idx, best[2])
        else:
            root, sl, sr = best[2]
            left, right = self._split(idx, root)
            self.tree_ = {"feature": self.features[root[0]], "threshold": root[1],
                          "n": int(len(idx)), "left": self._build1(left, sl),
                          "right": self._build1(right, sr)}
        self.objective_ = float(best[0])
        del self.X, self.G
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.zeros(len(X), int)

        def go(node, idx):
            if "action" in node:
                out[idx] = node["action"]
                return
            m = X[idx, self.features.index(node["feature"])] <= node["threshold"]
            go(node["left"], idx[m])
            go(node["right"], idx[~m])
        go(self.tree_, np.arange(len(X)))
        return out

    def rules(self, node=None, conds=()):
        node = node or self.tree_
        if "action" in node:
            return [{"rule": " and ".join(conds) or "always",
                     "action": ACTIONS[node["action"]], "n": node["n"]}]
        f, t = node["feature"], node["threshold"]
        return (self.rules(node["left"], conds + (f"{f} <= {t:g}",))
                + self.rules(node["right"], conds + (f"{f} > {t:g}",)))
