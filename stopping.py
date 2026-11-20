"""Executable evaluation of a pit rule: one stop per stint, off-policy.

The lap-level evaluation (run_experiment.py) scores every lap as a one-off deviation
from the team, so a tree that says PIT on 50% of laps looks "fine" although no car can
follow it. Here a lap-level rule pi(x) is turned into a stopping rule:

    in each stint, stay out until the first decision lap where pi says PIT
    (the "firing lap" F), then stop within the next g laps (lap F .. F+g),
    at the latest on lap F+g; after that stop, follow the team.

That regime is executable (it only uses the past) and makes at most one stop per stint.
Its value is estimated by clone-censor-weight with a grace period (Cain et al. 2010,
Hernan & Robins 2016): a stint "follows" the regime if the team's stop fell in
[F, F+g] (or, if pi never fires, the team never stopped in the stint). Followers are
re-weighted by the inverse probability of having followed:

    w = prod_{decision laps l < F} 1 / (1 - p_l)      (the team had to stay out)
        x 1 / p_{F+g}   if the stop is on the last lap of the window (forced stop)

where p_l is the cross-fitted pit propensity. During the window the regime lets the
team choose, so those laps add no weight. Staying out is the likely action
(1 - p ~ 0.95), so these weights are far tamer than the lap-level 1/p ~ 20-100.

The value of a stint is its rest-of-race outcome Y_s (build_rows(K="race")) at the
stint's first decision lap: seconds gained on the cars within +-3 positions from
there to the flag. The estimate is self-normalised (Hajek):
    V(pi) = sum_s w_s Y_s 1[follows] / sum_s w_s 1[follows],
and the gain over the teams is V(pi) - mean_s Y_s, with a bootstrap over races.

The policy only acts in the stint being evaluated; earlier and later stints follow the
team. So the gain is "what if the team had used this rule for this one stop".
"""

import numpy as np
import pandas as pd

LAP_KEY = 1000  # LapNumber < 1000, so stint * LAP_KEY + lap orders rows within stints


class Stints:
    """Decision rows (rest-of-race outcome, with p1) grouped into stints, precomputed
    so that evaluating a rule is a few vectorised operations."""

    def __init__(self, rows, clip=0.01, w_max=None, stratum="Stint"):
        """`rows`: decision rows. Action vectors passed to follow/value/evaluate are
        aligned with `rows` as given (they are re-ordered internally).
        w_max: truncate stint weights at this value (None: no truncation).
        stratum: column numbering the stint for the strata (P1: "DryStint")."""
        self.w_max = w_max
        rows = rows.reset_index(drop=True)
        self.order = rows.sort_values(["Event", "Driver", "Stint", "LapNumber"]).index.to_numpy()
        rows = rows.iloc[self.order].reset_index(drop=True)
        sid = rows.groupby(["Event", "Driver", "Stint"], sort=False).ngroup().to_numpy()
        self.sid = sid
        self.start = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
        self.end = np.r_[self.start[1:], len(rows)]           # exclusive
        self.n = len(self.start)
        self.lap = rows["LapNumber"].to_numpy()
        self.key = sid * LAP_KEY + self.lap
        p = rows["p1"].clip(clip, 1 - clip).to_numpy()
        self.p = p
        # C[i] = sum_{j <= i} -log(1 - p_j); C_pad[i] = sum_{j < i}
        self.C_pad = np.r_[0.0, np.cumsum(-np.log1p(-p))]
        # The team's stop: the stint's real last lap if it ended in a stop (it may lie
        # beyond the last decision row, e.g. a stop in the final laps or on a lap
        # with no timing), else none (the stint ran to the flag).
        stop_lap = rows["StintStopLap"].to_numpy()[self.start]
        self.has_obs = ~np.isnan(stop_lap)
        self.obs_lap = np.where(self.has_obs, stop_lap, -1).astype(int)
        self.last_lap = self.lap[self.end - 1]
        key_obs = self.sid[self.start] * LAP_KEY + self.obs_lap
        idx = np.searchsorted(self.key, key_obs)                 # decision row of the stop
        idx = np.minimum(idx, len(rows) - 1)
        self.obs_row = np.where(self.has_obs & (self.key[idx] == key_obs), idx, -1)
        self.Y = rows["y"].to_numpy()[self.start]               # rest-of-race at stint start
        self.race = rows["Event"].to_numpy()[self.start]
        self.stint_no = rows[stratum].to_numpy()[self.start]
        self.tyre = rows["TyreLife"].to_numpy()

    def follow(self, actions, grace=2):
        """For a lap-level action vector (aligned with self.rows): per stint
        (follows, weight, fires, firing row index)."""
        a = np.asarray(actions).astype(bool)[self.order]
        big = len(a)
        first_a = np.minimum.reduceat(np.where(a, np.arange(big), big), self.start)
        fires = first_a < self.end
        fire_idx = np.where(fires, first_a, self.start)
        fire_lap = self.lap[fire_idx]
        # pi never fires: follow iff the team never stopped; all laps were "stay".
        # (If the team stopped after the last decision row, what pi would have done
        # there is unknown: censored.)
        logw_never = self.C_pad[self.end] - self.C_pad[self.start]
        # pi fires at F: team must stay before F and stop in [F, F+g].
        before = self.C_pad[fire_idx] - self.C_pad[self.start]   # rows before the firing row
        hi = fire_lap + grace
        in_win = self.has_obs & (self.obs_lap >= fire_lap) & (self.obs_lap <= hi)
        # Forced stop on the window's last lap: weight 1/p there. If that lap is not a
        # decision row (no propensity), the stint cannot be weighted: censored.
        forced = in_win & (self.obs_lap == hi)
        p_obs = self.p[np.maximum(self.obs_row, 0)]
        in_win &= ~(forced & (self.obs_row < 0))
        logw_fire = before - np.where(forced & (self.obs_row >= 0), np.log(p_obs), 0.0)
        follows = np.where(fires, in_win, ~self.has_obs)
        w = np.exp(np.where(fires, logw_fire, logw_never)) * follows
        if self.w_max is not None:
            w = np.minimum(w, self.w_max)
        return follows, w, fires, fire_idx

    # Strata: first stint, second stint, third and later. The value is a Hajek mean
    # within each stratum, combined by stratum size. A single pooled normalisation
    # lets one stint type carry the estimate: e.g. a rule that pits at the first
    # chance in stint 1 has few (truncated-weight) followers there, and its pooled
    # value would mostly reflect later stints.
    STRATA = (1, 2, 3)

    def _stratum(self):
        return np.minimum(self.stint_no, self.STRATA[-1])

    def _agg(self, w):
        """(n_races, n_strata, 4): sums of w*Y, w, Y, count per race and stratum."""
        races, r_idx = np.unique(self.race, return_inverse=True)
        s_idx = self._stratum() - 1
        out = np.zeros((len(races), len(self.STRATA), 4))
        for k, v in enumerate((w * self.Y, w, self.Y, np.ones(self.n))):
            np.add.at(out[:, :, k], (r_idx, s_idx), v)
        return out

    @staticmethod
    def _gain(tot):
        """tot: (..., n_strata, 4) summed over races."""
        n = tot[..., 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = tot[..., 0] / tot[..., 1] - tot[..., 2] / n
        v = np.where(n > 0, v, 0.0)                     # empty stratum: no stints
        v = np.where((n > 0) & (tot[..., 1] <= 0), np.nan, v)  # no followers: undefined
        return (v * n).sum(-1) / n.sum(-1)

    def _ess(self, w):
        s = self._stratum()
        out = []
        for k in self.STRATA:
            wk = w[s == k]
            out.append(float(wk.sum() ** 2 / (wk**2).sum()) if wk.sum() > 0 else 0.0)
        return out

    def value(self, actions, grace=2):
        """Point estimate (no bootstrap), fast, for rule search:
        (gain, followers, smallest per-stratum ESS)."""
        follows, w, _, _ = self.follow(actions, grace)
        g = self._gain(self._agg(w).sum(0))
        return float(g), int(follows.sum()), min(self._ess(w))

    def evaluate(self, actions, grace=2, n_boot=1000, seed=0):
        """Gain over the teams with a 95% bootstrap interval over races."""
        follows, w, fires, fire_idx = self.follow(actions, grace)
        agg = self._agg(w)
        est = self._gain(agg.sum(0))
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(agg), size=(n_boot, len(agg)))
        boot = self._gain(agg[draws].sum(1))
        boot = boot[np.isfinite(boot)]
        st = self._stratum()
        fire_age = self.tyre[fire_idx][fires]
        obs_age = self.tyre[self.obs_row[self.obs_row >= 0]]
        per = {}
        for k in self.STRATA:
            m = st == k
            per[k] = {"stints": int(m.sum()), "followers": int(follows[m].sum()),
                      "mean_w": float(w[m].sum() / max(m.sum(), 1)),
                      "gain": float(self._gain(agg.sum(0)[[k - 1]][None])[0])
                      if m.any() else None}
        return {"gain": float(est),
                "lo": float(np.quantile(boot, 0.025)) if len(boot) else None,
                "hi": float(np.quantile(boot, 0.975)) if len(boot) else None,
                "followers": int(follows.sum()), "stints": int(self.n),
                "ess_by_stratum": self._ess(w),
                "mean_w": float(w.sum() / self.n),  # ~1 if the hazard model is calibrated
                "by_stratum": per,
                "fires_share": float(fires.mean()),
                "fire_tyre_age": float(np.median(fire_age)) if len(fire_age) else None,
                "team_stop_tyre_age": float(np.median(obs_age))}
