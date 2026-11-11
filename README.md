# Interpretable pit-stop policies for Formula 1

15.095 project (Georgia Abou Chebel, Nanrui Gong). We learn a short "stay out vs. pit"
rule from historical OpenF1 race data with an **optimal policy tree on doubly robust
rewards**, and evaluate it off-policy on held-out races. The method follows
`f1_proposal.pdf`; differences from it are listed under [Changes from the proposal](#changes-from-the-proposal).

We do not expect to beat the teams, who have far more data, private telemetry and their
own race models. The questions are: **how close does a 1–3 split rule get to what teams
did, and does the method recover known strategy facts** (cheap stops under a safety car,
the stop window opening as tyres age)?

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python build_dataset.py            # download 2023-2025 races -> data/laps.pkl (~5 min; rerun if rate-limited)
python run_experiment.py           # train 2023-24, evaluate 2025 -> results/ (~1.5 min; --quick skips sensitivity)
python -m pytest -q                # tests: tree solver is exact, DR rewards, SC/VSC parsing (~10 s)
streamlit run results_dashboard.py # results dashboard
streamlit run dashboard.py         # raw-data explorer (features, strategies, lap times)
```

`data/laps.pkl` is committed, so the experiment runs without downloading anything. The
API responses are cached in `cache/` (git-ignored); with a warm cache `build_dataset.py`
takes a few seconds and makes no network calls.

## The method in plain words

- **Decision point (row).** One (driver, lap) where a stop was a real option: dry tyres at
  least 5 laps old, not the first or last K laps, no rain. About 45k rows.
- **Action T.** 1 if the driver pitted at the end of that lap, else 0 (about 5% of rows).
- **Outcome y.** Seconds the driver gained over the next K = 10 laps on the median of the
  cars within ±3 positions, minus a "stop debt" if they still owe the mandatory stop
  when the window ends (otherwise staying out would look free).
- **Why not just compare pitters and non-pitters?** Teams pit *when it is a good idea*
  (worn tyres, a safety car), so the raw comparison mixes up the effect of pitting with
  the reasons for pitting. This is confounding.
- **Nuisance models.** `f0(x)` and `f1(x)` predict y if the driver stays out or pits;
  `p1(x)` is the probability a team pits in context x. They are "cross-fitted": each
  race's predictions come from models trained on other races.
- **Rewards Γ.** For each row and each action, an estimate of y under that action.
  Γ_DM just uses `f`. Γ_DR (doubly robust) adds a correction on the action actually
  taken: `(y − f_T) / p_T`. It is unbiased if *either* the outcome models or the
  propensity are right. Per row it is very noisy (dividing by p ≈ 0.05 multiplies
  errors by 20); only averages over many rows mean anything.
- **τ = Γ_pit − Γ_stay** is the estimated benefit of pitting on that row. The policy tree
  splits the rows into at most 8 boxes and says PIT in a box when τ sums to more than 0.
- **Gain vs. teams** = mean over test rows of Γ for the policy's action minus the actual
  y. Zero means "as good as what teams did", in seconds per decision over 10 laps.
- **Trimming.** Rows where the model says teams (almost) never pit, p̂ < 1%, carry no
  information about what pitting would have done, so they are dropped (about 30%).

## Architecture

```
OpenF1 REST API
      │  openf1.py          get() with on-disk cache (cache/openf1/), one table per race
      ▼
build_dataset.py  ──────►  data/laps.pkl        one row per (driver, lap), 70 races 2023-25
      │
      ▼
policy.py  (the method, no I/O)
  1. build_rows()            decision points x, action T, outcome y
  2. nuisance_predictions()  outcome models f_0, f_1 and propensity p, cross-fitted by race
  3. rewards()               Γ_DM and Γ_DR
  4. PolicyTree              exact optimal tree, depth ≤ 3
      │
      ▼
run_experiment.py  (the study)  ──────►  results/*.json, *.csv
  split by season, CV depth, baselines, off-policy evaluation, bootstrap CIs,
  sanity checks, sensitivity analyses
      │
      ▼
results_dashboard.py  (Streamlit, reads results/ only)

dashboard.py     exploratory view of the raw lap table (pre-modelling)
```

## Files

| File | What it does |
|---|---|
| `openf1.py` | OpenF1 client and lap-table builder. `get()` caches every API response in `cache/openf1/` and retries on rate limits. `load_session()` joins `laps`, `stints`, `position`, `intervals`, `race_control`, `weather` and `drivers` into one row per (driver, lap). It derives the pit flag from stint ends, tyre age, gap behind (from the following car's interval), SC/VSC laps from race-control messages, and weather as-of each lap end. `FEATURES` documents every column. |
| `build_dataset.py` | Loads every race of 2023–2025 through `openf1.py`, adds `LapStart`, `Circuit`, `Year`, and saves `data/laps.pkl`. Races that fail (usually HTTP 429) are listed; rerunning fills them from the cache. |
| `policy.py` | The method; see the table below. |
| `run_experiment.py` | The experiment: train/test split, depth selection, baselines, evaluation, sanity and sensitivity analyses. Writes `results/`. |
| `results_dashboard.py` | Streamlit dashboard of the results: learned rule, policy comparison, gain by race and circuit type, sanity checks, overlap plot, sensitivity, and a lap-by-lap "tree vs. team" view. |
| `dashboard.py` | Streamlit explorer of the raw lap data (feature table, distributions, pit rates, tyre-strategy charts). |
| `test_policy.py` | Tests: the tree search matches brute force at depths 1–3, the DR reward formula, and SC/VSC parsing edge cases (red flags, VSC upgraded to SC, race ending under SC). |

### Where each proposal step lives

| Proposal step | Code |
|---|---|
| Decision points: dry tyres, tyre age ≥ 5, lap ≥ 2, not on an out-lap, window inside the race, no wet laps in the window | `policy.build_rows` (`keep = ...`) |
| Context x | `policy.TREE_FEATURES`: tyre age, race fraction, laps remaining, compound, two-compounds-done flag, stint, position, gaps ahead/behind, SC, VSC, track temp, circuit pit loss. Nuisance models also get `Team` and `Circuit` (`CATEGORICAL`). |
| Outcome y: seconds gained over K laps | `policy.build_rows`: elapsed time from the start of lap L to the start of L+K, so the in-lap and out-lap are both inside, compared with the median of cars within ±3 positions (`local_median`), minus the stop debt |
| 1. Outcome models f_t, cross-fitted by race | `policy.fit_nuisances` (gradient-boosted trees, one per action), `policy.nuisance_predictions` (5-fold `GroupKFold` by race on train; train-only fit for test) |
| 2. Propensities | `policy._propensity_model`: logistic regression with splines on tyre age / race phase plus team and circuit effects |
| 3. Rewards Γ_DM, Γ_DR | `policy.rewards` (propensity clipping `clip`) |
| Overlap / common support | `run_experiment.trim_support` (drops rows with p̂ < `TRIM` = 0.01, counts reported); `overlap_table` for the p̂-by-action-and-phase plot |
| 4. Optimal policy tree, depth ≤ 3 | `policy.PolicyTree` (see below) |
| Evaluation: grouped k-fold + holdout season | `run_experiment.cv_depth` (5-fold by race on 2023–24 to choose depth); test = all 2025 races |
| V̂(π) − V̂(historical), bootstrap over races | `run_experiment.evaluate`, `policy.bootstrap_gain` |
| Baselines: historical, threshold rule, DM vs DR tree, black box | `run_experiment.fit_policies` (also "never pit") |
| By circuit type | `run_experiment.main` (`STREET` circuits vs. permanent) |
| Sensitivity: K, clipping, SC rows | `run_experiment.main` (`run(...)` calls), plus outcome reference and stop-debt variants |
| Sanity checks: SC stops cheaper, softer compounds degrade faster | `run_experiment.sanity_checks` (also nuisance AUC/R² and effect by tyre age) |

### The policy tree solver

With two actions, Σᵢ Γᵢ,π(xᵢ) = Σᵢ Γᵢ,stay + Σ_{π=pit} τᵢ, where τ = Γ_pit − Γ_stay, so each leaf only
checks the sign of its summed τ. Each feature is cut at up to 16 quantile thresholds, and
the search is **exhaustive (exactly optimal) over those splits**:

- depth 1: per-feature histograms + cumulative sums;
- depth 2 (`_depth2`): for each root feature, a joint (root bin × feature × bin) histogram
  gives the best child split for every root threshold at once, O(n·p²);
- depth 3: every root split, with exact depth-2 subtrees on each side.

Leaves must hold ≥ 2% of rows. `test_depth2_search_is_exact` and
`test_depth3_search_is_exact` check the solver against brute-force enumeration. This replaces a MIO solver (e.g. Interpretable AI), which we
don't have; for two actions and binned splits it gives the same optimum.

## Changes from the proposal

Each change came from a sanity check that failed on a known racing fact, before the
test-set results were looked at for that choice.

1. **Stop debt.** With a plain K-lap outcome, a driver who stays out pays their stop
   after the window ends, so pitting looks 8–20 s worse almost everywhere and the
   policy degenerates to "never pit". A row that ends the window still owing the
   mandatory second dry compound is charged the circuit's typical **in-window** net cost
   of a green-flag stop (`estimate_stop_cost`, ≈ 8.5 s at K = 10, training races only).
   Charging the full pit-lane loss (≈ 23 s) instead ignores the deferred stop's own
   fresh-tyre benefit and biases the policy toward pitting: trees then pit on about 60% of
   laps with a fake +3 s gain. That variant is kept in the sensitivity table as a warning.
2. **Local reference instead of the whole-field median.** Measured against the whole
   field, a safety car's bunching erases the leaders' margins whether they pit or not
   (e.g. 2025 Barcelona: the leaders "lost" 85–90 s by pitting under the late SC). The
   outcome compares each driver with the median of cars within ±3 positions. The
   whole-field version is a sensitivity option (`reference="field"`).
3. **Red-flag windows dropped.** Windows whose elapsed time exceeds twice the normal
   K-lap time (race suspensions, long repairs) are not strategy decisions.
4. **Compound choice** (second stage) is not implemented yet: actions are stay / pit.

## Current results (K = 10, train 2023–24 = 45 races, test 2025 = 22 races; races with no dry decision rows drop out)

Gain is in seconds per decision point over the next 10 laps, relative to what teams did,
with a 95% bootstrap interval over races. See `results/results.json` or the dashboard for everything.

| Policy | Gain | 95% interval | Laps it says "pit" |
|---|---|---|---|
| DR tree, depth 1 (chosen by CV): pit while on the first stint | −0.30 s | [−1.81, +0.92] | 52% |
| DR tree, depth 2 | −0.35 s | [−1.53, +0.56] | 31% |
| DR tree, depth 3 | −0.97 s | [−2.16, +0.16] | 52% |
| DM tree, depth 1 | +0.50 s | [+0.26, +0.76] | 7% |
| Gradient boosting on Γ | −0.07 s | [−0.50, +0.30] | 23% |
| Threshold (tyre age ≥ 44) | +0.31 s | [+0.13, +0.54] | 1% |
| Never pit | +0.18 s | [+0.07, +0.29] | 0% |

- No learned policy is clearly better than the teams, as expected. Every DR tree's
  interval contains 0, and the CV gain on 2023–24 is negative at every depth (−0.57,
  −1.01, −0.75 s), so in cross-validation no DR tree beats the teams either.
- **The learned rules are unstable.** Fixing the SC/VSC flags (about 250 driver-laps in
  11 races) and removing 41 duplicate rows moved the CV choice from depth 3 to depth 1
  and changed every tree's rule; the earlier depth-3 tree split on `PitLoss` and
  `TrackTemp`, the new one on `LapsRemaining`, `TrackTemp` and `Position`. The pit
  rates (31–52%) also show the trees are not plausible strategies. Treat any single
  rule as noise until it survives a stability check (see "Next steps").
- "Never pit" and the threshold rule come out slightly *ahead* of the teams. Over a
  10-lap window that mostly says the stop debt (≈ 8.5 s) undercharges a deferred stop,
  not that teams pit too much.
- Sanity checks: DR says a stop under SC/VSC is better than under green (+5.5 s vs −4.5 s)
  ✅. Pitting pays off more as tyres age (clearly positive beyond 40 laps, +7.5 s) ✅.
  Soft degrades fastest ✅ (weak, because fuel burn-off cancels most of the slope).
- Weak spots: the pit-lap outcome model has no skill on test (R² ≈ −0.1), so the DM
  estimates (including the DM tree's +0.50 s) are unreliable and still give the wrong
  SC sign. About 30% of rows fall outside common support (few of them actual pits).

## Next steps

1. **Stability before interpretation.** Refit the tree on bootstrap resamples of the
   training races and report how often each first split appears. A rule worth showing
   should appear in most resamples.
2. **Make the outcome less myopic.** The trees' 30–50% pit rates are a symptom of the
   one-shot K-lap evaluation. Options: evaluate a policy as "pit at the first lap the
   tree says PIT, then follow the team" (one decision per stint), or charge the debt
   with the measured value of a stop *at the end of the window* rather than a circuit
   constant.
3. **Drop circuit proxies.** `PitLoss` and `TrackTemp` mostly identify the circuit.
   Refit without them (and reselect depth by CV on 2023–24) to check the tree learns
   strategy, not tracks.
4. **Better pit-lap outcome model.** f1 has R² ≈ 0 on test. Only ~1.2k training stops
   exist, so try a single model with T as a feature (an S-learner) or one that shares
   structure across actions, and check its calibration against the DR estimates.
5. **Compound choice** as a second stage (proposal), once the stay/pit stage is stable.

## Limitations (see proposal §2)

Myopic K-lap outcome (a policy that pits on many consecutive laps isn't executable as-is);
unobserved confounding (team pace models, tyre temperatures, damage); interference
between cars; regulation changes across seasons; off-policy estimates only (no races can
be re-run).
