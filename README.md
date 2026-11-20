# F1 pit-stop policies

## The idea

In every Formula 1 race, teams decide on each lap whether their driver should **stay out**
or **pit** for fresh tyres. A stop costs about 20 seconds, but new tyres are faster, so
the timing matters a lot.

We use public race data from 2023–2025 to learn a **short, readable rule**, such as
"pit once the tyres are 26 laps old". Then we check how that rule would have done
compared with what the teams actually did.

We don't expect to beat the teams. They have far better information. The questions are:

- How close does a simple rule get to the teams?
- Does the data rediscover things every strategist knows, like "pitting under a safety car is cheap"?

## How it works, in four steps

1. **Collect the data.** Lap-by-lap timing, tyres, gaps and safety cars for every race,
   from the free [OpenF1](https://openf1.org) API.
2. **Correct for the teams' choices.** Teams pit *when it makes sense* (worn tyres, a
   safety car), so we can't just compare drivers who pitted with drivers who didn't.
   We use causal-inference methods to estimate what *would* have happened under each choice.
3. **Learn a rule.** We look for the best small decision tree: at most three yes/no
   questions, such as "is tyre age above 26?".
4. **Test it on races it has never seen.** We learn on 2023–24 and test on 2025.

## What we found so far

- ✅ **Pitting too early clearly costs time.** Stopping before about 16 laps of tyre age
  loses around 8–13 seconds over the rest of the race. This holds in both 2023–24 and 2025.
- ➖ **The best simple rule is about as good as the teams, not better.** "Pit once the
  tyres are 26 laps old" comes out level with the teams.
- ⚠️ **The learned rules are not stable yet.** Learning again from slightly different
  races gives a different rule, so we can't recommend a specific one yet.

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


## Try it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_experiment.py            # first study (~2 min)
python run_p0.py --quick            # P0 study, fast version (~2 min)
python run_p1.py --quick            # P1 study, fast version (~5 min)
python run_p2.py                    # P2 study (~1 min; the IAI step needs the IAI Julia image)
streamlit run results_dashboard.py  # interactive results
```

The race data is already included (`data/laps.pkl`), so nothing needs downloading.

## What's in the repo

| File or folder | What it is |
|---|---|
| `openf1.py`, `build_dataset.py` | Download the race data |
| `policy.py` | The method |
| `run_experiment.py` | First study |
| `run_p0.py`, `stopping.py` | P0 study: executable (one stop per stint) evaluation |
| `run_p1.py` | P1 study: no circuit proxies in the tree, better nuisance models, nested CV, second holdout (2023 → 2024) |
| `run_p2.py`, `compound.py`, `iai_compare.jl` | P2 study: compound choice at the stop, check against Interpretable AI, race replays |
| `results_dashboard.py`, `dashboard.py` | Interactive views of the results and the raw data |
| `report/` | The technical report (PDF) |
| `first_results/` | Saved copy of the first study's results |
