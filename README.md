# Calibra: calibrating building energy models with 48 simulations

Team Calibra's entry to the INTEGRATES building-energy calibration hackathon.

Our method is Bayesian optimization that works out, separately for each
building, which parameters matter and concentrates the search on them. With
the official configuration it scores **97.03 / 100**; the organizer defaults
score 12.92.

| Building | Score with organizer defaults | Score after calibration | Remaining error (WAPE) |
| --- | ---: | ---: | ---: |
| `building_01` | 38.76 | 96.03 | 3.97 % |
| `building_02` | 0.00 | 99.05 | 0.95 % |
| `building_03` | 0.00 | 96.02 | 3.98 % |
| **Mean (official score)** | **12.92** | **97.03** | |

The submission is the single file [`calibration.py`](calibration.py).

## The problem

A building energy model simulates how much heating energy a building needs,
given its geometry, construction, weather, and the way it is operated.
Geometry and construction are known from an energy audit. How the building is
really operated is not: when the heating runs, which setpoints are used, how
much of the floor area is fully heated. With the organizer's default
assumptions the model predicts about 1.6 to 2.5 times the gas that was
actually billed.

Calibration means finding the operating assumptions for which the simulation
reproduces the measurements. The model is fixed; we search for its inputs.

- **Data.** Three anonymized buildings in Bolzano, hourly weather for 2018 and
  2019, and monthly heating energy from gas bills. Each building is scored on
  four numbers: February and March of both years. January is simulated as
  warm-up and discarded.
- **Search space.** Twelve bounded parameters, listed in the table below. One
  value per parameter and building applies to all months and years.
- **Objective.** WAPE, the summed absolute monthly error divided by the summed
  measured energy. The score of a building is `max(0, 100 - WAPE%)`, and the
  official score is the mean over the three buildings.
- **Budget.** At most 48 different candidates may be simulated per building.
  Candidates can be simulated in parallel batches (12 at a time in our runs),
  and a batch must be chosen before any of its results is known.
- **Tools.** The search must be our own code. The official environment has
  NumPy, SciPy, and scikit-learn, but no optimization packages.

| Parameter | Default | Bounds | Unit |
| --- | ---: | ---: | --- |
| `weekday_heating_hours` | 8 | 4–12 | h/day ¹ |
| `non_workday_heating_hours` | 2 | 0–4 | h/day ¹ |
| `weekday_start_hour` | 7 | 5–9 | hour ¹ |
| `occupied_heating_setpoint_c` | 20 | 18.5–21.5 | °C |
| `night_setback_temperature_c` | 15 | 12–18 | °C |
| `outdoor_shutdown_temperature_c` | 15 | 12–18 | °C |
| `normally_heated_area_fraction` | 0.80 | 0.60–1.00 | fraction |
| `reduced_area_setpoint_c` | 15 | 12–17 | °C |
| `reduced_area_schedule_fraction` | 0.50 | 0.25–0.75 | fraction ¹ |
| `air_exchange_multiplier` | 1.00 | 0.70–1.30 | multiplier |
| `opaque_envelope_u_value_multiplier` | 1.00 | 0.90–1.10 | multiplier |
| `window_u_value_multiplier` | 1.00 | 0.90–1.10 | multiplier |

¹ The simulation uses whole-hour schedules, so these values are rounded.
Candidates that round to the same schedule are the same simulation and are
charged to the budget once.

What makes this hard is the ratio of unknowns to simulations. A grid with only
two values per parameter would already need 2¹² = 4096 simulations, and we
have 48. Every simulation has to be chosen using everything learned from the
earlier ones.

## Our solution: Bayesian optimization focused by ARD

Bayesian optimization replaces the expensive simulator with a cheap
statistical model of it and uses that model to decide what to simulate next.
Our model is a Gaussian process (GP): fitted to the candidates simulated so
far, it predicts the error of any untried candidate and says how uncertain
that prediction is.

1. **Initial design, 13 simulations.** The organizer defaults plus 12
   candidates from a Latin hypercube, which spreads them evenly over the
   whole search space. Starting with the defaults guarantees that the result
   is never worse than the baseline.
2. **Fit the GP** to all results so far. We model log(1 + WAPE) rather than
   WAPE, because errors range from 1 % to over 150 % and the logarithm keeps
   the few very bad candidates from dominating the fit.
3. **Choose the next batch by Expected Improvement.** This prefers candidates
   that are predicted to be good or are uncertain enough that they might be,
   so it balances refining the best region against exploring unknown ones.
   Since a batch of 12 runs in parallel, we fill it one candidate at a time
   and pretend each chosen candidate returned the GP's prediction (the
   "kriging believer"). The following picks then avoid that spot, and the
   batch spreads out.
4. **Repeat** steps 2 and 3 until the 48 simulations are used, and return the
   best candidate simulated.

The part that makes it our method is the focus on relevant parameters. Twelve
parameters are many for 48 simulations, and for any one building only some of
them change the error much. **ARD, Automatic Relevance Determination,** finds
out which. Our GP has one length scale per parameter, and fitting the GP sets
each length scale to the distance over which the error changes along that
parameter. A short length scale means the error reacts strongly to the
parameter; a long one means the parameter hardly matters. The ranking costs no
extra simulations, because it is read from the model we fit anyway.

After the initial design, each batch therefore varies only the six parameters
with the shortest length scales. The other six are pinned to the values of the
best candidate so far, not to the defaults. The ranking is redone before each
batch, because the first one rests on 13 points only, and it is made per
building, because the buildings differ. If budget is left near the end, all
twelve parameters are varied again.

The focus pays off most where plain Bayesian optimization struggles. The table
shows the best error found so far (WAPE in %) after each batch, for the
identical initial design:

| Simulations used | 13 (initial design) | 25 | 37 | 48 |
| --- | ---: | ---: | ---: | ---: |
| `building_01`, all parameters | 19.84 | 5.92 | 4.40 | 4.32 |
| `building_01`, ARD focus | 19.84 | 5.16 | 3.97 | 3.97 |
| `building_02`, all parameters | 34.35 | 2.57 | 1.33 | 0.96 |
| `building_02`, ARD focus | 34.35 | 2.07 | 1.07 | 0.95 |
| `building_03`, all parameters | 13.32 | 7.80 | 5.88 | 5.78 |
| `building_03`, ARD focus | 13.32 | 4.63 | 3.98 | 3.98 |

A calibration took two to five minutes per building on 12 cores.

### What the calibration found

| Parameter | `building_01` | `building_02` | `building_03` |
| --- | ---: | ---: | ---: |
| `weekday_heating_hours` | 4.00 | 4.00 | 4.00 |
| `non_workday_heating_hours` | 3.17 | 0.00 | 0.00 |
| `weekday_start_hour` | 6.57 | 5.00 | 8.05 |
| `occupied_heating_setpoint_c` | 18.54 | 20.84 | 18.50 |
| `night_setback_temperature_c` | 12.00 | 12.00 | 14.84 |
| `outdoor_shutdown_temperature_c` | 16.44 | 13.49 | 12.13 |
| `normally_heated_area_fraction` | 0.97 | 0.60 | 0.64 |
| `reduced_area_setpoint_c` | 13.94 | 12.00 | 14.43 |
| `reduced_area_schedule_fraction` | 0.25 | 0.55 | 0.37 |
| `air_exchange_multiplier` | 0.70 | 0.99 | 1.26 |
| `opaque_envelope_u_value_multiplier` | 0.94 | 1.10 | 0.95 |
| `window_u_value_multiplier` | 0.97 | 0.93 | 1.10 |

All three buildings end at the lower bound of four heating hours per weekday,
half of the default eight. That is the clearest signal in the data: the
defaults assume far more heating than the bills show.

The other values should be read with care. Twelve parameters are fitted to
four measurements per building, so different parameter sets can reproduce the
bills about equally well. For `building_01`, for example, the variant of our
method that scored highest (step 5 below) fits slightly better with a heated
area fraction of 0.73 instead of 0.97 and an air exchange multiplier of 0.95
instead of 0.70. The calibration shows that the model can match the bills,
not that each value is the true one.

## How we got there

All scores below are from the official configuration: three buildings, both
years, budget 48, seed 2026, batches of 12. Unless a subset is named, all
twelve parameters were active. Every run is saved in `outputs/runs/` and can
be listed with `python experiments.py compare --all`.

### 1. A baseline and a way to compare

We began with random search, and then built the experiment runner
[`experiments.py`](experiments.py). It runs any method under the official
configuration, saves every result, and skips experiments whose result is
already saved and whose code still behaves the same. That let us try about a
hundred combinations of method and settings overnight on a cluster without
rerunning what was already saved.

### 2. A range of search methods

We implemented the usual candidates for expensive black-box problems, with
NumPy, SciPy, and scikit-learn only.

| Method | Score | `building_01` | `building_02` | `building_03` |
| --- | ---: | ---: | ---: | ---: |
| Organizer defaults, no search | 12.92 | 38.76 | 0.00 | 0.00 |
| Hyperband over local-search arms | 84.21 | 90.28 | 76.88 | 85.48 |
| Bayesian optimization with a fixed, unfitted GP | 91.96 | 94.86 | 89.88 | 91.15 |
| CMA-ES (evolution strategy) | 93.44 | 94.44 | 95.66 | 90.23 |
| TPE (tree-structured Parzen estimator) | 94.46 | 93.87 | 97.58 | 91.93 |
| Bayesian optimization with a fitted GP | 96.32 | 95.68 | 99.04 | 94.22 |

Methods that build a model of the simulator used the budget best. Evolution
strategies and Hyperband need many more evaluations than 48 to pay off, and
Hyperband has nothing to exploit here because every simulation costs the
same. Fitting the GP to the data mattered too: the simple version with a
fixed GP scored more than four points lower.

We also varied how the fitted GP chooses its batch. None of the variants was
reliably better than Expected Improvement with the kriging believer:

| Batch selection | Score |
| --- | ---: |
| Expected Improvement, kriging believer | 96.32 |
| Expected Improvement, constant liar | 95.28 |
| Max-value Entropy Search | 95.42 |
| Confidence bound, exploration weight 2 | 95.49 |
| Confidence bound, exploration weight 0.1, 0.3, 0.6, 1, 4 | 97.24, 95.97, 96.80, 96.71, 96.01 |

The confidence bound scored between 95.5 and 97.2 depending on its weight,
without a trend. We read that spread of almost two points as the noise of a
single run.

### 3. Fewer parameters help, but only the right ones

Next we gave Bayesian optimization hand-picked subsets of the parameters and
left the rest at their defaults.

| Active parameters | Score |
| --- | ---: |
| All 12 | 96.32 |
| 9: without air exchange and the two U-value multipliers | 97.10 |
| 6: heating hours, three temperatures, heated area, air exchange | 88.69 |
| 8: without the four schedule parameters | 42.23 |

Dropping three weak parameters improved the score, so twelve dimensions are
indeed too many for the budget. Dropping the wrong ones was far worse than
keeping all of them: without the schedule parameters the search cannot reduce
the heating hours, and the score collapses. A subset picked by hand is a bet.
It also has to be the same for all buildings, although the buildings differ.

### 4. Let the data choose the parameters

So we looked for ways to choose the parameters automatically, per building,
during the run, and tried four.

| How the parameters are chosen | Score | Best score over the settings we tried |
| --- | ---: | ---: |
| Screening design first, then main effects | 94.07 | |
| Lasso regression on the results so far | 94.59 | 95.70 |
| CalibLab pipeline: Sobol sensitivity, GP metamodel, MCMC | 87.13 | 95.58 |
| **ARD: length scales of the GP** | **97.03** | **97.55** |

The screening design spends 17 of the 48 simulations on a fixed two-level
design, which leaves less budget for the search. The Lasso assumes that the error
depends linearly on each parameter. CalibLab, an open-source Bayesian
calibration framework that we re-implemented for this budget, fixes its
parameter choice after a first sensitivity analysis and depends strongly on
the random seed: five seeds gave scores from 79.06 to 87.85.

ARD needs none of that. It uses the model that Bayesian optimization fits
anyway, it captures non-linear effects, and it can change its mind as results
arrive.

### 5. Tuning the ARD method

Finally we varied each of the four settings of the ARD method on its own.

| Setting | Values tried and their scores (bold: the value in use) | In `calibration.py` |
| --- | --- | ---: |
| `ARD_TOP_FRACTION`, share of the parameters varied | 0.25: 96.20, **0.5: 97.03**, 0.75: 96.97 | 0.5 |
| `ARD_INITIAL_FRACTION`, largest share of the budget for the initial design | 0: 94.45, 0.2: 96.80, 0.25: 95.90, **0.35 to 1: 97.03** | 0.35 |
| `ARD_RNK`, share of the focus stage after which the ranking is redone | **0 to 0.5: 97.03**, 0.75 and 1: 96.84 | 0.25 |
| `ARD_FOCUS_END_FRACTION`, share of the budget after which all parameters are varied again | 0 and 0.25: 96.32, 0.5: 97.55, 0.75: 97.19, **0.8 and 1: 97.03** | 0.8 |

Three things follow from the sweep.

- The focus is what helps. With `ARD_FOCUS_END_FRACTION` at 0 or 0.25 no batch
  is focused, the method is plain Bayesian optimization, and the score is
  96.32.
- The method is not very sensitive to its settings. Apart from the two
  settings that switch a part of it off (no focused batch, no initial design
  beyond the defaults), every variant scores between 95.9 and 97.6.
- Ending the focus earlier scored highest: 97.55 with 0.5, where only the
  first batch after the initial design is focused. `calibration.py` uses 0.8,
  which focuses all three batches.

### How far to trust these numbers

Each score comes from one run with one seed. Judging by the spread in step 2
and by CalibLab's seeds, a difference of one or two points between two runs
can be chance. The ranking of the method families is clear. Whether ARD at
97.03 is really better than the best confidence-bound run at 97.24, or 97.55
really better than 97.03, is not established by these runs. What we rely on
is that five of the seven best distinct results among all saved runs are ARD
variants, and that the focus was at least as good as plain Bayesian
optimization on every building, from the identical initial design.

## Repository

| Path | Content |
| --- | --- |
| [`calibration.py`](calibration.py) | The submission: the ARD method only, with the reasoning in its docstrings. |
| [`alternative_algorithms.py`](alternative_algorithms.py) | Every other method from the sections above. Not submitted; it imports the shared building blocks from `calibration.py`. |
| [`experiments.py`](experiments.py) | Experiment runner: runs, saves, and compares experiments. The notebook uses it too. |
| [`experiments.json`](experiments.json) | The experiments of the most recent sweep. |
| `outputs/runs/` | One folder per saved run: the four standardized result files, the exact code that ran, and a record of the settings. |
| [`notebooks/hackathon_integrates.ipynb`](notebooks/hackathon_integrates.ipynb) | The organizer notebook: data, baseline, calibration, scoring, export, and a comparison of saved runs. |
| `slurm/experiments.sbatch` | Runs the experiments as a Slurm job array. |
| `src/`, `data/`, `tests/`, `scripts/` | Organizer code and data: simulation, scoring, export. Unchanged. |

## Reproducing the results

The environment is the organizer's: Python 3.11 and pinned package versions,
so that everyone simulates the same energy.

```powershell
conda env create -f environment.yml
conda activate integrates_hackathon
```

**The submission, through the notebook.** Open
`notebooks/hackathon_integrates.ipynb`. Section 01 currently holds a short
development configuration; for the official run set both years, budget 48, and
all three buildings there. Then run all cells. Section 04 runs
`calibration.py`, or loads the result from `outputs/runs/` if this exact
experiment is already saved.

**Any experiment, from the command line.**

```powershell
python experiments.py status            # which experiments of experiments.json are saved, which would run
python experiments.py run               # run and save the missing ones
python experiments.py compare --all     # tables and plots of all saved runs; never simulates
python experiments.py compare --runs all_12__ard --runs all_12__bayesian_optimization__
```

An experiment names an algorithm, either the one in `calibration.py` or one
from `alternative_algorithms.py`, and optionally buildings, years, budget,
seed, active parameters, and values for the algorithm's settings. For example,
this entry of a settings file repeats the last row of the sweep in step 5:

```json
{"name": "all_12", "algorithm": "ard_bayesian_optimisation",
 "constants": {"ARD_FOCUS_END_FRACTION": [1, 0.75, 0.5, 0.25, 0]}}
```

The docstring of `experiments.py` describes the settings file and the rule
for when a saved result is reused. The saved runs were produced on a Linux
cluster. On a machine with other package versions the runner first checks,
on cheap test problems, that the algorithm still proposes the same
candidates, and reruns the experiment if it does not.

**Tests.**

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

These are the organizer's tests. Three of them check the untouched starter kit
(a `calibration.py` that only evaluates the defaults, and a notebook with the
official configuration and no active parameter) and therefore fail in this
repository; the others pass.

## Submission rules we follow

- `calibration.py` defines `TEAM_NAME`, `ALGORITHM_NAME`, and
  `calibrate(objective, parameter_space, budget, seed)`, and returns one
  in-bounds value per parameter.
- It reads dimensions, bounds, defaults, budget, and seed from its arguments.
  No building, year, or parameter count is hard-coded.
- It evaluates candidates only through `objective`, in batches through
  `objective.evaluate_many`, and never asks for more than the remaining
  budget.
- It imports only NumPy, SciPy, scikit-learn, and the standard library, and
  does not depend on any other file of this repository.

## License

BSD 3-Clause; see [`LICENSE.md`](LICENSE.md).
