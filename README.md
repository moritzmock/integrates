# INTEGRATES Building-Energy Calibration Hackathon

This repository contains the participant workflow for a building-energy
calibration challenge. It provides three anonymized buildings, actual-year
weather, monthly measured delivered energy, a fixed pyBuildingEnergy simulation
workflow, and standardized scoring and export. Participants develop only the
search strategy in the repository-root `calibration.py`.

## Official challenge configuration

- Buildings: `building_01`, `building_02`, and `building_03`
- Weather and measurement years: 2018 and 2019
- Simulation window: every hour of January, February, and March
- Warm-up: January, discarded before scoring
- Scored months: February and March; results are not annualized
- Evaluation budget: 48 unique parameter candidates per building
- Seed: 2026, with a stable building-ID offset
- Score: `max(0, 100 - WAPE%)`, averaged equally across the three buildings
- Execution: a local process pool on each team's own computer

One scalar value is returned for each active parameter and building. That value
is shared by both scored months and both years; month-specific and year-specific
parameters are not accepted.

## Requirements and installation

The supported environment uses Python 3.11 and pyBuildingEnergy 2.1.0.
`environment.yml` pins the exact versions of pyBuildingEnergy and every package
it uses, so every team and the official verification run get the same simulated
energy. Use Conda with this file rather than installing packages separately;
Section 03 warns if installed versions differ from those used for the baseline.

From the repository root in PowerShell:

```powershell
conda env create -f environment.yml
conda activate integrates_hackathon
jupyter notebook notebooks/hackathon_integrates.ipynb
```

If the environment already exists, update it after pulling repository changes:

```powershell
conda env update -f environment.yml --prune
conda activate integrates_hackathon
```

Run the notebook from top to bottom. The notebook is intentionally thin;
challenge-owned Python modules handle data loading, simulation, parallel work,
budget enforcement, validation, plots, scoring, and export.

## What participants may change

1. Edit `TEAM_NAME`, `ALGORITHM_NAME`, and `calibrate(...)` in `calibration.py`.
2. Activate any subset of the twelve entries in `calibratable_params_init` in
   notebook Section 04 by replacing `None` with an in-bounds numeric value.
3. During development only, reduce the building list, year list, evaluation
   budget, or worker count in notebook Section 01 to shorten test runs.

Do not add, remove, or rename parameter keys, or change parameter bounds,
simulation code, input data, measurements, weather, baseline, scoring, or export
code. Official verification restores all three buildings, both years, budget 48, and
seed 2026 in a clean copy of this repository.

## Participant API

Preserve these public names in `calibration.py`:

```python
TEAM_NAME = "Your team"
ALGORITHM_NAME = "Your method"

def calibrate(objective, parameter_space, budget, seed) -> dict[str, float]:
    ...
```

`parameter_space` is an ordered sequence. Each item exposes `.name`, `.lower`,
`.upper`, and `.default`. The vector order is the order of this sequence.

- Use `objective(vector)` to evaluate one candidate.
- Use `objective.evaluate_many(vectors)` when a method can propose a batch.
- `objective.max_batch_size` reports the configured local worker count.
- `objective.remaining_budget` reports how many new candidates may still be
  evaluated for the current building.
- Return exactly one finite, in-bounds value for every supplied parameter name.
- Read dimensions, bounds, defaults, budget, and seed at runtime; do not
  hard-code buildings, months, years, or a particular parameter count.

Evaluate candidates only through `objective`. `calibration.py` may use any
installed optimization or scientific library (for example NumPy or SciPy), but
it is rejected before it runs if it imports `integrates_hackathon`,
`pybuildingenergy`, `multiprocessing`, `concurrent.futures`, or `subprocess`.
After each building, the organizer compares the objective's budget count with an
independent count of simulations; a mismatch, or more evaluations than the
budget, stops the run. Submitted code is also reviewed during verification.

The organizer objective caches duplicate candidates. Only unique candidates
count against the per-building budget, and a request beyond the remaining
budget is rejected. Candidates that produce the same whole-hour heating
schedules (see footnote ¹ below) are the same simulation and are charged once.
Every candidate value must be a finite real number; NaN, infinity, booleans,
and text raise an error without using budget. Finite values outside the bounds
are clipped to the nearest bound. The same candidate is evaluated against every scored month and year
for the current building before its loss is returned. The loss returned by the
objective is WAPE in percentage points, so lower is better.

The shipped starter is a deliberate baseline passthrough. With every selection
set to `None`, it evaluates the empty active vector once while the simulator
uses all organizer defaults. Consequently, its calibrated score equals the
organizer baseline. Participants must select parameters and replace the
passthrough with a search strategy to improve the score.

## Parameter space

Participants may activate any subset of these twelve parameters. `None` holds a
parameter at its organizer default; an in-bounds number activates it and becomes
its optimizer starting value.

| Parameter | Default | Bounds | Unit |
| --- | ---: | ---: | --- |
| `weekday_heating_hours` | 8 | 4–12 | h/day¹ |
| `non_workday_heating_hours` | 2 | 0–4 | h/day¹ |
| `weekday_start_hour` | 7 | 5–9 | hour¹ |
| `occupied_heating_setpoint_c` | 20 | 18.5–21.5 | °C |
| `night_setback_temperature_c` | 15 | 12–18 | °C |
| `outdoor_shutdown_temperature_c` | 15 | 12–18 | °C |
| `normally_heated_area_fraction` | 0.80 | 0.60–1.00 | fraction |
| `reduced_area_setpoint_c` | 15 | 12–17 | °C |
| `reduced_area_schedule_fraction` | 0.50 | 0.25–0.75 | fraction¹ |
| `air_exchange_multiplier` | 1.00 | 0.70–1.30 | multiplier |
| `opaque_envelope_u_value_multiplier` | 1.00 | 0.90–1.10 | multiplier |
| `window_u_value_multiplier` | 1.00 | 0.90–1.10 | multiplier |

¹ The simulation uses whole-hour schedules. Start hour and heating hours are
rounded to the nearest integer, and reduced-area hours are the rounded product
of heating hours and `reduced_area_schedule_fraction`. Continuous optimizers may
submit floats, but different floats can produce the same schedule: for example,
7.6 and 8.4 weekday hours both mean 8 h, and fractions 0.50 and 0.53 both give
4 reduced-area hours on an 8 h weekday. Such candidates return the same loss
and use budget only once.

## Simulation model and data

Audited envelope properties and generator efficiencies are fixed unless a team
activates one of the permitted bounded multipliers above. The model represents
normally heated space and lower-temperature space as two thermal regimes and
area-weights their hourly results. Maintained setback heating is counted.
Bolzano's legal heating season and dated municipal exceptions are fixed inputs.

The public Excel files contain anonymized building geometry and energy data for
the three challenge buildings. The buildings' gas bills combine space heating
and domestic hot water (DHW), while the model simulates space heating only.
The scored measurements are therefore the heating-only rows of
`data/energy_consumption_values.xlsx`: each month's combined consumption minus
that year's June–August average, taken as the monthly DHW load. Small negative
residuals in May–September, outside the legal heating season, are set to zero.
The EPW files contain actual meteorological years for Bolzano and do not encode
building locations.

Section 03 loads a version-checked baseline from:

- `data/precomputed_baseline.csv`
- `data/precomputed_baseline_metadata.json`

Participants do not rerun the baseline. The cache is tied to the fixed inputs,
simulation code, and pyBuildingEnergy version; the notebook stops if these do
not match.

## Scoring

For each building:

```text
WAPE (%) = 100 × Σ |simulated − measured| / Σ measured
score    = max(0, 100 − WAPE)
```

The official score is the arithmetic mean of the three building scores, so
larger buildings do not dominate. CV(RMSE), NMBE, and evaluation count
are exported as diagnostics and may support organizer review or tie-breaking.

## Parallel execution and development runs

`PARALLEL_WORKERS` controls the challenge-owned local process pool. Start with a
value no larger than the laptop's available logical processors and reduce it if
the computer becomes memory-constrained. Changing the worker count should
affect runtime, not numerical results. Participant code should not create its
own multiprocessing pool; use `objective.evaluate_many(...)` for batched work.

For a quick development run, teams may temporarily select one building, one
year, and a smaller evaluation budget in Section 01. Official verification
restores the configuration listed at the top of this README.

### Expected runtime

Each candidate runs two thermal regimes for each selected year. A single call to
`objective(vector)` can only run the two years in parallel, so most workers sit
idle. Batches from `objective.evaluate_many(...)` keep the pool busy. Measured
with the official configuration on one 14-logical-core laptop with
`PARALLEL_WORKERS = 12`:

| Calling pattern | Time per candidate | Budget 48 × 3 buildings |
| --- | ---: | ---: |
| `objective(vector)`, one at a time | 12–45 s | about 30 min to 2 h |
| `evaluate_many(...)`, batches of 6–12 | 5–8 s | about 12–20 min |

Times varied by about a factor of three on the same laptop with power mode and
background load, and will be longer on computers with fewer cores. The first
evaluation also starts the worker pool, and every building adds one final rerun
of the returned parameters. Plan development runs accordingly.

## Standardized output and submission

Section 07 writes four files to `outputs/<team-name>/`:

- `summary.csv`: one row per building;
- `monthly_results.csv`: actual, baseline, calibrated, normalized values, and
  monthly errors;
- `evaluation_trace.csv`: every unique candidate charged against the
  per-building evaluation budget, in execution order, with its parameters,
  WAPE loss, score, and an indication of whether it was returned as the final
  candidate;
- `submission_result.json`: configuration, score, active parameter selection,
  final calibrated values, and the SHA-256 hash of the exact `calibration.py`
  bytes executed for the run.

Keep the exported JSON from the team's final run because
`active_parameter_selection` records the selected parameters and
`calibration_py_sha256` identifies the exact submitted `calibration.py`. The
organizer verifies the submitted file hash and that selection in a clean
challenge copy under the official configuration.
Changes to organizer-owned files are not included in a submission.

## Verification

Run the automated checks from the repository root:

```powershell
conda activate integrates_hackathon
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Expected result: all tests pass. If notebook imports fail, confirm that Jupyter
is running from the `integrates_hackathon` environment and restart its kernel.

## Organizer-only baseline regeneration

After an authorized change to fixed inputs or simulation code, organizers can
regenerate the expensive baseline with:

```powershell
$env:PYTHONPATH = "src"
python scripts/generate_baseline_cache.py
```

Then confirm that a fresh pyBuildingEnergy run reproduces the cache:

```powershell
$env:INTEGRATES_SLOW_TESTS = "1"
python -m unittest discover -s tests -v
```

Participants should not run these commands.

When the repository is hosted on GitHub, `.github/workflows/tests.yml` runs the
same checks on Windows, macOS, and Linux for every push: the full test suite
including this baseline regression, then the release notebook from top to
bottom with the starter `calibration.py`. If a pinned package version is
changed, regenerate the baseline in the updated environment.

## License

This project is licensed under the BSD 3-Clause License. See `LICENSE.md`.
