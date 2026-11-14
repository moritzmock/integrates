"""Team Calibra's submission: Bayesian optimization focused by ARD.

The task
--------
For each building, find the values of up to twelve operating and envelope
parameters for which the simulated heating energy matches the gas bills. The
organizer supplies ``objective``: it simulates one candidate (or a batch of
candidates in parallel) and returns the error, WAPE in percent. Only 48
candidates per building may be simulated, so the search must learn as much as
possible from every simulation.

The method
----------
Bayesian optimization replaces the expensive simulator with a cheap
statistical model of it, a Gaussian process (GP), and uses the model to decide
which candidates are worth simulating:

1. Simulate an initial design: the organizer defaults plus a Latin hypercube
   sample that covers the whole parameter space.
2. Fit the GP to all results so far. It predicts the error of any untried
   candidate together with the uncertainty of that prediction.
3. Simulate the batch of candidates with the largest Expected Improvement:
   those predicted to be good, or uncertain enough that they might be.
4. Repeat from step 2 until the budget is used, then return the best
   simulated candidate.

Twelve parameters are a lot to explore with 48 simulations, and for a given
building only some of them change the error noticeably. The GP tells us which
ones, through Automatic Relevance Determination (ARD): its kernel has one
length scale per parameter, and the fit sets each length scale to the distance
over which the error changes along that parameter. A short length scale means
the error reacts strongly to the parameter; a long one means the parameter
hardly matters. In the middle of the run we therefore vary only the half of
the parameters with the shortest length scales and pin the others to the best
values found so far. The ranking is redone as results arrive, and it is made
separately for each building. See ``ard_bayesian_optimisation`` for the stages.

Layout of this file
-------------------
- ``_History``: budget bookkeeping and scaling of the parameters to [0, 1].
- ``ard_bayesian_optimisation``: the stages of the search.
- ``_propose_batch`` and the helpers after it: GP fit, Expected Improvement,
  candidate generation, and batch filling.
- ``calibrate``: the entry point called by the organizer workflow.

The other methods we tried are in ``alternative_algorithms.py``, which is not
part of the submission. Everything here uses NumPy, SciPy, and scikit-learn
only, because the official environment has no optimization packages.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


TEAM_NAME = "Calibra"
ALGORITHM_NAME = "ard_bayesian_optimisation"

# Stop if this many consecutive batches add no new (budget-charged) candidates,
# e.g. when all proposals collapse onto already-evaluated whole-hour schedules.
_MAX_STALLED_BATCHES = 5


class _History:
    """Evaluates candidates in the unit cube and remembers every result.

    The search works on [0, 1]^d; this class maps to physical bounds, sizes
    batches to the remaining budget, and tracks the best point so far.
    """

    def __init__(self, objective, lower: np.ndarray, upper: np.ndarray, budget: int):
        self.objective = objective
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.submitted = 0
        self.stalled = 0
        self.batches = 0
        self.X: list[np.ndarray] = []  # unit-cube points
        self.y: list[float] = []  # WAPE losses (%)

    def to_physical(self, unit: np.ndarray) -> np.ndarray:
        return self.lower + np.clip(unit, 0.0, 1.0) * (self.upper - self.lower)

    def to_unit(self, physical: np.ndarray) -> np.ndarray:
        span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
        return np.clip((physical - self.lower) / span, 0.0, 1.0)

    def remaining(self) -> int:
        remaining = self.objective.remaining_budget
        if remaining is None:  # Unlimited objective: use our own upper-bound count.
            remaining = self.budget - self.submitted
        return max(0, int(remaining))

    def can_continue(self) -> bool:
        return self.remaining() > 0 and self.stalled < _MAX_STALLED_BATCHES

    def next_batch_size(self) -> int:
        return max(1, min(self.objective.max_batch_size, self.remaining()))

    def evaluate(self, units: np.ndarray) -> None:
        units = np.atleast_2d(units)[: self.remaining()]
        if len(units) == 0:
            return
        before = self.objective.remaining_budget
        vectors = [tuple(map(float, self.to_physical(u))) for u in units]
        losses = self.objective.evaluate_many(vectors)
        self.batches += 1
        self.submitted += len(units)
        if before is not None:
            self.stalled = self.stalled + 1 if self.objective.remaining_budget == before else 0
        self.X.extend(np.clip(units, 0.0, 1.0))
        self.y.extend(float(v) for v in losses)

    def best(self) -> tuple[np.ndarray, float]:
        index = int(np.argmin(self.y))
        return self.to_physical(self.X[index]), self.y[index]


# --------------------------------------------------------------------------
# Bayesian optimization focused by Automatic Relevance Determination (ARD)
# --------------------------------------------------------------------------

# All four settings are shares, so they adapt to any budget and parameter count.
ARD_INITIAL_FRACTION = 0.35  # at most this share of the budget goes to the initial design
ARD_FOCUS_END_FRACTION = 0.8  # once this share of the budget is used, every parameter is searched again
ARD_TOP_FRACTION = 0.5  # share of the parameters searched during the focus stage
ARD_RNK = 0.25  # the parameters are ranked again after this share of the focus stage (0: before every batch)


def ard_bayesian_optimisation(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Bayesian optimization in three stages, focused by ARD length scales.

    ARD stands for Automatic Relevance Determination. The GP kernel has one
    length scale per parameter, and fitting the GP sets each one to the
    distance over which the loss changes along that parameter. Sorting the
    parameters by length scale therefore ranks them by relevance, shortest
    first, at no extra simulation cost (see ``_ard_active_dimensions``).

    1. Initial stage: the start vector plus a Latin hypercube sample of all
       parameters. The start vector (the organizer defaults) comes first, so
       the result is never worse than the starting point. The design has
       one point more than there are parameters, or one full batch if that is
       larger, but uses at most ARD_INITIAL_FRACTION of the budget.
    2. Focus stage, while less than ARD_FOCUS_END_FRACTION of the budget is
       used: only the ARD_TOP_FRACTION of the parameters with the shortest
       length scales are varied. The others are pinned to the values of the
       best point so far, not to the defaults. The ranking is redone each time
       another ARD_RNK of the focus stage has passed, because the first
       ranking rests on few points and the pinned values change with the best
       point.
    3. Final stage: every parameter is varied again, which fine-tunes the
       parameters that were pinned.

    In stages 2 and 3 each batch is proposed by ``_propose_batch``: Expected
    Improvement on a GP fitted to all results so far, filled greedily with the
    kriging believer.

    Example, the official run (12 parameters, budget 48, batches of 12): 13
    initial points, then batches of 12, 12, and 11. These start with 27 %,
    52 %, and 77 % of the budget used, which is below 80 %, so all three are
    focus batches on six parameters, each with a fresh ranking. The final
    stage runs only on budget left over after that, which happens when
    candidates were duplicates of earlier ones and were not charged.
    """

    if not 0 <= ARD_RNK <= 1:
        raise ValueError("ARD_RNK must be between 0 and 1")

    dim = len(start)
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, int(history.remaining() * ARD_INITIAL_FRACTION)))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    history.evaluate(np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))]))

    active = None  # indices of the parameters being varied; None means all of them
    next_ranking = 0.0  # focus-stage progress at which the ranking is redone

    while history.can_continue():
        progress = history.submitted / max(1, history.budget)
        if progress < ARD_FOCUS_END_FRACTION:
            # Progress within the focus stage. It is negative for the first
            # batch when the initial design was smaller than its allowance.
            focus_span = max(ARD_FOCUS_END_FRACTION - ARD_INITIAL_FRACTION, 1e-12)
            focus_progress = (progress - ARD_INITIAL_FRACTION) / focus_span
            if active is None or ARD_RNK == 0 or focus_progress >= next_ranking:
                y = np.log1p(np.maximum(np.array(history.y), 0.0))
                active = _ard_active_dimensions(_fit_gp(np.array(history.X), y, rng))
                next_ranking = focus_progress + ARD_RNK
                print(f"ard: searching parameters {active.tolist()} of {dim}")
        else:
            active = None  # final stage: all parameters

        batch = _propose_batch(
            np.array(history.X),
            np.array(history.y),
            history.next_batch_size(),
            rng,
            _expected_improvement,
            _kriging_believer,
            active=active,
        )
        history.evaluate(batch)


def _ard_active_dimensions(gp) -> np.ndarray:
    """Indices of the ARD_TOP_FRACTION of the parameters with the shortest length scales.

    All parameters are scaled to [0, 1], so their length scales are directly
    comparable: 0.2 means the loss changes over a fifth of the allowed range,
    while 10 or more means it is practically flat along that parameter.
    """

    # ConstantKernel * Matern + WhiteKernel: the Matern is kernel_.k1.k2.
    length_scales = np.asarray(gp.kernel_.k1.k2.length_scale, dtype=float).reshape(-1)
    count = max(1, int(np.ceil(len(length_scales) * ARD_TOP_FRACTION)))
    return np.sort(np.argsort(length_scales, kind="stable")[:count])


# --------------------------------------------------------------------------
# One round of Bayesian optimization: fit the GP, then choose a batch
# --------------------------------------------------------------------------


def _propose_batch(
    X: np.ndarray,
    losses: np.ndarray,
    size: int,
    rng: np.random.Generator,
    acquisition: Callable,
    fantasy: Callable,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Choose ``size`` candidates to simulate next, given all results so far.

    A GP is fitted to log(1 + WAPE) and the batch is filled greedily: pick the
    candidate that maximizes ``acquisition``, add it to the data with the fake
    loss returned by ``fantasy``, recondition the GP, and pick the next one.
    The fake results make the batch spread out instead of piling onto the
    same optimum, which matters because a whole batch is simulated in
    parallel before any of its results is known.

    ``acquisition(gp, candidates, best, rng)`` returns one score per candidate
    (higher is better). ``fantasy(gp, point, y_observed)`` returns the fake
    log-loss assumed for a point that is chosen but not yet evaluated. The
    submitted method uses ``_expected_improvement`` and ``_kriging_believer``;
    ``alternative_algorithms.py`` passes other choices.

    If ``active`` is given, only those parameters are varied and the others
    are pinned to the best observed point's values.
    """

    y = np.log1p(np.maximum(losses, 0.0))  # Compress large errors; smoother surface.
    gp = _fit_gp(X, y, rng)
    kernel = gp.kernel_
    pool = _candidate_pool(X, y, rng)
    if active is not None:
        pinned = np.tile(X[int(np.argmin(y))], (len(pool), 1))
        pinned[:, active] = pool[:, active]
        pool = pinned
    X_fantasy, y_fantasy = X.copy(), y.copy()
    batch: list[np.ndarray] = []
    for _ in range(size):
        score = acquisition(gp, pool, float(y_fantasy.min()), rng)
        choice = pool[int(np.argmax(score))]
        batch.append(choice)
        # Add a fake observation (kriging believer / constant liar) and
        # recondition the GP with the hyperparameters kept fixed.
        X_fantasy = np.vstack([X_fantasy, choice])
        y_fantasy = np.append(y_fantasy, fantasy(gp, choice, y))
        gp = _fit_gp(X_fantasy, y_fantasy, rng, kernel=kernel)
    return np.array(batch)


def _fit_gp(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, kernel=None):
    """Fit the GP surrogate to unit-cube points ``X`` and log-losses ``y``.

    The kernel is a Matern 5/2 with one length scale per parameter (this is
    what makes it an ARD kernel), times a signal variance, plus a small noise
    term. The hyperparameters are fitted by maximizing the marginal
    likelihood from four starting points. If ``kernel`` is given, it is used
    as it is: a fast refit on fantasy data with the hyperparameters fixed.
    """

    dim = X.shape[1]
    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.full(dim, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(1e-4, (1e-8, 1e-1))
        optimizer, restarts = "fmin_l_bfgs_b", 3
    else:  # Reuse fitted hyperparameters (fast refit on fantasy data).
        optimizer, restarts = None, 0
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=optimizer,
        n_restarts_optimizer=restarts,
        random_state=int(rng.integers(2**31 - 1)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def _expected_improvement(gp, candidates: np.ndarray, best: float, rng=None) -> np.ndarray:
    """Expected amount by which each candidate's loss falls below ``best``.

    It is large where the predicted loss is low (exploitation) and where the
    prediction is uncertain (exploration), which balances the two without a
    tuning parameter.
    """

    # ``rng`` is unused; it keeps the signature shared with the other acquisitions.
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    z = (best - mean) / std
    return (best - mean) * norm.cdf(z) + std * norm.pdf(z)


def _kriging_believer(gp, point: np.ndarray, y_observed: np.ndarray) -> float:
    """Fake loss = the GP's own prediction at the chosen point."""

    return gp.predict(point[None, :])[0]


def _candidate_pool(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Candidates scored by the acquisition: 2000 uniform random points plus
    6000 Gaussian perturbations (three widths) around the five best points."""

    dim = X.shape[1]
    global_points = rng.uniform(size=(2000, dim))
    elite = X[np.argsort(y)[: min(5, len(y))]]
    local = [
        e + rng.normal(scale=scale, size=(400, dim))
        for e in elite
        for scale in (0.03, 0.1, 0.25)
    ]
    return np.clip(np.vstack([global_points, *local]), 0.0, 1.0)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def calibrate(
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Return the best parameters found for one building.

    Called by the organizer workflow once per building. Dimensions, bounds,
    defaults, budget, and seed are read at run time from the arguments;
    nothing is hard-coded.
    """

    return _run_search(
        ard_bayesian_optimisation, ALGORITHM_NAME, objective, parameter_space, budget, seed
    )


def _run_search(
    search: Callable,
    name: str,
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Run ``search(history, start, rng)`` and return its best candidate by parameter name.

    Kept separate from ``calibrate`` so that ``alternative_algorithms.py`` can
    run its methods through exactly the same code.
    """

    if budget < 1:
        raise ValueError("budget must be at least one")

    lower = np.array([float(p.lower) for p in parameter_space])
    upper = np.array([float(p.upper) for p in parameter_space])
    start = np.array([float(p.default) for p in parameter_space])

    if lower.size == 0:
        # No active parameters: one evaluation of the organizer defaults.
        objective(())
        return {}

    history = _History(objective, lower, upper, budget)
    search(history, start, np.random.default_rng(seed))
    best_vector, best_loss = history.best()

    print(
        f"{name}: best WAPE {best_loss:.2f}% after "
        f"{history.batches} batches / {len(history.y)} candidates"
    )

    # Clip defensively so the returned values are always in bounds.
    best_vector = np.clip(best_vector, lower, upper)
    return {
        p.name: float(value)
        for p, value in zip(parameter_space, best_vector, strict=True)
    }
