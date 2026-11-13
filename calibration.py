"""Participant submission: batched random search or Bayesian optimization.

Teams replace the search strategy in ``calibrate`` while preserving the three
public names below. The notebook independently reruns and scores the returned
parameters, so submissions must return only the best parameter dictionary.

``ALGORITHM_NAME`` selects the search strategy (see ``_ALGORITHMS``).
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
ALGORITHM_NAME = "bayesian_optimization" #or "random_search" 

# Stop if this many consecutive batches add no new (budget-charged) candidates,
# e.g. when all proposals collapse onto already-evaluated whole-hour schedules.
_MAX_STALLED_BATCHES = 5


class _History:
    """Evaluates candidates in the unit cube and remembers every result.

    The search methods work on [0, 1]^d; this class maps to physical bounds,
    sizes batches to the remaining budget, and tracks the best point so far.
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


def random_search(history: _History, start: np.ndarray, rng: np.random.Generator) -> None:
    """Uniform random sampling within the bounds, in batches.

    The start vector (the notebook's initial values) is evaluated first, so
    the result is never worse than the starting point.
    """

    history.evaluate(history.to_unit(start))
    while history.can_continue():
        history.evaluate(rng.uniform(size=(history.next_batch_size(), len(start))))


def bayesian_optimization(
    history: _History, start: np.ndarray, rng: np.random.Generator
) -> None:
    """Batched Bayesian optimization with a Gaussian-process surrogate.

    1. Initial design: the start vector plus a Latin hypercube sample.
    2. Each round: fit a GP (Matern 5/2, one length scale per parameter) to
       log(1 + WAPE), then pick a batch by maximizing Expected Improvement
       with the "kriging believer" heuristic: after choosing a point, pretend
       its loss equals the GP prediction and choose the next one, so the batch
       spreads out instead of piling onto the same optimum.
    """

    dim = len(start)
    # Enough points to fit a first GP, but at most half the budget, so the
    # model-guided rounds always get the other half.
    n_init = max(dim + 1, history.objective.max_batch_size)
    n_init = max(1, min(n_init, history.remaining() // 2))
    design = qmc.LatinHypercube(d=dim, seed=rng)
    initial = np.vstack([history.to_unit(start), design.random(max(0, n_init - 1))])
    history.evaluate(initial)

    while history.can_continue():
        batch = _propose_batch(
            np.array(history.X), np.array(history.y), history.next_batch_size(), rng
        )
        history.evaluate(batch)


def _fit_gp(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, kernel=None):
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


def _expected_improvement(gp, candidates: np.ndarray, best: float) -> np.ndarray:
    mean, std = gp.predict(candidates, return_std=True)
    std = np.maximum(std, 1e-9)
    z = (best - mean) / std
    return (best - mean) * norm.cdf(z) + std * norm.pdf(z)


def _candidate_pool(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global random points plus local perturbations around the best points."""

    dim = X.shape[1]
    global_points = rng.uniform(size=(2000, dim))
    elite = X[np.argsort(y)[: min(5, len(y))]]
    local = [
        e + rng.normal(scale=scale, size=(400, dim))
        for e in elite
        for scale in (0.03, 0.1, 0.25)
    ]
    return np.clip(np.vstack([global_points, *local]), 0.0, 1.0)


def _propose_batch(
    X: np.ndarray, losses: np.ndarray, size: int, rng: np.random.Generator
) -> np.ndarray:
    y = np.log1p(np.maximum(losses, 0.0))  # Compress large errors; smoother surface.
    gp = _fit_gp(X, y, rng)
    kernel = gp.kernel_
    pool = _candidate_pool(X, y, rng)
    X_fantasy, y_fantasy = X.copy(), y.copy()
    batch: list[np.ndarray] = []
    for _ in range(size):
        ei = _expected_improvement(gp, pool, float(y_fantasy.min()))
        choice = pool[int(np.argmax(ei))]
        batch.append(choice)
        # Kriging believer: add the GP's own prediction as a fake observation.
        X_fantasy = np.vstack([X_fantasy, choice])
        y_fantasy = np.append(y_fantasy, gp.predict(choice[None, :])[0])
        gp = _fit_gp(X_fantasy, y_fantasy, rng, kernel=kernel)
    return np.array(batch)


_ALGORITHMS = {
    "random_search": random_search,
    "bayesian_optimization": bayesian_optimization,
}


def calibrate(
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Return the best parameters found by the method named in ALGORITHM_NAME.

    Dimensions, bounds, defaults, budget, and seed are read at run time from
    the organizer-supplied arguments; nothing is hard-coded.
    """

    if budget < 1:
        raise ValueError("budget must be at least one")
    if ALGORITHM_NAME not in _ALGORITHMS:
        raise ValueError(
            f"Unknown ALGORITHM_NAME {ALGORITHM_NAME!r}; choose from {sorted(_ALGORITHMS)}"
        )

    lower = np.array([float(p.lower) for p in parameter_space])
    upper = np.array([float(p.upper) for p in parameter_space])
    start = np.array([float(p.default) for p in parameter_space])

    if lower.size == 0:
        # No active parameters: one evaluation of the organizer defaults.
        objective(())
        return {}

    history = _History(objective, lower, upper, budget)
    _ALGORITHMS[ALGORITHM_NAME](history, start, np.random.default_rng(seed))
    best_vector, best_loss = history.best()

    print(
        f"{ALGORITHM_NAME}: best WAPE {best_loss:.2f}% after "
        f"{history.batches} batches / {len(history.y)} candidates"
    )

    # Clip defensively so the returned values are always in bounds.
    best_vector = np.clip(best_vector, lower, upper)
    return {
        p.name: float(value)
        for p, value in zip(parameter_space, best_vector, strict=True)
    }
