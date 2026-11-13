"""Participant submission: batched random search.

Teams replace the search strategy in ``calibrate`` while preserving the three
public names below. The notebook independently reruns and scores the returned
parameters, so submissions must return only the best parameter dictionary.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


TEAM_NAME = "Calibra"
ALGORITHM_NAME = "random_search"

# Stop if this many consecutive batches add no new (budget-charged) candidates,
# e.g. when all samples collapse onto already-evaluated whole-hour schedules.
_MAX_STALLED_BATCHES = 5


def random_search(
    objective,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Minimize ``objective`` by uniform random sampling within the bounds.

    The start vector (the notebook's initial values) is evaluated first, so
    the result is never worse than the starting point. Candidates are sent in
    batches of ``objective.max_batch_size`` to keep the worker pool busy.
    Returns the best vector, its loss (WAPE %), and the number of batches run.
    """

    best_loss = float(objective.evaluate_many([tuple(start)])[0])
    best_vector = start.copy()
    batches = 1
    submitted = 1
    stalled = 0

    while stalled < _MAX_STALLED_BATCHES:
        remaining = objective.remaining_budget
        if remaining is None:  # Unlimited objective: use our own (upper-bound) count.
            remaining = budget - submitted
        if remaining <= 0:
            break

        batch_size = max(1, min(objective.max_batch_size, remaining))
        candidates = rng.uniform(lower, upper, size=(batch_size, lower.size))
        losses = objective.evaluate_many([tuple(map(float, row)) for row in candidates])
        batches += 1
        submitted += batch_size

        if objective.remaining_budget is not None:
            stalled = stalled + 1 if objective.remaining_budget == remaining else 0

        index = int(np.argmin(losses))
        if losses[index] < best_loss:
            best_loss = float(losses[index])
            best_vector = candidates[index].copy()

    return best_vector, best_loss, batches


def calibrate(
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Return the best parameters found by batched random search.

    Dimensions, bounds, defaults, budget, and seed are read at run time from
    the organizer-supplied arguments; nothing is hard-coded.
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

    if ALGORITHM_NAME == "random_search":
        rng = np.random.default_rng(seed)
        best_vector, best_loss, batches = random_search(
            objective, lower, upper, start, budget, rng
        )

    print(f"Random search: best WAPE {best_loss:.2f}% after {batches} batches")

    # Clip defensively so the returned values are always in bounds.
    best_vector = np.clip(best_vector, lower, upper)
    return {
        p.name: float(value)
        for p, value in zip(parameter_space, best_vector, strict=True)
    }
