"""Starter participant submission.

Teams replace the search strategy in ``calibrate`` while preserving the three
public names below. The notebook independently reruns and scores the returned
parameters, so submissions must return only the best parameter dictionary.
"""

from __future__ import annotations

from typing import Callable, Sequence


TEAM_NAME = "Starter example"
ALGORITHM_NAME = "Baseline passthrough"


def calibrate(
    objective: Callable[[Sequence[float]], float],
    parameter_space,
    budget: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate and return the organizer-provided baseline unchanged.

    The organizer, not this file, defines the active parameters and limits.
    Read ``parameter.name``, ``.lower``, ``.upper``, and ``.default`` at run
    time; do not hard-code a dimension, bound, budget, building, month, or
    year. Participants should replace this passthrough with their own search
    strategy. The same returned scalar vector is independently rerun against
    every scored month/year for the current building.
    """

    if budget < 1:
        raise ValueError("budget must be at least one")
    del seed  # The starter is deterministic; participant methods may use it.
    baseline_vector = tuple(
        float(parameter.default) for parameter in parameter_space
    )
    objective(baseline_vector)

    return {
        parameter.name: float(value)
        for parameter, value in zip(parameter_space, baseline_vector, strict=True)
    }

