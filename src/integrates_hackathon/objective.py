"""Bounded and cached optimization objective."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Sequence

from .heating_schedule import profile_pair
from .metrics import CalibrationMetrics, compare_monthly
from .models import BuildingCase, MonthlyConsumption, finite_real
from .simulation import Simulator, resolved_operational_parameters


# Parameters that affect pyBuildingEnergy only through the whole-hour heating
# profiles built by ``heating_schedule.profile_pair``.
SCHEDULE_PARAMETERS = frozenset(
    {
        "weekday_heating_hours",
        "non_workday_heating_hours",
        "weekday_start_hour",
        "reduced_area_schedule_fraction",
    }
)


@dataclass(frozen=True)
class ObjectiveEvaluation:
    loss: float
    parameters: Mapping[str, float]
    metrics: CalibrationMetrics
    simulated: tuple[MonthlyConsumption, ...]


class CalibrationObjective:
    """Callable loss with deterministic clipping and memoization."""

    def __init__(
        self,
        case: BuildingCase,
        simulator: Simulator,
        *,
        max_evaluations: int | None = None,
    ) -> None:
        self.case = case
        self.simulator = simulator
        self.max_evaluations = max_evaluations
        self._cache: dict[tuple[float, ...], ObjectiveEvaluation] = {}
        self._evaluation_trace: list[ObjectiveEvaluation] = []
        self.evaluation_count = 0
        self._lock = RLock()

    @property
    def remaining_budget(self) -> int | None:
        if self.max_evaluations is None:
            return None
        return max(0, self.max_evaluations - self.evaluation_count)

    @property
    def max_batch_size(self) -> int:
        return int(getattr(self.simulator, "max_workers", 1))

    @property
    def evaluation_trace(self) -> tuple[ObjectiveEvaluation, ...]:
        """Unique, budget-charged evaluations in execution order."""

        with self._lock:
            return tuple(self._evaluation_trace)

    def candidate_key(self, parameters: Mapping[str, float]) -> tuple:
        """Identify candidates that produce the same simulation.

        The schedule parameters reach the simulator only through whole-hour
        heating profiles, so they are compared by those profiles. For example,
        7.6 and 8.4 weekday hours both mean 8 hours and are charged once.
        """

        others = tuple(
            round(float(parameters[name]), 12)
            for name in self.case.parameter_names
            if name not in SCHEDULE_PARAMETERS
        )
        resolved = resolved_operational_parameters(self.case, parameters)
        profiles = tuple(
            tuple(values)
            for reduced_regime in (False, True)
            for values in profile_pair(resolved, reduced_regime=reduced_regime).values()
        )
        return (others, profiles)

    def __call__(self, vector: Sequence[float]) -> float:
        return self.evaluate_vector(vector).loss

    def evaluate_vector(self, vector: Sequence[float]) -> ObjectiveEvaluation:
        parameters = self.case.parameters_from_vector(tuple(vector))
        return self.evaluate(parameters)

    def evaluate(self, parameters: Mapping[str, float]) -> ObjectiveEvaluation:
        return self.evaluate_many_parameters((parameters,))[0]

    def evaluate_many(self, vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Evaluate candidate vectors as one organizer-controlled batch."""

        parameters = tuple(
            self.case.parameters_from_vector(tuple(vector)) for vector in vectors
        )
        return tuple(item.loss for item in self.evaluate_many_parameters(parameters))

    def evaluate_many_parameters(
        self, parameter_sets: Sequence[Mapping[str, float]]
    ) -> tuple[ObjectiveEvaluation, ...]:
        """Validate, deduplicate, reserve budget, simulate, and restore order."""

        with self._lock:
            prepared: list[tuple[dict[str, float], tuple[float, ...]]] = []
            for parameters in parameter_sets:
                missing = set(self.case.parameter_names) - set(parameters)
                if missing:
                    raise ValueError(f"Missing parameters: {sorted(missing)}")
                clipped = {
                    spec.name: spec.clip(
                        finite_real(parameters[spec.name], f"Candidate value for {spec.name!r}")
                    )
                    for spec in self.case.parameters
                }
                prepared.append((clipped, self.candidate_key(clipped)))

            unseen: list[tuple[dict[str, float], tuple[float, ...]]] = []
            batch_keys: set[tuple[float, ...]] = set()
            for clipped, key in prepared:
                if key not in self._cache and key not in batch_keys:
                    unseen.append((clipped, key))
                    batch_keys.add(key)

            if (
                self.max_evaluations is not None
                and self.evaluation_count + len(unseen) > self.max_evaluations
            ):
                raise RuntimeError(
                    f"Simulation evaluation budget exhausted: requested {len(unseen)} "
                    f"new candidates with {self.remaining_budget} remaining."
                )

            if unseen:
                start = self.evaluation_count + 1
                total = self.max_evaluations or self.evaluation_count + len(unseen)
                if hasattr(self.simulator, "set_calibration_batch_progress"):
                    self.simulator.set_calibration_batch_progress(
                        tuple(range(start, start + len(unseen))), total
                    )
                simulated_batch = (
                    self.simulator.simulate_many(
                        self.case, tuple(clipped for clipped, _key in unseen)
                    )
                    if hasattr(self.simulator, "simulate_many")
                    else tuple(
                        self.simulator.simulate(self.case, clipped)
                        for clipped, _key in unseen
                    )
                )
                for (clipped, key), simulated in zip(
                    unseen, simulated_batch, strict=True
                ):
                    metrics = compare_monthly(self.case.measurements, simulated)
                    evaluation = ObjectiveEvaluation(
                        loss=metrics.weighted_absolute_percentage_error,
                        parameters=clipped,
                        metrics=metrics,
                        simulated=simulated,
                    )
                    self._cache[key] = evaluation
                    self._evaluation_trace.append(evaluation)
                    self.evaluation_count += 1

            return tuple(self._cache[key] for _clipped, key in prepared)


class ParticipantObjective:
    """The only objective interface handed to ``calibrate(...)``.

    It exposes evaluation and read-only budget information, but not the
    organizer objective, simulator, building case, measurements, or counters.
    This is a guard against accidental or casual misuse; official verification
    additionally counts simulations independently and reviews submitted code.
    """

    __slots__ = ("_evaluate", "_evaluate_many", "_remaining", "_max_batch_size")

    def __init__(self, objective: CalibrationObjective) -> None:
        set_attribute = object.__setattr__
        set_attribute(self, "_evaluate", lambda vector: objective(vector))
        set_attribute(
            self, "_evaluate_many", lambda vectors: objective.evaluate_many(vectors)
        )
        set_attribute(self, "_remaining", lambda: objective.remaining_budget)
        set_attribute(self, "_max_batch_size", objective.max_batch_size)

    def __call__(self, vector: Sequence[float]) -> float:
        return self._evaluate(vector)

    def evaluate_many(self, vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
        return self._evaluate_many(vectors)

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def remaining_budget(self) -> int | None:
        return self._remaining()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("The participant objective is read-only")

    def __repr__(self) -> str:
        return f"ParticipantObjective(remaining_budget={self.remaining_budget})"

