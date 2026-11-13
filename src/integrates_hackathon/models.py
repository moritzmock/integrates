"""Small, dependency-free data contracts shared by the challenge modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, Mapping


def finite_real(value: Any, label: str) -> float:
    """Return ``value`` as a float, rejecting booleans, text, NaN, and inf.

    NumPy scalars and 0-d arrays are accepted because optimizers return them.
    """

    if hasattr(value, "ndim") and getattr(value, "ndim") == 0 and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, not {type(value).__name__}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite, received {number}")
    return number


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded scalar that an optimization algorithm may change."""

    name: str
    lower: float
    upper: float
    default: float
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        values = (float(self.lower), float(self.upper), float(self.default))
        if not all(isfinite(value) for value in values):
            raise ValueError(f"Parameter {self.name!r} contains a non-finite value.")
        if self.lower >= self.upper:
            raise ValueError(f"Parameter {self.name!r} must have lower < upper.")
        if not self.lower <= self.default <= self.upper:
            raise ValueError(f"Default for {self.name!r} is outside its bounds.")

    def clip(self, value: float) -> float:
        return min(float(self.upper), max(float(self.lower), float(value)))


@dataclass(frozen=True, order=True)
class Measurement:
    year: int
    month: int
    consumption_kwh: float

    def __post_init__(self) -> None:
        if not 1 <= int(self.month) <= 12:
            raise ValueError("month must be between 1 and 12")
        if not isfinite(float(self.consumption_kwh)) or self.consumption_kwh < 0:
            raise ValueError("consumption_kwh must be finite and non-negative")


@dataclass(frozen=True, order=True)
class MonthlyConsumption:
    year: int
    month: int
    consumption_kwh: float


@dataclass(frozen=True)
class BuildingCase:
    """All organizer-owned inputs for one independently calibrated building."""

    building_id: str
    fixed_inputs: Mapping[str, Any]
    parameters: tuple[ParameterSpec, ...]
    measurements: tuple[Measurement, ...]
    weather_files: Mapping[int, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.building_id.strip():
            raise ValueError("building_id must not be empty")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names in {self.building_id!r}.")
        if len(names) > 12:
            raise ValueError("Each challenge case may expose at most 12 parameters.")
        periods = [(item.year, item.month) for item in self.measurements]
        if len(periods) != len(set(periods)):
            raise ValueError(f"Duplicate measured months in {self.building_id!r}.")
        if not periods:
            raise ValueError(f"No measurements supplied for {self.building_id!r}.")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def defaults(self) -> dict[str, float]:
        return {parameter.name: float(parameter.default) for parameter in self.parameters}

    def parameters_from_vector(self, vector: list[float] | tuple[float, ...]) -> dict[str, float]:
        if len(vector) != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} parameters, received {len(vector)}."
            )
        return {
            spec.name: spec.clip(finite_real(value, f"Candidate value for {spec.name!r}"))
            for spec, value in zip(self.parameters, vector, strict=True)
        }

