"""Transparent monthly diagnostics, kept separate from future official scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from .models import Measurement, MonthlyConsumption


@dataclass(frozen=True)
class CalibrationMetrics:
    mean_absolute_percentage_error: float
    mean_signed_percentage_error: float
    root_mean_squared_error_kwh: float
    mean_absolute_error_kwh: float
    weighted_absolute_percentage_error: float
    cv_rmse_pct: float
    nmbe_pct: float
    compared_months: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compare_monthly(
    measured: tuple[Measurement, ...],
    simulated: tuple[MonthlyConsumption, ...],
) -> CalibrationMetrics:
    actual_by_period = {(row.year, row.month): float(row.consumption_kwh) for row in measured}
    simulated_by_period = {(row.year, row.month): float(row.consumption_kwh) for row in simulated}
    missing = sorted(set(actual_by_period) - set(simulated_by_period))
    if missing:
        raise ValueError(f"Simulation is missing measured periods: {missing}")

    percentage_errors: list[float] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    for period in sorted(actual_by_period):
        actual = actual_by_period[period]
        predicted = simulated_by_period[period]
        error = predicted - actual
        absolute_errors.append(abs(error))
        squared_errors.append(error * error)
        if actual > 0:
            percentage_errors.append(100.0 * error / actual)

    if not percentage_errors:
        raise ValueError("At least one measured month must have consumption above zero.")
    count = len(absolute_errors)
    total_actual = sum(actual_by_period.values())
    mean_actual = total_actual / count
    if total_actual <= 0 or mean_actual <= 0:
        raise ValueError("Total and mean measured consumption must be above zero.")
    signed_error = sum(
        simulated_by_period[period] - actual_by_period[period]
        for period in actual_by_period
    )
    return CalibrationMetrics(
        mean_absolute_percentage_error=sum(abs(value) for value in percentage_errors)
        / len(percentage_errors),
        mean_signed_percentage_error=sum(percentage_errors) / len(percentage_errors),
        root_mean_squared_error_kwh=sqrt(sum(squared_errors) / count),
        mean_absolute_error_kwh=sum(absolute_errors) / count,
        weighted_absolute_percentage_error=100.0 * sum(absolute_errors) / total_actual,
        cv_rmse_pct=100.0 * sqrt(sum(squared_errors) / count) / mean_actual,
        nmbe_pct=100.0 * signed_error / total_actual,
        compared_months=count,
    )

