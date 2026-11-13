"""Transparent 0-100 example scoring for challenge discussions and demos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean

from .metrics import CalibrationMetrics, compare_monthly
from .models import Measurement, MonthlyConsumption


@dataclass(frozen=True)
class BuildingScore:
    building_id: str
    baseline_score: float
    calibrated_score: float
    improvement_points: float
    baseline_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics

    def as_dict(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "building_id": self.building_id,
            "baseline_score": self.baseline_score,
            "calibrated_score": self.calibrated_score,
            "improvement_points": self.improvement_points,
        }
        row.update({f"baseline_{key}": value for key, value in asdict(self.baseline_metrics).items()})
        row.update({f"calibrated_{key}": value for key, value in asdict(self.calibrated_metrics).items()})
        return row


def accuracy_score_from_wape(wape_pct: float) -> float:
    """Return 100 for exact agreement and floor the score at zero."""

    return max(0.0, 100.0 - float(wape_pct))


def score_building(
    building_id: str,
    actual: tuple[Measurement, ...],
    baseline: tuple[MonthlyConsumption, ...],
    calibrated: tuple[MonthlyConsumption, ...],
) -> BuildingScore:
    baseline_metrics = compare_monthly(actual, baseline)
    calibrated_metrics = compare_monthly(actual, calibrated)
    baseline_score = accuracy_score_from_wape(
        baseline_metrics.weighted_absolute_percentage_error
    )
    calibrated_score = accuracy_score_from_wape(
        calibrated_metrics.weighted_absolute_percentage_error
    )
    return BuildingScore(
        building_id=building_id,
        baseline_score=baseline_score,
        calibrated_score=calibrated_score,
        improvement_points=calibrated_score - baseline_score,
        baseline_metrics=baseline_metrics,
        calibrated_metrics=calibrated_metrics,
    )


def overall_score(scores: list[BuildingScore] | tuple[BuildingScore, ...]) -> float:
    """Weight buildings equally so large buildings do not dominate."""

    if not scores:
        raise ValueError("At least one building score is required.")
    return fmean(score.calibrated_score for score in scores)

