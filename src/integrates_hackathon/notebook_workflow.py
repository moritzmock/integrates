"""Notebook-facing orchestration, reporting, visualization, and export.

The challenge notebook intentionally contains only calls into this module.
Organizer code owns simulation, validation, metrics, scoring, and exported
schemas; participant code owns only the search performed in ``calibration.py``.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, replace
from math import cos, pi
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from .baseline_cache import load_precomputed_baseline
from .building_inputs import BUILDING_INPUTS_BY_ID
from .challenge_inputs import BUILDINGS, MAX_CALIBRATABLE_PARAMETERS, parameter_specs
from .data_io import load_building_parameter_frame, parse_geometry_rings
from .energy_consumption import normalized_rows
from .heating_schedule import YEAR_SPECIFIC_EXCEPTIONS
from .models import BuildingCase, MonthlyConsumption
from .objective import CalibrationObjective, ObjectiveEvaluation, ParticipantObjective
from .scoring import (
    BuildingScore,
    accuracy_score_from_wape,
    overall_score,
    score_building,
)
from .simulation import AnalyticDemoSimulator, PyBuildingEnergySimulator, Simulator
from .submission_api import Submission, load_submission, repository_root, validate_parameter_result
from .weather_inputs import WEATHER_SUMMARIES


@dataclass(frozen=True)
class ChallengeContext:
    cases: tuple[BuildingCase, ...]
    simulator: Simulator
    backend: str
    evaluation_budget: int
    seed: int
    output_dir: Path
    calibration_years: tuple[int, ...]
    score_months: tuple[int, ...]
    warmup_month: int
    parallel_workers: int


@dataclass(frozen=True)
class CalibrationCaseRun:
    building_id: str
    parameters: Mapping[str, float]
    simulated: tuple[MonthlyConsumption, ...]
    evaluations: int
    evaluation_trace: tuple[ObjectiveEvaluation, ...]
    runtime_seconds: float
    status: str = "ok"
    # 1-based trace position of the candidate equivalent to the returned
    # parameters, or None if the returned parameters were never evaluated.
    final_evaluation_number: int | None = None


@dataclass(frozen=True)
class SubmissionRun:
    submission: Submission
    cases: Mapping[str, CalibrationCaseRun]


@dataclass(frozen=True)
class ScoreReport:
    submission: Submission
    building_scores: tuple[BuildingScore, ...]
    overall_score: float
    summary_rows: tuple[dict[str, Any], ...]
    monthly_rows: tuple[dict[str, Any], ...]


def initialize_challenge(
    *,
    backend: str = "pbe",
    evaluation_budget: int = 48,
    seed: int = 2026,
    output_dir: str | Path | None = None,
    building_ids: tuple[str, ...] = ("building_01", "building_02", "building_03"),
    calibration_years: tuple[int, ...] = (2018, 2019),
    score_months: tuple[int, ...] = (2, 3),
    warmup_month: int = 1,
    parallel_workers: int = 12,
) -> ChallengeContext:
    """Validate organizer data and return the shared notebook context."""

    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be at least one")
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be at least one")
    if not calibration_years:
        raise ValueError("calibration_years must contain at least one year")
    if tuple(score_months) != (2, 3) or int(warmup_month) != 1:
        raise ValueError("This challenge window uses January warm-up and February/March scoring")

    selected_ids = tuple(building_ids)
    source_by_id = {case.building_id: case for case in BUILDINGS}
    missing = set(selected_ids) - set(source_by_id)
    if missing:
        raise KeyError(f"Unknown challenge buildings: {sorted(missing)}")
    selected_cases = []
    for building_id in selected_ids:
        source = source_by_id[building_id]
        measurements = tuple(
            row for row in source.measurements
            if row.year in set(calibration_years) and row.month in set(score_months)
        )
        found_years = {row.year for row in measurements}
        absent = set(calibration_years) - found_years
        if absent:
            raise ValueError(f"{building_id} has no February/March bills for {sorted(absent)}")
        selected_cases.append(
            replace(
                source,
                measurements=measurements,
                weather_files={year: source.weather_files[year] for year in calibration_years},
                metadata={
                    **source.metadata,
                    "warmup_month": int(warmup_month),
                    "score_months": tuple(score_months),
                },
            )
        )
    cases = tuple(selected_cases)
    resolved_output = (
        Path(output_dir) if output_dir else repository_root() / "outputs"
    )

    if backend == "demo":
        simulator: Simulator = AnalyticDemoSimulator()
    elif backend == "pbe":
        from .window_simulation import ParallelWindowedPyBuildingEnergySimulator
        simulator = ParallelWindowedPyBuildingEnergySimulator(
            output_dir=resolved_output,
            warmup_month=warmup_month,
            score_months=score_months,
            max_workers=parallel_workers,
        )
    else:
        raise ValueError("backend must be 'demo' or 'pbe'")

    for case in cases:
        periods = {(row.year, row.month) for row in case.measurements}
        if len(periods) != len(case.measurements):
            raise ValueError(f"Duplicate measurement period in {case.building_id}")
        weather_years = {
            row.year for row in WEATHER_SUMMARIES if row.building_id == case.building_id
        }
        measured_years = {year for year, _ in periods}
        if not measured_years.issubset(weather_years):
            raise ValueError(f"Missing actual-year weather for {case.building_id}")

    return ChallengeContext(
        cases=cases,
        simulator=simulator,
        backend=backend,
        evaluation_budget=evaluation_budget,
        seed=seed,
        output_dir=resolved_output,
        calibration_years=tuple(int(year) for year in calibration_years),
        score_months=tuple(int(month) for month in score_months),
        warmup_month=int(warmup_month),
        parallel_workers=int(getattr(simulator, "max_workers", parallel_workers)),
    )


def configure_parameter_contract(
    context: ChallengeContext,
    calibratable_params_init: Mapping[str, float | None],
) -> ChallengeContext:
    """Validate the notebook dictionary and activate its numeric entries.

    Every building receives the same parameter names, bounds, and starting
    values. The optimizer may return a different scalar value per building,
    but no parameter can vary by month or year.
    """

    specs = parameter_specs(dict(calibratable_params_init), announce=True)
    assert 0 <= len(specs) <= MAX_CALIBRATABLE_PARAMETERS
    configured_cases = tuple(
        replace(
            case,
            parameters=specs,
            metadata={
                **case.metadata,
                "active_parameter_names": tuple(spec.name for spec in specs),
            },
        )
        for case in context.cases
    )
    print(
        "Shared active-parameter contract applied to every building and every scored "
        "month/year; month-specific and year-specific parameter keys are not accepted."
    )
    return replace(context, cases=configured_cases)


def show_challenge_contract(context: ChallengeContext) -> None:
    pd, _, display, Markdown = _presentation_dependencies()
    display(
        Markdown(
            f"**Backend:** `{context.backend}`  \n"
            f"**Simulation-evaluation budget:** `{context.evaluation_budget}` per building  \n"
            f"**Seed:** `{context.seed}`  \n"
            f"**Buildings:** `{', '.join(case.building_id for case in context.cases)}`  \n"
            f"**Calibration years:** `{list(context.calibration_years)}`  \n"
            "**Window:** January warm-up; February and March scored (no annualization)  \n"
            f"**Local simulation workers:** `{context.parallel_workers}`  \n"
            "**Example score:** `max(0, 100 - WAPE%)`, averaged equally across buildings.  \n"
            "**Budget enforcement:** the organizer objective refuses any new candidate "
            "after the stated per-building budget."
        )
    )
    display(
        pd.DataFrame(
            [
                {"Required name": "TEAM_NAME", "Type": "str", "Purpose": "Name shown in standardized results"},
                {"Required name": "ALGORITHM_NAME", "Type": "str", "Purpose": "Short algorithm description"},
                {"Required name": "calibrate", "Type": "callable", "Purpose": "Returns {parameter_name: best_value}"},
            ]
        )
    )


def show_input_data(context: ChallengeContext) -> dict[str, Any]:
    pd, plt, display, Markdown = _presentation_dependencies()
    selected_ids = {case.building_id for case in context.cases}
    source_buildings = load_building_parameter_frame()
    source_buildings = source_buildings.loc[
        source_buildings["building_id"].isin(selected_ids)
    ].copy()
    buildings = pd.DataFrame(_building_rows(context.cases))
    parameters = pd.DataFrame(_parameter_rows(context.cases))
    selected_periods = {
        (case.building_id, row.year, row.month)
        for case in context.cases for row in case.measurements
    }
    energy = pd.DataFrame(
        row for row in normalized_rows()
        if (row["building_id"], row["year"], row["month"]) in selected_periods
    )
    weather = pd.DataFrame(
        row.as_dict() for row in WEATHER_SUMMARIES
        if row.building_id in selected_ids and row.year in set(context.calibration_years)
    )
    weather_by_year = (
        weather.groupby(["year", "month"], as_index=False)
        .agg(
            mean_outdoor_temperature_c=("mean_outdoor_temperature_c", "first"),
            heating_degree_days_18c=("heating_degree_days_18c", "first"),
        )
    )

    display(Markdown("### Parsed anonymized building inputs"))
    display(buildings)
    display(Markdown("### EUBUCCO footprints at the shared anonymous centroid"))
    _plot_footprints(source_buildings, plt)
    display(Markdown("### Calibratable parameters"))
    display(parameters)
    legal_exceptions = pd.DataFrame(
        [
            {
                "year": year,
                "start_date": start.isoformat(),
                "end_date_inclusive": end.isoformat(),
                "maximum_hours_per_day": 7,
                "calibratable": False,
            }
            for year, periods in YEAR_SPECIFIC_EXCEPTIONS.items()
            for start, end in periods
        ]
    )
    display(
        Markdown(
            "### Fixed legal heating permissions\n"
            "Ordinary Bolzano operation is limited to 15 October–15 April, "
            "05:00–23:00, at no more than 14 h/day. The dated exceptions below "
            "are fixed at 7 h/day; algorithms cannot create other out-of-season operation."
        )
    )
    display(legal_exceptions)

    selected_energy = (
        energy.groupby(["building_id", "year"], as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "sum"),
            consumption_kwh_m2_gfa=("consumption_kwh_m2_gfa", "sum"),
            source_unit=("source_unit", "first"),
            end_use_scope=("end_use_scope", "first"),
        )
    )
    display(
        Markdown(
            "### Selected February/March heating consumption (converted to kWh)\n"
            "Combined heating and DHW bills minus each year's June–August "
            "average, which is taken as the monthly DHW load."
        )
    )
    display(selected_energy)

    _plot_energy_inputs(energy, plt)

    display(
        Markdown(
            "### Weather summary\n"
            "Monthly temperature and heating-degree-day summaries calculated from "
            "the organizer-provided AMY EPW files for every reported consumption year."
        )
    )
    _, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for year, group in weather_by_year.groupby("year"):
        label = str(year)
        axes[0].plot(group["month"], group["mean_outdoor_temperature_c"], marker="o", label=label)
        axes[1].plot(group["month"], group["heating_degree_days_18c"], marker="o", label=label)
    axes[0].set(title="Monthly mean outdoor temperature", ylabel="°C")
    axes[1].set(title="Heating degree days (base 18°C)", ylabel="HDD", xlabel="Month")
    for axis in axes:
        axis.set_xticks(range(1, 13))
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    plt.show()
    return {
        "buildings": buildings,
        "building_source": source_buildings,
        "parameters": parameters,
        "legal_exceptions": legal_exceptions,
        "energy": energy,
        "selected_energy": selected_energy,
        "weather": weather_by_year,
    }


def run_baseline(context: ChallengeContext) -> dict[str, tuple[MonthlyConsumption, ...]]:
    """Run the expensive baseline directly (organizer use only)."""

    baseline: dict[str, tuple[MonthlyConsumption, ...]] = {}
    for case in context.cases:
        simulated = context.simulator.simulate(case, case.defaults)
        expected_periods = {(row.year, row.month) for row in case.measurements}
        simulated_periods = {(row.year, row.month) for row in simulated}
        if simulated_periods != expected_periods:
            raise ValueError(
                f"Baseline period mismatch for {case.building_id}: "
                f"expected {len(expected_periods)}, received {len(simulated_periods)}"
            )
        baseline[case.building_id] = simulated
    return baseline


def load_baseline(context: ChallengeContext) -> dict[str, tuple[MonthlyConsumption, ...]]:
    """Load the validated organizer baseline without running simulations."""

    if context.backend != "pbe":
        return run_baseline(context)
    return load_precomputed_baseline(context.cases)


def show_baseline_results(
    context: ChallengeContext,
    baseline: Mapping[str, tuple[MonthlyConsumption, ...]],
) -> Any:
    pd, plt, display, Markdown = _presentation_dependencies()
    frame = pd.DataFrame(_monthly_rows(context, baseline, calibrated=None))
    summary = (
        frame.groupby("building_id", as_index=False)
        .agg(
            measured_months=("actual_kwh", "size"),
            measured_kwh=("actual_kwh", "sum"),
            baseline_kwh=("baseline_kwh", "sum"),
        )
    )
    display(Markdown("### Organizer-provided baseline results"))
    display(summary)
    display(Markdown("### Baseline monthly comparison"))
    display(frame)
    _plot_monthly_comparison(frame, plt, include_calibrated=False)
    return frame


def run_calibration(
    context: ChallengeContext,
    *,
    submission_path: str | Path | None = None,
) -> SubmissionRun:
    submission = load_submission(submission_path)
    runs: dict[str, CalibrationCaseRun] = {}
    for case in context.cases:
        counting_simulator = _CountingSimulator(context.simulator)
        organizer_objective = CalibrationObjective(
            case,
            counting_simulator,
            max_evaluations=context.evaluation_budget,
        )
        seed = _building_seed(context.seed, case.building_id)
        started = perf_counter()
        raw_parameters = submission.calibrate(
            ParticipantObjective(organizer_objective),
            case.parameters,
            context.evaluation_budget,
            seed,
        )
        runtime = perf_counter() - started
        _check_budget_integrity(
            case.building_id,
            context.evaluation_budget,
            organizer_objective,
            counting_simulator,
        )
        parameters = validate_parameter_result(case, raw_parameters)
        # The independent rerun uses the uncounted organizer simulator.
        simulated = context.simulator.simulate(case, parameters)
        trace = organizer_objective.evaluation_trace
        final_key = organizer_objective.candidate_key(parameters)
        final_evaluation_number = next(
            (
                number
                for number, evaluation in enumerate(trace, start=1)
                if organizer_objective.candidate_key(evaluation.parameters) == final_key
            ),
            None,
        )
        runs[case.building_id] = CalibrationCaseRun(
            building_id=case.building_id,
            parameters=parameters,
            simulated=simulated,
            evaluations=organizer_objective.evaluation_count,
            evaluation_trace=trace,
            runtime_seconds=runtime,
            final_evaluation_number=final_evaluation_number,
        )
    return SubmissionRun(submission=submission, cases=runs)


class _CountingSimulator:
    """Count every candidate simulated during one building's calibration.

    The count is kept separately from the objective's own budget bookkeeping,
    so tampering with one is detected by comparison with the other.
    """

    def __init__(self, simulator: Simulator) -> None:
        self._simulator = simulator
        self.simulated_candidates = 0

    @property
    def max_workers(self) -> int:
        return int(getattr(self._simulator, "max_workers", 1))

    def set_calibration_batch_progress(self, indices, total: int) -> None:
        if hasattr(self._simulator, "set_calibration_batch_progress"):
            self._simulator.set_calibration_batch_progress(indices, total)

    def simulate(self, case: BuildingCase, parameters: Mapping[str, float]):
        return self.simulate_many(case, (parameters,))[0]

    def simulate_many(self, case: BuildingCase, parameter_sets):
        parameter_sets = tuple(parameter_sets)
        self.simulated_candidates += len(parameter_sets)
        if hasattr(self._simulator, "simulate_many"):
            return self._simulator.simulate_many(case, parameter_sets)
        return tuple(self._simulator.simulate(case, values) for values in parameter_sets)


def _check_budget_integrity(
    building_id: str,
    budget: int,
    objective: CalibrationObjective,
    simulator: _CountingSimulator,
) -> None:
    counts = (
        objective.evaluation_count,
        len(objective.evaluation_trace),
        simulator.simulated_candidates,
    )
    if len(set(counts)) != 1 or counts[0] > budget:
        raise RuntimeError(
            f"Evaluation budget integrity check failed for {building_id}: "
            f"objective count={counts[0]}, trace length={counts[1]}, "
            f"simulated candidates={counts[2]}, budget={budget}. "
            "calibrate(...) must evaluate candidates only through the objective."
        )


def _building_seed(base_seed: int, building_id: str) -> int:
    """Return a stable per-building seed independent of selection order."""

    try:
        building_number = int(building_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Building ID must end in a numeric suffix: {building_id!r}"
        ) from exc
    if building_number < 1:
        raise ValueError(f"Building number must be positive: {building_id!r}")
    return int(base_seed) + building_number - 1


def show_calibration_results(context: ChallengeContext, run: SubmissionRun) -> Any:
    pd, _, display, Markdown = _presentation_dependencies()
    rows = []
    for case in context.cases:
        case_run = run.cases[case.building_id]
        for spec in case.parameters:
            rows.append(
                {
                    "team_name": run.submission.team_name,
                    "algorithm_name": run.submission.algorithm_name,
                    "building_id": case.building_id,
                    "parameter": spec.name,
                    "lower": spec.lower,
                    "baseline": spec.default,
                    "calibrated": case_run.parameters[spec.name],
                    "upper": spec.upper,
                    "unit": spec.unit,
                }
            )
    frame = pd.DataFrame(rows)
    display(Markdown(f"### Calibration returned by {run.submission.team_name}\nAlgorithm: **{run.submission.algorithm_name}**"))
    display(frame)
    return frame


def show_calibrated_comparison(
    context: ChallengeContext,
    baseline: Mapping[str, tuple[MonthlyConsumption, ...]],
    run: SubmissionRun,
) -> Any:
    """Show the standardized actual/baseline/calibrated monthly comparison."""

    pd, plt, display, Markdown = _presentation_dependencies()
    rows = _monthly_rows(
        context,
        baseline,
        calibrated={key: value.simulated for key, value in run.cases.items()},
        submission=run.submission,
    )
    frame = pd.DataFrame(rows)
    display(Markdown("### Actual, baseline, and independently rerun calibrated results"))
    display(frame)
    _plot_monthly_comparison(frame, plt, include_calibrated=True)
    return frame


def score_and_visualize(
    context: ChallengeContext,
    baseline: Mapping[str, tuple[MonthlyConsumption, ...]],
    run: SubmissionRun,
) -> ScoreReport:
    pd, plt, display, Markdown = _presentation_dependencies()
    scores = [
        score_building(
            case.building_id,
            case.measurements,
            baseline[case.building_id],
            run.cases[case.building_id].simulated,
        )
        for case in context.cases
    ]
    summary_rows = []
    for score in scores:
        case_run = run.cases[score.building_id]
        row = score.as_dict()
        row.update(
            {
                "team_name": run.submission.team_name,
                "algorithm_name": run.submission.algorithm_name,
                "evaluations": case_run.evaluations,
                "runtime_seconds": case_run.runtime_seconds,
                "status": case_run.status,
            }
        )
        summary_rows.append(row)
    monthly_rows = _monthly_rows(
        context,
        baseline,
        calibrated={key: value.simulated for key, value in run.cases.items()},
        submission=run.submission,
    )
    overall = overall_score(scores)
    report = ScoreReport(
        submission=run.submission,
        building_scores=tuple(scores),
        overall_score=overall,
        summary_rows=tuple(summary_rows),
        monthly_rows=tuple(monthly_rows),
    )

    display(
        Markdown(
            f"## Example overall score: **{overall:.2f} / 100**\n"
            "The example uses `100 - WAPE%` with a zero floor and weights each building equally. "
            "CVRMSE and NMBE are reported as diagnostics and possible tie-breakers."
        )
    )
    summary = pd.DataFrame(summary_rows)
    display(
        summary[
            [
                "team_name",
                "building_id",
                "baseline_score",
                "calibrated_score",
                "improvement_points",
                "baseline_weighted_absolute_percentage_error",
                "calibrated_weighted_absolute_percentage_error",
                "calibrated_cv_rmse_pct",
                "calibrated_nmbe_pct",
                "evaluations",
                "runtime_seconds",
            ]
        ]
    )
    monthly = pd.DataFrame(monthly_rows)
    _plot_score_bars(summary, plt)
    _plot_error_heatmap(monthly, plt)
    return report


def export_standardized_results(
    context: ChallengeContext,
    run: SubmissionRun,
    report: ScoreReport,
) -> dict[str, Path]:
    output_dir = context.output_dir / _slug(run.submission.team_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    monthly_path = output_dir / "monthly_results.csv"
    trace_path = output_dir / "evaluation_trace.csv"
    json_path = output_dir / "submission_result.json"
    _write_csv(summary_path, list(report.summary_rows))
    _write_csv(monthly_path, list(report.monthly_rows))
    trace_rows, trace_fieldnames = _evaluation_trace_rows(context, run)
    _write_csv(trace_path, trace_rows, fieldnames=trace_fieldnames)
    payload = {
        "schema_version": "1.2",
        "team_name": run.submission.team_name,
        "algorithm_name": run.submission.algorithm_name,
        "calibration_py_sha256": run.submission.source_sha256,
        "backend": context.backend,
        "evaluation_budget_per_building": context.evaluation_budget,
        "seed": context.seed,
        "overall_score": report.overall_score,
        "building_results": list(report.summary_rows),
        "active_parameter_names": list(context.cases[0].parameter_names),
        "active_parameter_selection": {
            spec.name: float(spec.default)
            for spec in context.cases[0].parameters
        },
        "parameters": {building_id: dict(case_run.parameters) for building_id, case_run in run.cases.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "summary": summary_path,
        "monthly": monthly_path,
        "evaluation_trace": trace_path,
        "json": json_path,
    }


def _evaluation_trace_rows(
    context: ChallengeContext,
    run: SubmissionRun,
) -> tuple[list[dict[str, Any]], list[str]]:
    parameter_names = list(context.cases[0].parameter_names)
    fieldnames = [
        "team_name",
        "algorithm_name",
        "calibration_py_sha256",
        "building_id",
        "evaluation_number",
        "evaluation_budget",
        "loss_wape_pct",
        "score",
        "is_final_candidate",
        *parameter_names,
    ]
    rows: list[dict[str, Any]] = []
    for case in context.cases:
        case_run = run.cases[case.building_id]
        for evaluation_number, evaluation in enumerate(
            case_run.evaluation_trace, start=1
        ):
            # Equivalent candidates (for example 7.6 h and 8 h) count as final.
            is_final = evaluation_number == case_run.final_evaluation_number
            row: dict[str, Any] = {
                "team_name": run.submission.team_name,
                "algorithm_name": run.submission.algorithm_name,
                "calibration_py_sha256": run.submission.source_sha256,
                "building_id": case.building_id,
                "evaluation_number": evaluation_number,
                "evaluation_budget": context.evaluation_budget,
                "loss_wape_pct": evaluation.loss,
                "score": accuracy_score_from_wape(evaluation.loss),
                "is_final_candidate": is_final,
            }
            row.update(
                {
                    name: float(evaluation.parameters[name])
                    for name in parameter_names
                }
            )
            rows.append(row)
    return rows, fieldnames


def show_exported_results(paths: Mapping[str, Path]) -> None:
    _, _, display, Markdown = _presentation_dependencies()
    lines = [f"- **{name}:** `{path.resolve()}`" for name, path in paths.items()]
    display(Markdown("### Standardized result files\n" + "\n".join(lines)))


def _building_rows(cases: tuple[BuildingCase, ...]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        public = BUILDING_INPUTS_BY_ID[case.building_id]
        rows.append(
            {
                "building_id": public.building_id,
                "building_use": public.building_use,
                "construction_year": public.construction_year,
                "gross_floor_area_m2": public.gross_floor_area_m2,
                "conditioned_floor_area_m2": public.conditioned_floor_area_m2,
                "heated_volume_m3": public.heated_volume_m3,
                "floors_above_ground": public.floors_above_ground,
                "total_height_m": public.total_height_m,
                "footprint_area_m2": public.footprint_area_m2,
                "geometry_complete": public.geometry_complete,
                "source_status": public.source_status,
            }
        )
    return rows


def _plot_footprints(buildings, plt) -> None:
    count = len(buildings)
    figure, axes = plt.subplots(
        1,
        count,
        figsize=(3.2 * count, 3.6),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (_, row) in zip(axes[0], buildings.iterrows(), strict=True):
        points, holes = parse_geometry_rings(
            row["anonymized_geometry_positions_latlon"]
        )
        anchor_latitude = float(row["geometry_anchor_latitude"])
        anchor_longitude = float(row["geometry_anchor_longitude"])
        metres_per_degree_latitude = 2.0 * pi * 6_371_008.8 / 360.0
        metres_per_degree_longitude = metres_per_degree_latitude * cos(
            anchor_latitude * pi / 180.0
        )
        x = [
            (float(longitude) - anchor_longitude) * metres_per_degree_longitude
            for latitude, longitude in points
        ]
        y = [
            (float(latitude) - anchor_latitude) * metres_per_degree_latitude
            for latitude, longitude in points
        ]
        axis.fill(x, y, color="#4C78A8", alpha=0.28)
        axis.plot(x, y, color="#1F4E78", linewidth=1.8)
        for hole in holes:
            hole_x = [
                (longitude - anchor_longitude) * metres_per_degree_longitude
                for latitude, longitude in hole
            ]
            hole_y = [
                (latitude - anchor_latitude) * metres_per_degree_latitude
                for latitude, longitude in hole
            ]
            axis.fill(hole_x, hole_y, color="white", alpha=1.0, zorder=3)
            axis.plot(
                hole_x,
                hole_y,
                color="#C44E52",
                linewidth=1.5,
                linestyle="--",
                zorder=4,
            )
        axis.scatter([0], [0], color="#C44E52", s=18, label="Shared centroid")
        axis.annotate("N", xy=(0.92, 0.92), xytext=(0.92, 0.72),
                      xycoords="axes fraction", textcoords="axes fraction",
                      arrowprops={"arrowstyle": "->", "color": "#333333"},
                      ha="center", va="bottom")
        axis.set(
            title=str(row["building_id"]),
            xlabel="East-west offset (m)",
            ylabel="North-south offset (m)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    plt.show()


def _plot_energy_inputs(energy, plt) -> None:
    building_ids = sorted(energy["building_id"].unique())
    figure, axes = plt.subplots(
        len(building_ids),
        2,
        figsize=(13, 3.2 * len(building_ids)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, building_id in enumerate(building_ids):
        building = energy.loc[energy["building_id"] == building_id]
        for year, group in building.groupby("year"):
            label = str(int(year))
            axes[row_index, 0].plot(
                group["month"], group["consumption_kwh"], marker="o", label=label
            )
            axes[row_index, 1].plot(
                group["month"],
                group["consumption_kwh_m2_gfa"],
                marker="o",
                label=label,
            )
        axes[row_index, 0].set(
            title=f"{building_id}: measured monthly delivered energy",
            ylabel="kWh",
        )
        axes[row_index, 1].set(
            title=f"{building_id}: normalized measured energy",
            ylabel="kWh/m² GFA",
        )
        for axis in axes[row_index]:
            axis.set(xlabel="Month", xticks=range(1, 13))
            axis.grid(alpha=0.25)
            axis.legend(title="Year", ncol=2, fontsize=8)
    plt.show()


def _parameter_rows(cases: tuple[BuildingCase, ...]) -> list[dict[str, Any]]:
    return [
        {
            "building_id": case.building_id,
            "parameter": spec.name,
            "lower": spec.lower,
            "baseline": spec.default,
            "upper": spec.upper,
            "unit": spec.unit,
            "description": spec.description,
        }
        for case in cases
        for spec in case.parameters
    ]


def _monthly_rows(
    context: ChallengeContext,
    baseline: Mapping[str, tuple[MonthlyConsumption, ...]],
    *,
    calibrated: Mapping[str, tuple[MonthlyConsumption, ...]] | None,
    submission: Submission | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in context.cases:
        actual = {(row.year, row.month): row.consumption_kwh for row in case.measurements}
        base = {(row.year, row.month): row.consumption_kwh for row in baseline[case.building_id]}
        post = (
            {(row.year, row.month): row.consumption_kwh for row in calibrated[case.building_id]}
            if calibrated is not None
            else {}
        )
        area = float(case.fixed_inputs["gross_floor_area_m2"])
        for period in sorted(actual):
            actual_kwh = float(actual[period])
            baseline_kwh = float(base[period])
            calibrated_kwh = float(post[period]) if period in post else None
            rows.append(
                {
                    "team_name": submission.team_name if submission else "",
                    "algorithm_name": submission.algorithm_name if submission else "",
                    "building_id": case.building_id,
                    "year": period[0],
                    "month": period[1],
                    "gross_floor_area_m2": area,
                    "actual_kwh": actual_kwh,
                    "baseline_kwh": baseline_kwh,
                    "calibrated_kwh": calibrated_kwh,
                    "actual_kwh_m2_gfa": actual_kwh / area,
                    "baseline_kwh_m2_gfa": baseline_kwh / area,
                    "calibrated_kwh_m2_gfa": calibrated_kwh / area if calibrated_kwh is not None else None,
                    "baseline_error_pct": 100.0 * (baseline_kwh - actual_kwh) / actual_kwh if actual_kwh else None,
                    "calibrated_error_pct": (
                        100.0 * (calibrated_kwh - actual_kwh) / actual_kwh
                        if actual_kwh and calibrated_kwh is not None
                        else None
                    ),
                }
            )
    return rows


def _plot_score_bars(summary, plt) -> None:
    labels = summary["building_id"].tolist()
    x = list(range(len(labels)))
    width = 0.36
    _, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar([value - width / 2 for value in x], summary["baseline_score"], width, label="Baseline")
    axis.bar([value + width / 2 for value in x], summary["calibrated_score"], width, label="Calibrated")
    axis.set(title="Example score before and after calibration", ylabel="Score (0-100)", ylim=(0, 100))
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    plt.show()


def _plot_monthly_comparison(frame, plt, *, include_calibrated: bool) -> None:
    for building_id, group in frame.groupby("building_id"):
        ordered = group.sort_values(["year", "month"]).copy()
        ordered["period"] = ordered["year"].astype(str) + "-" + ordered["month"].astype(str).str.zfill(2)
        _, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        axis.plot(ordered["period"], ordered["actual_kwh"], label="Actual", linewidth=2.2)
        axis.plot(ordered["period"], ordered["baseline_kwh"], label="Baseline", linewidth=1.8)
        if include_calibrated:
            axis.plot(ordered["period"], ordered["calibrated_kwh"], label="Calibrated", linewidth=1.8)
        axis.set(title=f"Monthly delivered energy - {building_id}", ylabel="kWh", xlabel="Month")
        step = max(1, len(ordered) // 12)
        ticks = list(range(0, len(ordered), step))
        axis.set_xticks(ticks, [ordered["period"].iloc[index] for index in ticks], rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend()
        plt.show()


def _plot_error_heatmap(frame, plt) -> None:
    pivot = frame.assign(
        period=frame["year"].astype(str) + "-" + frame["month"].astype(str).str.zfill(2),
        row_label=frame["building_id"],
    ).pivot(index="row_label", columns="period", values="calibrated_error_pct")
    figure, axis = plt.subplots(
        figsize=(max(10, 0.24 * len(pivot.columns)), 2.5 + 0.45 * len(pivot.index)),
        constrained_layout=True,
    )
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-50, vmax=50)
    axis.set(title="Calibrated monthly percentage error", xlabel="Month", ylabel="Building")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    step = max(1, len(pivot.columns) // 12)
    tick_positions = list(range(0, len(pivot.columns), step))
    axis.set_xticks(tick_positions, [pivot.columns[index] for index in tick_positions], rotation=45, ha="right")
    figure.colorbar(image, ax=axis, label="Error (%)")
    plt.show()


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot export empty table to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames if fieldnames is not None else list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "team"


def _presentation_dependencies():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Notebook presentation requires pandas and matplotlib. "
            "Install the project notebook extras."
        ) from error
    try:
        from IPython.display import Markdown, display
    except ImportError:
        # This keeps command-line smoke tests useful; Jupyter supplies the rich
        # display functions when the workflow runs in the challenge notebook.
        Markdown = str

        def display(value):
            print(value)

    return pd, plt, display, Markdown

