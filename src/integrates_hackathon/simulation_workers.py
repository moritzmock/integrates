"""Importable process workers for the parallel calibration workflow.

Workers never write weather files or shared output.  The parent process creates
the shortened EPWs once, then passes read-only paths and immutable tasks here.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from .bui import build_bui
from .models import BuildingCase, MonthlyConsumption
from .epw_io import read_epw_table
from .pbe_compat import pbe_profile_year
from .simulation import _hourly_result, resolved_operational_parameters


@dataclass(frozen=True)
class ParallelYearTask:
    """One candidate/building/year calculation executed in a child process."""

    case: BuildingCase
    parameters: Mapping[str, float]
    year: int
    window_epw_path: Path
    window_hours: int
    warmup_hours: int
    score_months: tuple[int, ...]


@contextlib.contextmanager
def _quiet_solver():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


def simulate_year_worker(task: ParallelYearTask) -> tuple[MonthlyConsumption, ...]:
    """Simulate both thermal regimes for one candidate and one weather year."""

    try:
        from pybuildingenergy.source import utils as pbe_utils
        from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
        from pybuildingenergy.source.utils import ISO52016
    except ImportError as error:
        raise RuntimeError(
            "Parallel workers require the project environment from environment.yml"
        ) from error

    # All anonymized buildings use Bolzano weather and the Italian holiday
    # calendar. Do not call an external reverse-geocoding service from local
    # worker processes merely to rediscover that fixed organizer input.
    pbe_utils.get_country_code_from_latlon = lambda _latitude, _longitude: "IT"

    parameters = resolved_operational_parameters(task.case, task.parameters)

    def validated(reduced: bool):
        bui, issues = sanitize_and_validate_BUI(
            build_bui(task.case, parameters, reduced_regime=reduced), fix=True
        )
        errors = [item for item in issues if item.get("level") == "ERROR"]
        if errors:
            raise ValueError(f"Invalid BUI for {task.case.building_id}: {errors}")
        return bui

    def run(bui):
        with pbe_profile_year(int(task.year)), _quiet_solver():
            result = ISO52016().Temperature_and_Energy_needs_calculation(
                bui,
                weather_source="epw",
                path_weather_file=str(task.window_epw_path),
            )
        hourly = _hourly_result(result)
        score_hours = int(task.window_hours) - int(task.warmup_hours)
        if len(hourly) < score_hours:
            raise ValueError(
                f"Solver returned {len(hourly)} rows; expected at least "
                f"{score_hours} scored hours after January warm-up"
            )
        return hourly.iloc[-score_hours:].reset_index(drop=True)

    normal = run(validated(False))
    reduced = run(validated(True))
    normal_q = pd.to_numeric(normal["Q_H"], errors="coerce").fillna(0).to_numpy()
    reduced_q = pd.to_numeric(reduced["Q_H"], errors="coerce").fillna(0).to_numpy()
    normal_fraction = float(parameters["normally_heated_area_fraction"])
    efficiency = float(task.case.fixed_inputs["heating_system_efficiency"])
    # Q_H already reflects active setpoint versus maintained setback at every
    # hour. Count both. Applying the old permission mask here discarded real
    # setback energy after it had affected the thermal state.
    delivered = (
        normal_fraction * normal_q
        + (1.0 - normal_fraction) * reduced_q
    ) / 1000.0 / efficiency

    _header, epw = read_epw_table(task.window_epw_path)
    months = epw["Month"].astype(int).to_numpy()[-len(delivered) :]
    return tuple(
        MonthlyConsumption(
            int(task.year), int(month), float(delivered[months == month].sum())
        )
        for month in task.score_months
    )
