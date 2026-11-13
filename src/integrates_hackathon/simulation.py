"""Simulation backends for the INTEGRATES calibration challenge."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .bui import build_bui
from .models import BuildingCase, MonthlyConsumption


class Simulator(Protocol):
    def simulate(
        self, case: BuildingCase, parameters: Mapping[str, float]
    ) -> tuple[MonthlyConsumption, ...]: ...


def resolved_operational_parameters(
    case: BuildingCase, parameters: Mapping[str, float]
) -> dict[str, float]:
    """Combine organizer-fixed operation assumptions with tunable values."""

    resolved = {
        str(name): float(value)
        for name, value in case.fixed_inputs.get(
            "fixed_operational_parameters", {}
        ).items()
    }
    resolved.update({str(name): float(value) for name, value in parameters.items()})
    return resolved


class PyBuildingEnergySimulator:
    """Run ISO 52016 against organizer-provided EPW files.

    Heavy dependencies are imported only when this backend is selected, so the
    input validation, metrics, and demo optimizer remain lightweight.
    """

    def simulate(
        self, case: BuildingCase, parameters: Mapping[str, float]
    ) -> tuple[MonthlyConsumption, ...]:
        parameters = resolved_operational_parameters(case, parameters)
        if not case.weather_files:
            raise ValueError(
                f"{case.building_id!r} has no weather files. Add them in challenge_inputs.py."
            )
        try:
            import pandas as pd
            from pybuildingenergy.source import utils as pbe_utils
            from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
            from pybuildingenergy.source.utils import ISO52016
        except ImportError as error:
            raise RuntimeError(
                "The pbe backend requires the project environment from environment.yml."
            ) from error
        pbe_utils.get_country_code_from_latlon = lambda _latitude, _longitude: "IT"

        measured_years = {row.year for row in case.measurements}
        weather_years = {int(year) for year in case.weather_files}
        if weather_years != measured_years:
            raise ValueError(
                f"{case.building_id} AMY years {sorted(weather_years)} do not exactly "
                f"match consumption years {sorted(measured_years)}"
            )
        missing_weather = [
            str(path) for path in case.weather_files.values() if not Path(path).is_file()
        ]
        if missing_weather:
            raise ValueError(f"Missing AMY EPW files: {missing_weather}")

        normal_bui = _validated_bui(
            case, parameters, reduced_regime=False,
            validator=sanitize_and_validate_BUI,
        )
        reduced_bui = _validated_bui(
            case, parameters, reduced_regime=True,
            validator=sanitize_and_validate_BUI,
        )
        efficiency = float(case.fixed_inputs["heating_system_efficiency"])
        normal_fraction = float(parameters["normally_heated_area_fraction"])
        monthly: list[MonthlyConsumption] = []
        for year, epw_path in sorted(case.weather_files.items()):
            normal_hourly = _run_pbe_year(ISO52016, normal_bui, epw_path, year)
            reduced_hourly = _run_pbe_year(ISO52016, reduced_bui, epw_path, year)
            if len(normal_hourly) != len(reduced_hourly):
                raise ValueError("Normal and reduced-regime simulations returned different lengths.")

            # EPW files often contain 8760 rows even when their label is a leap
            # year. Reference calendars preserve the file's actual month lengths.
            reference_year = 2000 if len(normal_hourly) == 8784 else 2001
            index = pd.date_range(
                f"{reference_year}-01-01", periods=len(normal_hourly), freq="h"
            )
            normal_q = pd.to_numeric(normal_hourly["Q_H"], errors="coerce").fillna(0.0)
            reduced_q = pd.to_numeric(reduced_hourly["Q_H"], errors="coerce").fillna(0.0)
            # Option 1: represent lower-temperature wings/floors as a second
            # full-building regime, then area-weight the two thermal responses.
            # The solver's active/setback profile already applies the operating
            # controls. Count maintained setback heat instead of discarding it.
            delivered_kwh = (
                normal_fraction * normal_q
                + (1.0 - normal_fraction) * reduced_q
            ) / 1000.0 / efficiency
            grouped = delivered_kwh.groupby(index.month).sum()
            monthly.extend(
                MonthlyConsumption(int(year), int(month), float(value))
                for month, value in grouped.items()
            )
        return tuple(monthly)


def _validated_bui(case, parameters, *, reduced_regime, validator):
    bui, issues = validator(
        build_bui(case, parameters, reduced_regime=reduced_regime), fix=True
    )
    errors = [issue for issue in issues if issue.get("level") == "ERROR"]
    if errors:
        raise ValueError(f"Invalid BUI for {case.building_id}: {errors}")
    return bui


def _run_pbe_year(iso_class, bui, epw_path, year):
    result = iso_class().Temperature_and_Energy_needs_calculation(
        bui,
        weather_source="epw",
        path_weather_file=str(epw_path),
    )
    hourly = _hourly_result(result)
    raw_hours = len(hourly)
    if raw_hours in (8760 + 744, 8784 + 744):
        hourly = hourly.iloc[744:].reset_index(drop=True)
    if len(hourly) not in (8760, 8784):
        raise ValueError(
            f"Unexpected hourly result length for {year}: {raw_hours}. "
            "Expected 8760/8784 hours, optionally after a 744-hour warm-up."
        )
    if "Q_H" not in hourly.columns:
        raise ValueError("pyBuildingEnergy hourly output does not contain Q_H.")
    return hourly.reset_index(drop=True)


def _hourly_result(result):
    """Accept the tuple/list/dict return shapes seen in pyBuildingEnergy 2.1."""

    if isinstance(result, (tuple, list)):
        for item in result:
            if hasattr(item, "columns") and "Q_H" in item.columns:
                return item.copy()
    if isinstance(result, dict):
        for item in result.values():
            if hasattr(item, "columns") and "Q_H" in item.columns:
                return item.copy()
    raise ValueError(f"Could not identify hourly results in {type(result)!r}.")


class AnalyticDemoSimulator:
    """Fast deterministic stand-in used only to exercise the challenge loop."""

    _SEASONAL = (1.42, 1.24, 1.02, 0.67, 0.30, 0.10, 0.04, 0.05, 0.22, 0.65, 1.04, 1.34)

    def simulate(
        self, case: BuildingCase, parameters: Mapping[str, float]
    ) -> tuple[MonthlyConsumption, ...]:
        parameters = resolved_operational_parameters(case, parameters)
        area = float(case.fixed_inputs["conditioned_floor_area_m2"])
        schedule = (
            0.55 * float(parameters["weekday_heating_hours"]) / 10.0
            + 0.10 * float(parameters["non_workday_heating_hours"]) / 4.0
            + 0.10 * float(parameters["air_exchange_multiplier"])
            + 0.10 * (float(parameters["night_setback_temperature_c"]) - 8.0) / 7.0
            + 0.05 * (float(parameters["outdoor_shutdown_temperature_c"]) - 12.0) / 4.0
        )
        normal_fraction = float(parameters["normally_heated_area_fraction"])
        reduced = (
            float(parameters["reduced_area_schedule_fraction"])
            * (float(parameters["reduced_area_setpoint_c"]) - 8.0) / 12.0
        )
        operating_factor = normal_fraction * schedule + (1.0 - normal_fraction) * reduced
        efficiency = float(case.fixed_inputs["heating_system_efficiency"])
        # Calibrated only for a runnable demonstration at office-building scale;
        # competition results must use the pyBuildingEnergy backend.
        annual_scale = area * 70.0 * max(0.05, operating_factor) / efficiency
        periods = sorted({(row.year, row.month) for row in case.measurements})
        return tuple(
            MonthlyConsumption(year, month, annual_scale * seasonal / sum(self._SEASONAL))
            for year, month in periods
            for seasonal in (self._SEASONAL[month - 1],)
        )

