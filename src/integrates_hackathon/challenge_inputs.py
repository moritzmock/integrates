"""Assemble organizer-owned inputs into the calibration problem contract."""

from __future__ import annotations

from .building_inputs import BUILDING_INPUTS
from .data_io import amy_epw_path
from .energy_consumption import measurements_for
from .models import BuildingCase, ParameterSpec


CALIBRATABLE_PARAMETER_DEFINITIONS = (
    (
        "weekday_heating_hours",
        4.0,
        12.0,
        8.0,
        "h/day",
        "Active occupied-setpoint duration on weekdays",
    ),
    (
        "non_workday_heating_hours",
        0.0,
        4.0,
        2.0,
        "h/day",
        "Active occupied-setpoint duration on weekends and public holidays",
    ),
    ("weekday_start_hour", 5.0, 9.0, 7.0, "hour", "Start of active weekday heating"),
    (
        "occupied_heating_setpoint_c",
        18.5,
        21.5,
        20.0,
        "degC",
        "Maintained setpoint during active occupied hours",
    ),
    (
        "night_setback_temperature_c",
        12.0,
        18.0,
        15.0,
        "degC",
        "Maintained setpoint outside active occupied hours; its energy is counted",
    ),
    (
        "outdoor_shutdown_temperature_c",
        12.0,
        18.0,
        15.0,
        "degC",
        "24-hour mean outdoor threshold switching active heating to setback",
    ),
    (
        "normally_heated_area_fraction",
        0.60,
        1.00,
        0.80,
        "fraction",
        "Conditioned area following the normal rather than reduced regime",
    ),
    (
        "reduced_area_setpoint_c",
        12.0,
        17.0,
        15.0,
        "degC",
        "Active setpoint for wings/floors following the reduced regime",
    ),
    (
        "reduced_area_schedule_fraction",
        0.25,
        0.75,
        0.50,
        "fraction",
        "Reduced-regime active hours as a fraction of normal active hours",
    ),
    (
        "air_exchange_multiplier",
        0.70,
        1.30,
        1.00,
        "multiplier",
        "Multiplier applied to audited/default infiltration and ventilation",
    ),
    (
        "opaque_envelope_u_value_multiplier",
        0.90,
        1.10,
        1.00,
        "multiplier",
        "Common multiplier on audited wall, roof, and floor U-values",
    ),
    (
        "window_u_value_multiplier",
        0.90,
        1.10,
        1.00,
        "multiplier",
        "Multiplier on the audited window U-value",
    ),
)

PARAMETER_DEFINITIONS = CALIBRATABLE_PARAMETER_DEFINITIONS
MAX_CALIBRATABLE_PARAMETERS = len(CALIBRATABLE_PARAMETER_DEFINITIONS)


def parameter_defaults() -> dict[str, float]:
    """Return the organizer defaults used when a parameter is not selected."""

    return {
        name: float(default)
        for name, _lower, _upper, default, _unit, _description
        in CALIBRATABLE_PARAMETER_DEFINITIONS
    }


FIXED_OPERATIONAL_PARAMETERS = parameter_defaults()


def parameter_selection_defaults() -> dict[str, None]:
    """Return the participant-facing selection with every dimension inactive."""

    return {name: None for name, *_rest in CALIBRATABLE_PARAMETER_DEFINITIONS}


def parameter_specs(
    selection: dict[str, float | None] | None = None,
    *,
    announce: bool = True,
) -> tuple[ParameterSpec, ...]:
    """Create the active bounded parameter space for one submission.

    ``None`` keeps a parameter fixed at its organizer default. A numeric value
    activates that parameter and becomes its optimizer starting value. Bounds
    and parameter order always come from organizer code.
    """

    selected = parameter_selection_defaults() if selection is None else dict(selection)
    expected = parameter_selection_defaults()
    assert set(selected) == set(expected), (
        "calibratable_params_init keys are organizer-owned and must not be "
        "added, removed, or renamed"
    )

    specs: list[ParameterSpec] = []
    for name, lower, upper, _default, unit, description in CALIBRATABLE_PARAMETER_DEFINITIONS:
        initial = selected[name]
        if initial is None:
            continue
        if isinstance(initial, bool):
            raise AssertionError(f"{name!r} must be numeric, not boolean")
        try:
            initial_value = float(initial)
        except (TypeError, ValueError) as error:
            raise AssertionError(f"{name!r} must be None or a finite numeric value") from error
        if not lower <= initial_value <= upper:
            raise AssertionError(
                f"{name!r} initial value {initial_value} is outside [{lower}, {upper}]"
            )
        specs.append(ParameterSpec(name, lower, upper, initial_value, unit, description))

    if announce:
        names = [spec.name for spec in specs]
        print(
            f"Parameter rule check passed: {len(specs)}/{MAX_CALIBRATABLE_PARAMETERS} "
            "calibratable parameters activated"
            + (f" ({', '.join(names)})." if names else "; baseline passthrough selected.")
        )
    return tuple(specs)


def _parameters_for(building) -> tuple[ParameterSpec, ...]:
    del building
    return tuple(
        ParameterSpec(name, lower, upper, default, unit, description)
        for name, lower, upper, default, unit, description
        in CALIBRATABLE_PARAMETER_DEFINITIONS
    )


def _case_from_public_input(building) -> BuildingCase:
    measurements = measurements_for(building.building_id)
    measured_years = sorted({row.year for row in measurements})
    weather_files = {year: amy_epw_path(year) for year in measured_years}
    missing_weather = [year for year, path in weather_files.items() if not path.is_file()]
    if missing_weather:
        raise ValueError(
            f"{building.building_id} has consumption data but no AMY EPW for "
            f"{missing_weather}"
        )
    return BuildingCase(
        building_id=building.building_id,
        fixed_inputs={
            "name": building.building_id,
            "building_use": building.building_use,
            "construction_year": building.construction_year,
            "gross_floor_area_m2": building.gross_floor_area_m2,
            "conditioned_floor_area_m2": building.conditioned_floor_area_m2,
            "heated_volume_m3": building.heated_volume_m3,
            "n_floors": building.floors_above_ground,
            "total_height_m": building.total_height_m,
            "footprint_area_m2": building.footprint_area_m2,
            "geometry_positions_latlon": building.geometry_positions_latlon,
            "geometry_holes_latlon": building.geometry_holes_latlon,
            "geometry_wkt": building.geometry_wkt,
            "simulation_latitude": building.simulation_latitude,
            "simulation_longitude": building.simulation_longitude,
            "facades": building.facades,
            "roof_area_m2": building.roof_area_m2,
            "ground_floor_area_m2": building.ground_floor_area_m2,
            "geometry_complete": building.geometry_complete,
            "fixed_operational_parameters": FIXED_OPERATIONAL_PARAMETERS,
            **building.aggregate_envelope,
            **building.baseline_assumptions,
            # Stable BUI names for audited fabric properties. Participants may
            # choose only the tightly bounded multipliers defined above.
            "wall_u_value": building.baseline_assumptions["wall_u_value_w_m2k"],
            "roof_u_value": building.baseline_assumptions["roof_u_value_w_m2k"],
            "floor_u_value": building.baseline_assumptions["floor_u_value_w_m2k"],
            "window_u_value": building.baseline_assumptions["window_u_value_w_m2k"],
        },
        parameters=_parameters_for(building),
        measurements=measurements,
        weather_files=weather_files,
        metadata={
            "source_status": building.source_status,
            "weather_file_ids": dict(building.weather_file_ids),
            "end_use_scope": building.baseline_assumptions.get("end_use_scope"),
        },
    )


BUILDINGS = tuple(_case_from_public_input(building) for building in BUILDING_INPUTS)
BUILDINGS_BY_ID = {case.building_id: case for case in BUILDINGS}


def get_building_case(building_id: str) -> BuildingCase:
    try:
        return BUILDINGS_BY_ID[building_id]
    except KeyError as error:
        available = ", ".join(sorted(BUILDINGS_BY_ID))
        raise KeyError(f"Unknown building {building_id!r}. Available: {available}") from error

