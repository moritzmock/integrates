"""Build a pyBuildingEnergy BUI from scalar inputs, without spatial libraries."""

from __future__ import annotations

from typing import Any, Mapping

from .heating_schedule import profile_pair
from .models import BuildingCase


def _positive(inputs: Mapping[str, Any], name: str) -> float:
    value = float(inputs[name])
    if value <= 0:
        raise ValueError(f"{name!r} must be positive.")
    return value


def build_bui(
    case: BuildingCase,
    parameters: Mapping[str, float],
    *,
    reduced_regime: bool = False,
) -> dict[str, Any]:
    """Build a BUI from explicit oriented areas; never infer 2D/3D geometry."""

    resolved_parameters = {
        **{
            str(name): float(value)
            for name, value in case.fixed_inputs.get(
                "fixed_operational_parameters", {}
            ).items()
        },
        **{str(name): float(value) for name, value in parameters.items()},
    }
    cfg = dict(case.fixed_inputs)
    cfg.update(resolved_parameters)

    floor_area = _positive(cfg, "conditioned_floor_area_m2")
    n_floors = _positive(cfg, "n_floors")
    if not cfg.get("geometry_complete"):
        raise ValueError(
            f"{case.building_id} has incomplete geometry. Fill total height, roof/floor "
            "areas, and N/E/S/W opaque and window areas in building_inputs.py."
        )
    height = _positive(cfg, "total_height_m")
    roof_area = _positive(cfg, "roof_area_m2")
    ground_floor_area = _positive(cfg, "ground_floor_area_m2")
    internal_gains = _positive(cfg, "internal_gains_w_m2")
    facades = cfg["facades"]
    heating_profile = profile_pair(resolved_parameters, reduced_regime=reduced_regime)
    normal_setpoint = float(
        resolved_parameters.get(
            "occupied_heating_setpoint_c",
            cfg.get("heating_setpoint_c", 20.0),
        )
    )
    active_setpoint = (
        min(normal_setpoint, float(resolved_parameters["reduced_area_setpoint_c"]))
        if reduced_regime
        else normal_setpoint
    )
    setback = min(
        active_setpoint - 0.5,
        float(resolved_parameters["night_setback_temperature_c"]),
    )
    air_multiplier = float(resolved_parameters["air_exchange_multiplier"])
    opaque_u_multiplier = float(
        resolved_parameters.get("opaque_envelope_u_value_multiplier", 1.0)
    )
    window_u_multiplier = float(
        resolved_parameters.get("window_u_value_multiplier", 1.0)
    )
    # Match pyBuildingEnergy's Office profile used by the earlier fallback,
    # while making the schedule explicit for versions that require custom
    # internal-gain entries to carry their own 24-hour arrays.
    gain_weekday = [
        0, 0, 0, 0, 0, 0, 0, 0.2, 0.6, 0.6, 0.7, 0.7,
        0.4, 0.6, 0.7, 0.7, 0.6, 0.2, 0, 0, 0, 0, 0, 0,
    ]
    gain_weekend = [0.0] * 24

    surfaces: list[dict[str, Any]] = [
        _opaque_surface(
            "Roof", roof_area, float(cfg["roof_u_value"]) * opaque_u_multiplier,
            0.0, 1.0, 0.0,
        ),
        _opaque_surface(
            "Ground floor", ground_floor_area,
            float(cfg["floor_u_value"]) * opaque_u_multiplier, 90.0, 0.0, 0.0,
        ),
    ]
    facade_lengths: list[float] = []
    for orientation in ("N", "E", "S", "W"):
        facade = facades[orientation]
        opaque_area = float(facade.opaque_area_m2)
        window_area = float(facade.window_area_m2)
        azimuth = float(facade.azimuth_deg)
        if opaque_area < 0 or window_area < 0:
            raise ValueError(f"Facade areas must be non-negative for {orientation}")
        length = (opaque_area + window_area) / height
        facade_lengths.append(length)
        surfaces.append(
            _opaque_surface(
                f"Opaque {orientation} wall",
                opaque_area,
                float(cfg["wall_u_value"]) * opaque_u_multiplier,
                90.0,
                0.5,
                length,
                azimuth,
                height,
            )
        )
        if window_area > 0:
            surfaces.append(
                {
                    "name": f"Window {orientation}",
                    "type": "transparent",
                    "area": window_area,
                    "sky_view_factor": 0.5,
                    "u_value": float(cfg["window_u_value"]) * window_u_multiplier,
                    "g_value": float(cfg["window_g_value"]),
                    "solar_absorptance": 0.0,
                    "thermal_capacity": 0.0,
                    "orientation": {"azimuth": azimuth, "tilt": 90.0},
                    "name_adj_zone": None,
                    "height": height,
                    "length": length,
                    "width": window_area / height,
                }
            )

    return {
        "building": {
            "name": str(cfg.get("name", case.building_id)),
            "azimuth_relative_to_true_north": 0.0,
            "latitude": float(cfg.get("simulation_latitude", 0.0)),
            "longitude": float(cfg.get("simulation_longitude", 0.0)),
            "exposed_perimeter": sum(facade_lengths),
            "height": height,
            "wall_thickness": float(cfg.get("wall_thickness_m", 0.30)),
            "n_floors": n_floors,
            "building_type_class": str(cfg.get("building_type_class", "Office")),
            "adj_zones_present": False,
            "number_adj_zone": 0,
            "net_floor_area": floor_area,
            "construction_class": str(cfg.get("construction_class", "class_i")),
        },
        "adjacent_zones": [],
        "building_surface": surfaces,
        "units": {
            "area": "m2",
            "length": "m",
            "height": "m",
            "u_value": "W/m2K",
            "thermal_capacity": "J/m2K",
            "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
            "tilt": "degrees (0=horizontal, 90=vertical)",
            "internal_gain": "W/m2",
            "internal_gain_profile": "Normalized to 0-1",
            "HVAC_profile": "0: off, 1: on",
        },
        "building_parameters": {
            "temperature_setpoints": {
                "heating_setpoint": active_setpoint,
                "heating_setback": setback,
                "cooling_setpoint": float(cfg.get("cooling_setpoint_c", 26.0)),
                "cooling_setback": float(cfg.get("cooling_setback_c", 30.0)),
                "units": "degC",
            },
            "system_capacities": {
                "heating_capacity": float(
                    cfg.get("heating_capacity_w", cfg.get("heating_system_capacity_kw", floor_area * 0.1) * 1000.0)
                ),
                "cooling_capacity": float(cfg.get("cooling_capacity_w", 0.0)),
                "units": "W",
            },
            "system_metadata": {
                "heat_generator_type": str(cfg.get("heat_generator_type", "gas boiler")),
                "cop": None,
                "seasonal_performance_factor": None,
            },
            "system_efficiencies": {
                "heating_generator_efficiency": float(cfg["heating_system_efficiency"]),
                "system_efficiency_factor": 1.0,
                "heating_system_efficiency": float(cfg["heating_system_efficiency"]),
                "units": "fraction",
            },
            "ventilation": {
                "ventilation_type": str(cfg.get("ventilation_type", "temp_wind")),
                "flow_rate_per_person": float(cfg.get("flow_rate_per_person", 0.5)),
                "units": "l/(s m2)",
                "custom_heat_transfer_coefficient_ventilation": None,
            },
            "airflow_rates": {
                "infiltration_rate": float(cfg["infiltration_ach"]) * air_multiplier,
                "ventilation_rate_extra": float(cfg.get("ventilation_rate_extra_ach", 0.0)) * air_multiplier,
                "units": "ACH (air changes per hour)",
            },
            "heating_profile": heating_profile,
            "ventilation_profile": heating_profile,
            # The pyBuildingEnergy compatibility adapter applies these controls
            # to the solver profile. A zero profile selects maintained setback;
            # it does not mean free-floating or uncounted heat.
            "heating_controls": {
                "heating_season_start_doy": 288,  # 15 October
                "heating_season_end_doy": 105,  # 15 April
                "heating_cutoff_temperature_c": float(
                    resolved_parameters["outdoor_shutdown_temperature_c"]
                ),
                "heating_cutoff_hysteresis_c": 0.0,
            },
            # Keep the audited/default total fixed while retaining the
            # engine's category-specific gain channels. Shares sum to 1.0.
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": internal_gains * 0.375,
                    "weekday": gain_weekday,
                    "weekend": gain_weekend,
                },
                {
                    "name": "appliances",
                    "full_load": internal_gains * 0.500,
                    "weekday": gain_weekday,
                    "weekend": gain_weekend,
                },
                {
                    "name": "lighting",
                    "full_load": internal_gains * 0.125,
                    "weekday": gain_weekday,
                    "weekend": gain_weekend,
                },
            ],
        },
    }


def _opaque_surface(
    name: str,
    area: float,
    u_value: float,
    tilt: float,
    sky_view_factor: float,
    length: float,
    azimuth: float = 0.0,
    height: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "type": "opaque",
        "area": float(area),
        "sky_view_factor": float(sky_view_factor),
        "u_value": float(u_value),
        "solar_absorptance": 0.6 if sky_view_factor else 0.0,
        "thermal_capacity": 165000.0,
        "orientation": {"azimuth": float(azimuth), "tilt": float(tilt)},
        "name_adj_zone": None,
        "height": float(height),
        "length": float(length),
    }

