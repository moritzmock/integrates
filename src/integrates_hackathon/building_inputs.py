"""Public, anonymized building inputs loaded from ``data``.

Only ``building_id`` is an identifier. The geometry coordinates are translated
to one shared reference centroid and preserve each EUBUCCO footprint's shape
and orientation without exposing its original position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Mapping

from .data_io import available_amy_years, load_building_parameter_frame, parse_geometry_rings


ORIENTATIONS = ("N", "E", "S", "W")


@dataclass(frozen=True)
class FacadeGeometry:
    opaque_area_m2: float | None
    window_area_m2: float | None
    azimuth_deg: float
    gross_area_m2: float | None = None
    wwr: float | None = None
    wwr_basis: str = "dimensionless_pixel_area_ratio"


@dataclass(frozen=True)
class BuildingInput:
    building_id: str
    building_use: str
    construction_year: int
    gross_floor_area_m2: float
    conditioned_floor_area_m2: float
    heated_volume_m3: float
    floors_above_ground: int
    floors_below_ground: int
    total_height_m: float | None
    facades: Mapping[str, FacadeGeometry]
    roof_area_m2: float | None
    ground_floor_area_m2: float | None
    weather_file_ids: Mapping[int, str]
    source_status: str
    footprint_area_m2: float | None = None
    geometry_positions_latlon: tuple[tuple[float, float], ...] = ()
    geometry_holes_latlon: tuple[tuple[tuple[float, float], ...], ...] = ()
    geometry_wkt: str | None = None
    simulation_latitude: float | None = None
    simulation_longitude: float | None = None
    aggregate_envelope: Mapping[str, float] = field(default_factory=dict)
    baseline_assumptions: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.facades) != set(ORIENTATIONS):
            raise ValueError("facades must contain exactly N, E, S, and W")
        if self.gross_floor_area_m2 <= 0 or self.conditioned_floor_area_m2 <= 0:
            raise ValueError("floor areas must be positive")
        if self.heated_volume_m3 <= 0:
            raise ValueError("heated_volume_m3 must be positive")
        if self.geometry_positions_latlon:
            if self.geometry_positions_latlon[0] != self.geometry_positions_latlon[-1]:
                raise ValueError("geometry_positions_latlon must be a closed ring")
        for ring in self.geometry_holes_latlon:
            if ring[0] != ring[-1]:
                raise ValueError("Every geometry hole must be a closed ring")

    @property
    def geometry_complete(self) -> bool:
        surfaces = [
            self.roof_area_m2,
            self.ground_floor_area_m2,
            *(
                value
                for facade in self.facades.values()
                for value in (facade.opaque_area_m2, facade.window_area_m2)
            ),
        ]
        return all(value is not None and value >= 0 for value in surfaces)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if isnan(number) else number


def _required_float(row, field: str) -> float:
    value = _optional_float(row.get(field))
    if value is None or value <= 0:
        raise ValueError(f"{row.get('building_id')} is missing positive {field}")
    return value


def _facades(row) -> dict[str, FacadeGeometry]:
    facades: dict[str, FacadeGeometry] = {}
    for orientation in ORIENTATIONS:
        prefix = f"facade_{orientation.lower()}"
        gross_area = _optional_float(row.get(f"{prefix}_gross_area_m2"))
        wwr = _optional_float(row.get(f"{prefix}_wwr"))
        if gross_area is not None and wwr is not None:
            if gross_area <= 0:
                raise ValueError(f"{row.get('building_id')} has invalid {prefix}_gross_area_m2")
            if not 0 <= wwr <= 1:
                raise ValueError(f"{row.get('building_id')} has invalid unitless {prefix}_wwr")
            # The takeoff window/wall pixel counts are deliberately ignored.
            # Only their dimensionless ratio is applied to physical EUBUCCO area.
            window_area = gross_area * wwr
            opaque_area = gross_area * (1.0 - wwr)
        else:
            window_area = _optional_float(row.get(f"{prefix}_window_area_m2"))
            opaque_area = _optional_float(row.get(f"{prefix}_opaque_area_m2"))
        facades[orientation] = FacadeGeometry(
            opaque_area_m2=opaque_area,
            window_area_m2=window_area,
            azimuth_deg=float(row.get(f"{prefix}_azimuth_deg")),
            gross_area_m2=gross_area,
            wwr=wwr,
        )
    return facades


def _assumption(row, field: str, fallback: float) -> float:
    value = _optional_float(row.get(field))
    return fallback if value is None else value


def _load_buildings() -> tuple[BuildingInput, ...]:
    frame = load_building_parameter_frame()
    buildings: list[BuildingInput] = []
    for row in frame.to_dict(orient="records"):
        building_id = str(row["building_id"])
        positions, holes = parse_geometry_rings(
            str(row["anonymized_geometry_positions_latlon"])
        )
        baseline = {
            "wall_u_value_w_m2k": _assumption(row, "wall_u_value_w_m2k", 0.78),
            "roof_u_value_w_m2k": _assumption(row, "roof_u_value_w_m2k", 1.08),
            "floor_u_value_w_m2k": _assumption(row, "floor_u_value_w_m2k", 0.96),
            "window_u_value_w_m2k": _assumption(row, "window_u_value_w_m2k", 2.51),
            "window_g_value": _assumption(row, "window_g_value", 0.55),
            "infiltration_ach": _assumption(row, "infiltration_ach", 0.20),
            "heating_system_efficiency": _assumption(
                row, "heating_system_efficiency", 0.88
            ),
            "internal_gains_w_m2": _assumption(row, "internal_gains_w_m2", 8.0),
            "heating_setpoint_c": _assumption(row, "heating_setpoint_c", 21.0),
            "heating_hours_per_day": _assumption(
                row, "heating_hours_per_day", 14.0
            ),
            "heating_system_capacity_kw": _assumption(
                row, "heating_system_capacity_kw", 100.0
            ),
            "dhw_average_power_kw": _assumption(row, "dhw_average_power_kw", 0.0),
            "heat_generator_type": str(row.get("heat_generator_type") or "gas boiler"),
            "end_use_scope": str(row.get("end_use_scope") or "heating_only"),
        }
        buildings.append(
            BuildingInput(
                building_id=building_id,
                building_use=str(row["building_use"]),
                construction_year=int(row["construction_year"]),
                gross_floor_area_m2=_required_float(row, "gross_floor_area_m2"),
                conditioned_floor_area_m2=_required_float(
                    row, "conditioned_floor_area_m2"
                ),
                heated_volume_m3=_required_float(row, "heated_volume_m3"),
                floors_above_ground=int(row["floors_above_ground"]),
                floors_below_ground=int(row["floors_below_ground"]),
                total_height_m=_optional_float(row.get("total_height_m")),
                facades=_facades(row),
                roof_area_m2=_optional_float(row.get("roof_area_m2")),
                ground_floor_area_m2=_optional_float(row.get("ground_floor_area_m2")),
                weather_file_ids={
                    year: f"ita_39100_bolzano_local_{year}.epw"
                    for year in available_amy_years()
                },
                source_status=str(row.get("release_ready_status") or ""),
                footprint_area_m2=_optional_float(row.get("footprint_area_m2")),
                geometry_positions_latlon=positions,
                geometry_holes_latlon=holes,
                geometry_wkt=str(row.get("anonymized_geometry_wkt") or ""),
                simulation_latitude=_optional_float(
                    row.get("geometry_anchor_latitude")
                ),
                simulation_longitude=_optional_float(
                    row.get("geometry_anchor_longitude")
                ),
                aggregate_envelope={
                    key: value
                    for key in (
                        "total_envelope_area_m2",
                        "total_window_area_m2",
                        "total_opaque_wall_area_m2",
                    )
                    if (value := _optional_float(row.get(key))) is not None
                },
                baseline_assumptions=baseline,
            )
        )
    return tuple(buildings)


BUILDING_INPUTS = _load_buildings()
BUILDING_INPUTS_BY_ID = {
    building.building_id: building for building in BUILDING_INPUTS
}


def get_building_input(building_id: str) -> BuildingInput:
    try:
        return BUILDING_INPUTS_BY_ID[building_id]
    except KeyError as error:
        raise KeyError(
            f"Unknown building {building_id!r}; available: {sorted(BUILDING_INPUTS_BY_ID)}"
        ) from error
