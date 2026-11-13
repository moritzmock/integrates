"""Raw and normalized monthly energy consumption for anonymized buildings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .building_inputs import BUILDING_INPUTS_BY_ID
from .data_io import load_energy_source_rows
from .models import Measurement


SMC_TO_KWH = 9.59

# Only space heating is simulated, so only the heating-only block is scored.
# Organizers derive it from combined heating+DHW bills by subtracting that
# year's June-August average as the assumed monthly DHW load.
SCORED_END_USE_SCOPE = "heating_only"

# Months wholly outside Bolzano's legal heating season (15 October-15 April).
# The DHW subtraction can leave small negative residuals here; heating cannot
# be negative, so they are clipped to zero. Elsewhere a negative is an error.
NON_HEATING_MONTHS = frozenset({5, 6, 7, 8, 9})
CLIPPED_QUALITY_FLAG = "negative_after_dhw_subtraction_clipped_to_zero"


@dataclass(frozen=True)
class EnergyObservation:
    building_id: str
    year: int
    month: int
    source_value: float
    source_unit: str
    kwh_per_source_unit: float
    end_use_scope: str
    quality_flag: str = "reported"

    def __post_init__(self) -> None:
        if self.source_unit not in {"Smc", "kWh"}:
            raise ValueError(
                "Only Smc and kWh are currently configured. Supply a documented "
                "conversion factor before accepting ordinary m3."
            )
        if not 1 <= self.month <= 12:
            raise ValueError("month must be between 1 and 12")
        if self.source_value < 0 or self.kwh_per_source_unit <= 0:
            raise ValueError(
                "energy must be non-negative and its conversion factor must be positive"
            )

    @property
    def consumption_kwh(self) -> float:
        return float(self.source_value) * float(self.kwh_per_source_unit)

    @property
    def gross_floor_area_m2(self) -> float:
        return float(BUILDING_INPUTS_BY_ID[self.building_id].gross_floor_area_m2)

    @property
    def consumption_kwh_m2_gfa(self) -> float:
        return self.consumption_kwh / self.gross_floor_area_m2

    def as_normalized_dict(self) -> dict[str, float | int | str]:
        row = asdict(self)
        row["consumption_kwh"] = self.consumption_kwh
        row["gross_floor_area_m2"] = self.gross_floor_area_m2
        row["consumption_kwh_m2_gfa"] = self.consumption_kwh_m2_gfa
        return row


def _load_observations() -> tuple[EnergyObservation, ...]:
    source_rows = load_energy_source_rows()
    scored_years = {
        (row["building_id"], row["year"])
        for row in source_rows
        if row["end_use_scope"] == SCORED_END_USE_SCOPE
    }
    unscored_years = sorted(
        {(row["building_id"], row["year"]) for row in source_rows} - scored_years
    )
    if unscored_years:
        raise ValueError(
            f"No heating-only values for {unscored_years}. Enter heating-only "
            "values (combined heating+DHW minus the June-August average) in the "
            "heating-only block of energy_consumption_values.xlsx."
        )

    observations = []
    for row in source_rows:
        if row["end_use_scope"] != SCORED_END_USE_SCOPE:
            continue
        source_value = float(row["source_value"])
        quality_flag = str(row["quality_flag"])
        if source_value < 0:
            if int(row["month"]) not in NON_HEATING_MONTHS:
                raise ValueError(
                    f"{row['building_id']} {row['year']}-{int(row['month']):02d} "
                    f"has negative heating-only consumption ({source_value}) "
                    "inside the heating season"
                )
            source_value = 0.0
            quality_flag = CLIPPED_QUALITY_FLAG
        unit = str(row["source_unit"])
        if unit == "Smc":
            factor = SMC_TO_KWH
        elif unit == "kWh":
            factor = 1.0
        else:
            raise ValueError(
                f"{row['building_id']} {row['year']} uses m3 without a documented "
                "kWh conversion factor"
            )
        observations.append(
            EnergyObservation(
                building_id=str(row["building_id"]),
                year=int(row["year"]),
                month=int(row["month"]),
                source_value=source_value,
                source_unit=unit,
                kwh_per_source_unit=factor,
                end_use_scope=str(row["end_use_scope"]),
                quality_flag=quality_flag,
            )
        )
    unknown = {
        row.building_id
        for row in observations
        if row.building_id not in BUILDING_INPUTS_BY_ID
    }
    if unknown:
        raise ValueError(f"Energy data contains unknown building IDs: {sorted(unknown)}")
    return tuple(sorted(observations, key=lambda row: (row.building_id, row.year, row.month)))


ENERGY_OBSERVATIONS = _load_observations()


def observations_for(building_id: str) -> tuple[EnergyObservation, ...]:
    rows = tuple(row for row in ENERGY_OBSERVATIONS if row.building_id == building_id)
    if not rows:
        raise KeyError(f"No energy observations for {building_id!r}")
    return rows


def measurements_for(building_id: str) -> tuple[Measurement, ...]:
    return tuple(
        Measurement(row.year, row.month, row.consumption_kwh)
        for row in observations_for(building_id)
    )


def normalized_rows(
    observations: Iterable[EnergyObservation] = ENERGY_OBSERVATIONS,
) -> list[dict[str, float | int | str]]:
    return [row.as_normalized_dict() for row in observations]
