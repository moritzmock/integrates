"""Monthly summaries derived from the organizer's actual-year EPW files."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass

from .building_inputs import BUILDING_INPUTS
from .data_io import amy_epw_path, available_amy_years


@dataclass(frozen=True)
class MonthlyWeather:
    building_id: str
    year: int
    month: int
    mean_outdoor_temperature_c: float
    heating_degree_days_18c: float
    data_status: str = "actual_year_epw"

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


WEATHER_YEARS = available_amy_years()


def _summarize_epw(year: int) -> tuple[tuple[float, float], ...]:
    path = amy_epw_path(year)
    hourly_by_day: dict[tuple[int, int], list[float]] = defaultdict(list)
    hourly_by_month: dict[int, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(8):
            next(reader)
        for row in reader:
            month, day, dry_bulb_c = int(row[1]), int(row[2]), float(row[6])
            hourly_by_day[(month, day)].append(dry_bulb_c)
            hourly_by_month[month].append(dry_bulb_c)

    summaries = []
    for month in range(1, 13):
        temperatures = hourly_by_month[month]
        mean_temperature = sum(temperatures) / len(temperatures)
        degree_days = sum(
            max(0.0, 18.0 - sum(day_values) / len(day_values))
            for (day_month, _), day_values in hourly_by_day.items()
            if day_month == month
        )
        summaries.append((mean_temperature, degree_days))
    return tuple(summaries)


def _actual_weather() -> tuple[MonthlyWeather, ...]:
    rows: list[MonthlyWeather] = []
    yearly = {year: _summarize_epw(year) for year in WEATHER_YEARS}
    for building in BUILDING_INPUTS:
        for year, summaries in yearly.items():
            for month, (temperature, degree_days) in enumerate(summaries, start=1):
                rows.append(
                    MonthlyWeather(
                        building_id=building.building_id,
                        year=year,
                        month=month,
                        mean_outdoor_temperature_c=temperature,
                        heating_degree_days_18c=degree_days,
                    )
                )
    return tuple(rows)


WEATHER_SUMMARIES = _actual_weather()


def weather_for(building_id: str) -> tuple[MonthlyWeather, ...]:
    return tuple(row for row in WEATHER_SUMMARIES if row.building_id == building_id)

