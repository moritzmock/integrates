"""Organizer-owned heating-season law and schedule construction.

The municipal rules are fixed inputs, not calibration variables.  Bolzano's
ordinary season is 15 October through 15 April, with operation between 05:00
and 23:00 for at most 14 hours per day.  The dated seven-hour permissions
below are the year-specific municipal exceptions identified for the measured
period.  Participants may optimize operation *within* these permissions, but
cannot create additional out-of-season operation.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping


ORDINARY_START = (10, 15)
ORDINARY_END = (4, 15)
LEGAL_WINDOW_START_HOUR = 5
LEGAL_WINDOW_END_HOUR = 23
ORDINARY_MAX_HOURS = 14.0
EXCEPTION_MAX_HOURS = 7.0

# Inclusive dates.  These are organizer evidence and deliberately immutable
# from calibration.py.
YEAR_SPECIFIC_EXCEPTIONS: Mapping[int, tuple[tuple[date, date], ...]] = {
    2019: ((date(2019, 4, 16), date(2019, 5, 26)),),
    2020: ((date(2020, 10, 1), date(2020, 10, 14)),),
    2021: (
        (date(2021, 4, 16), date(2021, 4, 22)),
        (date(2021, 10, 8), date(2021, 10, 14)),
    ),
}


def permitted_heating_hours(day: date) -> float:
    """Return the legal maximum plant-operating hours for one date."""

    if (day.month, day.day) >= ORDINARY_START or (day.month, day.day) <= ORDINARY_END:
        return ORDINARY_MAX_HOURS
    for start, end in YEAR_SPECIFIC_EXCEPTIONS.get(day.year, ()):
        if start <= day <= end:
            return EXCEPTION_MAX_HOURS
    return 0.0


def daily_schedule_hours(
    day: date,
    parameters: Mapping[str, float],
    *,
    reduced_regime: bool = False,
) -> tuple[float, float] | None:
    """Return an allowed [start, end) interval, or ``None`` when plant is off."""

    legal_cap = permitted_heating_hours(day)
    if legal_cap <= 0:
        return None
    is_non_workday = day.weekday() >= 5 or day in italian_public_holidays(day.year)
    requested = (
        float(parameters["non_workday_heating_hours"])
        if is_non_workday
        else float(parameters["weekday_heating_hours"])
    )
    if reduced_regime:
        requested *= float(parameters["reduced_area_schedule_fraction"])
    duration = max(0.0, min(requested, legal_cap))
    if duration <= 0:
        return None
    start = max(LEGAL_WINDOW_START_HOUR, float(parameters["weekday_start_hour"]))
    end = min(float(LEGAL_WINDOW_END_HOUR), start + duration)
    if end <= start:
        return None
    return start, end


def hourly_permission_mask(
    epw_path: str | Path,
    year: int,
    parameters: Mapping[str, float],
    *,
    reduced_regime: bool = False,
) -> tuple[bool, ...]:
    """Build the exact hourly permission mask on the EPW row calendar.

    A 24-hour rolling outdoor-temperature mean implements the calibratable
    weather-compensated shutdown.  It can switch heating off inside a legal
    period; it can never switch heating on outside one.
    """

    rows: list[tuple[datetime, float]] = []
    with Path(epw_path).open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(8):
            next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            month, day = int(row[1]), int(row[2])
            # EPW hours are 1..24 and describe the preceding hourly interval.
            hour = max(0, min(23, int(row[3]) - 1))
            rows.append((datetime(int(year), month, day, hour), float(row[6])))

    cutoff = float(parameters["outdoor_shutdown_temperature_c"])
    temperatures = [temperature for _, temperature in rows]
    rolling: list[float] = []
    running = 0.0
    for index, temperature in enumerate(temperatures):
        running += temperature
        if index >= 24:
            running -= temperatures[index - 24]
        rolling.append(running / min(index + 1, 24))

    mask = []
    for (timestamp, _), mean_temperature in zip(rows, rolling, strict=True):
        interval = daily_schedule_hours(
            timestamp.date(), parameters, reduced_regime=reduced_regime
        )
        in_interval = bool(
            interval is not None and interval[0] <= timestamp.hour < interval[1]
        )
        mask.append(in_interval and mean_temperature < cutoff)
    return tuple(mask)


def profile_pair(
    parameters: Mapping[str, float], *, reduced_regime: bool = False
) -> dict[str, list[float]]:
    """Return PBE's 24-hour weekday/non-workday heating profile pair."""

    start = int(round(float(parameters["weekday_start_hour"])))
    weekday_duration = float(parameters["weekday_heating_hours"])
    non_workday_duration = float(parameters["non_workday_heating_hours"])
    if reduced_regime:
        factor = float(parameters["reduced_area_schedule_fraction"])
        weekday_duration *= factor
        non_workday_duration *= factor

    def values(duration: float) -> list[float]:
        end = min(LEGAL_WINDOW_END_HOUR, start + max(0, int(round(duration))))
        return [1.0 if start <= hour < end else 0.0 for hour in range(24)]

    return {"weekday": values(weekday_duration), "weekend": values(non_workday_duration)}


def italian_public_holidays(year: int) -> frozenset[date]:
    """National public holidays relevant to the working/non-working profile."""

    easter = _gregorian_easter(year)
    fixed = ((1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15), (11, 1), (12, 8), (12, 25), (12, 26))
    return frozenset({date(year, month, day) for month, day in fixed} | {easter, easter + timedelta(days=1)})


def _gregorian_easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)
