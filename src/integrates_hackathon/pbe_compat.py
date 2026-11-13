"""Compatibility helpers for shortened pyBuildingEnergy 2.1 EPW runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np
import pandas as pd
from pybuildingenergy.source import utils as pbe_utils
from pybuildingenergy.source.generate_profile import HourlyProfileGenerator


def _shared_context(name: str) -> ContextVar:
    attribute = f"_integrates_context_{name}"
    context = getattr(pbe_utils, attribute, None)
    if context is None:
        context = ContextVar(f"integrates_pbe_{name}", default=None)
        setattr(pbe_utils, attribute, context)
    return context


_ACTIVE_WEATHER_INDEX = _shared_context("active_weather_index")
_ACTIVE_PROFILE_YEAR = _shared_context("active_profile_year")
_ACTIVE_WEATHER_TEMPERATURE_C = _shared_context("active_weather_temperature_c")
_ACTIVE_HEATING_CONTROLS = _shared_context("active_heating_controls")


@contextmanager
def pbe_profile_year(year: Any):
    token = _ACTIVE_PROFILE_YEAR.set(int(year))
    try:
        yield
    finally:
        _ACTIVE_PROFILE_YEAR.reset(token)


def _profile_calendar(index: pd.DatetimeIndex, year: int | None) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(index)
    if year is None or index.empty:
        return index
    leading_december = 0
    for month in index.month:
        if int(month) != 12:
            break
        leading_december += 1
    years = np.full(len(index), int(year), dtype=int)
    if 0 < leading_december < len(index):
        years[:leading_december] = int(year) - 1
    return pd.DatetimeIndex(pd.to_datetime({
        "year": years,
        "month": index.month,
        "day": index.day,
        "hour": index.hour,
        "minute": index.minute,
        "second": index.second,
    }))


def _patch_variable_weather_length() -> None:
    iso52010 = getattr(pbe_utils, "ISO52010", None)
    if iso52010 is None:
        return
    original_solar = getattr(iso52010, "Solar_irradiance_calculation", None)
    if original_solar is not None and not getattr(
        original_solar, "_integrates_weather_length_patch", False
    ):
        def solar_with_calendar_length(cls, *args, **kwargs):
            if "calendar" in kwargs:
                kwargs = dict(kwargs)
                kwargs["n_timesteps"] = len(kwargs["calendar"])
            elif len(args) > 9:
                args = list(args)
                args[9] = len(args[8])
                args = tuple(args)
            return original_solar(*args, **kwargs)

        solar_with_calendar_length._integrates_weather_length_patch = True
        iso52010.Solar_irradiance_calculation = classmethod(solar_with_calendar_length)

    original_shading = getattr(iso52010, "Shading_reduction_factor_window", None)
    if original_shading is not None and not getattr(
        original_shading, "_integrates_weather_length_patch", False
    ):
        def shading_with_calendar_length(cls, *args, **kwargs):
            if "calendar" in kwargs:
                kwargs = dict(kwargs)
                kwargs["n_timesteps"] = len(kwargs["calendar"])
            elif len(args) > 5:
                args = list(args)
                args[5] = len(args[4])
                args = tuple(args)
            return original_shading(*args, **kwargs)

        shading_with_calendar_length._integrates_weather_length_patch = True
        iso52010.Shading_reduction_factor_window = classmethod(
            shading_with_calendar_length
        )


def _patch_epw_weather_source() -> None:
    original = getattr(pbe_utils, "Calculation_ISO_52010", None)
    if original is None or getattr(original, "_integrates_epw_patch", False):
        return

    def calculation_with_epw_default(
        building_object, path_weather_file=None, weather_source="pvgis"
    ):
        resolved_source = "epw" if path_weather_file and weather_source == "pvgis" else weather_source
        result = original(
            building_object, path_weather_file, weather_source=resolved_source
        )
        frame = result.sim_df
        if 0 < len(frame) < 8760:
            leading_december = 0
            for is_december in pd.DatetimeIndex(frame.index).month == 12:
                if not is_december:
                    break
                leading_december += 1
            if 0 < leading_december < 744:
                warmup = frame.iloc[:leading_december]
                missing = 744 - leading_december
                copies = (missing + leading_december - 1) // leading_december
                padding = pd.concat([warmup] * copies).iloc[-missing:]
                frame = pd.concat([padding, frame])
                result.sim_df = frame
        _ACTIVE_WEATHER_INDEX.set(pd.DatetimeIndex(frame.index).copy())
        if "T2m" in frame:
            _ACTIVE_WEATHER_TEMPERATURE_C.set(
                pd.to_numeric(frame["T2m"], errors="coerce").reset_index(drop=True)
            )
        return result

    calculation_with_epw_default._integrates_epw_patch = True
    pbe_utils.Calculation_ISO_52010 = calculation_with_epw_default


def _patch_profiles() -> None:
    generator_class = HourlyProfileGenerator
    iso52016 = getattr(pbe_utils, "ISO52016", None)
    if generator_class is None or iso52016 is None:
        return
    original_generate = getattr(generator_class, "generate", None)
    if original_generate is not None and not getattr(
        original_generate, "_integrates_profile_patch", False
    ):
        def generate_for_active_weather(self):
            weather_index = _ACTIVE_WEATHER_INDEX.get()
            if weather_index is None:
                return original_generate(self)
            import holidays

            date_index = _profile_calendar(
                pd.DatetimeIndex(weather_index), _ACTIVE_PROFILE_YEAR.get()
            )
            profile = pd.DataFrame({"datetime": date_index})
            profile["date"] = profile["datetime"].dt.date
            profile["hour"] = profile["datetime"].dt.hour
            profile["day_of_week"] = profile["datetime"].dt.dayofweek
            profile["day_name"] = profile["datetime"].dt.day_name()
            self.country_holidays = holidays.country_holidays(
                self.country, years=sorted({int(value) for value in date_index.year})
            )
            profile["is_holiday"] = profile["date"].map(
                lambda value: value in self.country_holidays
            )
            profile["is_weekend"] = profile["day_of_week"].isin([5, 6])
            profile["is_working_day"] = ~(
                profile["is_holiday"] | profile["is_weekend"]
            )
            profile["holiday_name"] = profile["date"].map(
                lambda value: self.country_holidays.get(value, "")
            )
            hours = profile["hour"].to_numpy(dtype=int)
            working = profile["is_working_day"].to_numpy(dtype=bool)
            for category, pair in self.profiles.items():
                weekday = np.asarray(pair["weekday"], dtype=float)
                holiday = np.asarray(pair["holiday"], dtype=float)
                profile[f"{category}_profile"] = np.where(
                    working, weekday[hours], holiday[hours]
                )

            controls = _ACTIVE_HEATING_CONTROLS.get() or {}
            if controls and "heating_profile" in profile:
                temperatures = _ACTIVE_WEATHER_TEMPERATURE_C.get()
                if temperatures is not None and len(temperatures) == len(profile):
                    rolling = temperatures.rolling(24, min_periods=1).mean().to_numpy(float)
                else:
                    rolling = np.full(len(profile), np.nan)
                day_of_year = profile["datetime"].dt.dayofyear.to_numpy(int)
                start_day = int(controls.get("heating_season_start_doy", 274))
                end_day = int(controls.get("heating_season_end_doy", 135))
                season = (
                    (day_of_year >= start_day) | (day_of_year <= end_day)
                    if start_day > end_day
                    else (day_of_year >= start_day) & (day_of_year <= end_day)
                )
                cutoff = float(controls.get("heating_cutoff_temperature_c", 16.0))
                hysteresis = max(0.0, float(controls.get("heating_cutoff_hysteresis_c", 1.5)))
                turn_on = cutoff - hysteresis / 2.0
                turn_off = cutoff + hysteresis / 2.0
                available = np.zeros(len(profile), dtype=bool)
                state = False
                previous = None
                for index, (timestamp, in_season, temperature) in enumerate(
                    zip(profile["datetime"], season, rolling)
                ):
                    if previous is not None and timestamp < previous:
                        state = False
                    previous = timestamp
                    if not in_season:
                        state = False
                    elif np.isfinite(temperature):
                        if temperature <= turn_on:
                            state = True
                        elif temperature >= turn_off:
                            state = False
                    available[index] = state and bool(in_season)
                profile.loc[~available, "heating_profile"] = 0.0
                profile["heating_active_setpoint_available"] = available
                profile["heating_regime"] = np.where(available, "cold", "setback")

            profile["profile_value"] = profile["occupancy_profile"]
            profile["day_type"] = np.where(
                profile["is_working_day"], "Working Day", "Holiday/Weekend"
            )
            self.df = profile
            return profile

        generate_for_active_weather._integrates_profile_patch = True
        generator_class.generate = generate_for_active_weather

    original_temperature = getattr(
        iso52016, "Temperature_and_Energy_needs_calculation", None
    )
    if original_temperature is not None and not getattr(
        original_temperature, "_integrates_profile_patch", False
    ):
        def temperature_with_scoped_weather(cls, *args, **kwargs):
            building = args[0] if args else kwargs.get("building_object", {})
            controls = (
                building.get("building_parameters", {}).get("heating_controls")
                if isinstance(building, dict) else None
            )
            index_token = _ACTIVE_WEATHER_INDEX.set(None)
            temperature_token = _ACTIVE_WEATHER_TEMPERATURE_C.set(None)
            controls_token = _ACTIVE_HEATING_CONTROLS.set(controls)
            try:
                return original_temperature(*args, **kwargs)
            finally:
                _ACTIVE_WEATHER_INDEX.reset(index_token)
                _ACTIVE_WEATHER_TEMPERATURE_C.reset(temperature_token)
                _ACTIVE_HEATING_CONTROLS.reset(controls_token)

        temperature_with_scoped_weather._integrates_profile_patch = True
        iso52016.Temperature_and_Energy_needs_calculation = classmethod(
            temperature_with_scoped_weather
        )


_patch_variable_weather_length()
_patch_epw_weather_source()
_patch_profiles()
