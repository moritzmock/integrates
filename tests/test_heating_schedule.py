from __future__ import annotations

import unittest
from datetime import date

from integrates_hackathon.heating_schedule import (
    EXCEPTION_MAX_HOURS,
    ORDINARY_MAX_HOURS,
    daily_schedule_hours,
    permitted_heating_hours,
)


class HeatingScheduleTests(unittest.TestCase):
    def test_ordinary_season_and_summer_shutdown(self) -> None:
        self.assertEqual(permitted_heating_hours(date(2020, 1, 10)), ORDINARY_MAX_HOURS)
        self.assertEqual(permitted_heating_hours(date(2020, 7, 10)), 0.0)

    def test_year_specific_exceptions_are_fixed(self) -> None:
        self.assertEqual(permitted_heating_hours(date(2019, 5, 26)), EXCEPTION_MAX_HOURS)
        self.assertEqual(permitted_heating_hours(date(2020, 10, 1)), EXCEPTION_MAX_HOURS)
        self.assertEqual(permitted_heating_hours(date(2021, 4, 22)), EXCEPTION_MAX_HOURS)
        self.assertEqual(permitted_heating_hours(date(2021, 10, 8)), EXCEPTION_MAX_HOURS)
        self.assertEqual(permitted_heating_hours(date(2021, 4, 23)), 0.0)

    def test_exception_caps_calibrated_duration(self) -> None:
        parameters = {
            "weekday_start_hour": 6.0,
            "weekday_heating_hours": 14.0,
            "non_workday_heating_hours": 10.0,
            "reduced_area_schedule_fraction": 0.5,
        }
        self.assertEqual(
            daily_schedule_hours(date(2021, 4, 20), parameters),
            (6.0, 13.0),
        )


if __name__ == "__main__":
    unittest.main()
