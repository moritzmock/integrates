from __future__ import annotations

import unittest

from integrates_hackathon.challenge_inputs import BUILDINGS
from integrates_hackathon.data_io import available_amy_years


class BuildingInputTests(unittest.TestCase):
    def test_current_example_is_anonymized(self) -> None:
        self.assertEqual(
            [case.building_id for case in BUILDINGS],
            [f"building_{index:02d}" for index in range(1, 4)],
        )
        for case in BUILDINGS:
            self.assertNotIn("address", case.fixed_inputs)
            self.assertNotIn("pre_hackathon_name", case.fixed_inputs)
            self.assertAlmostEqual(
                case.fixed_inputs["simulation_latitude"], 46.49831458980591
            )
            self.assertAlmostEqual(
                case.fixed_inputs["simulation_longitude"], 11.354762256861017
            )
            self.assertEqual(sorted(case.weather_files), sorted({row.year for row in case.measurements}))
            self.assertTrue(all(path.is_file() for path in case.weather_files.values()))
            self.assertTrue(case.fixed_inputs["geometry_complete"])
            for facade in case.fixed_inputs["facades"].values():
                self.assertEqual(facade.wwr_basis, "dimensionless_pixel_area_ratio")
                self.assertIsNotNone(facade.gross_area_m2)
                self.assertIsNotNone(facade.wwr)
                self.assertAlmostEqual(
                    facade.window_area_m2,
                    facade.gross_area_m2 * facade.wwr,
                )
                self.assertAlmostEqual(
                    facade.opaque_area_m2 + facade.window_area_m2,
                    facade.gross_area_m2,
                )
        building_03 = BUILDINGS[-1]
        self.assertEqual(len(building_03.fixed_inputs["geometry_holes_latlon"]), 2)
        self.assertLess(building_03.fixed_inputs["footprint_area_m2"], 2924.85)

    def test_all_shipped_actual_weather_years_are_discovered(self) -> None:
        self.assertEqual(available_amy_years(), (2018, 2019))

    def test_every_case_exposes_all_twelve_available_parameters(self) -> None:
        for case in BUILDINGS:
            self.assertEqual(len(case.parameters), 12)
            self.assertGreaterEqual(len(case.measurements), 24)
            self.assertEqual(len(case.measurements) % 12, 0)


if __name__ == "__main__":
    unittest.main()

