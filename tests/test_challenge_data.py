from __future__ import annotations

import unittest
from statistics import fmean
from unittest import mock

from integrates_hackathon import energy_consumption
from integrates_hackathon.data_io import load_energy_source_rows
from integrates_hackathon.energy_consumption import (
    CLIPPED_QUALITY_FLAG,
    ENERGY_OBSERVATIONS,
    NON_HEATING_MONTHS,
    SMC_TO_KWH,
)
from integrates_hackathon.scoring import accuracy_score_from_wape


def _source_row(month: int, value: float, scope: str = "heating_only") -> dict:
    return {
        "building_id": "building_01",
        "year": 2018,
        "month": month,
        "source_value": value,
        "source_unit": "Smc",
        "end_use_scope": scope,
        "quality_flag": "reported",
    }


class ChallengeDataTests(unittest.TestCase):
    def test_consumption_is_complete_and_converted_to_kwh(self) -> None:
        self.assertEqual(len(ENERGY_OBSERVATIONS), 72)
        counts = {
            building_id: sum(
                row.building_id == building_id for row in ENERGY_OBSERVATIONS
            )
            for building_id in {row.building_id for row in ENERGY_OBSERVATIONS}
        }
        self.assertEqual(
            counts,
            {
                "building_01": 24,
                "building_02": 24,
                "building_03": 24,
            },
        )
        first = ENERGY_OBSERVATIONS[0]
        self.assertEqual((first.year, first.month, first.source_unit), (2018, 1, "Smc"))
        self.assertAlmostEqual(first.consumption_kwh, first.source_value * SMC_TO_KWH)
        self.assertGreater(first.consumption_kwh_m2_gfa, 0)

    def test_scored_values_are_combined_minus_summer_dhw_average(self) -> None:
        combined = {
            (row["building_id"], row["year"], row["month"]): row["source_value"]
            for row in load_energy_source_rows()
            if row["end_use_scope"] == "heating_plus_dhw"
        }
        self.assertTrue(combined)
        for row in ENERGY_OBSERVATIONS:
            self.assertEqual(row.end_use_scope, "heating_only")
            summer_dhw = fmean(
                combined[(row.building_id, row.year, month)] for month in (6, 7, 8)
            )
            expected = combined[(row.building_id, row.year, row.month)] - summer_dhw
            if row.quality_flag == CLIPPED_QUALITY_FLAG:
                self.assertLess(expected, 0)
                self.assertEqual(row.source_value, 0.0)
            else:
                self.assertAlmostEqual(row.source_value, expected, places=6)

    def test_negative_residuals_are_clipped_only_outside_heating_season(self) -> None:
        clipped = [row for row in ENERGY_OBSERVATIONS if row.quality_flag == CLIPPED_QUALITY_FLAG]
        self.assertTrue(clipped)
        self.assertTrue(all(row.month in NON_HEATING_MONTHS for row in clipped))
        self.assertTrue(all(row.consumption_kwh >= 0 for row in ENERGY_OBSERVATIONS))

    def test_negative_value_in_heating_season_is_rejected(self) -> None:
        rows = [_source_row(month, 100.0) for month in range(1, 13)]
        rows[2] = _source_row(3, -1.0)
        with mock.patch.object(energy_consumption, "load_energy_source_rows", return_value=rows):
            with self.assertRaisesRegex(ValueError, "inside the heating season"):
                energy_consumption._load_observations()

    def test_combined_only_year_is_rejected(self) -> None:
        rows = [_source_row(month, 100.0, "heating_plus_dhw") for month in range(1, 13)]
        with mock.patch.object(energy_consumption, "load_energy_source_rows", return_value=rows):
            with self.assertRaisesRegex(ValueError, "No heating-only values"):
                energy_consumption._load_observations()

    def test_example_score_is_positive_and_bounded(self) -> None:
        self.assertEqual(accuracy_score_from_wape(0), 100)
        self.assertEqual(accuracy_score_from_wape(12.5), 87.5)
        self.assertEqual(accuracy_score_from_wape(150), 0)


if __name__ == "__main__":
    unittest.main()
