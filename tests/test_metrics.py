from __future__ import annotations

import unittest

from integrates_hackathon.metrics import compare_monthly
from integrates_hackathon.models import Measurement, MonthlyConsumption


class MetricsTests(unittest.TestCase):
    def test_exact_match_has_zero_error(self) -> None:
        actual = (Measurement(2025, 1, 100.0), Measurement(2025, 2, 200.0))
        predicted = (
            MonthlyConsumption(2025, 1, 100.0),
            MonthlyConsumption(2025, 2, 200.0),
        )
        metrics = compare_monthly(actual, predicted)
        self.assertEqual(metrics.mean_absolute_percentage_error, 0.0)
        self.assertEqual(metrics.root_mean_squared_error_kwh, 0.0)

    def test_missing_month_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_monthly(
                (Measurement(2025, 1, 100.0), Measurement(2025, 2, 200.0)),
                (MonthlyConsumption(2025, 1, 100.0),),
            )


if __name__ == "__main__":
    unittest.main()

