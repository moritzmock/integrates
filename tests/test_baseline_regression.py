"""Slow check that the shipped baseline reproduces with the current code.

Run before every release that touches simulation code, defaults, inputs, or
pinned package versions (CI runs it on every push):

    $env:INTEGRATES_SLOW_TESTS = "1"
    python -m unittest tests.test_baseline_regression -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

from integrates_hackathon.baseline_cache import load_precomputed_baseline
from integrates_hackathon.notebook_workflow import initialize_challenge


@unittest.skipUnless(
    os.environ.get("INTEGRATES_SLOW_TESTS") == "1",
    "set INTEGRATES_SLOW_TESTS=1 to run pyBuildingEnergy regression checks",
)
class BaselineRegressionTests(unittest.TestCase):
    def test_default_simulation_reproduces_every_cached_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = initialize_challenge(
                backend="pbe",
                output_dir=directory,
                parallel_workers=os.cpu_count() or 1,
            )
            cached = load_precomputed_baseline(context.cases)
            try:
                for case in context.cases:
                    simulated = context.simulator.simulate(case, case.defaults)
                    self.assertEqual(len(simulated), len(cached[case.building_id]))
                    for fresh, stored in zip(
                        simulated, cached[case.building_id], strict=True
                    ):
                        with self.subTest(
                            building=case.building_id, year=fresh.year, month=fresh.month
                        ):
                            self.assertEqual(
                                (fresh.year, fresh.month), (stored.year, stored.month)
                            )
                            self.assertAlmostEqual(
                                fresh.consumption_kwh,
                                stored.consumption_kwh,
                                delta=1e-6 * stored.consumption_kwh,
                            )
            finally:
                context.simulator.close()


if __name__ == "__main__":
    unittest.main()
