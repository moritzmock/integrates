from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from integrates_hackathon.challenge_inputs import parameter_defaults
from integrates_hackathon.notebook_workflow import (
    configure_parameter_contract,
    initialize_challenge,
    run_calibration,
)
from integrates_hackathon.objective import CalibrationObjective
from integrates_hackathon.submission_api import validate_parameter_result


def _context(directory: str, budget: int = 10):
    context = initialize_challenge(
        backend="demo",
        evaluation_budget=budget,
        output_dir=directory,
        building_ids=("building_01",),
        calibration_years=(2018,),
    )
    return configure_parameter_contract(context, dict(parameter_defaults()))


class CandidateValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.context = _context(self._directory.name)
        self.case = self.context.cases[0]
        self.names = list(self.case.parameter_names)
        self.objective = CalibrationObjective(
            self.case, self.context.simulator, max_evaluations=10
        )

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _vector(self, **changes):
        values = dict(self.case.defaults)
        values.update(changes)
        return [values[name] for name in self.names]

    def test_non_finite_and_non_numeric_candidates_are_rejected(self) -> None:
        for bad, error in (
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            ("8", TypeError),
            (True, TypeError),
            (None, TypeError),
        ):
            with self.subTest(value=bad):
                with self.assertRaises(error):
                    self.objective(self._vector(weekday_heating_hours=bad))
        self.assertEqual(self.objective.evaluation_count, 0)

    def test_numpy_scalars_are_accepted(self) -> None:
        vector = np.asarray(self._vector(), dtype=float)
        self.objective(vector)
        self.objective([np.array(value) for value in vector])
        self.assertEqual(self.objective.evaluation_count, 1)

    def test_equivalent_whole_hour_schedules_are_charged_once(self) -> None:
        self.objective.evaluate_many(
            [
                self._vector(weekday_heating_hours=7.6),
                self._vector(weekday_heating_hours=8.4),
                self._vector(weekday_start_hour=6.8),
                # 8 h x 0.53 = 4.24 h rounds to the same 4 h as 8 h x 0.5.
                self._vector(reduced_area_schedule_fraction=0.53),
            ]
        )
        self.assertEqual(self.objective.evaluation_count, 1)

        self.objective(self._vector(weekday_heating_hours=8.6))  # 9 h
        self.objective(self._vector(reduced_area_schedule_fraction=0.6))  # 4.8 -> 5 h
        self.objective(self._vector(air_exchange_multiplier=1.0 + 1e-6))
        self.assertEqual(self.objective.evaluation_count, 4)

    def test_returned_values_must_be_finite_real_numbers(self) -> None:
        result = dict(self.case.defaults)
        result["window_u_value_multiplier"] = np.float64(1.05)
        self.assertEqual(
            validate_parameter_result(self.case, result)["window_u_value_multiplier"], 1.05
        )
        for bad, error in ((True, TypeError), ("8", TypeError), (float("nan"), ValueError)):
            with self.subTest(value=bad):
                invalid = dict(self.case.defaults)
                invalid["weekday_heating_hours"] = bad
                with self.assertRaises(error):
                    validate_parameter_result(self.case, invalid)

    def test_equivalent_returned_candidate_is_marked_final_in_trace(self) -> None:
        path = Path(self._directory.name) / "calibration.py"
        path.write_text(
            textwrap.dedent(
                """
                TEAM_NAME = "Trace"
                ALGORITHM_NAME = "Test"

                def calibrate(objective, space, budget, seed):
                    vector = [7.6 if p.name == "weekday_heating_hours" else p.default for p in space]
                    objective(vector)
                    return {p.name: p.default for p in space}
                """
            ),
            encoding="utf-8",
        )
        run = run_calibration(self.context, submission_path=path)
        case_run = run.cases["building_01"]
        self.assertEqual(case_run.evaluations, 1)
        self.assertEqual(case_run.final_evaluation_number, 1)


if __name__ == "__main__":
    unittest.main()
