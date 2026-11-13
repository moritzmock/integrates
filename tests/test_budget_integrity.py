from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from integrates_hackathon.challenge_inputs import parameter_defaults
from integrates_hackathon.notebook_workflow import (
    configure_parameter_contract,
    initialize_challenge,
    run_calibration,
)
from integrates_hackathon.submission_api import check_submission_imports

HEADER = 'TEAM_NAME = "Integrity"\nALGORITHM_NAME = "Test"\n'


class BudgetIntegrityTests(unittest.TestCase):
    def _run(self, body: str, budget: int = 3):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.py"
            path.write_text(HEADER + textwrap.dedent(body), encoding="utf-8")
            context = initialize_challenge(
                backend="demo",
                evaluation_budget=budget,
                output_dir=directory,
                building_ids=("building_01",),
                calibration_years=(2018,),
            )
            context = configure_parameter_contract(context, dict(parameter_defaults()))
            return run_calibration(context, submission_path=path)

    def test_participant_objective_hides_organizer_internals(self) -> None:
        run = self._run(
            """
            def calibrate(objective, space, budget, seed):
                for name in ("simulator", "case", "max_evaluations", "evaluation_count"):
                    assert not hasattr(objective, name), name
                try:
                    objective.max_evaluations = 10**6
                except AttributeError:
                    pass
                else:
                    raise AssertionError("objective accepted a new attribute")
                assert objective.remaining_budget == budget
                losses = objective.evaluate_many(
                    [[p.lower for p in space], [p.upper for p in space]]
                )
                assert len(losses) == 2 and objective.remaining_budget == budget - 2
                objective([p.default for p in space])
                return {p.name: p.default for p in space}
            """
        )
        self.assertEqual(run.cases["building_01"].evaluations, 3)

    def test_resetting_the_hidden_counter_is_detected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
            self._run(
                """
                def calibrate(objective, space, budget, seed):
                    inner = objective._evaluate.__closure__[0].cell_contents
                    for step in range(budget + 2):
                        inner.evaluation_count = 0
                        fraction = step / (budget + 1)
                        objective([p.lower + fraction * (p.upper - p.lower) for p in space])
                    return {p.name: p.default for p in space}
                """
            )

    def test_raising_the_hidden_budget_is_detected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
            self._run(
                """
                def calibrate(objective, space, budget, seed):
                    objective._evaluate.__closure__[0].cell_contents.max_evaluations = 10**6
                    for step in range(budget + 2):
                        fraction = step / (budget + 1)
                        objective([p.lower + fraction * (p.upper - p.lower) for p in space])
                    return {p.name: p.default for p in space}
                """
            )

    def test_forbidden_imports_are_rejected_before_execution(self) -> None:
        for source in (
            "import integrates_hackathon\n",
            "from integrates_hackathon.simulation import AnalyticDemoSimulator\n",
            "import pybuildingenergy.source.utils\n",
            "from multiprocessing import Pool\n",
            "from concurrent import futures\n",
            "import concurrent.futures as cf\n",
            "def calibrate():\n    import subprocess\n",
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ImportError, "not allowed"):
                    check_submission_imports(source)

    def test_ordinary_scientific_imports_are_allowed(self) -> None:
        check_submission_imports(
            "import math, random\nimport numpy as np\n"
            "from scipy.optimize import minimize\nfrom concurrent import interpreters\n"
        )


if __name__ == "__main__":
    unittest.main()
