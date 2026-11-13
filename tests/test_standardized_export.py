from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from integrates_hackathon.challenge_inputs import (
    parameter_defaults,
    parameter_selection_defaults,
)
from integrates_hackathon.notebook_workflow import (
    ScoreReport,
    configure_parameter_contract,
    export_standardized_results,
    initialize_challenge,
    run_calibration,
)


class StandardizedExportTests(unittest.TestCase):
    def test_export_includes_selection_hash_and_evaluation_trace(self) -> None:
        selection = parameter_selection_defaults()
        selection["weekday_heating_hours"] = parameter_defaults()[
            "weekday_heating_hours"
        ]
        with TemporaryDirectory() as directory:
            context = initialize_challenge(
                backend="demo",
                evaluation_budget=48,
                output_dir=directory,
                building_ids=("building_01",),
                calibration_years=(2018,),
            )
            context = configure_parameter_contract(context, selection)
            run = run_calibration(context)
            report = ScoreReport(
                submission=run.submission,
                building_scores=(),
                overall_score=0.0,
                summary_rows=({"building_id": "building_01"},),
                monthly_rows=({"building_id": "building_01"},),
            )

            paths = export_standardized_results(context, run, report)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            expected_hash = sha256(
                (Path(__file__).resolve().parents[1] / "calibration.py").read_bytes()
            ).hexdigest()
            with paths["evaluation_trace"].open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                trace = list(csv.DictReader(stream))

        self.assertEqual(payload["schema_version"], "1.2")
        self.assertEqual(payload["calibration_py_sha256"], expected_hash)
        self.assertEqual(
            payload["active_parameter_selection"],
            {"weekday_heating_hours": parameter_defaults()["weekday_heating_hours"]},
        )
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["evaluation_number"], "1")
        self.assertEqual(trace[0]["evaluation_budget"], "48")
        self.assertEqual(trace[0]["calibration_py_sha256"], expected_hash)
        self.assertEqual(trace[0]["is_final_candidate"], "True")


if __name__ == "__main__":
    unittest.main()
