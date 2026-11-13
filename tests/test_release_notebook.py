from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


class ReleaseNotebookTests(unittest.TestCase):
    def test_only_official_notebook_is_shipped_with_clean_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebooks = sorted((root / "notebooks").glob("*.ipynb"))
        self.assertEqual(
            [path.name for path in notebooks], ["hackathon_integrates.ipynb"]
        )

        notebook = json.loads(notebooks[0].read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ]
        self.assertTrue(code_cells)
        for cell in code_cells:
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            ast.parse("".join(cell["source"]))

        initialization = "".join(code_cells[0]["source"])
        self.assertIn("CALIBRATION_YEARS = [2018, 2019]", initialization)
        self.assertIn("EVALUATION_BUDGET = 48", initialization)
        self.assertIn(
            'building_ids=("building_01", "building_02", "building_03")',
            initialization,
        )
        calibration = "".join(code_cells[3]["source"])
        self.assertIn("calibratable_params_init = {", calibration)
        self.assertEqual(calibration.count('": None,'), 12)
        self.assertNotIn("required_params_init", calibration)
        self.assertNotIn("optional_params_init", calibration)


if __name__ == "__main__":
    unittest.main()
