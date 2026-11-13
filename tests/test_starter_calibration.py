from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


class _Parameter:
    def __init__(self, name: str, default: float) -> None:
        self.name = name
        self.default = default


class StarterCalibrationTests(unittest.TestCase):
    def test_starter_evaluates_and_returns_only_the_baseline(self) -> None:
        path = Path(__file__).resolve().parents[1] / "calibration.py"
        spec = importlib.util.spec_from_file_location("starter_calibration", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        evaluated: list[tuple[float, ...]] = []

        def objective(vector):
            evaluated.append(tuple(vector))
            return 99.0

        parameter_space = (_Parameter("a", 1.5), _Parameter("b", 2.5))
        result = module.calibrate(objective, parameter_space, budget=6, seed=2026)

        self.assertEqual(module.ALGORITHM_NAME, "Baseline passthrough")
        self.assertEqual(evaluated, [(1.5, 2.5)])
        self.assertEqual(result, {"a": 1.5, "b": 2.5})


if __name__ == "__main__":
    unittest.main()
