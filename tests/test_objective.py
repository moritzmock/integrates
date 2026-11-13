from __future__ import annotations

import unittest

from integrates_hackathon.challenge_inputs import BUILDINGS
from integrates_hackathon.models import MonthlyConsumption
from integrates_hackathon.notebook_workflow import _building_seed
from integrates_hackathon.objective import CalibrationObjective


class _BatchSimulator:
    max_workers = 4

    def __init__(self):
        self.batch_sizes = []

    def simulate_many(self, case, parameter_sets):
        self.batch_sizes.append(len(parameter_sets))
        values = tuple(
            MonthlyConsumption(row.year, row.month, row.consumption_kwh)
            for row in case.measurements
        )
        return tuple(values for _parameters in parameter_sets)


class ParallelObjectiveTests(unittest.TestCase):
    def test_batch_deduplicates_and_preserves_order(self):
        case = BUILDINGS[0]
        simulator = _BatchSimulator()
        objective = CalibrationObjective(case, simulator, max_evaluations=2)
        vector = tuple(case.defaults[name] for name in case.parameter_names)
        losses = objective.evaluate_many((vector, vector))
        self.assertEqual(losses, (0.0, 0.0))
        self.assertEqual(objective.evaluation_count, 1)
        self.assertEqual(simulator.batch_sizes, [1])
        self.assertEqual(objective.max_batch_size, 4)
        self.assertEqual(objective.remaining_budget, 1)
        self.assertEqual(len(objective.evaluation_trace), 1)
        self.assertEqual(objective.evaluation_trace[0].loss, 0.0)
        self.assertEqual(
            dict(objective.evaluation_trace[0].parameters),
            case.defaults,
        )

    def test_building_seed_is_independent_of_selection_order(self):
        self.assertEqual(_building_seed(2026, "building_01"), 2026)
        self.assertEqual(_building_seed(2026, "building_02"), 2027)
        self.assertEqual(_building_seed(2026, "building_03"), 2028)


if __name__ == "__main__":
    unittest.main()
