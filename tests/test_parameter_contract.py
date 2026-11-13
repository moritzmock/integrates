from __future__ import annotations

import unittest

from integrates_hackathon.challenge_inputs import (
    MAX_CALIBRATABLE_PARAMETERS,
    parameter_defaults,
    parameter_selection_defaults,
    parameter_specs,
)


class ParameterContractTests(unittest.TestCase):
    def test_no_selection_creates_baseline_passthrough(self) -> None:
        specs = parameter_specs(parameter_selection_defaults(), announce=False)
        self.assertEqual(specs, ())

    def test_keys_cannot_be_added_removed_or_renamed(self) -> None:
        selection = parameter_selection_defaults()
        selection.pop("weekday_heating_hours")
        with self.assertRaisesRegex(AssertionError, "keys are organizer-owned"):
            parameter_specs(selection, announce=False)

    def test_initial_value_must_respect_organizer_bounds(self) -> None:
        selection = parameter_selection_defaults()
        selection["air_exchange_multiplier"] = 1.5
        with self.assertRaisesRegex(AssertionError, "outside"):
            parameter_specs(selection, announce=False)

    def test_boolean_is_not_a_numeric_selection(self) -> None:
        selection = parameter_selection_defaults()
        selection["weekday_heating_hours"] = True
        with self.assertRaisesRegex(AssertionError, "not boolean"):
            parameter_specs(selection, announce=False)

    def test_all_twelve_parameters_can_be_activated(self) -> None:
        selection = parameter_defaults()
        specs = parameter_specs(selection, announce=False)
        self.assertEqual(len(specs), MAX_CALIBRATABLE_PARAMETERS)
        self.assertEqual(
            [spec.name for spec in specs],
            list(parameter_selection_defaults()),
        )
        self.assertEqual(
            {spec.name: spec.default for spec in specs},
            parameter_defaults(),
        )

    def test_arbitrary_subset_preserves_organizer_order(self) -> None:
        selection = parameter_selection_defaults()
        selection["window_u_value_multiplier"] = 1.0
        selection["weekday_start_hour"] = 7.0
        specs = parameter_specs(selection, announce=False)
        self.assertEqual(
            [spec.name for spec in specs],
            ["weekday_start_hour", "window_u_value_multiplier"],
        )


if __name__ == "__main__":
    unittest.main()
