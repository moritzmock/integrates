from __future__ import annotations

import unittest
from dataclasses import replace

from integrates_hackathon.bui import build_bui
from integrates_hackathon.building_inputs import FacadeGeometry
from integrates_hackathon.challenge_inputs import BUILDINGS


class BuiTests(unittest.TestCase):
    def test_unselected_parameters_use_all_organizer_defaults(self) -> None:
        case = BUILDINGS[0]
        fixed = build_bui(case, {})
        explicit = build_bui(case, case.defaults)
        self.assertEqual(fixed, explicit)

    def test_incomplete_geometry_is_rejected_before_simulation(self) -> None:
        case = BUILDINGS[0]
        incomplete = replace(
            case,
            fixed_inputs={**case.fixed_inputs, "geometry_complete": False},
        )
        with self.assertRaises((TypeError, ValueError, KeyError)):
            build_bui(incomplete, incomplete.defaults)

    def test_explicit_oriented_areas_are_preserved(self) -> None:
        case = BUILDINGS[0]
        complete_inputs = {
            **case.fixed_inputs,
            "geometry_complete": True,
            "total_height_m": 18.0,
            "roof_area_m2": 300.0,
            "ground_floor_area_m2": 300.0,
            "facades": {
                orientation: FacadeGeometry(100.0 + index, 20.0 + index, azimuth)
                for index, (orientation, azimuth) in enumerate(
                    (("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0))
                )
            },
        }
        bui = build_bui(replace(case, fixed_inputs=complete_inputs), case.defaults)
        surfaces = {surface["name"]: surface for surface in bui["building_surface"]}
        self.assertEqual(surfaces["Opaque S wall"]["area"], 102.0)
        self.assertEqual(surfaces["Window W"]["area"], 23.0)
        self.assertEqual(surfaces["Window E"]["orientation"]["azimuth"], 90.0)
        gains = bui["building_parameters"]["internal_gains"]
        self.assertEqual([gain["name"] for gain in gains], ["occupants", "appliances", "lighting"])
        self.assertAlmostEqual(
            sum(gain["full_load"] for gain in gains),
            case.fixed_inputs["internal_gains_w_m2"],
        )
        self.assertIn("heating_profile", bui["building_parameters"])
        self.assertEqual(
            bui["building_parameters"]["temperature_setpoints"]["heating_setback"],
            case.defaults["night_setback_temperature_c"],
        )

    def test_selected_parameters_modify_the_bui(self) -> None:
        case = BUILDINGS[0]
        parameters = {
            **case.defaults,
            "occupied_heating_setpoint_c": 19.25,
            "opaque_envelope_u_value_multiplier": 1.10,
            "window_u_value_multiplier": 0.90,
        }
        bui = build_bui(case, parameters)
        surfaces = {surface["name"]: surface for surface in bui["building_surface"]}
        self.assertEqual(
            bui["building_parameters"]["temperature_setpoints"]["heating_setpoint"],
            19.25,
        )
        self.assertAlmostEqual(
            surfaces["Roof"]["u_value"],
            case.fixed_inputs["roof_u_value"] * 1.10,
        )
        self.assertAlmostEqual(
            surfaces["Window N"]["u_value"],
            case.fixed_inputs["window_u_value"] * 0.90,
        )
        self.assertEqual(
            bui["building_parameters"]["heating_controls"][
                "heating_cutoff_temperature_c"
            ],
            case.defaults["outdoor_shutdown_temperature_c"],
        )


if __name__ == "__main__":
    unittest.main()

